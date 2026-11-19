#include "TemporalPoseTracker.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <cmath>

namespace tmcad {

namespace {
constexpr double kPi = 3.14159265358979323846;

double rotationAngleDeg(const cv::Matx33d& R_diff) {
    const double tr = R_diff(0, 0) + R_diff(1, 1) + R_diff(2, 2);
    const double c = std::clamp((tr - 1.0) * 0.5, -1.0, 1.0);
    return std::acos(c) * 180.0 / kPi;
}
} // namespace

void TemporalPoseTracker::configure(double cad_diag_m,
                                    const cv::Vec3d& cad_min,
                                    const cv::Vec3d& cad_max,
                                    bool is_floor_standing,
                                    const std::vector<cv::Matx33d>& symmetry_matrices)
{
    _cad_diag_m = std::max(0.05, cad_diag_m);
    _cad_min = cad_min;
    _cad_max = cad_max;
    _is_floor_standing = is_floor_standing;
    _symmetry_matrices = symmetry_matrices;
    if (_symmetry_matrices.empty()) {
        _symmetry_matrices.push_back(cv::Matx33d::eye());
    }
    reset();
}

void TemporalPoseTracker::reset() noexcept
{
    _state = TrackState::SEARCHING;
    _confirm_streak = 0;
    _lost_streak = 0;
    _weak_frames = 0;
    _tracked_frames = 0;
    _has_floor = false;
    _floor_plane = cv::Vec4d(0.0, -1.0, 0.0, 0.0);
    _track_pose = ObjectPose{};
    _prev_global_pose = ObjectPose{};
    _prev_gray.release();
    _kpts_2d.clear();
    _kpts_3d_cad.clear();
}

double TemporalPoseTracker::symmetryAwareRotationDiffDeg(const cv::Matx33d& R_prev,
                                                         const cv::Matx33d& R_curr) const
{
    const cv::Matx33d R_rel = R_prev.t() * R_curr;
    double best_deg = rotationAngleDeg(R_rel);
    for (const auto& S : _symmetry_matrices) {
        const double d = rotationAngleDeg(R_rel * S);
        if (d < best_deg) best_deg = d;
    }
    return best_deg;
}

bool TemporalPoseTracker::passesPoseJumpGuard(const ObjectPose& cand) const
{
    if (!_track_pose.valid || !cand.valid) return true;
    const double max_dt = std::max(0.08, 0.40 * _cad_diag_m);
    const double dt = cv::norm(cand.t - _track_pose.t);
    if (dt > max_dt) return false;
    const double dtheta = symmetryAwareRotationDiffDeg(_track_pose.R, cand.R);
    if (dtheta > 25.0) return false;
    return true;
}

void TemporalPoseTracker::seedKeypoints2D(const cv::Mat& gray,
                                          const ObjectPose& pose,
                                          const CameraIntrinsics& intr,
                                          const PoseRefinerCallbacks& cb)
{
    _kpts_2d.clear();
    _kpts_3d_cad.clear();
    if (gray.empty() || !pose.valid || !cb.getVisibleCadPoints) return;

    const int w = gray.cols;
    const int h = gray.rows;
    const double fx = (intr.fx > 100.0) ? intr.fx : 608.215;
    const double fy = (intr.fy > 100.0) ? intr.fy : 608.128;
    const double cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : 0.5 * w);
    const double cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : 0.5 * h);

    std::vector<cv::Vec3d> vis_pts;
    cb.getVisibleCadPoints(pose, intr, w, h, vis_pts);
    if (vis_pts.size() < 20) return;

    cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
    std::vector<cv::Point2f> proj_uv;
    proj_uv.reserve(vis_pts.size());
    std::vector<cv::Point3f> proj_cad;
    proj_cad.reserve(vis_pts.size());

    for (const auto& vp : vis_pts) {
        const cv::Vec3d pc = pose.R * vp + pose.t;
        if (pc[2] <= 0.10) continue;
        const float u = static_cast<float>(fx * pc[0] / pc[2] + cx);
        const float v = static_cast<float>(fy * pc[1] / pc[2] + cy);
        const int ui = static_cast<int>(std::round(u));
        const int vi = static_cast<int>(std::round(v));
        if (ui >= 2 && ui < w - 2 && vi >= 2 && vi < h - 2) {
            cv::circle(mask, cv::Point(ui, vi), 3, cv::Scalar(255), -1);
            proj_uv.emplace_back(u, v);
            proj_cad.emplace_back(static_cast<float>(vp[0]), static_cast<float>(vp[1]), static_cast<float>(vp[2]));
        }
    }
    if (proj_uv.size() < 20) return;

    // Erode slightly so keypoints stay well inside the object boundary (away from background edges)
    cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    if (cv::countNonZero(mask) < 64) return;

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray, corners, 80, 0.02, 7.0, mask);
    if (corners.size() < 12) return;

    _kpts_2d.reserve(corners.size());
    _kpts_3d_cad.reserve(corners.size());
    for (const auto& pt : corners) {
        float best_d2 = 36.0f; // within 6 px of a visible CAD surface point
        int best_idx = -1;
        for (size_t i = 0; i < proj_uv.size(); ++i) {
            const float du = pt.x - proj_uv[i].x;
            const float dv = pt.y - proj_uv[i].y;
            const float d2 = du * du + dv * dv;
            if (d2 < best_d2) {
                best_d2 = d2;
                best_idx = static_cast<int>(i);
            }
        }
        if (best_idx >= 0) {
            _kpts_2d.push_back(pt);
            _kpts_3d_cad.push_back(proj_cad[best_idx]);
        }
    }
    _prev_gray = gray.clone();
}

bool TemporalPoseTracker::trackKeypointsPyrLKAndPnP(const cv::Mat& curr_gray,
                                                    const CameraIntrinsics& intr,
                                                    cv::Matx33d& out_R,
                                                    cv::Vec3d& out_t)
{
    if (_prev_gray.empty() || curr_gray.empty() || _prev_gray.size() != curr_gray.size() ||
        _kpts_2d.size() < 15 || _kpts_2d.size() != _kpts_3d_cad.size()) {
        return false;
    }

    std::vector<cv::Point2f> next_pts, back_pts;
    std::vector<uchar> status_fwd, status_bwd;
    std::vector<float> err_fwd, err_bwd;

    cv::calcOpticalFlowPyrLK(_prev_gray, curr_gray, _kpts_2d, next_pts, status_fwd, err_fwd,
                             cv::Size(15, 15), 2);
    cv::calcOpticalFlowPyrLK(curr_gray, _prev_gray, next_pts, back_pts, status_bwd, err_bwd,
                             cv::Size(15, 15), 2);

    std::vector<cv::Point2f> good_2d;
    std::vector<cv::Point3f> good_3d;
    good_2d.reserve(_kpts_2d.size());
    good_3d.reserve(_kpts_3d_cad.size());

    for (size_t i = 0; i < _kpts_2d.size(); ++i) {
        if (!status_fwd[i] || !status_bwd[i]) continue;
        const float fb_err = static_cast<float>(cv::norm(_kpts_2d[i] - back_pts[i]));
        if (fb_err < 1.5f &&
            next_pts[i].x >= 0.0f && next_pts[i].x < curr_gray.cols &&
            next_pts[i].y >= 0.0f && next_pts[i].y < curr_gray.rows) {
            good_2d.push_back(next_pts[i]);
            good_3d.push_back(_kpts_3d_cad[i]);
        }
    }

    _kpts_2d = std::move(good_2d);
    _kpts_3d_cad = std::move(good_3d);
    _prev_gray = curr_gray.clone();

    if (_kpts_2d.size() < 15) return false;

    const double fx = (intr.fx > 100.0) ? intr.fx : 608.215;
    const double fy = (intr.fy > 100.0) ? intr.fy : 608.128;
    const double cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : 0.5 * curr_gray.cols);
    const double cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : 0.5 * curr_gray.rows);

    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    cv::Mat rvec, tvec;
    cv::Rodrigues(_track_pose.R, rvec);
    tvec = (cv::Mat_<double>(3, 1) << _track_pose.t[0], _track_pose.t[1], _track_pose.t[2]);

    std::vector<int> inliers;
    const bool ok = cv::solvePnPRansac(
        _kpts_3d_cad, _kpts_2d, K, cv::Mat(), rvec, tvec,
        true, 60, 3.0f, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
    if (!ok || inliers.size() < 15) return false;

    cv::Mat R_mat;
    cv::Rodrigues(rvec, R_mat);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out_R(r, c) = R_mat.at<double>(r, c);
        }
    }
    out_t = cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    // Keep only PnP inlier keypoints for next frame
    std::vector<cv::Point2f> in_2d;
    std::vector<cv::Point3f> in_3d;
    in_2d.reserve(inliers.size());
    in_3d.reserve(inliers.size());
    for (int idx : inliers) {
        in_2d.push_back(_kpts_2d[idx]);
        in_3d.push_back(_kpts_3d_cad[idx]);
    }
    _kpts_2d = std::move(in_2d);
    _kpts_3d_cad = std::move(in_3d);
    return true;
}

bool TemporalPoseTracker::trackFrame(const cv::Mat& rgb,
                                     const cv::Mat& depth_z16,
                                     const CameraIntrinsics& intr,
                                     const PoseRefinerCallbacks& cb,
                                     ObjectPose& out_pose,
                                     std::vector<cv::Vec3f>& out_local_fg)
{
    out_local_fg.clear();
    if (!isTrackingActive() || depth_z16.empty()) return false;

    const int w = depth_z16.cols;
    const int h = depth_z16.rows;
    const double fx = (intr.fx > 100.0) ? intr.fx : 608.215;
    const double fy = (intr.fy > 100.0) ? intr.fy : 608.128;
    const double cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : 0.5 * w);
    const double cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : 0.5 * h);
    const double scale = (intr.depth_scale > 0.0) ? intr.depth_scale : 0.001;
    const double diag = _cad_diag_m;
    const double max_ws_z = _is_floor_standing ? 2.95 : std::clamp(diag * 3.8 + 0.15, 0.95, 2.50);

    // 1. Local ROI + Background-Depth-Gated Mask Cleanup (MORPH_OPEN + Largest Connected Component)
    const cv::Rect2f& pb = _track_pose.bbox_2d;
    const int pad_x = std::max(48, static_cast<int>(std::round(0.35f * pb.width)));
    const int pad_y = std::max(48, static_cast<int>(std::round(0.35f * pb.height)));
    const int x0 = (pb.width > 8.0f) ? std::max(0, static_cast<int>(std::floor(pb.x)) - pad_x) : 0;
    const int y0 = (pb.height > 8.0f) ? std::max(0, static_cast<int>(std::floor(pb.y)) - pad_y) : 0;
    const int x1 = (pb.width > 8.0f) ? std::min(w, static_cast<int>(std::ceil(pb.x + pb.width)) + pad_x) : w;
    const int y1 = (pb.height > 8.0f) ? std::min(h, static_cast<int>(std::ceil(pb.y + pb.height)) + pad_y) : h;
    if (x1 <= x0 + 8 || y1 <= y0 + 8) return false;

    const cv::Matx33d R_inv = _track_pose.R.t();
    const double obb_pad = _is_floor_standing
        ? std::clamp(0.16 * diag, 0.040, 0.120)
        : std::clamp(0.075 * diag, 0.010, 0.018);
    const cv::Vec3d obb_min = _cad_min - cv::Vec3d(obb_pad, obb_pad, obb_pad);
    const cv::Vec3d obb_max = _cad_max + cv::Vec3d(obb_pad, obb_pad, obb_pad);
    const double floor_norm = cv::norm(cv::Vec3d(_floor_plane[0], _floor_plane[1], _floor_plane[2]));
    const bool use_floor_filter = _has_floor && (floor_norm > 0.5);
    const double z_bg_cut = _track_pose.t[2] + 0.60 * diag;

    const int roi_w = x1 - x0;
    const int roi_h = y1 - y0;
    cv::Mat roi_fg_mask = cv::Mat::zeros(roi_h, roi_w, CV_8UC1);
    const long long area = static_cast<long long>(roi_w) * roi_h;
    const int stride = (area < 90000) ? 1 : 2;

    for (int v = y0; v < y1; v += stride) {
        const uint16_t* drow = depth_z16.ptr<uint16_t>(v);
        uint8_t* mrow = roi_fg_mask.ptr<uint8_t>(v - y0);
        for (int u = x0; u < x1; u += stride) {
            const uint16_t dr = drow[u];
            if (dr == 0) continue;
            const double z = static_cast<double>(dr) * scale;
            if (z < 0.15 || z > max_ws_z || z > z_bg_cut) continue;
            const double x = (static_cast<double>(u) - cx) * z / fx;
            const double y = (static_cast<double>(v) - cy) * z / fy;
            if (use_floor_filter) {
                const double signed_d = _floor_plane[0] * x + _floor_plane[1] * y +
                                        _floor_plane[2] * z + _floor_plane[3];
                if (std::abs(signed_d) <= 0.018) continue;
                if (_floor_plane[1] < -0.5 && signed_d > 0.018) continue;
            }
            const cv::Vec3d pc = R_inv * (cv::Vec3d(x, y, z) - _track_pose.t);
            if (pc[0] >= obb_min[0] && pc[0] <= obb_max[0] &&
                pc[1] >= obb_min[1] && pc[1] <= obb_max[1] &&
                pc[2] >= obb_min[2] && pc[2] <= obb_max[2]) {
                mrow[u - x0] = 255;
            }
        }
    }

    // Morphological Opening + Largest Connected Component cleanup when stride == 1
    if (stride == 1 && cv::countNonZero(roi_fg_mask) >= 64) {
        cv::Mat cleaned;
        cv::morphologyEx(roi_fg_mask, cleaned, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
        cv::Mat labels, stats, centroids;
        const int n_cc = cv::connectedComponentsWithStats(cleaned, labels, stats, centroids, 8, CV_32S);
        if (n_cc > 2) {
            int best_cc = -1, best_area = 0;
            for (int c = 1; c < n_cc; ++c) {
                const int a = stats.at<int>(c, cv::CC_STAT_AREA);
                if (a > best_area) { best_area = a; best_cc = c; }
            }
            if (best_cc > 0 && best_area >= 32) {
                roi_fg_mask = (labels == best_cc);
            }
        }
    }

    out_local_fg.reserve(2048);
    for (int v = y0; v < y1; v += stride) {
        const uint16_t* drow = depth_z16.ptr<uint16_t>(v);
        const uint8_t* mrow = roi_fg_mask.ptr<uint8_t>(v - y0);
        for (int u = x0; u < x1; u += stride) {
            if (mrow[u - x0] == 0) continue;
            const double z = static_cast<double>(drow[u]) * scale;
            const double x = (static_cast<double>(u) - cx) * z / fx;
            const double y = (static_cast<double>(v) - cy) * z / fy;
            out_local_fg.emplace_back(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
        }
    }

    const size_t min_fg_pts = _is_floor_standing ? 48 : 24;
    if (out_local_fg.size() < min_fg_pts) return false;

    // 2. Sparse 2D Keypoint Tracking (PyrLK + FB < 1.5 px + solvePnPRansac)
    cv::Mat curr_gray;
    if (!rgb.empty() && !_is_floor_standing) {
        if (rgb.channels() == 3) cv::cvtColor(rgb, curr_gray, cv::COLOR_BGR2GRAY);
        else curr_gray = rgb;
    }

    ObjectPose cand = _track_pose;
    cand.valid = true;
    cand.method = "warm_start_icp";
    cand.valid_depth_pts = static_cast<int>(out_local_fg.size());
    cand.sparse_depth_fallback = (out_local_fg.size() < 250);

    bool pnp_used = false;
    if (!curr_gray.empty() && !_is_floor_standing) {
        cv::Matx33d R_pnp = cand.R;
        cv::Vec3d t_pnp = cand.t;
        if (trackKeypointsPyrLKAndPnP(curr_gray, intr, R_pnp, t_pnp)) {
            ObjectPose pnp_probe = cand;
            pnp_probe.R = R_pnp;
            pnp_probe.t = t_pnp;
            if (passesPoseJumpGuard(pnp_probe)) {
                cand.R = R_pnp;
                cand.t = t_pnp;
                pnp_used = true;
            }
        }
    }

    // 3. Adaptive Warm-Start 3D ICP + 2D Silhouette (only when eta_depth < 0.65)
    const double eta_depth = cb.computeDepthCoverage
        ? cb.computeDepthCoverage(depth_z16, cand, intr)
        : 0.0;

    if (cb.runLocalIcp && (eta_depth >= 0.30 || !pnp_used) && out_local_fg.size() >= 20) {
        cb.runLocalIcp(out_local_fg, cand, intr);
    }

    double sil_cost = 0.0;
    if (cb.runLocalSilhouette && !_is_floor_standing && !rgb.empty() && eta_depth < 0.65) {
        sil_cost = cb.runLocalSilhouette(rgb, depth_z16, _floor_plane, _has_floor, cand, intr);
    }

    if (cb.finalizeGeometry) {
        cb.finalizeGeometry(cand, intr, w, h);
    }

    // 4. Symmetry-Aware Pose Jump Guard
    if (!passesPoseJumpGuard(cand)) {
        return false;
    }

    // 5. Independent Verification
    if (cb.evaluatePose && !cb.evaluatePose(out_local_fg, depth_z16, sil_cost, cand, intr)) {
        return false;
    }

    // Tracking Sanity Watchdog: exit if quality degrades 3 frames in a row
    const bool weak_frame = (sil_cost > 6.8) || (cand.inliers_count < 24 && cand.confidence < 0.58f);
    _weak_frames = weak_frame ? (_weak_frames + 1) : 0;
    if (_weak_frames >= 3) {
        reset();
        return false;
    }

    // Re-seed 2D keypoints whenever count drops below 25
    if (!curr_gray.empty() && !_is_floor_standing && _kpts_2d.size() < 25) {
        seedKeypoints2D(curr_gray, cand, intr, cb);
    }

    _state = TrackState::TRACKING;
    _track_pose = cand;
    _lost_streak = 0;
    ++_tracked_frames;
    out_pose = std::move(cand);
    return true;
}

void TemporalPoseTracker::updateFromGlobal(ObjectPose& global_pose,
                                           const cv::Mat& rgb,
                                           const cv::Vec4d& floor_plane,
                                           bool floor_detected,
                                           const CameraIntrinsics& intr,
                                           const PoseRefinerCallbacks& cb)
{
    if (!_enabled) return;

    if (global_pose.valid) {
        if (_state == TrackState::TRACKING || _state == TrackState::HOLD) {
            _track_pose = global_pose;
            _state = TrackState::TRACKING;
            _lost_streak = 0;
            _weak_frames = 0;
        } else {
            bool consistent = false;
            if (_prev_global_pose.valid) {
                const double dt = cv::norm(global_pose.t - _prev_global_pose.t);
                const double dtheta = symmetryAwareRotationDiffDeg(_prev_global_pose.R, global_pose.R);
                consistent = (dt <= std::max(0.12, 0.60 * _cad_diag_m) && dtheta <= 25.0);
            }
            _confirm_streak = consistent ? (_confirm_streak + 1) : 1;
            _prev_global_pose = global_pose;
            const bool strong_first = (global_pose.confidence >= 0.72f && global_pose.inliers_count >= 100);
            if (_confirm_streak >= 2 || strong_first) {
                _state = TrackState::TRACKING;
                _track_pose = global_pose;
                _lost_streak = 0;
                _weak_frames = 0;
                _tracked_frames = 0;
            }
        }
        if (floor_detected) {
            _has_floor = true;
            _floor_plane = floor_plane;
        }
        if (_state == TrackState::TRACKING && !rgb.empty() && !_is_floor_standing && _kpts_2d.size() < 25) {
            cv::Mat gray;
            if (rgb.channels() == 3) cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
            else gray = rgb;
            seedKeypoints2D(gray, _track_pose, intr, cb);
        }
    } else if (_state == TrackState::TRACKING || _state == TrackState::HOLD) {
        ++_lost_streak;
        if (_lost_streak <= 3 && _track_pose.valid) {
            _state = TrackState::HOLD;
            global_pose = _track_pose;
            global_pose.confidence = std::max(0.50f, _track_pose.confidence - 0.05f * _lost_streak);
            global_pose.method = "TRACKED/HOLD";
        } else {
            _state = TrackState::LOST;
            reset();
        }
    } else {
        _confirm_streak = 0;
        _prev_global_pose = ObjectPose{};
    }
}

} // namespace tmcad

