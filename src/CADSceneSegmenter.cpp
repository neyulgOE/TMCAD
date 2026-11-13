#include "CADSceneSegmenter.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <queue>
#include <random>
#include <regex>

namespace tmcad {

namespace {

struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& o) const noexcept {
        return x == o.x && y == o.y && z == o.z;
    }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& k) const noexcept {
        std::size_t h1 = std::hash<int>{}(k.x);
        std::size_t h2 = std::hash<int>{}(k.y);
        std::size_t h3 = std::hash<int>{}(k.z);
        return h1 ^ (h2 * 0x9e3779b9 + (h1 << 6) + (h1 >> 2)) ^ (h3 * 0xc2b2ae35);
    }
};

static bool parseClassicDouble(const std::string& s, double& out_val) {
    std::istringstream iss(s);
    iss.imbue(std::locale::classic());
    iss >> out_val;
    return !iss.fail();
}

static bool extractJsonNumber(const std::string& content, const std::string& key, double& out_val) {
    std::regex re("\"" + key + "\"\\s*:\\s*([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)");
    std::smatch match;
    if (std::regex_search(content, match, re) && match.size() >= 2) {
        return parseClassicDouble(match[1].str(), out_val);
    }
    return false;
}

static bool extractJsonString(const std::string& content, const std::string& key, std::string& out_str) {
    std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (std::regex_search(content, match, re) && match.size() >= 2) {
        out_str = match[1].str();
        return true;
    }
    return false;
}

static bool extractJsonVec3(const std::string& content, const std::string& key, double& x, double& y, double& z) {
    std::regex re("\"" + key + "\"\\s*:\\s*\\[\\s*([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\\s*,\\s*([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\\s*,\\s*([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\\s*\\]");
    std::smatch match;
    if (std::regex_search(content, match, re) && match.size() >= 4) {
        return parseClassicDouble(match[1].str(), x) &&
               parseClassicDouble(match[2].str(), y) &&
               parseClassicDouble(match[3].str(), z);
    }
    return false;
}

static float computeQuantileDiag(const std::vector<cv::Vec3f>& pts, double q_lo = 0.05, double q_hi = 0.95) {
    if (pts.size() < 12) return 0.0f;
    const size_t step = std::max<size_t>(1, pts.size() / 512);
    std::vector<float> xs, ys, zs;
    xs.reserve(pts.size() / step + 1);
    ys.reserve(pts.size() / step + 1);
    zs.reserve(pts.size() / step + 1);
    for (size_t i = 0; i < pts.size(); i += step) {
        xs.push_back(pts[i][0]);
        ys.push_back(pts[i][1]);
        zs.push_back(pts[i][2]);
    }
    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());
    std::sort(zs.begin(), zs.end());
    const size_t n = xs.size();
    const size_t lo = std::min(n - 1, static_cast<size_t>(q_lo * n));
    const size_t hi = std::min(n - 1, static_cast<size_t>(q_hi * n));
    const float dx = xs[hi] - xs[lo];
    const float dy = ys[hi] - ys[lo];
    const float dz = zs[hi] - zs[lo];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

static void computePcaSpread(const std::vector<cv::Vec3f>& pts, float& s0, float& s1, float& s2) {
    s0 = s1 = s2 = 0.0f;
    if (pts.size() < 16) return;
    const size_t step = std::max<size_t>(1, pts.size() / 200);
    cv::Vec3d mu(0.0, 0.0, 0.0);
    size_t n = 0;
    for (size_t i = 0; i < pts.size(); i += step) {
        mu += cv::Vec3d(pts[i][0], pts[i][1], pts[i][2]);
        ++n;
    }
    if (n < 8) return;
    mu *= (1.0 / static_cast<double>(n));
    cv::Matx33d cov = cv::Matx33d::zeros();
    for (size_t i = 0; i < pts.size(); i += step) {
        const cv::Vec3d d(pts[i][0] - mu[0], pts[i][1] - mu[1], pts[i][2] - mu[2]);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                cov(r, c) += d[r] * d[c];
    }
    cov *= (1.0 / static_cast<double>(n));
    cv::Mat ev, evec;
    cv::eigen(cv::Mat(cov), ev, evec);
    s0 = static_cast<float>(std::sqrt(std::max(0.0, ev.at<double>(0))));
    s1 = static_cast<float>(std::sqrt(std::max(0.0, ev.at<double>(1))));
    s2 = static_cast<float>(std::sqrt(std::max(0.0, ev.at<double>(2))));
}

static std::vector<cv::Vec3f> voxelDownsampleCentroids(const std::vector<cv::Vec3f>& pts, float leaf) {
    if (pts.empty() || leaf <= 1e-5f) return pts;
    const float inv = 1.0f / leaf;
    struct Acc {
        cv::Vec3d sum{0.0, 0.0, 0.0};
        int count{0};
    };
    std::unordered_map<VoxelKey, Acc, VoxelKeyHash> grid;
    grid.reserve(pts.size() / 4 + 16);
    for (const auto& p : pts) {
        VoxelKey k{
            static_cast<int>(std::floor(p[0] * inv)),
            static_cast<int>(std::floor(p[1] * inv)),
            static_cast<int>(std::floor(p[2] * inv))
        };
        auto& acc = grid[k];
        acc.sum += cv::Vec3d(p[0], p[1], p[2]);
        ++acc.count;
    }
    std::vector<cv::Vec3f> out;
    out.reserve(grid.size());
    for (const auto& kv : grid) {
        const double inv_n = 1.0 / static_cast<double>(kv.second.count);
        out.emplace_back(static_cast<float>(kv.second.sum[0] * inv_n),
                         static_cast<float>(kv.second.sum[1] * inv_n),
                         static_cast<float>(kv.second.sum[2] * inv_n));
    }
    return out;
}

static void extractStridedScenePoints(const cv::Mat& depth_z16,
                                      const CameraIntrinsics& intr,
                                      int stride,
                                      float min_depth_m,
                                      float max_depth_m,
                                      float left_margin_ratio,
                                      float right_margin_ratio,
                                      std::vector<cv::Vec3f>& out_pts) {
    out_pts.clear();
    if (depth_z16.empty() || depth_z16.type() != CV_16UC1) return;
    const int w = depth_z16.cols;
    const int h = depth_z16.rows;
    stride = std::max(1, stride);

    const float fx = (intr.fx > 100.0) ? static_cast<float>(intr.fx)
                                       : ((w == 1024 && h == 576) ? 739.41f : 0.95f * w);
    const float fy = (intr.fy > 100.0) ? static_cast<float>(intr.fy) : fx;
    const float cx = (intr.ppx > 10.0) ? static_cast<float>(intr.ppx)
                                       : ((intr.cx > 10.0) ? static_cast<float>(intr.cx) : (0.5f * w));
    const float cy = (intr.ppy > 10.0) ? static_cast<float>(intr.ppy)
                                       : ((intr.cy > 10.0) ? static_cast<float>(intr.cy) : (0.5f * h));
    const float scale = (intr.depth_scale > 0.0) ? static_cast<float>(intr.depth_scale) : 0.001f;

    const int x0 = std::clamp(static_cast<int>(std::round(left_margin_ratio * w)), 0, std::max(0, w - 1));
    const int x1 = std::clamp(static_cast<int>(std::round((1.0f - right_margin_ratio) * w)), x0 + 1, w);
    out_pts.reserve(static_cast<size_t>((x1 - x0) / stride + 1) * static_cast<size_t>(h / stride + 1));

    for (int v = 0; v < h; v += stride) {
        const uint16_t* row_ptr = depth_z16.ptr<uint16_t>(v);
        for (int u = x0; u < x1; u += stride) {
            const uint16_t d_raw = row_ptr[u];
            if (d_raw == 0) continue;
            const float z = d_raw * scale;
            if (z < min_depth_m || z > max_depth_m) continue;
            out_pts.emplace_back((static_cast<float>(u) - cx) * z / fx,
                                 (static_cast<float>(v) - cy) * z / fy,
                                 z);
        }
    }
}

static void extractGlobalSupportPlanesRANSAC(const std::vector<cv::Vec3f>& pts,
                                             float bounding_diag_m,
                                             double dist_thresh,
                                             std::vector<cv::Vec4d>& out_planes) {
    out_planes.clear();
    if (pts.size() < 320) return;

    std::vector<cv::Vec3f> rem = pts;
    std::mt19937 rng(42);
    const float min_plane_diag = std::max(0.30f, 1.75f * bounding_diag_m);

    for (int pass = 0; pass < 3; ++pass) {
        if (rem.size() < 320) break;
        const size_t cand_step = std::max<size_t>(1, rem.size() / 4096);
        std::vector<cv::Vec3f> cands;
        cands.reserve(rem.size() / cand_step + 1);
        for (size_t i = 0; i < rem.size(); i += cand_step) cands.push_back(rem[i]);
        if (cands.size() < 120) break;

        std::uniform_int_distribution<size_t> dist(0, cands.size() - 1);
        const size_t eval_step = std::max<size_t>(1, cands.size() / 2048);
        size_t best_cand_hits = 0;
        cv::Vec4d best_plane(0.0, -1.0, 0.0, 0.0);

        for (int iter = 0; iter < 180; ++iter) {
            const cv::Vec3f& p1 = cands[dist(rng)];
            const cv::Vec3f& p2 = cands[dist(rng)];
            const cv::Vec3f& p3 = cands[dist(rng)];
            cv::Vec3f n = (p2 - p1).cross(p3 - p1);
            const float norm_n = cv::norm(n);
            if (norm_n < 1e-6f) continue;
            n /= norm_n;
            if (n[1] > 0.0f) n = -n;
            const double d = -(n[0] * p1[0] + n[1] * p1[1] + n[2] * p1[2]);
            size_t count = 0;
            for (size_t k = 0; k < cands.size(); k += eval_step) {
                const auto& q = cands[k];
                if (std::abs(n[0] * q[0] + n[1] * q[1] + n[2] * q[2] + d) <= dist_thresh) ++count;
            }
            if (count > best_cand_hits) {
                best_cand_hits = count;
                best_plane = cv::Vec4d(n[0], n[1], n[2], d);
            }
        }
        if (best_cand_hits < 40) break;

        std::vector<cv::Vec3f> inl_pts, nxt_rem;
        inl_pts.reserve(rem.size() / 4);
        nxt_rem.reserve(rem.size());
        for (const auto& q : rem) {
            if (std::abs(best_plane[0] * q[0] + best_plane[1] * q[1] + best_plane[2] * q[2] + best_plane[3]) <= dist_thresh) {
                inl_pts.push_back(q);
            } else {
                nxt_rem.push_back(q);
            }
        }
        const float inl_qdiag = computeQuantileDiag(inl_pts, 0.05, 0.95);
        if (inl_pts.size() < std::max<size_t>(450, pts.size() / 8) || inl_qdiag <= min_plane_diag) {
            break;
        }

        // SVD refinement on global support plane inliers
        const size_t svd_step = std::max<size_t>(1, inl_pts.size() / 4096);
        cv::Vec3d mean(0.0, 0.0, 0.0);
        size_t svd_cnt = 0;
        for (size_t i = 0; i < inl_pts.size(); i += svd_step) {
            mean += cv::Vec3d(inl_pts[i][0], inl_pts[i][1], inl_pts[i][2]);
            ++svd_cnt;
        }
        if (svd_cnt >= 20) {
            mean *= (1.0 / static_cast<double>(svd_cnt));
            cv::Matx33d cov = cv::Matx33d::zeros();
            for (size_t i = 0; i < inl_pts.size(); i += svd_step) {
                const auto& q = inl_pts[i];
                cv::Vec3d dq(q[0] - mean[0], q[1] - mean[1], q[2] - mean[2]);
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        cov(r, c) += dq[r] * dq[c];
            }
            cv::Mat eig_vals, eig_vecs;
            cv::eigen(cv::Mat(cov), eig_vals, eig_vecs);
            cv::Vec3d n_svd(eig_vecs.at<double>(2, 0), eig_vecs.at<double>(2, 1), eig_vecs.at<double>(2, 2));
            if (n_svd[1] > 0.0) n_svd = -n_svd;
            n_svd *= (1.0 / std::max(1e-9, cv::norm(n_svd)));
            best_plane = cv::Vec4d(n_svd[0], n_svd[1], n_svd[2], -n_svd.dot(mean));
        }

        out_planes.push_back(best_plane);
        if (nxt_rem.size() < 120) break;
        rem = std::move(nxt_rem);
    }
}

} // anonymous namespace

bool CADSceneSegmenter::loadProfileConfig(const std::string& profile_json_path) {
    std::ifstream ifs(profile_json_path);
    if (!ifs.is_open()) {
        return false;
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string content = ss.str();

    double lx = 0.0, ly = 0.0, lz = 0.0, diag = 0.0;
    if (extractJsonVec3(content, "extents_xyz_m", lx, ly, lz) ||
        (extractJsonNumber(content, "extent_x_m", lx) &&
         extractJsonNumber(content, "extent_y_m", ly) &&
         extractJsonNumber(content, "extent_z_m", lz))) {
        _expected_width_m  = static_cast<float>(std::min(lx, ly));
        _expected_length_m = static_cast<float>(std::max(lx, ly));
        _expected_height_m = static_cast<float>(lz);
    }
    if (extractJsonNumber(content, "real_bounding_diag_m", diag) && diag > 0.01) {
        _bounding_diag_m = static_cast<float>(diag);
    } else {
        _bounding_diag_m = std::sqrt(_expected_length_m * _expected_length_m +
                                     _expected_width_m * _expected_width_m +
                                     _expected_height_m * _expected_height_m);
    }

    std::string placement_mode;
    if (extractJsonString(content, "placement_mode", placement_mode)) {
        _floor_standing_mode = (placement_mode == "FLOOR_STANDING");
    }

    // Automatically adapt depth range for small tabletop/handheld objects vs floor-standing frames
    if (!_floor_standing_mode && _bounding_diag_m < 0.45f) {
        _min_depth_m = 0.25f;
        _max_depth_m = 1.85f;
        _left_margin_ratio = 0.02f;
        _right_margin_ratio = 0.02f;
    } else if (!_floor_standing_mode) {
        _min_depth_m = 0.30f;
        _max_depth_m = 2.50f;
        _left_margin_ratio = 0.02f;
        _right_margin_ratio = 0.02f;
    } else {
        _min_depth_m = 0.45f;
        _max_depth_m = 2.10f; // Reject background clutter > 2.10m visible through hollow table frame
        _left_margin_ratio = 0.08f; // Reject left-edge vertical aluminum post
        _right_margin_ratio = 0.04f;
    }
    return true;
}

void CADSceneSegmenter::setDepthRange(float min_depth_m, float max_depth_m) {
    _min_depth_m = min_depth_m;
    _max_depth_m = max_depth_m;
}

void CADSceneSegmenter::setBorderMarginRatio(float left_margin, float right_margin) {
    _left_margin_ratio = std::clamp(left_margin, 0.0f, 0.35f);
    _right_margin_ratio = std::clamp(right_margin, 0.0f, 0.35f);
}

void CADSceneSegmenter::setExpectedObjectDimensions(float length_m, float width_m, float height_m, bool floor_standing) {
    _expected_length_m = length_m;
    _expected_width_m = width_m;
    _expected_height_m = height_m;
    _bounding_diag_m = std::sqrt(length_m * length_m + width_m * width_m + height_m * height_m);
    _floor_standing_mode = floor_standing;

    if (!_floor_standing_mode && _bounding_diag_m < 0.45f) {
        _min_depth_m = 0.25f;
        _max_depth_m = 1.85f;
        _left_margin_ratio = 0.02f;
        _right_margin_ratio = 0.02f;
    } else if (!_floor_standing_mode) {
        _min_depth_m = 0.30f;
        _max_depth_m = 2.50f;
        _left_margin_ratio = 0.02f;
        _right_margin_ratio = 0.02f;
    } else {
        _min_depth_m = 0.45f;
        _max_depth_m = 2.10f;
        _left_margin_ratio = 0.08f;
        _right_margin_ratio = 0.04f;
    }
}

void CADSceneSegmenter::setMultiInstanceMode(bool enable, int max_instances) {
    _enable_multi_instance = enable;
    _max_instances = std::max(1, max_instances);
}

void CADSceneSegmenter::extractDepthPoints(const cv::Mat& depth_z16,
                                           const CameraIntrinsics& intr,
                                           const cv::Rect& roi_hint,
                                           std::vector<cv::Vec3f>& out_pts) const {
    out_pts.clear();
    if (depth_z16.empty() || depth_z16.type() != CV_16UC1) return;

    const int w = depth_z16.cols;
    const int h = depth_z16.rows;

    // Auto-Intrinsics Fallback if caller passed uninitialized intrinsics
    const float fx = (intr.fx > 100.0) ? static_cast<float>(intr.fx)
                                       : ((w == 1024 && h == 576) ? 739.41f : 0.95f * w);
    const float fy = (intr.fy > 100.0) ? static_cast<float>(intr.fy) : fx;
    const float cx = (intr.ppx > 10.0) ? static_cast<float>(intr.ppx)
                                       : ((intr.cx > 10.0) ? static_cast<float>(intr.cx) : (0.5f * w));
    const float cy = (intr.ppy > 10.0) ? static_cast<float>(intr.ppy)
                                       : ((intr.cy > 10.0) ? static_cast<float>(intr.cy) : (0.5f * h));
    const float scale = (intr.depth_scale > 0.0) ? static_cast<float>(intr.depth_scale) : 0.001f;

    int x0 = static_cast<int>( std::round(_left_margin_ratio * w) );
    int x1 = static_cast<int>( std::round((1.0f - _right_margin_ratio) * w) );
    int y0 = 0;
    int y1 = h;

    const bool has_roi = (roi_hint.width > 16 && roi_hint.height > 16);
    if (has_roi) {
        x0 = std::clamp(roi_hint.x, 0, std::max(0, w - 1));
        y0 = std::clamp(roi_hint.y, 0, std::max(0, h - 1));
        x1 = std::clamp(roi_hint.x + roi_hint.width, x0 + 1, w);
        y1 = std::clamp(roi_hint.y + roi_hint.height, y0 + 1, h);
    }

    const long long roi_area = static_cast<long long>(x1 - x0) * static_cast<long long>(y1 - y0);
    const int stride = ((has_roi && roi_area < 75000) || (!_floor_standing_mode && _bounding_diag_m < 0.35f && w <= 848)) ? 1 : 2;
    out_pts.reserve(static_cast<size_t>((x1 - x0) / stride + 1) * static_cast<size_t>((y1 - y0) / stride + 1));

    const int max_jump_raw = static_cast<int>(std::round(0.018f / std::max(1e-6f, scale)));
    for (int v = y0; v < y1; v += stride) {
        const uint16_t* row_ptr = depth_z16.ptr<uint16_t>(v);
        const uint16_t* row_up = (v >= 2) ? depth_z16.ptr<uint16_t>(v - 2) : row_ptr;
        const uint16_t* row_dn = (v + 2 < h) ? depth_z16.ptr<uint16_t>(v + 2) : row_ptr;
        for (int u = x0; u < x1; u += stride) {
            uint16_t d_raw = row_ptr[u];
            if (d_raw == 0) continue;

            // Khử "Flying Pixels" ở mép vật thể nhựa bóng / trụ tròn (như chai Chin-su)
            if (!_floor_standing_mode && u >= 2 && u + 2 < w) {
                const uint16_t d_l = row_ptr[u - 2];
                const uint16_t d_r = row_ptr[u + 2];
                const uint16_t d_u = row_up[u];
                const uint16_t d_d = row_dn[u];
                if (d_l == 0 || d_r == 0 || d_u == 0 || d_d == 0) continue;
                if (std::abs(static_cast<int>(d_raw) - static_cast<int>(d_l)) > max_jump_raw ||
                    std::abs(static_cast<int>(d_raw) - static_cast<int>(d_r)) > max_jump_raw ||
                    std::abs(static_cast<int>(d_raw) - static_cast<int>(d_u)) > max_jump_raw ||
                    std::abs(static_cast<int>(d_raw) - static_cast<int>(d_d)) > max_jump_raw) {
                    continue;
                }
            }

            float z = d_raw * scale;
            if (z < _min_depth_m || z > _max_depth_m) continue;

            float x = (static_cast<float>(u) - cx) * z / fx;
            float y = (static_cast<float>(v) - cy) * z / fy;
            out_pts.emplace_back(x, y, z);
        }
    }
}

bool CADSceneSegmenter::fitSupportPlaneRANSAC(const std::vector<cv::Vec3f>& pts,
                                              bool has_roi_hint,
                                              cv::Vec4d& out_plane,
                                              std::vector<cv::Vec3f>& out_above_floor) const {
    out_above_floor.clear();
    if (pts.size() < 80) {
        out_above_floor = pts;
        return false;
    }

    if (_floor_standing_mode) {
        // FLOOR_STANDING branch preserved verbatim for TABLE-DEMO accuracy
        std::vector<cv::Vec3f> floor_cands;
        floor_cands.reserve(pts.size() / 2);
        for (const auto& p : pts) {
            if (p[1] > -0.05f) floor_cands.push_back(p);
        }
        if (floor_cands.size() < 50) floor_cands = pts;

        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> dist(0, floor_cands.size() - 1);

        const double dist_thresh = 0.022;
        size_t best_inliers = 0;
        cv::Vec4d best_plane(0.0, -1.0, 0.0, 0.0);

        for (int iter = 0; iter < 140; ++iter) {
            const cv::Vec3f& p1 = floor_cands[dist(rng)];
            const cv::Vec3f& p2 = floor_cands[dist(rng)];
            const cv::Vec3f& p3 = floor_cands[dist(rng)];

            cv::Vec3f v1 = p2 - p1;
            cv::Vec3f v2 = p3 - p1;
            cv::Vec3f n = v1.cross(v2);
            float norm_n = cv::norm(n);
            if (norm_n < 1e-6f) continue;
            n /= norm_n;
            if (n[1] > 0.0f) n = -n;
            // Hỗ trợ camera đặt cao > 1.36m nhìn chúc xuống tới ~72 độ
            if (n[1] > -0.30f || std::abs(n[0]) > 0.45f || std::abs(n[2]) > 0.95f) continue;

            double d = -(n[0] * p1[0] + n[1] * p1[1] + n[2] * p1[2]);
            size_t count = 0;
            for (size_t k = 0; k < floor_cands.size(); k += 2) {
                const auto& q = floor_cands[k];
                double err = std::abs(n[0] * q[0] + n[1] * q[1] + n[2] * q[2] + d);
                if (err <= dist_thresh) ++count;
            }

            if (count > best_inliers) {
                best_inliers = count;
                best_plane = cv::Vec4d(n[0], n[1], n[2], d);
            }
        }

        if (best_inliers < 35) {
            out_above_floor = pts;
            return false;
        }

        cv::Vec3d mean(0.0, 0.0, 0.0);
        std::vector<cv::Vec3f> inliers;
        inliers.reserve(best_inliers * 2);
        for (const auto& q : floor_cands) {
            double err = std::abs(best_plane[0] * q[0] + best_plane[1] * q[1] + best_plane[2] * q[2] + best_plane[3]);
            if (err <= dist_thresh) {
                inliers.push_back(q);
                mean += cv::Vec3d(q[0], q[1], q[2]);
            }
        }
        if (inliers.size() >= 15) {
            mean *= (1.0 / static_cast<double>(inliers.size()));
            cv::Matx33d cov = cv::Matx33d::zeros();
            for (const auto& q : inliers) {
                cv::Vec3d dq(q[0] - mean[0], q[1] - mean[1], q[2] - mean[2]);
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        cov(r, c) += dq[r] * dq[c];
            }
            cv::Mat eig_vals, eig_vecs;
            cv::eigen(cv::Mat(cov), eig_vals, eig_vecs);
            cv::Vec3d n_svd(eig_vecs.at<double>(2, 0), eig_vecs.at<double>(2, 1), eig_vecs.at<double>(2, 2));
            if (n_svd[1] > 0.0) n_svd = -n_svd;
            n_svd *= (1.0 / std::max(1e-9, cv::norm(n_svd)));
            double d_svd = -n_svd.dot(mean);
            best_plane = cv::Vec4d(n_svd[0], n_svd[1], n_svd[2], d_svd);
        }

        if (best_plane[1] > -0.72) {
            out_above_floor = pts;
            return false;
        }

        const double min_h = 0.020;
        const double max_h = _expected_height_m + 0.18;
        out_above_floor.reserve(pts.size());
        for (const auto& p : pts) {
            double h_above = (best_plane[0] * p[0] + best_plane[1] * p[1] + best_plane[2] * p[2] + best_plane[3]);
            if (h_above >= min_h && h_above <= max_h) {
                out_above_floor.push_back(p);
            }
        }

        out_plane = best_plane;
        return true;
    }

    // FREE_6DOF: Multi-plane tabletop & background peeling with strict CAD-size isolation guard
    const float raw_qdiag = computeQuantileDiag(pts, 0.03, 0.97);

    // Never slice an already-isolated target object (e.g. validation render or tight foreground crop)
    if (pts.size() <= 350 || raw_qdiag <= 1.32f * _bounding_diag_m ||
        (!has_roi_hint && pts.size() <= 2400 && raw_qdiag <= 1.42f * _bounding_diag_m)) {
        out_above_floor = pts;
        return false;
    }

    std::vector<cv::Vec3f> rem = pts;
    std::mt19937 rng(42);
    const bool wide_full_scene = (!has_roi_hint && pts.size() > 60000);
    const double dist_thresh = 0.009;
    bool found_horizontal = false;
    size_t best_horiz_count = 0;

    for (int pass = 0; pass < 3; ++pass) {
        if (rem.size() < 400) break;

        // Uniform subsample up to 4096 points for O(1)-fast RANSAC hypothesis search
        const size_t cand_step = std::max<size_t>(1, rem.size() / 4096);
        std::vector<cv::Vec3f> cands;
        cands.reserve(rem.size() / cand_step + 1);
        for (size_t i = 0; i < rem.size(); i += cand_step) {
            cands.push_back(rem[i]);
        }
        if (cands.size() < 120) break;

        std::uniform_int_distribution<size_t> dist(0, cands.size() - 1);
        const size_t eval_step = std::max<size_t>(1, cands.size() / 2048);
        size_t best_cand_hits = 0;
        cv::Vec4d best_plane(0.0, -1.0, 0.0, 0.0);

        for (int iter = 0; iter < 160; ++iter) {
            const cv::Vec3f& p1 = cands[dist(rng)];
            const cv::Vec3f& p2 = cands[dist(rng)];
            const cv::Vec3f& p3 = cands[dist(rng)];

            cv::Vec3f n = (p2 - p1).cross(p3 - p1);
            float norm_n = cv::norm(n);
            if (norm_n < 1e-6f) continue;
            n /= norm_n;
            if (n[1] > 0.0f) n = -n;

            double d = -(n[0] * p1[0] + n[1] * p1[1] + n[2] * p1[2]);
            size_t count = 0;
            for (size_t k = 0; k < cands.size(); k += eval_step) {
                const auto& q = cands[k];
                if (std::abs(n[0] * q[0] + n[1] * q[1] + n[2] * q[2] + d) <= dist_thresh) {
                    ++count;
                }
            }
            if (count > best_cand_hits) {
                best_cand_hits = count;
                best_plane = cv::Vec4d(n[0], n[1], n[2], d);
            }
        }

        if (best_cand_hits < 40) break;

        // Collect all inliers across rem, compute trimmed quantile span, and SVD-refine plane
        std::vector<cv::Vec3f> inl_pts;
        inl_pts.reserve(rem.size() / 4);
        for (const auto& q : rem) {
            double err = std::abs(best_plane[0] * q[0] + best_plane[1] * q[1] + best_plane[2] * q[2] + best_plane[3]);
            if (err <= dist_thresh) {
                inl_pts.push_back(q);
            }
        }

        const float inl_qdiag = computeQuantileDiag(inl_pts, 0.05, 0.95);
        if (inl_pts.size() < 240 || inl_qdiag <= std::max(0.30f, 1.75f * _bounding_diag_m)) {
            break;
        }

        // SVD refinement on strided inliers
        const size_t svd_step = std::max<size_t>(1, inl_pts.size() / 4096);
        cv::Vec3d mean(0.0, 0.0, 0.0);
        size_t svd_cnt = 0;
        for (size_t i = 0; i < inl_pts.size(); i += svd_step) {
            mean += cv::Vec3d(inl_pts[i][0], inl_pts[i][1], inl_pts[i][2]);
            ++svd_cnt;
        }
        if (svd_cnt >= 20) {
            mean *= (1.0 / static_cast<double>(svd_cnt));
            cv::Matx33d cov = cv::Matx33d::zeros();
            for (size_t i = 0; i < inl_pts.size(); i += svd_step) {
                const auto& q = inl_pts[i];
                cv::Vec3d dq(q[0] - mean[0], q[1] - mean[1], q[2] - mean[2]);
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        cov(r, c) += dq[r] * dq[c];
            }
            cv::Mat eig_vals, eig_vecs;
            cv::eigen(cv::Mat(cov), eig_vals, eig_vecs);
            cv::Vec3d n_svd(eig_vecs.at<double>(2, 0), eig_vecs.at<double>(2, 1), eig_vecs.at<double>(2, 2));
            if (n_svd[1] > 0.0) n_svd = -n_svd;
            n_svd *= (1.0 / std::max(1e-9, cv::norm(n_svd)));
            best_plane = cv::Vec4d(n_svd[0], n_svd[1], n_svd[2], -n_svd.dot(mean));
        }

        const bool is_horizontal = (std::abs(best_plane[1]) >= 0.55);
        bool accept_peel = false;
        const size_t min_dom_inliers = std::max<size_t>(450, pts.size() / 8);
        if (is_horizontal) {
            accept_peel = (inl_pts.size() >= min_dom_inliers) &&
                          (inl_qdiag > std::max(0.30f, 1.75f * _bounding_diag_m));
        } else {
            accept_peel = (inl_pts.size() >= min_dom_inliers) &&
                          (inl_qdiag > std::max(0.32f, 1.80f * _bounding_diag_m));
        }
        if (!accept_peel) break;

        // For massive architectural/sofa walls (>4x object diag & >0.75m), widen peel slightly to suppress padded rim halos
        const double peel_thresh = (wide_full_scene && inl_qdiag > std::max(0.75f, 4.0f * _bounding_diag_m))
                                       ? 0.022
                                       : dist_thresh;

        std::vector<cv::Vec3f> next_rem;
        next_rem.reserve(rem.size() - std::min(rem.size(), inl_pts.size()));
        for (const auto& q : rem) {
            double err = std::abs(best_plane[0] * q[0] + best_plane[1] * q[1] + best_plane[2] * q[2] + best_plane[3]);
            if (err > peel_thresh) {
                next_rem.push_back(q);
            }
        }
        if (next_rem.size() < 80) break;

        if (is_horizontal && inl_pts.size() > best_horiz_count) {
            best_horiz_count = inl_pts.size();
            out_plane = best_plane;
            found_horizontal = true;
        }
        rem = std::move(next_rem);
    }

    out_above_floor = std::move(rem);
    return found_horizontal;
}

void CADSceneSegmenter::clusterAndMergeTarget(const std::vector<cv::Vec3f>& fg_pts,
                                              bool has_roi_hint,
                                              std::vector<cv::Vec3f>& out_merged_dense,
                                              std::vector<SegmentedCluster>& out_multi_clusters) const {
    out_merged_dense.clear();
    out_multi_clusters.clear();
    if (fg_pts.empty()) return;

    auto computeClusterStats = [this, has_roi_hint](std::vector<cv::Vec3f>&& pts) -> SegmentedCluster {
        SegmentedCluster cl;
        cl.points = std::move(pts);
        if (cl.points.empty()) return cl;
        cv::Vec3d sum(0.0, 0.0, 0.0);
        cv::Vec3f bmin(1e9f, 1e9f, 1e9f), bmax(-1e9f, -1e9f, -1e9f);
        for (const auto& pt : cl.points) {
            sum += cv::Vec3d(pt[0], pt[1], pt[2]);
            for (int d = 0; d < 3; ++d) {
                bmin[d] = std::min(bmin[d], pt[d]);
                bmax[d] = std::max(bmax[d], pt[d]);
            }
        }
        cl.centroid = cv::Vec3f(
            static_cast<float>(sum[0] / cl.points.size()),
            static_cast<float>(sum[1] / cl.points.size()),
            static_cast<float>(sum[2] / cl.points.size())
        );
        cl.bbox_min = bmin;
        cl.bbox_max = bmax;
        double radial_offset = std::hypot(cl.centroid[0], cl.centroid[1] * 0.5);
        if (_floor_standing_mode) {
            cl.score = static_cast<double>(cl.points.size()) / (cl.centroid[2] + 1.8 * radial_offset + 0.35);
        } else {
            double c_diag = cv::norm(bmax - bmin);
            double size_match = 1.0 / (std::abs(c_diag - static_cast<double>(_bounding_diag_m)) + 0.025);
            if (c_diag > 1.45 * _bounding_diag_m || c_diag < 0.35 * _bounding_diag_m) size_match *= 0.15;
            if (c_diag > 1.90 * _bounding_diag_m) size_match *= 0.20;
            double pt_support = std::sqrt(static_cast<double>(std::min<size_t>(cl.points.size(), 4000)));
            double rad_w = has_roi_hint ? 0.8 : 1.3;
            double shape_factor = 1.0;
            if (_expected_width_m > 0.015f && _expected_height_m > 0.015f &&
                c_diag <= 1.85 * static_cast<double>(_bounding_diag_m)) {
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
                computePcaSpread(cl.points, s0, s1, s2);
                const float min_dim = std::min(_expected_width_m, _expected_length_m);
                const float mid_dim = std::max(_expected_width_m, _expected_length_m);
                if (s1 < 0.14f * min_dim || (s2 < 0.0014f && s1 < 0.28f * mid_dim)) {
                    shape_factor = 0.04; // Suppress 1D plane-peel remnant strips
                }
            }
            cl.score = (pt_support * size_match * shape_factor) / (cl.centroid[2] + rad_w * radial_offset + 0.35);
        }
        return cl;
    };

    auto runVoxelBFS = [&](float cell_sz, size_t min_pts, float max_abs_x) -> std::vector<SegmentedCluster> {
        const float inv_cell = 1.0f / std::max(1e-4f, cell_sz);
        std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> grid;
        grid.reserve(fg_pts.size() / 4 + 16);

        for (int i = 0; i < static_cast<int>(fg_pts.size()); ++i) {
            const auto& p = fg_pts[i];
            VoxelKey key{
                static_cast<int>(std::floor(p[0] * inv_cell)),
                static_cast<int>(std::floor(p[1] * inv_cell)),
                static_cast<int>(std::floor(p[2] * inv_cell))
            };
            grid[key].push_back(i);
        }

        std::unordered_map<VoxelKey, bool, VoxelKeyHash> visited;
        visited.reserve(grid.size());
        std::vector<SegmentedCluster> clusters;

        for (const auto& kv : grid) {
            if (visited[kv.first]) continue;

            std::queue<VoxelKey> q;
            q.push(kv.first);
            visited[kv.first] = true;

            std::vector<cv::Vec3f> cl_pts;
            while (!q.empty()) {
                VoxelKey cur = q.front();
                q.pop();

                auto it = grid.find(cur);
                if (it != grid.end()) {
                    for (int idx : it->second) {
                        cl_pts.push_back(fg_pts[idx]);
                    }
                }

                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dz = -1; dz <= 1; ++dz) {
                            if (dx == 0 && dy == 0 && dz == 0) continue;
                            VoxelKey nb{cur.x + dx, cur.y + dy, cur.z + dz};
                            if (grid.find(nb) != grid.end() && !visited[nb]) {
                                visited[nb] = true;
                                q.push(nb);
                            }
                        }
                    }
                }
            }

            if (cl_pts.size() < min_pts) continue;
            SegmentedCluster cl = computeClusterStats(std::move(cl_pts));
            if (std::abs(cl.centroid[0]) > max_abs_x || cl.centroid[2] > _max_depth_m) continue;
            clusters.push_back(std::move(cl));
        }
        return clusters;
    };

    if (_floor_standing_mode) {
        const float cell_size = std::clamp(_bounding_diag_m * 0.075f, 0.04f, 0.09f);
        const float max_abs_x = has_roi_hint ? 2.20f : 1.15f;
        std::vector<SegmentedCluster> raw_clusters = runVoxelBFS(cell_size, 15, max_abs_x);
        if (raw_clusters.empty()) {
            out_merged_dense = fg_pts;
            return;
        }
        std::sort(raw_clusters.begin(), raw_clusters.end(),
                  [](const SegmentedCluster& a, const SegmentedCluster& b) { return a.score > b.score; });

        // Co-box Cluster Merging (Fixes fragmented table legs when front aluminum beam has depth == 0)
        const SegmentedCluster& seed = raw_clusters[0];
        const float max_dx = _expected_length_m + 0.12f; // 0.95m for 0.83m table
        const float max_dz = _expected_width_m + 0.14f;  // 0.58m for 0.44m table

        std::vector<cv::Vec3f> merged_pts = seed.points;
        for (size_t i = 1; i < raw_clusters.size(); ++i) {
            const auto& cand = raw_clusters[i];
            float dx = std::abs(cand.centroid[0] - seed.centroid[0]);
            float dz = std::abs(cand.centroid[2] - seed.centroid[2]);
            if (dx <= max_dx && dz <= max_dz) {
                merged_pts.insert(merged_pts.end(), cand.points.begin(), cand.points.end());
            }
        }
        SegmentedCluster merged = computeClusterStats(std::move(merged_pts));
        out_merged_dense = merged.points;
        out_multi_clusters.push_back(std::move(merged));
    } else {
        // FREE_6DOF: isolated single-object preservation OR scale-adaptive CC + K-Means spherical windows
        const float fg_qdiag = computeQuantileDiag(fg_pts, 0.02, 0.98);
        const bool is_isolated_cloud = (!has_roi_hint && fg_pts.size() <= 2400 && fg_qdiag <= 1.42f * _bounding_diag_m);
        if (is_isolated_cloud) {
            // Keep isolated point clouds intact (preserves grazing neck/cap points like s01969)
            std::vector<float> xs, ys, zs;
            xs.reserve(fg_pts.size());
            ys.reserve(fg_pts.size());
            zs.reserve(fg_pts.size());
            for (const auto& p : fg_pts) {
                xs.push_back(p[0]);
                ys.push_back(p[1]);
                zs.push_back(p[2]);
            }
            const size_t mid_idx = xs.size() / 2;
            std::nth_element(xs.begin(), xs.begin() + static_cast<std::ptrdiff_t>(mid_idx), xs.end());
            std::nth_element(ys.begin(), ys.begin() + static_cast<std::ptrdiff_t>(mid_idx), ys.end());
            std::nth_element(zs.begin(), zs.begin() + static_cast<std::ptrdiff_t>(mid_idx), zs.end());
            const cv::Vec3f med(xs[mid_idx], ys[mid_idx], zs[mid_idx]);
            const float keep_r2 = (0.78f * _bounding_diag_m) * (0.78f * _bounding_diag_m);
            std::vector<cv::Vec3f> kept_pts;
            kept_pts.reserve(fg_pts.size());
            for (const auto& p : fg_pts) {
                const cv::Vec3f d = p - med;
                if (d.dot(d) <= keep_r2) {
                    kept_pts.push_back(p);
                }
            }
            if (kept_pts.size() * 10 < fg_pts.size() * 8) {
                kept_pts = fg_pts;
            }
            SegmentedCluster iso_cl = computeClusterStats(std::move(kept_pts));
            out_merged_dense = iso_cl.points;
            out_multi_clusters.push_back(std::move(iso_cl));
            return;
        }

        const bool fine_roi_sep = (has_roi_hint && _bounding_diag_m < 0.35f && fg_pts.size() > 35000);
        float cell_size = std::clamp(_bounding_diag_m * 0.036f, 0.005f, 0.014f);
        size_t min_cluster_pts = 20;
        if (fine_roi_sep) {
            // When unpeeled background cushion borders an object inside a tight ROI, separate at fine scale
            cell_size = std::clamp(_bounding_diag_m * 0.018f, 0.0028f, 0.0055f);
            min_cluster_pts = 100;
        }

        const float max_abs_x = has_roi_hint ? 2.20f : 1.65f;
        std::vector<SegmentedCluster> raw_clusters = runVoxelBFS(cell_size, min_cluster_pts, max_abs_x);
        if (raw_clusters.empty()) {
            const float coarse_cell = std::clamp(_bounding_diag_m * 0.075f, 0.012f, 0.032f);
            raw_clusters = runVoxelBFS(coarse_cell, 15, max_abs_x);
        }
        if (raw_clusters.empty()) {
            out_merged_dense = fg_pts;
            return;
        }

        std::sort(raw_clusters.begin(), raw_clusters.end(),
                  [](const SegmentedCluster& a, const SegmentedCluster& b) { return a.score > b.score; });

        // Generate K-Means interior spherical windows for oversized/merged blobs
        auto generateSphericalProposals = [&](const SegmentedCluster& big_cl, bool add_centroid, int K_spheres) -> std::vector<SegmentedCluster> {
            std::vector<SegmentedCluster> props;
            const size_t n_pts = big_cl.points.size();
            if (n_pts < 60 || K_spheres <= 0) return props;

            std::vector<cv::Vec3f> sub;
            if (n_pts > 400) {
                const float leaf = std::clamp(0.08f * _bounding_diag_m, 0.008f, 0.025f);
                sub = voxelDownsampleCentroids(big_cl.points, leaf);
            }
            if (sub.size() < 24) {
                sub.clear();
                const size_t step = std::max<size_t>(1, n_pts / 600);
                sub.reserve(n_pts / step + 1);
                for (size_t i = 0; i < n_pts; i += step) {
                    sub.push_back(big_cl.points[i]);
                }
            }
            if (sub.size() < 20) return props;

            cv::Vec3d sum_sub(0.0, 0.0, 0.0);
            for (const auto& sp : sub) sum_sub += cv::Vec3d(sp);

            const cv::Vec3f cen(
                static_cast<float>(sum_sub[0] / sub.size()),
                static_cast<float>(sum_sub[1] / sub.size()),
                static_cast<float>(sum_sub[2] / sub.size())
            );

            int best0 = 0;
            float bd0 = 1e9f;
            for (int i = 0; i < static_cast<int>(sub.size()); ++i) {
                const float d = static_cast<float>(cv::norm(sub[i] - cen));
                if (d < bd0) {
                    bd0 = d;
                    best0 = i;
                }
            }

            const int K = std::min<int>(K_spheres, static_cast<int>(sub.size()));
            std::vector<cv::Vec3f> centers;
            centers.reserve(static_cast<size_t>(K));
            centers.push_back(sub[best0]);

            std::vector<float> min_d2(sub.size(), 1e9f);
            const float stop_d2 = has_roi_hint ? 0.0f : (0.35f * _bounding_diag_m) * (0.35f * _bounding_diag_m);
            for (int k = 1; k < K; ++k) {
                const cv::Vec3f& last = centers.back();
                int far_i = -1;
                float far_d2 = -1.0f;
                for (int i = 0; i < static_cast<int>(sub.size()); ++i) {
                    const cv::Vec3f df = sub[i] - last;
                    const float d2 = df.dot(df);
                    if (d2 < min_d2[i]) min_d2[i] = d2;
                    if (min_d2[i] > far_d2) {
                        far_d2 = min_d2[i];
                        far_i = i;
                    }
                }
                if (far_i < 0 || (!has_roi_hint && k >= 4 && far_d2 < stop_d2)) break;
                centers.push_back(sub[far_i]);
            }

            // 3 Lloyd K-means iterations to pull centers inside local object bodies
            for (int it = 0; it < 3; ++it) {
                std::vector<cv::Vec3d> csum(centers.size(), cv::Vec3d(0.0, 0.0, 0.0));
                std::vector<int> ccnt(centers.size(), 0);
                for (const auto& q : sub) {
                    int bj = 0;
                    float bd = 1e9f;
                    for (size_t j = 0; j < centers.size(); ++j) {
                        const cv::Vec3f df = q - centers[j];
                        const float d2 = df.dot(df);
                        if (d2 < bd) {
                            bd = d2;
                            bj = static_cast<int>(j);
                        }
                    }
                    csum[bj] += cv::Vec3d(q);
                    ++ccnt[bj];
                }
                for (size_t j = 0; j < centers.size(); ++j) {
                    if (ccnt[j] > 0) {
                        centers[j] = cv::Vec3f(
                            static_cast<float>(csum[j][0] / ccnt[j]),
                            static_cast<float>(csum[j][1] / ccnt[j]),
                            static_cast<float>(csum[j][2] / ccnt[j])
                        );
                    }
                }
            }

            std::vector<cv::Vec3f> all_c;
            all_c.reserve(centers.size() + (add_centroid ? 1 : 0));
            if (add_centroid) all_c.push_back(cen);
            for (const auto& c : centers) all_c.push_back(c);

            const float sph_r = (has_roi_hint ? 0.48f : 0.50f) * _bounding_diag_m;
            const float rad_sq = sph_r * sph_r;
            const size_t min_patch_pts = std::clamp(n_pts / 40, static_cast<size_t>(45), static_cast<size_t>(180));
            for (const auto& seed_c : all_c) {
                std::vector<cv::Vec3f> patch;
                patch.reserve(n_pts / 4 + 16);
                for (const auto& q : big_cl.points) {
                    const cv::Vec3f df = q - seed_c;
                    if (df.dot(df) <= rad_sq) {
                        patch.push_back(q);
                    }
                }
                if (patch.size() >= min_patch_pts) {
                    props.push_back(computeClusterStats(std::move(patch)));
                }
            }
            return props;
        };

        std::vector<SegmentedCluster> candidate_clusters;
        candidate_clusters.reserve(raw_clusters.size() + 24);

        if (has_roi_hint) {
            for (const auto& cl : raw_clusters) {
                const float c_diag = static_cast<float>(cv::norm(cl.bbox_max - cl.bbox_min));
                if (c_diag > 1.22f * _bounding_diag_m && c_diag <= 4.50f * _bounding_diag_m && cl.points.size() >= 120) {
                    const int K = std::clamp(static_cast<int>(std::ceil(c_diag / (0.20f * _bounding_diag_m))), 5, 14);
                    const bool add_cen = (c_diag <= 1.95f * _bounding_diag_m);
                    auto props = generateSphericalProposals(cl, add_cen, K);
                    candidate_clusters.insert(candidate_clusters.end(),
                                              std::make_move_iterator(props.begin()),
                                              std::make_move_iterator(props.end()));
                    if (c_diag <= 1.48f * _bounding_diag_m) {
                        candidate_clusters.push_back(cl);
                    }
                } else {
                    candidate_clusters.push_back(cl);
                }
            }
        } else {
            const float free_max_z = std::clamp(5.0f * _bounding_diag_m + 0.35f, 0.95f, _max_depth_m);
            // Keep object-sized standalone clusters first
            for (const auto& cl : raw_clusters) {
                const float c_diag = static_cast<float>(cv::norm(cl.bbox_max - cl.bbox_min));
                if (c_diag >= 0.35f * _bounding_diag_m &&
                    c_diag <= 1.65f * _bounding_diag_m &&
                    cl.centroid[2] <= free_max_z) {
                    candidate_clusters.push_back(cl);
                }
            }
            // Prioritize oversized workspace blobs by point support so dominant piles are NEVER starved by tiny strips
            std::vector<size_t> by_size(raw_clusters.size());
            std::iota(by_size.begin(), by_size.end(), 0);
            std::sort(by_size.begin(), by_size.end(), [&](size_t a, size_t b) {
                return raw_clusters[a].points.size() > raw_clusters[b].points.size();
            });
            int expanded_oversized = 0;
            for (size_t idx : by_size) {
                if (expanded_oversized >= 8) break;
                const auto& cl = raw_clusters[idx];
                const float c_diag = static_cast<float>(cv::norm(cl.bbox_max - cl.bbox_min));
                if (c_diag > 1.35f * _bounding_diag_m &&
                    c_diag <= 8.00f * _bounding_diag_m &&
                    cl.points.size() >= 200 &&
                    cl.centroid[2] <= free_max_z &&
                    std::abs(cl.centroid[0]) <= 0.95f) {
                    const int K = std::clamp(static_cast<int>(std::ceil(c_diag / (0.19f * _bounding_diag_m))), 5, 20);
                    auto props = generateSphericalProposals(cl, false, K);
                    for (auto& pr : props) {
                        if (pr.centroid[2] <= free_max_z) {
                            candidate_clusters.push_back(std::move(pr));
                        }
                    }
                    ++expanded_oversized;
                }
            }
        }

        if (candidate_clusters.empty()) {
            candidate_clusters = std::move(raw_clusters);
        }

        std::sort(candidate_clusters.begin(), candidate_clusters.end(),
                  [](const SegmentedCluster& a, const SegmentedCluster& b) { return a.score > b.score; });

        out_multi_clusters.push_back(candidate_clusters[0]);
        const int cand_cap = has_roi_hint ? std::max(_max_instances * 3, 16) : std::max(_max_instances * 9, 36);
        const double best_score = candidate_clusters[0].score;
        const float min_prop_sep = has_roi_hint ? 0.0f : (0.22f * _bounding_diag_m);
        const float min_prop_sep_sq = min_prop_sep * min_prop_sep;
        for (size_t i = 1; i < candidate_clusters.size() && static_cast<int>(out_multi_clusters.size()) < cand_cap; ++i) {
            const auto& cl = candidate_clusters[i];
            const float c_diag = static_cast<float>(cv::norm(cl.bbox_max - cl.bbox_min));
            const double min_rel = has_roi_hint ? 0.08 : 0.03;
            if (cl.score >= min_rel * best_score &&
                c_diag >= 0.32f * _bounding_diag_m &&
                c_diag <= 1.85f * _bounding_diag_m) {
                if (min_prop_sep_sq > 0.0f) {
                    bool near_dup = false;
                    for (const auto& kept : out_multi_clusters) {
                        const cv::Vec3f df = cl.centroid - kept.centroid;
                        if (df.dot(df) < min_prop_sep_sq) {
                            near_dup = true;
                            break;
                        }
                    }
                    if (near_dup) continue;
                }
                out_multi_clusters.push_back(cl);
            }
        }
        out_merged_dense = out_multi_clusters.front().points;
    }
}

std::vector<cv::Vec3f> CADSceneSegmenter::samplePointsCurvatureAware(const std::vector<cv::Vec3f>& pts,
                                                                     size_t target_count) const {
    if (pts.empty()) return {};
    if (pts.size() <= target_count) {
        std::vector<cv::Vec3f> padded = pts;
        std::mt19937 rng(1337);
        std::uniform_int_distribution<size_t> dist(0, pts.size() - 1);
        while (padded.size() < target_count) {
            padded.push_back(pts[dist(rng)]);
        }
        return padded;
    }

    // O(N) 3D Voxel-Neighborhood Geometric Deviation + Spatially Stratified Sampling:
    // Avoids order-dependent array neighbor bugs on hash-BFS extracted clusters.
    struct VoxelAccum {
        cv::Vec3d sum{0.0, 0.0, 0.0};
        int count{0};
    };

    const float v_size = std::clamp(_bounding_diag_m * 0.03f, 0.004f, 0.04f);
    const float inv_v = 1.0f / v_size;

    std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash> vgrid;
    vgrid.reserve(pts.size() / 2 + 16);
    std::vector<VoxelKey> pt_keys(pts.size());

    for (size_t i = 0; i < pts.size(); ++i) {
        const auto& p = pts[i];
        VoxelKey k{
            static_cast<int>(std::floor(p[0] * inv_v)),
            static_cast<int>(std::floor(p[1] * inv_v)),
            static_cast<int>(std::floor(p[2] * inv_v))
        };
        pt_keys[i] = k;
        auto& acc = vgrid[k];
        acc.sum += cv::Vec3d(p[0], p[1], p[2]);
        ++acc.count;
    }

    struct ScoredIdx {
        float deviation;
        VoxelKey key;
        size_t idx;
    };
    std::vector<ScoredIdx> scored;
    scored.reserve(pts.size());

    for (size_t i = 0; i < pts.size(); ++i) {
        const VoxelKey& k0 = pt_keys[i];
        cv::Vec3d nb_sum(0.0, 0.0, 0.0);
        int nb_cnt = 0;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    VoxelKey nb{k0.x + dx, k0.y + dy, k0.z + dz};
                    auto it = vgrid.find(nb);
                    if (it != vgrid.end()) {
                        nb_sum += it->second.sum;
                        nb_cnt += it->second.count;
                    }
                }
            }
        }
        float dev = 0.0f;
        if (nb_cnt > 1) {
            const auto& p = pts[i];
            cv::Vec3d mu = (nb_sum - cv::Vec3d(p[0], p[1], p[2])) * (1.0 / static_cast<double>(nb_cnt - 1));
            dev = static_cast<float>(cv::norm(cv::Vec3d(p[0], p[1], p[2]) - mu));
        }
        scored.push_back(ScoredIdx{dev, k0, i});
    }

    const size_t half = target_count / 2;
    std::nth_element(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(half), scored.end(),
                     [](const ScoredIdx& a, const ScoredIdx& b) { return a.deviation > b.deviation; });

    std::vector<cv::Vec3f> out;
    out.reserve(target_count);
    for (size_t i = 0; i < half; ++i) {
        out.push_back(pts[scored[i].idx]);
    }

    // Sort remaining points in 3D Morton-like voxel order before strided spatial selection
    std::sort(scored.begin() + static_cast<std::ptrdiff_t>(half), scored.end(),
              [](const ScoredIdx& a, const ScoredIdx& b) {
                  if (a.key.x != b.key.x) return a.key.x < b.key.x;
                  if (a.key.y != b.key.y) return a.key.y < b.key.y;
                  return a.key.z < b.key.z;
              });

    const size_t rem = target_count - half;
    const size_t avail = scored.size() - half;
    const double step = static_cast<double>(avail) / static_cast<double>(rem);
    for (size_t i = 0; i < rem; ++i) {
        size_t pick = half + std::min(avail - 1, static_cast<size_t>(i * step));
        out.push_back(pts[scored[pick].idx]);
    }
    return out;
}

SegmentationResult CADSceneSegmenter::segment(const cv::Mat& depth_z16,
                                              const CameraIntrinsics& intr,
                                              const cv::Rect& roi_hint,
                                              size_t target_num_points) const {
    SegmentationResult res;
    const bool has_roi = (roi_hint.width > 16 && roi_hint.height > 16);
    std::vector<cv::Vec3f> raw_pts;
    extractDepthPoints(depth_z16, intr, roi_hint, raw_pts);
    res.raw_points_count = raw_pts.size();
    if (raw_pts.size() < 20) return res;

    std::vector<cv::Vec3f> fg_pts;
    if (!_floor_standing_mode && has_roi) {
        // Fit support planes on a strided full-frame scene so small ROIs cannot self-peel the target object
        const int global_stride = std::clamp(std::min(depth_z16.cols, depth_z16.rows) / 240, 3, 6);
        std::vector<cv::Vec3f> global_pts;
        extractStridedScenePoints(depth_z16, intr, global_stride,
                                  _min_depth_m, _max_depth_m,
                                  _left_margin_ratio, _right_margin_ratio, global_pts);
        std::vector<cv::Vec4d> global_planes;
        extractGlobalSupportPlanesRANSAC(global_pts, _bounding_diag_m, 0.0085, global_planes);

        if (!global_planes.empty()) {
            const int roi_x0 = std::clamp(roi_hint.x, 0, std::max(0, depth_z16.cols - 1));
            const int roi_y0 = std::clamp(roi_hint.y, 0, std::max(0, depth_z16.rows - 1));
            const int roi_w = std::clamp(roi_hint.x + roi_hint.width, roi_x0 + 1, depth_z16.cols) - roi_x0;
            const int roi_h = std::clamp(roi_hint.y + roi_hint.height, roi_y0 + 1, depth_z16.rows) - roi_y0;
            const cv::Rect core_rect(roi_x0 + roi_w / 3, roi_y0 + roi_h / 3,
                                     std::max(8, roi_w / 3), std::max(8, roi_h / 3));
            std::vector<cv::Vec3f> core_pts;
            extractDepthPoints(depth_z16, intr, core_rect, core_pts);

            fg_pts = raw_pts;
            for (const auto& pl : global_planes) {
                if (!core_pts.empty()) {
                    size_t core_hits = 0;
                    for (const auto& cp : core_pts) {
                        if (std::abs(pl[0] * cp[0] + pl[1] * cp[1] + pl[2] * cp[2] + pl[3]) <= 0.0085) {
                            ++core_hits;
                        }
                    }
                    const double core_frac = static_cast<double>(core_hits) / static_cast<double>(core_pts.size());
                    if (core_frac > 0.52) {
                        continue; // Plane cuts through ROI target center (e.g. tilted/sunken object face)
                    }
                }

                std::vector<cv::Vec3f> next_fg;
                next_fg.reserve(fg_pts.size());
                for (const auto& q : fg_pts) {
                    if (std::abs(pl[0] * q[0] + pl[1] * q[1] + pl[2] * q[2] + pl[3]) > 0.0085) {
                        next_fg.push_back(q);
                    }
                }
                if (next_fg.size() >= std::max<size_t>(200, raw_pts.size() / 5)) {
                    if (!res.floor_detected && std::abs(pl[1]) >= 0.55) {
                        res.floor_detected = true;
                        res.floor_plane = pl;
                    }
                    fg_pts = std::move(next_fg);
                }
            }
        } else {
            fg_pts = raw_pts;
        }
    } else {
        res.floor_detected = fitSupportPlaneRANSAC(raw_pts, has_roi, res.floor_plane, fg_pts);
    }

    if (res.floor_detected) {
        const double nz = std::clamp(res.floor_plane[2], -1.0, 1.0);
        res.floor_pitch_deg = std::asin(nz) * 180.0 / 3.14159265358979323846;
    }
    if (fg_pts.size() < 15) {
        fg_pts = raw_pts;
    }
    res.foreground_points = fg_pts;

    clusterAndMergeTarget(fg_pts, has_roi, res.dense_object_points, res.multi_clusters);
    if (res.dense_object_points.empty()) return res;

    res.object_points = samplePointsCurvatureAware(res.dense_object_points, target_num_points);
    for (auto& cl : res.multi_clusters) {
        cl.dense_points = cl.points;
        cl.points = samplePointsCurvatureAware(cl.dense_points, target_num_points);
    }
    res.valid = !res.object_points.empty();
    return res;
}

} // namespace tmcad

