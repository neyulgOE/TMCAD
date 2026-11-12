#include "CADSceneSegmenter.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
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

static bool extractJsonNumber(const std::string& content, const std::string& key, double& out_val) {
    std::regex re("\"" + key + "\"\\s*:\\s*([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)");
    std::smatch match;
    if (std::regex_search(content, match, re) && match.size() >= 2) {
        out_val = std::stod(match[1].str());
        return true;
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
        x = std::stod(match[1].str());
        y = std::stod(match[2].str());
        z = std::stod(match[3].str());
        return true;
    }
    return false;
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

    if (roi_hint.width > 16 && roi_hint.height > 16) {
        x0 = std::max(0, roi_hint.x);
        y0 = std::max(0, roi_hint.y);
        x1 = std::min(w, roi_hint.x + roi_hint.width);
        y1 = std::min(h, roi_hint.y + roi_hint.height);
    }

    const int stride = (!_floor_standing_mode && _bounding_diag_m < 0.35f && w <= 848) ? 1 : 2;
    out_pts.reserve(((x1 - x0) / stride) * ((y1 - y0) / stride));

    for (int v = y0; v < y1; v += stride) {
        const uint16_t* row_ptr = depth_z16.ptr<uint16_t>(v);
        for (int u = x0; u < x1; u += stride) {
            uint16_t d_raw = row_ptr[u];
            if (d_raw == 0) continue;
            float z = d_raw * scale;
            if (z < _min_depth_m || z > _max_depth_m) continue;

            float x = (static_cast<float>(u) - cx) * z / fx;
            float y = (static_cast<float>(v) - cy) * z / fy;
            out_pts.emplace_back(x, y, z);
        }
    }
}

bool CADSceneSegmenter::fitSupportPlaneRANSAC(const std::vector<cv::Vec3f>& pts,
                                              cv::Vec4d& out_plane,
                                              std::vector<cv::Vec3f>& out_above_floor) const {
    out_above_floor.clear();
    if (pts.size() < 80) {
        out_above_floor = pts;
        return false;
    }

    // Candidate floor/support points (lower part of image / +Y in camera frame)
    std::vector<cv::Vec3f> floor_cands;
    floor_cands.reserve(pts.size() / 2);
    for (const auto& p : pts) {
        if (_floor_standing_mode) {
            if (p[1] > -0.05f) floor_cands.push_back(p);
        } else {
            floor_cands.push_back(p);
        }
    }
    if (floor_cands.size() < 50) floor_cands = pts;

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, floor_cands.size() - 1);

    const double dist_thresh = _floor_standing_mode ? 0.022 : 0.012;
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

        // Ensure normal points UPWARD in physical world (negative Y in OpenCV camera frame)
        if (n[1] > 0.0f) n = -n;

        // For FLOOR_STANDING mode, enforce upright floor tilt constraint
        if (_floor_standing_mode && n[1] > -0.78f) continue;

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

    // SVD refinement on floor inliers
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
        n_svd *= (1.0 / cv::norm(n_svd));
        double d_svd = -n_svd.dot(mean);
        best_plane = cv::Vec4d(n_svd[0], n_svd[1], n_svd[2], d_svd);
    }

    out_plane = best_plane;

    // Filter points strictly above the support plane within expected height envelope
    const double min_h = _floor_standing_mode ? 0.020 : 0.008;
    const double max_h = _floor_standing_mode ? (_expected_height_m + 0.18) : (_bounding_diag_m + 0.15);

    out_above_floor.reserve(pts.size());
    for (const auto& p : pts) {
        // Signed height above plane (since normal points upward in world / -Y in cam)
        double h_above = (best_plane[0] * p[0] + best_plane[1] * p[1] + best_plane[2] * p[2] + best_plane[3]);
        if (h_above >= min_h && h_above <= max_h) {
            out_above_floor.push_back(p);
        }
    }
    return true;
}

void CADSceneSegmenter::clusterAndMergeTarget(const std::vector<cv::Vec3f>& fg_pts,
                                              std::vector<cv::Vec3f>& out_merged_dense,
                                              std::vector<SegmentedCluster>& out_multi_clusters) const {
    out_merged_dense.clear();
    out_multi_clusters.clear();
    if (fg_pts.empty()) return;

    // Scale-adaptive voxel grid size: 8cm for table/pallets, 1.2cm for small objects like BBQ Sauce
    const float cell_size = _floor_standing_mode
        ? std::clamp(_bounding_diag_m * 0.075f, 0.04f, 0.09f)
        : std::clamp(_bounding_diag_m * 0.060f, 0.008f, 0.035f);
    const float inv_cell = 1.0f / cell_size;

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

    // 26-connectivity BFS clustering in Voxel space
    std::unordered_map<VoxelKey, bool, VoxelKeyHash> visited;
    visited.reserve(grid.size());

    // Lowered minimum cluster threshold from 40 -> 15 points so fragmented table legs survive depth dropout
    const size_t min_cluster_pts = _floor_standing_mode ? 15 : 20;
    std::vector<SegmentedCluster> raw_clusters;

    for (const auto& kv : grid) {
        if (visited[kv.first]) continue;

        std::queue<VoxelKey> q;
        q.push(kv.first);
        visited[kv.first] = true;

        std::vector<cv::Vec3f> cl_pts;
        cv::Vec3d sum(0.0, 0.0, 0.0);
        cv::Vec3f bmin(1e9f, 1e9f, 1e9f), bmax(-1e9f, -1e9f, -1e9f);

        while (!q.empty()) {
            VoxelKey cur = q.front();
            q.pop();

            auto it = grid.find(cur);
            if (it != grid.end()) {
                for (int idx : it->second) {
                    const auto& pt = fg_pts[idx];
                    cl_pts.push_back(pt);
                    sum += cv::Vec3d(pt[0], pt[1], pt[2]);
                    for (int d = 0; d < 3; ++d) {
                        bmin[d] = std::min(bmin[d], pt[d]);
                        bmax[d] = std::max(bmax[d], pt[d]);
                    }
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

        if (cl_pts.size() < min_cluster_pts) continue;

        cv::Vec3f centroid(
            static_cast<float>(sum[0] / cl_pts.size()),
            static_cast<float>(sum[1] / cl_pts.size()),
            static_cast<float>(sum[2] / cl_pts.size())
        );

        // Reject extreme off-center or far background clusters
        if (std::abs(centroid[0]) > 1.15f || centroid[2] > _max_depth_m) continue;

        // Centrality + density score
        double radial_offset = std::hypot(centroid[0], centroid[1] * 0.5);
        double score = static_cast<double>(cl_pts.size()) / (centroid[2] + 1.8 * radial_offset + 0.35);

        raw_clusters.push_back(SegmentedCluster{std::move(cl_pts), centroid, bmin, bmax, score});
    }

    if (raw_clusters.empty()) {
        out_merged_dense = fg_pts;
        return;
    }

    std::sort(raw_clusters.begin(), raw_clusters.end(),
              [](const SegmentedCluster& a, const SegmentedCluster& b) { return a.score > b.score; });

    if (_floor_standing_mode) {
        // Co-box Cluster Merging (Fixes fragmented table legs when front aluminum beam has depth == 0)
        const SegmentedCluster& seed = raw_clusters[0];
        const float max_dx = _expected_length_m + 0.12f; // 0.95m for 0.83m table
        const float max_dz = _expected_width_m + 0.14f;  // 0.58m for 0.44m table

        SegmentedCluster merged = seed;
        for (size_t i = 1; i < raw_clusters.size(); ++i) {
            const auto& cand = raw_clusters[i];
            float dx = std::abs(cand.centroid[0] - seed.centroid[0]);
            float dz = std::abs(cand.centroid[2] - seed.centroid[2]);
            if (dx <= max_dx && dz <= max_dz) {
                merged.points.insert(merged.points.end(), cand.points.begin(), cand.points.end());
                for (int d = 0; d < 3; ++d) {
                    merged.bbox_min[d] = std::min(merged.bbox_min[d], cand.bbox_min[d]);
                    merged.bbox_max[d] = std::max(merged.bbox_max[d], cand.bbox_max[d]);
                }
            }
        }
        out_merged_dense = merged.points;
        out_multi_clusters.push_back(std::move(merged));
    } else {
        // Multi-Instance Mode (e.g. 2 TANGY BBQ Sauce bottles lying next to each other on sofa)
        int count = std::min(static_cast<int>(raw_clusters.size()), _max_instances);
        for (int i = 0; i < count; ++i) {
            cv::Vec3f ext = raw_clusters[i].bbox_max - raw_clusters[i].bbox_min;
            float cluster_diag = cv::norm(ext);
            // If two identical objects touch each other (cluster_diag > 1.35 * bounding_diag),
            // split cluster along its principal horizontal axis so each instance gets its own pose!
            if (_enable_multi_instance && cluster_diag > 1.35f * _bounding_diag_m && raw_clusters[i].points.size() >= 60) {
                float mid_x = raw_clusters[i].centroid[0];
                SegmentedCluster left_inst, right_inst;
                for (const auto& pt : raw_clusters[i].points) {
                    if (pt[0] <= mid_x) left_inst.points.push_back(pt);
                    else right_inst.points.push_back(pt);
                }
                if (left_inst.points.size() >= 20) out_multi_clusters.push_back(std::move(left_inst));
                if (right_inst.points.size() >= 20) out_multi_clusters.push_back(std::move(right_inst));
            } else {
                out_multi_clusters.push_back(raw_clusters[i]);
            }
        }
        if (out_multi_clusters.empty()) {
            out_multi_clusters.push_back(raw_clusters[0]);
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

    // 50% High-gradient / boundary points + 50% Stratified spatial points
    std::vector<std::pair<float, size_t>> score_idx;
    score_idx.reserve(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        const auto& p = pts[i];
        const auto& p_prev = pts[(i > 0) ? (i - 1) : i];
        const auto& p_next = pts[(i + 1 < pts.size()) ? (i + 1) : i];
        float local_var = cv::norm(p - 0.5f * (p_prev + p_next));
        score_idx.emplace_back(local_var, i);
    }
    std::sort(score_idx.begin(), score_idx.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<cv::Vec3f> out;
    out.reserve(target_count);
    size_t half = target_count / 2;
    for (size_t i = 0; i < half; ++i) {
        out.push_back(pts[score_idx[i].second]);
    }

    size_t rem = target_count - half;
    double step = static_cast<double>(score_idx.size() - half) / static_cast<double>(rem);
    for (size_t i = 0; i < rem; ++i) {
        size_t pick = half + std::min(score_idx.size() - half - 1, static_cast<size_t>(i * step));
        out.push_back(pts[score_idx[pick].second]);
    }
    return out;
}

SegmentationResult CADSceneSegmenter::segment(const cv::Mat& depth_z16,
                                              const CameraIntrinsics& intr,
                                              const cv::Rect& roi_hint,
                                              size_t target_num_points) const {
    SegmentationResult res;
    std::vector<cv::Vec3f> raw_pts;
    extractDepthPoints(depth_z16, intr, roi_hint, raw_pts);
    if (raw_pts.size() < 20) return res;

    std::vector<cv::Vec3f> fg_pts;
    res.floor_detected = fitSupportPlaneRANSAC(raw_pts, res.floor_plane, fg_pts);
    if (fg_pts.size() < 15) {
        fg_pts = raw_pts;
    }

    clusterAndMergeTarget(fg_pts, res.dense_object_points, res.multi_clusters);
    if (res.dense_object_points.empty()) return res;

    res.object_points = samplePointsCurvatureAware(res.dense_object_points, target_num_points);
    for (auto& cl : res.multi_clusters) {
        cl.points = samplePointsCurvatureAware(cl.points, target_num_points);
    }
    res.valid = !res.object_points.empty();
    return res;
}

} // namespace tmcad

