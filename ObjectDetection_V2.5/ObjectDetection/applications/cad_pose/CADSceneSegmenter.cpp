#include "CADSceneSegmenter.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

namespace {

struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct VoxelKeyHasher {
    size_t operator()(const VoxelKey& k) const {
        return (static_cast<size_t>(k.x * 73856093) ^
                static_cast<size_t>(k.y * 19349663) ^
                static_cast<size_t>(k.z * 83492791));
    }
};

struct VoxelAccum {
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_z = 0.0f;
    int count = 0;
};

} // namespace

void CADSceneSegmenter::configure(double half_w, double half_l, double total_height,
                                  double min_depth, double max_depth)
{
    _half_w = half_w;
    _half_l = half_l;
    _total_h = total_height;
    _min_depth_m = min_depth;
    _max_depth_m = max_depth;
}

std::vector<cv::Vec3f> CADSceneSegmenter::extractDepthPoints(const RgbdFrame& frame,
                                                             const cv::Rect2f* rgb_roi) const
{
    const int w = static_cast<int>(frame.width);
    const int h = static_cast<int>(frame.height);
    const double fx = frame.intrinsics.fx;
    const double fy = frame.intrinsics.fy;
    const double cx = frame.intrinsics.ppx;
    const double cy = frame.intrinsics.ppy;
    const double scale = frame.depth_scale;

    int x0 = 0, x1 = w, y0 = 0, y1 = h;
    if (rgb_roi && rgb_roi->width > 10.0f && rgb_roi->height > 10.0f)
    {
        int pad_x = static_cast<int>(rgb_roi->width * 0.12f);
        int pad_y = static_cast<int>(rgb_roi->height * 0.12f);
        x0 = std::max(0, static_cast<int>(rgb_roi->x) - pad_x);
        y0 = std::max(0, static_cast<int>(rgb_roi->y) - pad_y);
        x1 = std::min(w, static_cast<int>(rgb_roi->x + rgb_roi->width) + pad_x);
        y1 = std::min(h, static_cast<int>(rgb_roi->y + rgb_roi->height) + pad_y);
    }

    const float voxel_m = static_cast<float>(_voxel_size_m);
    const float inv_voxel_m = 1.0f / voxel_m;

    std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHasher> voxels;
    voxels.reserve(25000);

    const float min_z = static_cast<float>(_workspace_min_z);
    const float max_z = static_cast<float>(_workspace_max_z);
    const float max_abs_x = static_cast<float>(_workspace_max_abs_x);
    const uint16_t min_d = static_cast<uint16_t>(min_z / scale);
    const uint16_t max_d = static_cast<uint16_t>(max_z / scale);

    for (int y = y0; y < y1; ++y)
    {
        const uint16_t* depth_row = frame.depth_z16.ptr<uint16_t>(y);
        for (int x = x0; x < x1; ++x)
        {
            uint16_t d = depth_row[x];
            if (d < min_d || d > max_d) continue;

            float z = static_cast<float>(d * scale);
            float px = static_cast<float>((x - cx) * z / fx);
            if (std::abs(px) > max_abs_x) continue;

            float py = static_cast<float>((y - cy) * z / fy);

            VoxelKey key{
                static_cast<int>(std::floor(px * inv_voxel_m)),
                static_cast<int>(std::floor(py * inv_voxel_m)),
                static_cast<int>(std::floor(z * inv_voxel_m))
            };
            auto& v = voxels[key];
            v.sum_x += px;
            v.sum_y += py;
            v.sum_z += z;
            ++v.count;
        }
    }

    std::vector<cv::Vec3f> points;
    points.reserve(voxels.size());
    for (const auto& kv : voxels)
    {
        if (kv.second.count > 0)
        {
            float inv = 1.0f / static_cast<float>(kv.second.count);
            points.emplace_back(kv.second.sum_x * inv,
                                kv.second.sum_y * inv,
                                kv.second.sum_z * inv);
        }
    }

    return points;
}

bool CADSceneSegmenter::fitFloorPlane(const std::vector<cv::Vec3f>& points,
                                      cv::Vec4d& out_plane,
                                      double& out_height,
                                      double& out_tilt_deg) const
{
    const int N = static_cast<int>(points.size());
    if (N < 200) return false;

    // Optical frame convention: Y is down towards the floor.
    // Floor normal points up along -Y. Expected normal = [0, -1, 0].
    const cv::Vec3d expected_floor_normal(0.0, -1.0, 0.0);
    const double floor_thresh = _ransac_distance_thresh_m;

    std::vector<float> xs(N), ys(N), zs(N);
    for (int i = 0; i < N; ++i)
    {
        xs[i] = points[i][0];
        ys[i] = points[i][1];
        zs[i] = points[i][2];
    }

    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> dist(0, N - 1);

    double best_score = -1.0;
    cv::Vec3d best_normal(0.0, -1.0, 0.0);
    double best_d = 0.0;
    std::vector<uint8_t> best_inliers(N, 0);

    for (int iter = 0; iter < _ransac_iterations; ++iter)
    {
        int i0 = dist(rng), i1 = dist(rng), i2 = dist(rng);
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;

        cv::Vec3d p0(points[i0]), p1(points[i1]), p2(points[i2]);
        cv::Vec3d v1 = p1 - p0;
        cv::Vec3d v2 = p2 - p0;
        cv::Vec3d normal = v1.cross(v2);

        double n_len = cv::norm(normal);
        if (n_len < 1e-7) continue;
        normal *= (1.0 / n_len);

        // Orient normal UP (-Y in camera optical frame)
        if (normal.dot(expected_floor_normal) < 0.0) normal = -normal;

        // Constraint: cos(n, [0, -1, 0]) >= 0.88 (within ~28 deg of vertical)
        if (normal.dot(expected_floor_normal) < 0.88) continue;

        double d_cand = -normal.dot(p0);
        double h_cand = std::abs(d_cand);
        if (h_cand < _min_cam_height_m || h_cand > _max_cam_height_m) continue;

        int count = 0, n_above = 0, n_below = 0;
        std::vector<uint8_t> inlier_mask(N, 0);

        const float nx = static_cast<float>(normal[0]);
        const float ny = static_cast<float>(normal[1]);
        const float nz = static_cast<float>(normal[2]);
        const float dc = static_cast<float>(d_cand);
        const float th = static_cast<float>(floor_thresh);

        for (int i = 0; i < N; ++i)
        {
            float h = nx * xs[i] + ny * ys[i] + nz * zs[i] + dc;
            if (std::abs(h) < th)
            {
                ++count;
                inlier_mask[i] = 1;
            }
            else if (h > th)
            {
                ++n_above;
            }
            else
            {
                ++n_below;
            }
        }

        // A true floor must have objects standing ABOVE it, not below it!
        // If n_below > n_above * 0.25, this plane is a tabletop/ceiling, NOT the floor.
        if (n_above < 200 || n_below > static_cast<int>(n_above * 0.25)) continue;

        double score = static_cast<double>(count) + static_cast<double>(n_above) * 0.3;
        if (score > best_score)
        {
            best_score = score;
            best_normal = normal;
            best_d = d_cand;
            best_inliers = std::move(inlier_mask);
        }
    }

    if (best_score < 0.0) return false;

    // Refit via SVD on all inlier points
    int inlier_cnt = 0;
    cv::Vec3d centroid(0.0, 0.0, 0.0);
    for (int i = 0; i < N; ++i)
    {
        if (best_inliers[i])
        {
            centroid += cv::Vec3d(points[i]);
            ++inlier_cnt;
        }
    }

    if (inlier_cnt < 100) return false;
    centroid *= (1.0 / inlier_cnt);

    cv::Matx33d cov = cv::Matx33d::zeros();
    for (int i = 0; i < N; ++i)
    {
        if (best_inliers[i])
        {
            cv::Vec3d diff = cv::Vec3d(points[i]) - centroid;
            cov += diff * diff.t();
        }
    }

    cv::Matx33d U, Vt;
    cv::Vec3d W;
    cv::SVD::compute(cov, W, U, Vt);

    cv::Vec3d refined_normal(Vt(2, 0), Vt(2, 1), Vt(2, 2));
    double n_norm = cv::norm(refined_normal);
    if (n_norm > 1e-6) refined_normal *= (1.0 / n_norm);
    if (refined_normal.dot(expected_floor_normal) < 0.0) refined_normal = -refined_normal;

    double refined_d = -refined_normal.dot(centroid);
    out_plane = cv::Vec4d(refined_normal[0], refined_normal[1], refined_normal[2], refined_d);
    out_height = std::abs(refined_d);

    double dot_up = std::clamp(refined_normal.dot(expected_floor_normal), -1.0, 1.0);
    out_tilt_deg = std::acos(dot_up) * 180.0 / CV_PI;

    bool is_floor = (out_tilt_deg < 28.5) &&
                    (out_height >= _min_cam_height_m) &&
                    (out_height <= _max_cam_height_m) &&
                    (inlier_cnt >= 200);

    return is_floor;
}

std::vector<std::vector<cv::Vec3f>> CADSceneSegmenter::clusterCentralTable(const std::vector<cv::Vec3f>& candidates,
                                                                          cv::Rect2f& out_roi,
                                                                          const CameraIntrinsics& intr,
                                                                          int img_w, int img_h) const
{
    const int N = static_cast<int>(candidates.size());
    if (N < _dbscan_min_points)
    {
        out_roi = cv::Rect2f(0, 0, 0, 0);
        return {};
    }

    const float eps = static_cast<float>(_dbscan_eps);
    const float eps2 = eps * eps;
    const float inv_eps = 1.0f / eps;
    const int min_pts = _dbscan_min_points;

    // Spatial hash grid for accelerated neighbor search
    std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHasher> grid;
    grid.reserve(N);

    for (int i = 0; i < N; ++i)
    {
        VoxelKey k{
            static_cast<int>(std::floor(candidates[i][0] * inv_eps)),
            static_cast<int>(std::floor(candidates[i][1] * inv_eps)),
            static_cast<int>(std::floor(candidates[i][2] * inv_eps))
        };
        grid[k].push_back(i);
    }

    auto get_neighbors = [&](int idx, std::vector<int>& out_nbrs) {
        out_nbrs.clear();
        const auto& pi = candidates[idx];
        int cx = static_cast<int>(std::floor(pi[0] * inv_eps));
        int cy = static_cast<int>(std::floor(pi[1] * inv_eps));
        int cz = static_cast<int>(std::floor(pi[2] * inv_eps));

        for (int dx = -1; dx <= 1; ++dx)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dz = -1; dz <= 1; ++dz)
                {
                    VoxelKey nk{cx + dx, cy + dy, cz + dz};
                    auto it = grid.find(nk);
                    if (it != grid.end())
                    {
                        for (int j : it->second)
                        {
                            float d2 = (pi[0] - candidates[j][0]) * (pi[0] - candidates[j][0]) +
                                       (pi[1] - candidates[j][1]) * (pi[1] - candidates[j][1]) +
                                       (pi[2] - candidates[j][2]) * (pi[2] - candidates[j][2]);
                            if (d2 <= eps2) out_nbrs.push_back(j);
                        }
                    }
                }
            }
        }
    };

    // Standard DBSCAN clustering algorithm
    std::vector<int> labels(N, -1);
    int cur_label = 0;
    std::vector<int> nbrs, sub_nbrs;
    nbrs.reserve(256);
    sub_nbrs.reserve(256);

    for (int i = 0; i < N; ++i)
    {
        if (labels[i] != -1) continue;
        get_neighbors(i, nbrs);
        if (static_cast<int>(nbrs.size()) < min_pts)
        {
            continue; // Noise point or border
        }

        labels[i] = cur_label;
        std::vector<int> seed_set = nbrs;
        size_t qi = 0;
        while (qi < seed_set.size())
        {
            int q = seed_set[qi++];
            if (labels[q] == -1) labels[q] = cur_label;
            else if (labels[q] >= 0) continue;
            labels[q] = cur_label;

            get_neighbors(q, sub_nbrs);
            if (static_cast<int>(sub_nbrs.size()) >= min_pts)
            {
                for (int sn : sub_nbrs)
                {
                    if (labels[sn] < 0)
                    {
                        seed_set.push_back(sn);
                    }
                }
            }
        }
        cur_label++;
    }

    if (cur_label == 0)
    {
        out_roi = cv::Rect2f(0, 0, 0, 0);
        return {};
    }

    // Rank clusters matching table geometry in valid operational range
    std::vector<cv::Vec3d> cluster_centroids(cur_label, cv::Vec3d(0, 0, 0));
    std::vector<int> cluster_counts(cur_label, 0);

    for (int i = 0; i < N; ++i)
    {
        int l = labels[i];
        if (l >= 0)
        {
            cluster_centroids[l] += cv::Vec3d(candidates[i]);
            cluster_counts[l]++;
        }
    }

    int best_label = -1;
    double best_score = -1.0;

    for (int l = 0; l < cur_label; ++l)
    {
        int n_pts = cluster_counts[l];
        if (n_pts < min_pts) continue;

        cluster_centroids[l] *= (1.0 / n_pts);
        cv::Vec3d c = cluster_centroids[l];

        // Central camera field of view constraints
        if (c[2] > 2.05 || std::abs(c[0]) > 0.70) continue;

        double d_center = std::hypot(c[0], c[1]) + c[2] * 0.5;
        double score = static_cast<double>(n_pts) / (d_center + 0.2);
        if (score > best_score)
        {
            best_score = score;
            best_label = l;
        }
    }

    if (best_label < 0)
    {
        out_roi = cv::Rect2f(0, 0, 0, 0);
        return {};
    }

    // Merge central table components within 0.85m of best cluster centroid
    cv::Vec3d c_best = cluster_centroids[best_label];
    std::vector<cv::Vec3f> merged_table_pts;
    for (int l = 0; l < cur_label; ++l)
    {
        if (cluster_counts[l] < min_pts) continue;
        cv::Vec3d cl = cluster_centroids[l];
        if (cv::norm(cl - c_best) < 0.85 && cl[2] <= 2.05)
        {
            for (int i = 0; i < N; ++i)
            {
                if (labels[i] == l)
                {
                    merged_table_pts.push_back(candidates[i]);
                }
            }
        }
    }

    std::vector<std::vector<cv::Vec3f>> result_clusters;
    result_clusters.push_back(std::move(merged_table_pts));

    // Also include top individual clusters as multi-candidate fallback
    std::vector<std::pair<double, int>> ranked_indices;
    for (int l = 0; l < cur_label; ++l)
    {
        if (cluster_counts[l] >= min_pts)
        {
            cv::Vec3d c = cluster_centroids[l];
            double d_center = std::hypot(c[0], c[1]) + c[2] * 0.5;
            double score = static_cast<double>(cluster_counts[l]) / (d_center + 0.2);
            ranked_indices.emplace_back(score, l);
        }
    }
    std::sort(ranked_indices.rbegin(), ranked_indices.rend());

    int num_to_take = std::min(3, static_cast<int>(ranked_indices.size()));
    for (int k = 0; k < num_to_take; ++k)
    {
        int l = ranked_indices[k].second;
        std::vector<cv::Vec3f> pts;
        pts.reserve(cluster_counts[l]);
        for (int i = 0; i < N; ++i)
        {
            if (labels[i] == l) pts.push_back(candidates[i]);
        }
        result_clusters.push_back(std::move(pts));
    }

    // Project 2D ROI from merged table points
    float min_u = static_cast<float>(img_w);
    float max_u = 0.0f;
    float min_v = static_cast<float>(img_h);
    float max_v = 0.0f;

    for (const auto& p : result_clusters[0])
    {
        float u = static_cast<float>(intr.fx * p[0] / p[2] + intr.ppx);
        float v = static_cast<float>(intr.fy * p[1] / p[2] + intr.ppy);

        min_u = std::min(min_u, u);
        max_u = std::max(max_u, u);
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }

    min_u = std::clamp(min_u, 0.0f, static_cast<float>(img_w - 1));
    max_u = std::clamp(max_u, 0.0f, static_cast<float>(img_w - 1));
    min_v = std::clamp(min_v, 0.0f, static_cast<float>(img_h - 1));
    max_v = std::clamp(max_v, 0.0f, static_cast<float>(img_h - 1));

    if (max_u > min_u && max_v > min_v)
    {
        out_roi = cv::Rect2f(min_u, min_v, max_u - min_u, max_v - min_v);
    }
    else
    {
        out_roi = cv::Rect2f(0, 0, static_cast<float>(img_w), static_cast<float>(img_h));
    }

    return result_clusters;
}

CADSceneSegmenter::SegmentResult CADSceneSegmenter::segment(const RgbdFrame& frame,
                                                          int target_points,
                                                          const cv::Rect2f* rgb_roi) const
{
    SegmentResult res;
    if (!frame.valid())
    {
        res.status = "invalid_frame";
        return res;
    }

    // Step 1: 5mm voxel downsampling and target workspace ROI extraction
    std::vector<cv::Vec3f> raw_pts = extractDepthPoints(frame, rgb_roi);
    res.raw_points_count = static_cast<int>(raw_pts.size());

    if (raw_pts.size() < 100)
    {
        res.status = "too_few_depth_points";
        return res;
    }

    // Step 2: Fit floor plane via gravity-constrained RANSAC (1000 iter, seed 1337)
    cv::Vec4d floor_plane;
    double floor_height = 0.0;
    double floor_tilt = 0.0;
    bool floor_found = fitFloorPlane(raw_pts, floor_plane, floor_height, floor_tilt);

    res.floor_detected = floor_found;
    res.floor_plane = floor_plane;
    res.floor_height_m = floor_height;
    res.floor_tilt_deg = floor_tilt;

    // Step 3: Elevation band filtering (matches real_sensor_pipeline.py: dists in [0.03m, 1.40m])
    std::vector<cv::Vec3f> table_candidates;
    table_candidates.reserve(raw_pts.size() / 2);

    if (floor_found)
    {
        cv::Vec3d normal(floor_plane[0], floor_plane[1], floor_plane[2]);
        double d = floor_plane[3];

        for (const auto& p : raw_pts)
        {
            cv::Vec3d pt(p);
            double h_above_floor = normal.dot(pt) + d;

            if (h_above_floor > 0.03 && h_above_floor < 1.40 &&
                p[2] <= 2.10f && std::abs(p[0]) <= 0.75f)
            {
                table_candidates.push_back(p);
            }
        }
    }
    else
    {
        // Fallback: depth band filter if floor was not distinct
        for (const auto& p : raw_pts)
        {
            if (p[2] >= 0.50f && p[2] <= 2.10f && std::abs(p[0]) <= 0.75f)
            {
                table_candidates.push_back(p);
            }
        }
    }

    if (table_candidates.size() < 50)
    {
        res.status = "no_table_candidates_above_floor";
        return res;
    }

    // Step 4: True DBSCAN spatial clustering to isolate discrete table cluster
    int img_w = static_cast<int>(frame.width);
    int img_h = static_cast<int>(frame.height);
    std::vector<std::vector<cv::Vec3f>> all_cl_pts = clusterCentralTable(table_candidates,
                                                                         res.roi_2d,
                                                                         frame.intrinsics,
                                                                         img_w, img_h);

    if (all_cl_pts.empty() || all_cl_pts[0].size() < 50)
    {
        res.status = "table_cluster_too_small";
        return res;
    }

    res.table_cluster_points_count = static_cast<int>(all_cl_pts[0].size());

    if (rgb_roi && rgb_roi->width > 10.0f && rgb_roi->height > 10.0f)
    {
        res.roi_2d = *rgb_roi;
    }

    // Step 5: Subsample cluster candidates to exactly target_points (1024)
    for (const auto& cl_pts : all_cl_pts)
    {
        std::vector<cv::Vec3f> sampled;
        sampled.reserve(target_points);

        if (cl_pts.size() >= static_cast<size_t>(target_points))
        {
            double stride = static_cast<double>(cl_pts.size()) / static_cast<double>(target_points);
            for (int i = 0; i < target_points; ++i)
            {
                size_t idx = static_cast<size_t>(i * stride);
                if (idx >= cl_pts.size()) idx = cl_pts.size() - 1;
                sampled.push_back(cl_pts[idx]);
            }
        }
        else
        {
            for (int i = 0; i < target_points; ++i)
            {
                sampled.push_back(cl_pts[i % cl_pts.size()]);
            }
        }
        res.cluster_candidates.push_back(std::move(sampled));
    }

    res.dense_cluster_points = all_cl_pts[0];
    res.table_points = res.cluster_candidates[0];
    res.valid = true;
    res.status = "ok";
    return res;
}
