#include "CADSceneSegmenter.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

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
                                                             int step,
                                                             const cv::Rect2f* rgb_roi) const
{
    std::vector<cv::Vec3f> points;
    const int w = static_cast<int>(frame.width);
    const int h = static_cast<int>(frame.height);
    const double fx = frame.intrinsics.fx;
    const double fy = frame.intrinsics.fy;
    const double cx = frame.intrinsics.ppx;
    const double cy = frame.intrinsics.ppy;
    const double scale = frame.depth_scale;

    int x0 = 0, x1 = w, y0 = 0, y1 = h;
    int cur_step = step;

    if (rgb_roi && rgb_roi->width > 10.0f && rgb_roi->height > 10.0f)
    {
        // Align RGB ROI to Depth with 12% safety margin
        // Bounding depth extraction strictly inside the RGB box eliminates 90% background noise/flying pixels!
        int pad_x = static_cast<int>(rgb_roi->width * 0.12f);
        int pad_y = static_cast<int>(rgb_roi->height * 0.12f);
        x0 = std::max(0, static_cast<int>(rgb_roi->x) - pad_x);
        y0 = std::max(0, static_cast<int>(rgb_roi->y) - pad_y);
        x1 = std::min(w, static_cast<int>(rgb_roi->x + rgb_roi->width) + pad_x);
        y1 = std::min(h, static_cast<int>(rgb_roi->y + rgb_roi->height) + pad_y);
        cur_step = std::max(1, step / 2); // Sample denser inside ROI
    }

    points.reserve((w / cur_step) * (h / cur_step));

    for (int y = y0; y < y1; y += cur_step)
    {
        const uint16_t* depth_row = frame.depth_z16.ptr<uint16_t>(y);
        for (int x = x0; x < x1; x += cur_step)
        {
            uint16_t d_raw = depth_row[x];
            if (d_raw == 0) continue;

            float z = static_cast<float>(d_raw * scale);
            if (z < static_cast<float>(_min_depth_m) || z > static_cast<float>(_max_depth_m)) continue;

            float px = static_cast<float>((x - cx) * z / fx);
            float py = static_cast<float>((y - cy) * z / fy);

            // Workspace bounding: filter extreme lateral points
            if (std::abs(px) > 2.0f || py < -1.4f || py > 1.4f) continue;

            points.emplace_back(px, py, z);
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

    // Seed pool: Sample from all valid scene depth points
    std::vector<int> seed_indices;
    seed_indices.reserve(N);
    for (int i = 0; i < N; ++i)
    {
        if (points[i][2] >= 0.40f && points[i][2] <= 3.20f) seed_indices.push_back(i);
    }
    if (seed_indices.size() < 100) return false;

    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(seed_indices.size()) - 1);

    int best_inliers = 0;
    cv::Vec3d best_normal(0.0, -1.0, 0.0);
    std::vector<uint8_t> best_mask(N, 0);

    for (int iter = 0; iter < _ransac_iterations; ++iter)
    {
        int i0 = seed_indices[dist(rng)];
        int i1 = seed_indices[dist(rng)];
        int i2 = seed_indices[dist(rng)];
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;

        cv::Vec3d p0(points[i0]);
        cv::Vec3d p1(points[i1]);
        cv::Vec3d p2(points[i2]);

        cv::Vec3d v1 = p1 - p0;
        cv::Vec3d v2 = p2 - p0;
        cv::Vec3d normal = v1.cross(v2);

        double n_len = cv::norm(normal);
        if (n_len < 1e-6) continue;
        normal *= (1.0 / n_len);

        // Orient normal UP (-Y in camera optical frame)
        if (normal.dot(expected_floor_normal) < 0.0) normal = -normal;

        // Check alignment with vertical: dot product with [0, -1, 0] must be >= 0.70 (tilt up to 45 deg)
        // Exactly matches real_sensor_pipeline.py constraint
        if (normal.dot(expected_floor_normal) < 0.70) continue;

        double d = -normal.dot(p0);
        double cam_h = std::abs(d);
        if (cam_h < _min_cam_height_m || cam_h > _max_cam_height_m) continue;

        int current_inliers = 0;
        int n_above = 0;
        int n_below = 0;
        std::vector<uint8_t> current_mask(N, 0);

        for (int i = 0; i < N; ++i)
        {
            cv::Vec3d pt(points[i]);
            double h = normal.dot(pt) + d;
            if (std::abs(h) < _ransac_distance_thresh_m)
            {
                ++current_inliers;
                current_mask[i] = 1;
            }
            else if (h > _ransac_distance_thresh_m)
            {
                ++n_above;
            }
            else
            {
                ++n_below;
            }
        }

        // A true floor must have objects standing ABOVE it, not below it!
        // If n_below > n_above * 0.25, this candidate is a tabletop/ceiling, NOT the floor.
        if (n_above < 100 || n_below > n_above * 0.25) continue;

        if (current_inliers > best_inliers)
        {
            best_inliers = current_inliers;
            best_normal = normal;
            best_mask = std::move(current_mask);

            if (best_inliers > static_cast<int>(0.60 * N))
            {
                break; // Early exit on strong consensus
            }
        }
    }

    if (best_inliers < static_cast<int>(_min_floor_inlier_ratio * N) || best_inliers < 100)
    {
        return false;
    }

    // Refinement via SVD on all inlier points
    cv::Vec3d centroid{0.0, 0.0, 0.0};
    for (int i = 0; i < N; ++i)
    {
        if (best_mask[i]) centroid += cv::Vec3d(points[i]);
    }
    centroid *= (1.0 / best_inliers);

    cv::Matx33d cov = cv::Matx33d::zeros();
    for (int i = 0; i < N; ++i)
    {
        if (best_mask[i])
        {
            cv::Vec3d diff = cv::Vec3d(points[i]) - centroid;
            cov += diff * diff.t();
        }
    }

    cv::Matx33d U, Vt;
    cv::Vec3d W;
    cv::SVD::compute(cov, W, U, Vt);

    // Normal corresponds to smallest eigenvalue (row 2 of Vt)
    cv::Vec3d refined_normal(Vt(2, 0), Vt(2, 1), Vt(2, 2));
    double n_norm = cv::norm(refined_normal);
    if (n_norm > 1e-6) refined_normal *= (1.0 / n_norm);
    if (refined_normal.dot(expected_floor_normal) < 0.0) refined_normal = -refined_normal;

    double refined_d = -refined_normal.dot(centroid);
    out_plane = cv::Vec4d(refined_normal[0], refined_normal[1], refined_normal[2], refined_d);
    out_height = std::abs(refined_d);

    double dot_up = std::clamp(refined_normal.dot(expected_floor_normal), -1.0, 1.0);
    out_tilt_deg = std::acos(dot_up) * 180.0 / CV_PI;

    return true;
}

std::vector<std::vector<cv::Vec3f>> CADSceneSegmenter::clusterCentralTable(const std::vector<cv::Vec3f>& candidates,
                                                                            cv::Rect2f& out_roi,
                                                                            const CameraIntrinsics& intr,
                                                                            int img_w, int img_h,
                                                                            const cv::Vec4d& floor_plane,
                                                                            bool floor_detected) const
{
    if (candidates.size() < 40)
    {
        out_roi = cv::Rect2f(0, 0, 0, 0);
        return {};
    }

    // Fast 3D Voxel Grid Hashing (Cell size = 0.08m, matching DBSCAN eps=0.08m)
    const float cell_size = 0.08f;
    const float inv_cell = 1.0f / cell_size;

    struct CellKey
    {
        int x, y, z;
        bool operator==(const CellKey& o) const { return x == o.x && y == o.y && z == o.z; }
    };

    struct KeyHasher
    {
        size_t operator()(const CellKey& k) const
        {
            return (static_cast<size_t>(k.x * 73856093) ^
                    static_cast<size_t>(k.y * 19349663) ^
                    static_cast<size_t>(k.z * 83492791));
        }
    };

    std::unordered_map<CellKey, std::vector<int>, KeyHasher> grid;
    grid.reserve(candidates.size() / 2);

    for (int i = 0; i < static_cast<int>(candidates.size()); ++i)
    {
        const auto& p = candidates[i];
        CellKey key{
            static_cast<int>(std::floor(p[0] * inv_cell)),
            static_cast<int>(std::floor(p[1] * inv_cell)),
            static_cast<int>(std::floor(p[2] * inv_cell))
        };
        grid[key].push_back(i);
    }

    // Connected components on voxel grid (26-connectivity)
    std::vector<bool> visited(candidates.size(), false);
    std::vector<std::vector<int>> clusters;

    for (const auto& pair : grid)
    {
        for (int pt_idx : pair.second)
        {
            if (visited[pt_idx]) continue;

            std::vector<int> current_cluster;
            std::vector<CellKey> queue;

            const auto& pt = candidates[pt_idx];
            CellKey start_key{
                static_cast<int>(std::floor(pt[0] * inv_cell)),
                static_cast<int>(std::floor(pt[1] * inv_cell)),
                static_cast<int>(std::floor(pt[2] * inv_cell))
            };
            queue.push_back(start_key);

            std::unordered_set<CellKey, KeyHasher> visited_cells;
            visited_cells.insert(start_key);

            // Seed cluster with all points in the starting cell
            for (int idx : pair.second)
            {
                if (!visited[idx])
                {
                    visited[idx] = true;
                    current_cluster.push_back(idx);
                }
            }

            while (!queue.empty())
            {
                CellKey cur = queue.back();
                queue.pop_back();

                for (int dx = -1; dx <= 1; ++dx)
                {
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        for (int dz = -1; dz <= 1; ++dz)
                        {
                            if (dx == 0 && dy == 0 && dz == 0) continue;
                            CellKey nb{cur.x + dx, cur.y + dy, cur.z + dz};
                            if (visited_cells.find(nb) != visited_cells.end()) continue;
                            visited_cells.insert(nb);

                            auto it = grid.find(nb);
                            if (it != grid.end())
                            {
                                queue.push_back(nb);
                                for (int idx : it->second)
                                {
                                    if (!visited[idx])
                                    {
                                        visited[idx] = true;
                                        current_cluster.push_back(idx);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (current_cluster.size() >= 20)
            {
                clusters.push_back(std::move(current_cluster));
            }
        }
    }

    if (clusters.empty())
    {
        // Fallback: take all candidates if clustering did not separate clusters
        std::vector<int> all_idx(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) all_idx[i] = static_cast<int>(i);
        clusters.push_back(all_idx);
    }

    // Rank clusters matching table geometry in valid operational range [0.65m, 3.20m]
    struct ScoredCluster
    {
        int cluster_idx;
        double score;
        cv::Vec3d center;
        double max_h;
    };
    std::vector<ScoredCluster> ranked;

    for (size_t c = 0; c < clusters.size(); ++c)
    {
        const auto& cl = clusters[c];
        if (cl.size() < 40) continue;

        cv::Vec3d center{0.0, 0.0, 0.0};
        double max_h = -1e9;
        double min_h = 1e9;

        for (int idx : cl) {
            const auto& p = candidates[idx];
            center += cv::Vec3d(p);

            double h = 0.0;
            if (floor_detected) {
                h = floor_plane[0] * p[0] + floor_plane[1] * p[1] + floor_plane[2] * p[2] + floor_plane[3];
            } else {
                h = -p[1] + 0.35;
            }
            if (h > max_h) max_h = h;
            if (h < min_h) min_h = h;
        }
        center *= (1.0 / cl.size());

        // Working operational range [0.65m, 3.20m], lateral |px| <= 1.20m
        if (center[2] < 0.65 || center[2] > 3.20 || std::abs(center[0]) > 1.20) {
            continue;
        }

        // REJECT FLOOR OBJECTS / PALLETS:
        // Pallet alone is only ~14cm tall. Table must reach at least 25cm above floor!
        if (floor_detected && max_h < 0.25) {
            continue;
        }

        // Scoring: Exactly matches real_sensor_pipeline.py:
        // d_center = np.linalg.norm(c[:2]) + c[2] * 0.2
        // score = n_pts / (d_center + 0.5)
        double d_center = std::sqrt(center[0] * center[0] + center[1] * center[1]) + center[2] * 0.20;
        double score = static_cast<double>(cl.size()) / (d_center + 0.50);

        ranked.push_back({static_cast<int>(c), score, center, max_h});
    }

    std::sort(ranked.begin(), ranked.end(), [](const ScoredCluster& a, const ScoredCluster& b) {
        return a.score > b.score;
    });

    if (ranked.empty())
    {
        // When individual clusters are sparse or separated across table profiles (legs/beams),
        // take all table candidates together as the unified table structure!
        if (candidates.size() >= 10) {
            std::vector<std::vector<cv::Vec3f>> result_all;
            result_all.push_back(candidates);
            return result_all;
        }
        return {};
    }

    std::vector<std::vector<cv::Vec3f>> result_clusters;
    int num_to_take = std::min(3, static_cast<int>(ranked.size()));
    for (int k = 0; k < num_to_take; ++k)
    {
        const auto& cl = clusters[ranked[k].cluster_idx];
        std::vector<cv::Vec3f> pts;
        pts.reserve(cl.size());
        for (int idx : cl) pts.push_back(candidates[idx]);
        result_clusters.push_back(std::move(pts));
    }

    // 2D ROI from top cluster
    const auto& best_cl = clusters[ranked[0].cluster_idx];
    float min_u = static_cast<float>(img_w);
    float max_u = 0.0f;
    float min_v = static_cast<float>(img_h);
    float max_v = 0.0f;

    for (int idx : best_cl)
    {
        const auto& p = candidates[idx];
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

    // Step 1: Sub-sample depth points from full depth frame for robust floor plane fitting and table extraction
    std::vector<cv::Vec3f> raw_pts = extractDepthPoints(frame, 1, nullptr);
    res.raw_points_count = static_cast<int>(raw_pts.size());

    if (raw_pts.size() < 100)
    {
        res.status = "too_few_depth_points";
        return res;
    }

    // Step 2: Fit floor plane
    cv::Vec4d floor_plane;
    double floor_height = 0.0;
    double floor_tilt = 0.0;
    bool floor_found = fitFloorPlane(raw_pts, floor_plane, floor_height, floor_tilt);

    res.floor_detected = floor_found;
    res.floor_plane = floor_plane;
    res.floor_height_m = floor_height;
    res.floor_tilt_deg = floor_tilt;

    // Step 3: Elevation & 2D/3D Frustum ROI filtering
    std::vector<cv::Vec3f> table_candidates;
    table_candidates.reserve(raw_pts.size() / 2);

    const bool use_roi_frustum = (rgb_roi && rgb_roi->width > 20.0f && rgb_roi->height > 20.0f);
    float roi_u0 = 0.0f, roi_u1 = static_cast<float>(frame.width);
    float roi_v0 = 0.0f, roi_v1 = static_cast<float>(frame.height);
    float min_z_filter = 0.50f, max_z_filter = 3.20f;
    double max_elev_above_floor = 1.40;

    if (use_roi_frustum) {
        // Tight 25px lateral margin to strictly exclude foreground peripheral clutter
        const float pad_x = 25.0f;
        const float pad_y = 25.0f;
        roi_u0 = std::max(0.0f, rgb_roi->x - pad_x);
        roi_u1 = std::min(static_cast<float>(frame.width), rgb_roi->x + rgb_roi->width + pad_x);
        roi_v0 = std::max(0.0f, rgb_roi->y - pad_y);
        roi_v1 = std::min(static_cast<float>(frame.height), rgb_roi->y + rgb_roi->height + pad_y);

        const float est_z = static_cast<float>((frame.intrinsics.fx * 2.0 * _half_l) / rgb_roi->width);
        if (est_z >= 0.50f && est_z <= 2.50f) {
            min_z_filter = std::max(0.50f, est_z - 0.15f);
            max_z_filter = est_z + static_cast<float>(2.0 * _half_w) + 0.30f;
        }
        max_elev_above_floor = _total_h + 0.12; // 0.625m max height above floor for table
    }

    if (floor_found)
    {
        cv::Vec3d normal(floor_plane[0], floor_plane[1], floor_plane[2]);
        double d = floor_plane[3];

        for (const auto& p : raw_pts)
        {
            if (p[2] < min_z_filter || p[2] > max_z_filter) continue;

            if (use_roi_frustum) {
                float u = static_cast<float>(frame.intrinsics.fx * p[0] / p[2] + frame.intrinsics.ppx);
                float v = static_cast<float>(frame.intrinsics.fy * p[1] / p[2] + frame.intrinsics.ppy);
                if (u < roi_u0 || u > roi_u1 || v < roi_v0 || v > roi_v1) continue;
            }

            cv::Vec3d pt(p);
            double h_above_floor = normal.dot(pt) + d; // Signed distance along UP vector

            if (h_above_floor >= 0.025 && h_above_floor <= max_elev_above_floor && std::abs(p[0]) <= 1.50f)
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
            if (p[2] >= min_z_filter && p[2] <= max_z_filter && std::abs(p[0]) <= 1.50f)
            {
                if (use_roi_frustum) {
                    float u = static_cast<float>(frame.intrinsics.fx * p[0] / p[2] + frame.intrinsics.ppx);
                    float v = static_cast<float>(frame.intrinsics.fy * p[1] / p[2] + frame.intrinsics.ppy);
                    if (u < roi_u0 || u > roi_u1 || v < roi_v0 || v > roi_v1) continue;
                }
                table_candidates.push_back(p);
            }
        }
    }

    if (table_candidates.size() < 10)
    {
        res.status = "no_table_candidates_above_floor";
        return res;
    }

    // Step 4: When Frustum 2D/3D ROI is active, table points are already isolated from peripheral clutter.
    // Preserving all table candidates as a unified structure prevents fracturing front/rear frames and legs.
    if (use_roi_frustum && table_candidates.size() >= 50)
    {
        std::vector<cv::Vec3f> sampled;
        sampled.reserve(target_points);
        const double stride = static_cast<double>(table_candidates.size()) / static_cast<double>(target_points);
        for (int i = 0; i < target_points; ++i)
        {
            size_t idx = static_cast<size_t>(i * stride);
            if (idx >= table_candidates.size()) idx = table_candidates.size() - 1;
            sampled.push_back(table_candidates[idx]);
        }
        res.roi_2d = *rgb_roi;
        res.table_cluster_points_count = static_cast<int>(table_candidates.size());
        res.table_points = sampled;
        res.cluster_candidates.push_back(std::move(sampled));
        res.valid = true;
        res.status = "ok";
        return res;
    }

    // Fallback: Spatial voxel clustering to isolate table candidate clusters when ROI is not provided
    int img_w = static_cast<int>(frame.width);
    int img_h = static_cast<int>(frame.height);
    std::vector<std::vector<cv::Vec3f>> all_cl_pts = clusterCentralTable(table_candidates,
                                                                         res.roi_2d,
                                                                         frame.intrinsics,
                                                                         img_w, img_h,
                                                                         floor_plane,
                                                                         floor_found);

    if (all_cl_pts.empty() || all_cl_pts[0].size() < 10)
    {
        res.status = "table_cluster_too_small";
        return res;
    }

    res.table_cluster_points_count = static_cast<int>(all_cl_pts[0].size());

    if (rgb_roi && rgb_roi->width > 10.0f && rgb_roi->height > 10.0f)
    {
        res.roi_2d = *rgb_roi;
    }

    // Step 5: Subsample each cluster candidate to target_points (1024)
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

    res.table_points = res.cluster_candidates[0];
    res.valid = true;
    res.status = "ok";
    return res;
}
