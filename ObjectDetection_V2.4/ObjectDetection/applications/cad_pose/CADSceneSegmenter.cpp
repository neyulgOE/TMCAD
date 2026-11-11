#include "CADSceneSegmenter.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
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
    // The ground plane normal pointing UP has expected vector [0, -1, 0].
    const cv::Vec3d expected_up(0.0, -1.0, 0.0);

    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> dist(0, N - 1);

    int best_inliers = 0;
    cv::Vec3d best_normal(0.0, -1.0, 0.0);
    double best_d = 0.0;
    std::vector<uint8_t> best_mask(N, 0);

    for (int iter = 0; iter < _ransac_iterations; ++iter)
    {
        int i0 = dist(rng);
        int i1 = dist(rng);
        int i2 = dist(rng);
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

        // Orient normal UP (-Y)
        if (normal[1] > 0.0) normal = -normal;

        // Check alignment with vertical: dot product with [0, -1, 0] is -normal[1]
        double cos_tilt = -normal[1];
        if (cos_tilt < 0.707) continue; // Tilt angle > 45 deg, not a floor

        double d = -normal.dot(p0);
        double cam_h = std::abs(d);
        if (cam_h < _min_cam_height_m || cam_h > _max_cam_height_m) continue;

        int current_inliers = 0;
        std::vector<uint8_t> current_mask(N, 0);

        for (int i = 0; i < N; ++i)
        {
            cv::Vec3d pt(points[i]);
            double dist_plane = std::abs(normal.dot(pt) + d);
            if (dist_plane < _ransac_distance_thresh_m)
            {
                ++current_inliers;
                current_mask[i] = 1;
            }
        }

        if (current_inliers > best_inliers)
        {
            best_inliers = current_inliers;
            best_normal = normal;
            best_d = d;
            best_mask = std::move(current_mask);

            if (best_inliers > static_cast<int>(0.45 * N))
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
    if (refined_normal[1] > 0.0) refined_normal = -refined_normal;

    double refined_d = -refined_normal.dot(centroid);
    out_plane = cv::Vec4d(refined_normal[0], refined_normal[1], refined_normal[2], refined_d);
    out_height = std::abs(refined_d);

    double dot_up = std::clamp(-refined_normal[1], -1.0, 1.0);
    out_tilt_deg = std::acos(dot_up) * 180.0 / CV_PI;

    return true;
}

std::vector<cv::Vec3f> CADSceneSegmenter::clusterCentralTable(const std::vector<cv::Vec3f>& candidates,
                                                             cv::Rect2f& out_roi,
                                                             const CameraIntrinsics& intr,
                                                             int img_w, int img_h) const
{
    if (candidates.size() < 40)
    {
        out_roi = cv::Rect2f(0, 0, 0, 0);
        return {};
    }

    // Fast 3D Voxel Grid Hashing (Cell size = 0.09m)
    const float cell_size = 0.09f;
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
                            CellKey nb{cur.x + dx, cur.y + dy, cur.z + dz};
                            auto it = grid.find(nb);
                            if (it != grid.end())
                            {
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

            if (current_cluster.size() >= 50)
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

    // Select the best cluster matching table center-line
    int best_cluster_idx = 0;
    double best_score = -1e9;

    for (size_t c = 0; c < clusters.size(); ++c)
    {
        const auto& cl = clusters[c];
        cv::Vec3d center{0.0, 0.0, 0.0};
        for (int idx : cl) center += cv::Vec3d(candidates[idx]);
        center *= (1.0 / cl.size());

        // Prioritize clusters right in front of camera (small X offset, reasonable Z distance)
        double dist_center = std::abs(center[0]) + center[2] * 0.25;
        double score = static_cast<double>(cl.size()) / (dist_center + 0.35);

        if (score > best_score)
        {
            best_score = score;
            best_cluster_idx = static_cast<int>(c);
        }
    }

    const auto& best_cl = clusters[best_cluster_idx];
    std::vector<cv::Vec3f> table_pts;
    table_pts.reserve(best_cl.size());

    float min_u = static_cast<float>(img_w);
    float max_u = 0.0f;
    float min_v = static_cast<float>(img_h);
    float max_v = 0.0f;

    for (int idx : best_cl)
    {
        const auto& p = candidates[idx];
        table_pts.push_back(p);

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

    return table_pts;
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

    // Step 1: Sub-sample depth points (bounded by RGB ROI if available)
    std::vector<cv::Vec3f> raw_pts = extractDepthPoints(frame, 4, rgb_roi);
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

    // Step 3: Elevation band filtering
    std::vector<cv::Vec3f> table_candidates;
    table_candidates.reserve(raw_pts.size() / 2);

    if (floor_found)
    {
        cv::Vec3d normal(floor_plane[0], floor_plane[1], floor_plane[2]);
        double d = floor_plane[3];

        for (const auto& p : raw_pts)
        {
            cv::Vec3d pt(p);
            double h_above_floor = normal.dot(pt) + d; // Signed distance along UP vector

            // Table elevation slice: between 4 cm and 1.25 m above floor
            if (h_above_floor >= 0.04 && h_above_floor <= 1.25)
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
            if (p[2] >= 0.40f && p[2] <= 2.80f && p[1] >= -0.60f && p[1] <= 0.60f)
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

    // Step 4: Spatial voxel clustering to isolate central table
    int img_w = static_cast<int>(frame.width);
    int img_h = static_cast<int>(frame.height);
    std::vector<cv::Vec3f> clustered_pts = clusterCentralTable(table_candidates,
                                                               res.roi_2d,
                                                               frame.intrinsics,
                                                               img_w, img_h);

    res.table_cluster_points_count = static_cast<int>(clustered_pts.size());

    if (rgb_roi && rgb_roi->width > 10.0f && rgb_roi->height > 10.0f)
    {
        res.roi_2d = *rgb_roi;
    }

    if (clustered_pts.size() < 40)
    {
        res.status = "table_cluster_too_small";
        return res;
    }

    // Step 5: Subsample exactly target_points (1024)
    res.table_points.reserve(target_points);

    if (clustered_pts.size() >= static_cast<size_t>(target_points))
    {
        double stride = static_cast<double>(clustered_pts.size()) / static_cast<double>(target_points);
        for (int i = 0; i < target_points; ++i)
        {
            size_t idx = static_cast<size_t>(i * stride);
            if (idx >= clustered_pts.size()) idx = clustered_pts.size() - 1;
            res.table_points.push_back(clustered_pts[idx]);
        }
    }
    else
    {
        // Wrap-around sampling when points are between 40 and 1024
        for (int i = 0; i < target_points; ++i)
        {
            res.table_points.push_back(clustered_pts[i % clustered_pts.size()]);
        }
    }

    res.valid = true;
    res.status = "ok";
    return res;
}
