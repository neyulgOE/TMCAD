#include "CADTemplateDetector.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

namespace
{
constexpr double kPi = 3.14159265358979323846;

cv::Matx33d rotationX(double rad)
{
    double c = std::cos(rad);
    double s = std::sin(rad);
    return cv::Matx33d(
        1.0, 0.0,  0.0,
        0.0,   c,   -s,
        0.0,   s,    c
    );
}

cv::Matx33d rotationZ(double rad)
{
    double c = std::cos(rad);
    double s = std::sin(rad);
    return cv::Matx33d(
          c,  -s,  0.0,
          s,   c,  0.0,
        0.0, 0.0,  1.0
    );
}
}

bool CADTemplateDetector::loadBinaryTemplates(const std::string& bin_path, std::string& error)
{
    error.clear();
    std::ifstream ifs(bin_path, std::ios::binary);
    if (!ifs.is_open())
    {
        error = "Failed to open binary templates file: " + bin_path;
        return false;
    }

    char magic[8];
    ifs.read(magic, 8);
    if (ifs.gcount() < 8 || std::memcmp(magic, "CADTPL01", 8) != 0)
    {
        error = "Invalid CAD template binary header in " + bin_path;
        return false;
    }

    uint32_t num_templates = 0;
    uint32_t desc_bytes = 0;
    ifs.read(reinterpret_cast<char*>(&num_templates), sizeof(num_templates));
    ifs.read(reinterpret_cast<char*>(&desc_bytes), sizeof(desc_bytes));

    if (num_templates == 0 || desc_bytes != 32)
    {
        error = "Invalid template count or descriptor length in " + bin_path;
        return false;
    }

    _templates.clear();
    _templates.reserve(num_templates);

    for (uint32_t i = 0; i < num_templates; ++i)
    {
        CADTemplate tmpl;
        float yaw = 0.0f;
        float bx = 0.0f, by = 0.0f, bw = 0.0f, bh = 0.0f;
        uint32_t n_kpts = 0;

        ifs.read(reinterpret_cast<char*>(&yaw), sizeof(yaw));
        ifs.read(reinterpret_cast<char*>(&bx), sizeof(bx));
        ifs.read(reinterpret_cast<char*>(&by), sizeof(by));
        ifs.read(reinterpret_cast<char*>(&bw), sizeof(bw));
        ifs.read(reinterpret_cast<char*>(&bh), sizeof(bh));
        ifs.read(reinterpret_cast<char*>(&n_kpts), sizeof(n_kpts));

        tmpl.yaw_deg = yaw;
        tmpl.bbox = cv::Rect2f(bx, by, bw, bh);

        tmpl.keypoints.resize(n_kpts);
        for (uint32_t k = 0; k < n_kpts; ++k)
        {
            float pt_x, pt_y, size, angle, resp;
            ifs.read(reinterpret_cast<char*>(&pt_x), sizeof(pt_x));
            ifs.read(reinterpret_cast<char*>(&pt_y), sizeof(pt_y));
            ifs.read(reinterpret_cast<char*>(&size), sizeof(size));
            ifs.read(reinterpret_cast<char*>(&angle), sizeof(angle));
            ifs.read(reinterpret_cast<char*>(&resp), sizeof(resp));
            tmpl.keypoints[k] = cv::KeyPoint(pt_x, pt_y, size, angle, resp);
        }

        tmpl.descriptors.create(static_cast<int>(n_kpts), 32, CV_8UC1);
        ifs.read(reinterpret_cast<char*>(tmpl.descriptors.data), n_kpts * 32);

        _templates.push_back(std::move(tmpl));
    }

    _initialized = !_templates.empty();
    return _initialized;
}

bool CADTemplateDetector::initialize(const std::string& cad_dir,
                                      const std::vector<cv::Vec3f>& cad_centroids,
                                      double half_w, double half_l,
                                      double top_z, double bot_z,
                                      std::string& error)
{
    error.clear();
    if (!cad_centroids.empty())
    {
        float min_x = cad_centroids[0][0], max_x = min_x;
        float min_y = cad_centroids[0][1], max_y = min_y;
        float min_z = cad_centroids[0][2], max_z = min_z;
        for (const auto& p : cad_centroids)
        {
            min_x = std::min(min_x, p[0]); max_x = std::max(max_x, p[0]);
            min_y = std::min(min_y, p[1]); max_y = std::max(max_y, p[1]);
            min_z = std::min(min_z, p[2]); max_z = std::max(max_z, p[2]);
        }
        _half_w = half_w > 0.01 ? half_w : static_cast<double>(max_x - min_x) * 0.5;
        _half_l = half_l > 0.01 ? half_l : static_cast<double>(max_y - min_y) * 0.5;
        _top_z  = (std::abs(top_z) > 1e-4) ? top_z : static_cast<double>(max_z);
        _bot_z  = (std::abs(bot_z) > 1e-4) ? bot_z : static_cast<double>(min_z);
    }
    else
    {
        _half_w = half_w > 0.01 ? half_w : 0.220;
        _half_l = half_l > 0.01 ? half_l : 0.415;
        _top_z  = top_z;
        _bot_z  = bot_z;
    }

    _orb = cv::ORB::create(400, 1.2f, 4, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 15);
    _matcher = cv::BFMatcher::create(cv::NORM_HAMMING);

    // 1. Zero-Boot Latency: Check if pre-baked binary templates exist
    std::vector<std::string> candidate_bins = {
        cad_dir + "/cad_data/TABLE-DEMO_cad_templates.bin",
        cad_dir + "/TABLE-DEMO_cad_templates.bin",
        cad_dir + "/cad_templates.bin",
        "/home/huyen1905/TMCAD/ObjectDetection_V2.5/models/object_detection/cad_pose/table_cad/v1/cad_data/TABLE-DEMO_cad_templates.bin",
        "/home/huyen1905/TMCAD/ObjectDetection_V2.4/models/object_detection/cad_pose/table_cad/v1/cad_data/TABLE-DEMO_cad_templates.bin",
        "/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO_cad_templates.bin"
    };

    for (const auto& bin_path : candidate_bins)
    {
        std::string load_err;
        if (loadBinaryTemplates(bin_path, load_err))
        {
            // Zero-latency warm-up (<0.2 ms) so Frame 0 has zero jitter
            cv::Mat warmup_mat(32, 32, CV_8UC1, cv::Scalar(0));
            std::vector<cv::KeyPoint> dummy_kpts;
            cv::Mat dummy_desc;
            _orb->detectAndCompute(warmup_mat, cv::noArray(), dummy_kpts, dummy_desc);
            cv::Mat dummy_q(1, 32, CV_8UC1, cv::Scalar(0));
            cv::Mat dummy_t(1, 32, CV_8UC1, cv::Scalar(0));
            std::vector<std::vector<cv::DMatch>> dummy_m;
            _matcher->knnMatch(dummy_q, dummy_t, dummy_m, 1);

            _initialized = true;
            return true; // Instant startup (~1.5 ms)!
        }
    }

    // Build 3D points representing the table geometry
    std::vector<cv::Vec3f> table_3d_points = cad_centroids;

    // Add 8 structural corners of the CAD bounding box
    table_3d_points.emplace_back(-_half_w, -_half_l, _top_z);
    table_3d_points.emplace_back( _half_w, -_half_l, _top_z);
    table_3d_points.emplace_back( _half_w,  _half_l, _top_z);
    table_3d_points.emplace_back(-_half_w,  _half_l, _top_z);
    table_3d_points.emplace_back(-_half_w, -_half_l, _bot_z);
    table_3d_points.emplace_back( _half_w, -_half_l, _bot_z);
    table_3d_points.emplace_back( _half_w,  _half_l, _bot_z);
    table_3d_points.emplace_back(-_half_w,  _half_l, _bot_z);

    // Add dense points along tabletop perimeter
    const int perimeter_samples = 20;
    for (int i = 0; i <= perimeter_samples; ++i)
    {
        double alpha = static_cast<double>(i) / perimeter_samples;
        double x = -_half_w + 2.0 * _half_w * alpha;
        double y = -_half_l + 2.0 * _half_l * alpha;
        table_3d_points.emplace_back(x, -_half_l, _top_z);
        table_3d_points.emplace_back(x,  _half_l, _top_z);
        table_3d_points.emplace_back(-_half_w, y, _top_z);
        table_3d_points.emplace_back( _half_w, y, _top_z);
    }

    // Reference camera intrinsics for template generation (standard 640x480 RealSense D435)
    cv::Matx33d K(
        605.0,   0.0, 320.0,
          0.0, 605.0, 240.0,
          0.0,   0.0,   1.0
    );

    _templates.clear();
    _templates.reserve(_num_templates);

    // Generate multi-view templates (every 10 degrees from 0 to 350)
    for (int i = 0; i < _num_templates; ++i)
    {
        float yaw_deg = static_cast<float>(i * (360 / _num_templates));
        CADTemplate tmpl;
        renderTemplate(yaw_deg, table_3d_points, K, tmpl);
        if (!tmpl.descriptors.empty() && tmpl.keypoints.size() >= 8)
        {
            _templates.push_back(std::move(tmpl));
        }
    }

    if (_templates.empty())
    {
        error = "Failed to generate any valid CAD templates";
        _initialized = false;
        return false;
    }

    _initialized = true;
    return true;
}

void CADTemplateDetector::renderTemplate(float yaw_deg,
                                         const std::vector<cv::Vec3f>& points_3d,
                                         const cv::Matx33d& K,
                                         CADTemplate& out_tmpl)
{
    out_tmpl.yaw_deg = yaw_deg;

    // Viewing angle: typical robot camera looking slightly downward (pitch ~ 20 deg)
    double pitch_rad = 20.0 * kPi / 180.0;
    double yaw_rad   = yaw_deg * kPi / 180.0;

    cv::Matx33d Rx = rotationX(pitch_rad);
    cv::Matx33d Rz = rotationZ(yaw_rad);
    cv::Matx33d R  = Rx * Rz;

    // Translation: table located at distance ~1.25m in front of camera
    cv::Vec3d t(0.0, 0.25, 1.25);

    cv::Mat canvas(480, 640, CV_8UC1, cv::Scalar(0));
    std::vector<cv::Point2f> projected_pts;
    projected_pts.reserve(points_3d.size());

    float min_u = 1e6f, max_u = -1e6f;
    float min_v = 1e6f, max_v = -1e6f;

    for (const auto& pt : points_3d)
    {
        cv::Vec3d p_cam = R * cv::Vec3d(pt) + t;
        if (p_cam[2] <= 0.1) continue;

        float u = static_cast<float>(K(0, 0) * p_cam[0] / p_cam[2] + K(0, 2));
        float v = static_cast<float>(K(1, 1) * p_cam[1] / p_cam[2] + K(1, 2));

        if (u >= 0 && u < 640 && v >= 0 && v < 480)
        {
            projected_pts.emplace_back(u, v);
            min_u = std::min(min_u, u);
            max_u = std::max(max_u, u);
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
    }

    if (projected_pts.size() < 10) return;

    // Draw table wireframe and surface textures onto canvas
    // Project 8 corners specifically for wireframe lines
    auto projCorner = [&](double cx, double cy, double cz) -> cv::Point {
        cv::Vec3d pc = R * cv::Vec3d(cx, cy, cz) + t;
        int pu = static_cast<int>(K(0, 0) * pc[0] / pc[2] + K(0, 2));
        int pv = static_cast<int>(K(1, 1) * pc[1] / pc[2] + K(1, 2));
        return cv::Point(pu, pv);
    };

    cv::Point c1 = projCorner(-_half_w, -_half_l, _top_z);
    cv::Point c2 = projCorner( _half_w, -_half_l, _top_z);
    cv::Point c3 = projCorner( _half_w,  _half_l, _top_z);
    cv::Point c4 = projCorner(-_half_w,  _half_l, _top_z);
    cv::Point c5 = projCorner(-_half_w, -_half_l, _bot_z);
    cv::Point c6 = projCorner( _half_w, -_half_l, _bot_z);
    cv::Point c7 = projCorner( _half_w,  _half_l, _bot_z);
    cv::Point c8 = projCorner(-_half_w,  _half_l, _bot_z);

    // Tabletop polygon fill and outline
    std::vector<cv::Point> top_poly = {c1, c2, c3, c4};
    cv::fillConvexPoly(canvas, top_poly, cv::Scalar(180));
    cv::polylines(canvas, top_poly, true, cv::Scalar(255), 2);

    // Table legs
    cv::line(canvas, c1, c5, cv::Scalar(220), 2);
    cv::line(canvas, c2, c6, cv::Scalar(220), 2);
    cv::line(canvas, c3, c7, cv::Scalar(220), 2);
    cv::line(canvas, c4, c8, cv::Scalar(220), 2);

    // Internal texture dots for rich ORB corner extraction
    for (const auto& pt : projected_pts)
    {
        cv::circle(canvas, cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)),
                   1, cv::Scalar(240), -1);
    }

    out_tmpl.bbox = cv::Rect2f(min_u, min_v, max_u - min_u, max_v - min_v);

    // Extract ORB features
    _orb->detectAndCompute(canvas, cv::noArray(), out_tmpl.keypoints, out_tmpl.descriptors);
}

CADTemplateDetector::TemplateMatch
CADTemplateDetector::detect(const cv::Mat& rgb,
                             const cv::Mat& depth_z16,
                             double depth_scale,
                             const CameraIntrinsics& intr) const
{
    TemplateMatch result;

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 1: Visual ORB Matching on RGB Frame
    // ─────────────────────────────────────────────────────────────────────────
    if (!rgb.empty() && _initialized && !_templates.empty())
    {
        cv::Mat gray;
        if (rgb.channels() == 3)
        {
            cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
        }
        else
        {
            gray = rgb;
        }

        std::vector<cv::KeyPoint> frame_kpts;
        cv::Mat frame_desc;
        _orb->detectAndCompute(gray, cv::noArray(), frame_kpts, frame_desc);

        if (!frame_desc.empty() && frame_kpts.size() >= 10)
        {
            int best_inliers = 0;
            int best_tmpl_idx = -1;
            std::vector<cv::DMatch> best_good_matches;

            for (size_t ti = 0; ti < _templates.size(); ++ti)
            {
                const auto& tmpl = _templates[ti];
                if (tmpl.descriptors.empty()) continue;

                std::vector<std::vector<cv::DMatch>> knn_matches;
                _matcher->knnMatch(tmpl.descriptors, frame_desc, knn_matches, 2);

                std::vector<cv::DMatch> good;
                for (const auto& m : knn_matches)
                {
                    if (m.size() >= 2 && m[0].distance < 0.78f * m[1].distance)
                    {
                        good.push_back(m[0]);
                    }
                }

                if (static_cast<int>(good.size()) > best_inliers)
                {
                    best_inliers = static_cast<int>(good.size());
                    best_tmpl_idx = static_cast<int>(ti);
                    best_good_matches = std::move(good);
                }
            }

            if (best_tmpl_idx >= 0 && best_inliers >= _min_good_matches)
            {
                float score = std::min(1.0f, static_cast<float>(best_inliers) / 32.0f);
                if (score >= _min_match_score)
                {
                    std::vector<cv::Point2f> matched_pts;
                    matched_pts.reserve(best_good_matches.size());
                    for (const auto& gm : best_good_matches)
                    {
                        matched_pts.push_back(frame_kpts[gm.trainIdx].pt);
                    }

                    cv::Rect box = cv::boundingRect(matched_pts);

                    // Add 15% margin padding
                    int pad_x = static_cast<int>(box.width * 0.15f);
                    int pad_y = static_cast<int>(box.height * 0.15f);

                    int x0 = std::max(0, box.x - pad_x);
                    int y0 = std::max(0, box.y - pad_y);
                    int x1 = std::min(rgb.cols, box.x + box.width + pad_x);
                    int y1 = std::min(rgb.rows, box.y + box.height + pad_y);

                    if ((x1 - x0) >= 30 && (y1 - y0) >= 20)
                    {
                        result.roi = cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
                        result.match_score = score;
                        result.template_index = best_tmpl_idx;
                        result.estimated_yaw_deg = _templates[best_tmpl_idx].yaw_deg;
                        result.good_matches = best_inliers;
                        result.valid = true;
                        result.method = "rgb_orb_template";
                        return result;
                    }
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 2: Geometric Elevation Fallback on Depth Frame
    // (Used when table has low visual texture, dim lighting, or monochrome surface)
    // ─────────────────────────────────────────────────────────────────────────
    if (!depth_z16.empty() && depth_scale > 0.0 && intr.valid())
    {
        TemplateMatch geom_match = detectByGeometricElevation(depth_z16, depth_scale, intr);
        if (geom_match.valid)
        {
            return geom_match;
        }
    }

    result.valid = false;
    return result;
}

CADTemplateDetector::TemplateMatch
CADTemplateDetector::detectByGeometricElevation(const cv::Mat& depth_z16,
                                                 double depth_scale,
                                                 const CameraIntrinsics& intr) const
{
    TemplateMatch result;
    const int w = depth_z16.cols;
    const int h = depth_z16.rows;
    const double fx = intr.fx;
    const double fy = intr.fy;
    const double cx = intr.ppx;
    const double cy = intr.ppy;

    std::vector<cv::Point> cluster_pixels;
    cluster_pixels.reserve(5000);

    // Step 4 pixels for fast scanning (<0.4 ms)
    const int step = 4;
    for (int y = 0; y < h; y += step)
    {
        const uint16_t* row = depth_z16.ptr<uint16_t>(y);
        for (int x = 0; x < w; x += step)
        {
            uint16_t d_raw = row[x];
            if (d_raw == 0) continue;

            float z = static_cast<float>(d_raw * depth_scale);
            if (z < 0.35f || z > 3.50f) continue; // Operating table depth range up to 3.5m

            float py = static_cast<float>((y - cy) * z / fy);
            float px = static_cast<float>((x - cx) * z / fx);

            // Elevated table filter:
            // Table sits above ground, supporting camera height from 0.15m to 1.80m
            if (std::abs(px) < 1.50f && py >= -1.00f && py <= 0.85f)
            {
                cluster_pixels.emplace_back(x, y);
            }
        }
    }

    if (cluster_pixels.size() >= 80)
    {
        cv::Rect box = cv::boundingRect(cluster_pixels);

        // Check if box dimensions are consistent with a table at operating distance
        if (box.width >= 50 && box.height >= 35 &&
            box.width <= w * 0.95 && box.height <= h * 0.95)
        {
            // Add 10% padding margin
            int pad_x = static_cast<int>(box.width * 0.10f);
            int pad_y = static_cast<int>(box.height * 0.10f);

            int x0 = std::max(0, box.x - pad_x);
            int y0 = std::max(0, box.y - pad_y);
            int x1 = std::min(w, box.x + box.width + pad_x);
            int y1 = std::min(h, box.y + box.height + pad_y);

            result.roi = cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
            result.match_score = 0.50f; // Geometric prior confidence
            result.valid = true;
            result.method = "geometric_depth_cluster";
            return result;
        }
    }

    result.valid = false;
    return result;
}
