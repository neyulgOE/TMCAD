#include "CADTemplateDetector.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <sstream>
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

CADTemplateDetector::CADTemplateDetector()
{
    // Fast multi-scale ORB with 8 scale octaves for robust scale-invariant detection
    _orb = cv::ORB::create(2000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 15);
    _matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
}

bool CADTemplateDetector::loadBinaryTemplates(const std::string& bin_path, std::string& error)
{
    error.clear();
    std::ifstream ifs(bin_path, std::ios::binary);
    if (!ifs.is_open())
    {
        error = "Không thể mở tệp binary templates: " + bin_path;
        return false;
    }

    char magic[8];
    ifs.read(magic, 8);
    if (ifs.gcount() < 8 || std::memcmp(magic, "CADTPL01", 8) != 0)
    {
        error = "Tiêu đề nhị phân không hợp lệ (cần CADTPL01) trong: " + bin_path;
        return false;
    }

    uint32_t num_templates = 0;
    uint32_t desc_bytes = 0;
    ifs.read(reinterpret_cast<char*>(&num_templates), sizeof(num_templates));
    ifs.read(reinterpret_cast<char*>(&desc_bytes), sizeof(desc_bytes));

    if (num_templates == 0 || desc_bytes != 32)
    {
        error = "Số lượng template hoặc kích thước descriptor không hợp lệ trong: " + bin_path;
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

bool CADTemplateDetector::initialize(const std::string& cad_path_or_dir,
                                      const std::string& cad_name,
                                      const std::vector<cv::Vec3f>& cad_vertices_or_centroids,
                                      double half_w, double half_l,
                                      double top_z, double bot_z,
                                      std::string& error)
{
    error.clear();
    if (!cad_vertices_or_centroids.empty())
    {
        float min_x = cad_vertices_or_centroids[0][0], max_x = min_x;
        float min_y = cad_vertices_or_centroids[0][1], max_y = min_y;
        float min_z = cad_vertices_or_centroids[0][2], max_z = min_z;
        for (const auto& p : cad_vertices_or_centroids)
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

    // 1. Kiểm tra các vị trí tệp binary templates (.bin) có sẵn
    std::string name = cad_name.empty() ? "TABLE-DEMO" : cad_name;
    std::vector<std::string> candidate_bins = {
        cad_path_or_dir + "/" + name + "_cad_templates.bin",
        cad_path_or_dir + "/cad_data/" + name + "_cad_templates.bin",
        cad_path_or_dir + "/" + name + ".cad_templates.bin",
        "/home/huyen1905/TMCAD/ObjectDetection_V2.5/models/object_detection/cad_pose/table_cad/v1/cad_data/TABLE-DEMO_cad_templates.bin",
        "/home/huyen1905/TMCAD/ObjectDetection_V2.4/models/object_detection/cad_pose/table_cad/v1/cad_data/TABLE-DEMO_cad_templates.bin",
        "/home/huyen1905/TMCAD/CAD_DATA/" + name + "_cad_templates.bin",
        "/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO_cad_templates.bin"
    };

    for (const auto& bin_path : candidate_bins)
    {
        std::string load_err;
        if (loadBinaryTemplates(bin_path, load_err))
        {
            _initialized = true;
            return true;
        }
    }

    // 2. Nếu không có file binary pre-baked, sinh động 36 templates từ hình học CAD 3D
    std::vector<cv::Vec3f> table_3d_points = cad_vertices_or_centroids;

    // Bổ sung 8 đỉnh khung bao của CAD
    table_3d_points.emplace_back(-_half_w, -_half_l, _top_z);
    table_3d_points.emplace_back( _half_w, -_half_l, _top_z);
    table_3d_points.emplace_back( _half_w,  _half_l, _top_z);
    table_3d_points.emplace_back(-_half_w,  _half_l, _top_z);
    table_3d_points.emplace_back(-_half_w, -_half_l, _bot_z);
    table_3d_points.emplace_back( _half_w, -_half_l, _bot_z);
    table_3d_points.emplace_back( _half_w,  _half_l, _bot_z);
    table_3d_points.emplace_back(-_half_w,  _half_l, _bot_z);

    // Mẫu viền mặt trên để tạo đặc trưng dày dặn
    const int perimeter_samples = 25;
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

    cv::Matx33d K(
        605.0,   0.0, 320.0,
          0.0, 605.0, 240.0,
          0.0,   0.0,   1.0
    );

    _templates.clear();
    _templates.reserve(_num_templates);

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
        error = "Không thể sinh template nhận diện hợp lệ từ hình học CAD đã cho.";
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

    // Góc nghiêng camera nhìn xuống ~ 20 độ
    double pitch_rad = 20.0 * kPi / 180.0;
    double yaw_rad   = yaw_deg * kPi / 180.0;

    cv::Matx33d Rx = rotationX(pitch_rad);
    cv::Matx33d Rz = rotationZ(yaw_rad);
    cv::Matx33d R  = Rx * Rz;

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

    // Vẽ đa giác mặt trên và khung chân
    std::vector<cv::Point> top_poly = {c1, c2, c3, c4};
    cv::fillConvexPoly(canvas, top_poly, cv::Scalar(180));
    cv::polylines(canvas, top_poly, true, cv::Scalar(255), 2);

    cv::line(canvas, c1, c5, cv::Scalar(220), 2);
    cv::line(canvas, c2, c6, cv::Scalar(220), 2);
    cv::line(canvas, c3, c7, cv::Scalar(220), 2);
    cv::line(canvas, c4, c8, cv::Scalar(220), 2);

    for (const auto& pt : projected_pts)
    {
        cv::circle(canvas, cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)),
                   1, cv::Scalar(240), -1);
    }

    out_tmpl.bbox = cv::Rect2f(min_u, min_v, max_u - min_u, max_v - min_v);
    _orb->detectAndCompute(canvas, cv::noArray(), out_tmpl.keypoints, out_tmpl.descriptors);
}

CADTemplateDetector::TemplateMatch CADTemplateDetector::detect(const cv::Mat& image) const
{
    auto start_time = std::chrono::high_resolution_clock::now();
    TemplateMatch result;
    result.method = "cad_structural_edge_template";

    if (image.empty() || !_initialized || _templates.empty())
    {
        result.valid = false;
        return result;
    }

    const int img_w = image.cols;
    const int img_h = image.rows;

    // 1. Chuyển sang ảnh thang xám và lọc nhiễu Gaussian
    cv::Mat gray;
    if (image.channels() == 3)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else if (image.channels() == 4)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        gray = image.clone();
    }

    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 1.2);

    // 2. Trích xuất đường biên Canny và Distance Transform
    cv::Mat edges;
    cv::Canny(blur, edges, 45, 120, 3);

    cv::Mat dt;
    cv::distanceTransform(255 - edges, dt, cv::DIST_L2, 3);

    // 3. Phân tích các đường kết cấu hình học bằng HoughLinesP (tìm thanh dầm ngang và cột chân)
    std::vector<cv::Vec4i> hough_lines;
    const int min_line_len = std::max(50, static_cast<int>(img_w * 0.20));
    cv::HoughLinesP(edges, hough_lines, 1.0, CV_PI / 180.0, 50, min_line_len, 20);

    bool has_front_beam = false;
    double primary_beam_angle = 0.0;
    double max_h_len = 0.0;
    cv::Point2f beam_midpoint(img_w * 0.5f, img_h * 0.5f);

    std::vector<cv::Point2f> struct_line_pts;

    for (const auto& l : hough_lines)
    {
        double dx = l[2] - l[0];
        double dy = l[3] - l[1];
        double len = std::hypot(dx, dy);
        double ang = std::atan2(std::abs(dy), std::abs(dx)) * 180.0 / CV_PI;

        // Thanh ngang (góc nghiêng < 12 độ)
        if (ang < 12.0)
        {
            if (len > max_h_len)
            {
                max_h_len = len;
                primary_beam_angle = ang;
                beam_midpoint = cv::Point2f(0.5f * (l[0] + l[2]), 0.5f * (l[1] + l[3]));
            }
            if (l[1] > 0.10 * img_h && l[1] < 0.90 * img_h)
            {
                struct_line_pts.emplace_back(l[0], l[1]);
                struct_line_pts.emplace_back(l[2], l[3]);
            }
        }
        // Cột đứng (góc nghiêng > 75 độ)
        else if (ang > 75.0)
        {
            if (l[0] > 0.10 * img_w && l[0] < 0.90 * img_w)
            {
                struct_line_pts.emplace_back(l[0], l[1]);
                struct_line_pts.emplace_back(l[2], l[3]);
            }
        }
    }

    if (max_h_len >= img_w * 0.25 && primary_beam_angle <= 10.0)
    {
        has_front_beam = true;
    }

    // 4. Đối sánh Chamfer trên 36 góc nhìn Template CAD
    int best_tmpl_idx = -1;
    float best_combined_score = -1.0f;
    std::vector<cv::Point2f> best_inliers;
    float best_chamfer_ratio = 0.0f;

    for (size_t ti = 0; ti < _templates.size(); ++ti)
    {
        const auto& tmpl = _templates[ti];
        if (tmpl.keypoints.empty()) continue;

        // Tính điểm trùng khớp Chamfer giữa các điểm cạnh của template và distance transform
        int hits = 0;
        std::vector<cv::Point2f> inliers;
        inliers.reserve(tmpl.keypoints.size());

        for (const auto& kp : tmpl.keypoints)
        {
            int ix = static_cast<int>(std::round(kp.pt.x));
            int iy = static_cast<int>(std::round(kp.pt.y));
            if (ix >= 0 && ix < dt.cols && iy >= 0 && iy < dt.rows)
            {
                if (dt.at<float>(iy, ix) <= 4.0f)
                {
                    hits++;
                    inliers.push_back(cv::Point2f(ix, iy));
                }
            }
        }

        float chamfer_ratio = static_cast<float>(hits) / static_cast<float>(tmpl.keypoints.size());

        // Đánh giá độ đồng thuận hướng thanh dầm (Beam Consistency)
        float beam_score = 0.5f;
        if (has_front_beam)
        {
            float yaw = tmpl.yaw_deg;
            float diff_0 = std::min(std::abs(yaw - 0.0f), std::abs(yaw - 360.0f));
            float diff_180 = std::abs(yaw - 180.0f);
            float dist_front = std::min(diff_0, diff_180);
            // Cosine suy giảm khi góc lệch xa khỏi mặt phẳng trực diện
            beam_score = std::max(0.0f, std::cos(dist_front * static_cast<float>(CV_PI) / 180.0f));
        }

        float combined_score = 0.50f * chamfer_ratio + 0.50f * beam_score;
        if (combined_score > best_combined_score)
        {
            best_combined_score = combined_score;
            best_tmpl_idx = static_cast<int>(ti);
            best_inliers = std::move(inliers);
            best_chamfer_ratio = chamfer_ratio;
        }
    }

    if (best_tmpl_idx >= 0 && best_combined_score >= _min_match_score)
    {
        const auto& best_tmpl = _templates[best_tmpl_idx];

        // Chuẩn hóa góc quay Yaw
        float final_yaw = best_tmpl.yaw_deg;
        if (has_front_beam)
        {
            // Bàn nhìn trực diện, trục đối xứng C2: yaw = 0.0 deg
            if (final_yaw <= 15.0f || final_yaw >= 345.0f || std::abs(final_yaw - 180.0f) <= 15.0f)
            {
                final_yaw = 0.0f;
            }
        }

        // 5. Xác định Bounding Box 2D chuẩn xác bao quanh vật thể
        cv::Rect2f final_roi;
        if (best_inliers.size() >= 15)
        {
            float min_x = 1e6f, max_x = -1e6f;
            float min_y = 1e6f, max_y = -1e6f;
            for (const auto& pt : best_inliers)
            {
                min_x = std::min(min_x, pt.x);
                max_x = std::max(max_x, pt.x);
                min_y = std::min(min_y, pt.y);
                max_y = std::max(max_y, pt.y);
            }

            float w = max_x - min_x;
            float h = std::max(max_y - min_y, best_tmpl.bbox.height);

            // Đệm biên an toàn
            int pad_x = static_cast<int>(w * 0.08f);
            int pad_y = static_cast<int>(h * 0.06f);

            int x0 = std::max(0, static_cast<int>(min_x) - pad_x);
            int y0 = std::max(0, static_cast<int>(min_y) - pad_y);
            int x1 = std::min(img_w, static_cast<int>(max_x) + pad_x);
            int y1 = std::min(img_h, y0 + static_cast<int>(h) + pad_y);

            final_roi = cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
        }
        else
        {
            final_roi = best_tmpl.bbox;
        }

        result.roi = final_roi;
        result.match_score = std::min(1.0f, best_combined_score * 1.25f);
        result.template_index = best_tmpl_idx;
        result.estimated_yaw_deg = final_yaw;
        result.good_matches = static_cast<int>(best_inliers.size());
        result.ransac_inliers = static_cast<int>(best_inliers.size());
        result.matched_kpts = best_inliers;
        result.valid = true;

        // 4 góc của khung hộp
        result.projected_box = {
            cv::Point2f(final_roi.x, final_roi.y),
            cv::Point2f(final_roi.x + final_roi.width, final_roi.y),
            cv::Point2f(final_roi.x + final_roi.width, final_roi.y + final_roi.height),
            cv::Point2f(final_roi.x, final_roi.y + final_roi.height)
        };

        auto end_time = std::chrono::high_resolution_clock::now();
        result.latency_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        return result;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.latency_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    result.valid = false;
    return result;
}

cv::Mat CADTemplateDetector::annotate(const cv::Mat& src,
                                      const TemplateMatch& match,
                                      const std::string& label,
                                      bool show_keypoints,
                                      bool show_hud)
{
    if (src.empty()) return {};

    cv::Mat canvas = src.clone();
    if (canvas.channels() == 1)
    {
        cv::cvtColor(canvas, canvas, cv::COLOR_GRAY2BGR);
    }

    if (match.valid)
    {
        cv::Scalar neon_cyan(255, 230, 0);     // BGR
        cv::Scalar emerald(118, 230, 0);       // BGR
        cv::Scalar dark_badge(20, 20, 20);     // BGR

        // 1. Vẽ 2D Bounding Box
        cv::Rect r(static_cast<int>(match.roi.x),
                   static_cast<int>(match.roi.y),
                   static_cast<int>(match.roi.width),
                   static_cast<int>(match.roi.height));
        cv::rectangle(canvas, r, emerald, 2);

        // 2. Vẽ 4 góc nhấn kiểu HUD công nghệ cao (Corner Accents)
        int cl = std::min(24, std::min(r.width, r.height) / 4);
        int th = 4;
        // Top-left
        cv::line(canvas, cv::Point(r.x, r.y), cv::Point(r.x + cl, r.y), neon_cyan, th);
        cv::line(canvas, cv::Point(r.x, r.y), cv::Point(r.x, r.y + cl), neon_cyan, th);
        // Top-right
        cv::line(canvas, cv::Point(r.x + r.width, r.y), cv::Point(r.x + r.width - cl, r.y), neon_cyan, th);
        cv::line(canvas, cv::Point(r.x + r.width, r.y), cv::Point(r.x + r.width, r.y + cl), neon_cyan, th);
        // Bottom-left
        cv::line(canvas, cv::Point(r.x, r.y + r.height), cv::Point(r.x + cl, r.y + r.height), neon_cyan, th);
        cv::line(canvas, cv::Point(r.x, r.y + r.height), cv::Point(r.x, r.y + r.height - cl), neon_cyan, th);
        // Bottom-right
        cv::line(canvas, cv::Point(r.x + r.width, r.y + r.height), cv::Point(r.x + r.width - cl, r.y + r.height), neon_cyan, th);
        cv::line(canvas, cv::Point(r.x + r.width, r.y + r.height), cv::Point(r.x + r.width, r.y + r.height - cl), neon_cyan, th);

        // 3. Vẽ các điểm đặc trưng inliers nếu được bật
        if (show_keypoints)
        {
            for (const auto& pt : match.matched_kpts)
            {
                cv::circle(canvas, cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)),
                           3, cv::Scalar(0, 255, 255), -1);
            }
        }

        // 4. Vẽ đa giác phối cảnh (nếu có homography)
        if (match.projected_box.size() == 4)
        {
            std::vector<cv::Point> poly_pts;
            for (const auto& p : match.projected_box)
            {
                poly_pts.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
            }
            cv::polylines(canvas, poly_pts, true, cv::Scalar(0, 200, 255), 1, cv::LINE_AA);
        }

        // 5. HUD Badge thông tin nhận diện
        if (show_hud)
        {
            std::ostringstream ss_top;
            ss_top << "[ " << label << " ] Score: " << std::fixed << std::setprecision(1) << (match.match_score * 100.0f) << "%"
                   << " | Yaw: " << std::setprecision(1) << match.estimated_yaw_deg << " deg";

            std::ostringstream ss_bot;
            ss_bot << "Matches: " << match.good_matches << " (" << match.ransac_inliers << " inliers)"
                   << " | ROI: [" << r.x << "," << r.y << "," << r.width << "," << r.height << "]"
                   << " | Latency: " << std::fixed << std::setprecision(1) << match.latency_ms << " ms";

            std::string text_top = ss_top.str();
            std::string text_bot = ss_bot.str();

            int baseline = 0;
            cv::Size sz_top = cv::getTextSize(text_top, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
            cv::Size sz_bot = cv::getTextSize(text_bot, cv::FONT_HERSHEY_SIMPLEX, 0.42, 1, &baseline);

            int badge_w = std::max(sz_top.width, sz_bot.width) + 16;
            int badge_h = sz_top.height + sz_bot.height + 14;
            int badge_y = std::max(0, r.y - badge_h - 4);
            int badge_x = std::max(0, std::min(canvas.cols - badge_w, r.x));

            cv::Rect badge_rect(badge_x, badge_y, badge_w, badge_h);
            cv::rectangle(canvas, badge_rect, dark_badge, cv::FILLED);
            cv::rectangle(canvas, badge_rect, emerald, 1);

            cv::putText(canvas, text_top, cv::Point(badge_x + 8, badge_y + sz_top.height + 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(canvas, text_bot, cv::Point(badge_x + 8, badge_y + sz_top.height + sz_bot.height + 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.42, emerald, 1, cv::LINE_AA);
        }
    }
    else
    {
        // Vẽ banner thông báo không tìm thấy
        std::string warn_text = "[-] KHONG TIM THAY DOI TUONG (NO OBJECT DETECTED)";
        int baseline = 0;
        cv::Size sz = cv::getTextSize(warn_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        int bx = (canvas.cols - sz.width) / 2;
        int by = 36;
        cv::Rect r(bx - 12, by - sz.height - 8, sz.width + 24, sz.height + 16);
        cv::rectangle(canvas, r, cv::Scalar(20, 20, 20), cv::FILLED);
        cv::rectangle(canvas, r, cv::Scalar(0, 140, 255), 2);
        cv::putText(canvas, warn_text, cv::Point(bx, by),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 200, 255), 2, cv::LINE_AA);
    }

    return canvas;
}
