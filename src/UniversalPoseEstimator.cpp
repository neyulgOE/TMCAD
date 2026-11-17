#include "UniversalPoseEstimator.h"
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <unordered_map>

namespace tmcad {

namespace {
constexpr double kPi = 3.14159265358979323846;

bool fileExists(const std::string& p) {
    return !p.empty() && std::filesystem::exists(p) && std::filesystem::is_regular_file(p);
}

std::string extractStem(const std::string& p) {
    return std::filesystem::path(p).stem().string();
}

void computeVisibleCadSubset(const cv::Matx33d& R,
                             const cv::Vec3d& t,
                             const std::vector<cv::Vec3f>& mesh_pts,
                             const std::vector<cv::Vec3f>& mesh_normals,
                             double fx,
                             double fy,
                             double cx,
                             double cy,
                             int img_w,
                             int img_h,
                             size_t mesh_stride,
                             std::vector<cv::Vec3d>& out_pts,
                             std::vector<cv::Vec3d>& out_norms)
{
    out_pts.clear();
    out_norms.clear();
    if (mesh_pts.empty()) return;
    mesh_stride = std::max<size_t>(1, mesh_stride);

    // Lưới Z-buffer thô (8x8 pixel/ô) để loại bỏ các vách ngăn/hốc xi-lanh bị che khuất bên trong ruột CAD
    const int grid_w = std::max(32, img_w / 8);
    const int grid_h = std::max(24, img_h / 8);
    std::vector<float> z_buf(static_cast<size_t>(grid_w) * grid_h, 1e6f);

    struct CandPt {
        size_t idx;
        int gx;
        int gy;
        float z;
    };
    std::vector<CandPt> cands;
    cands.reserve(mesh_pts.size() / mesh_stride + 1);

    for (size_t i = 0; i < mesh_pts.size(); i += mesh_stride) {
        cv::Vec3d p_cam = R * cv::Vec3d(mesh_pts[i]) + t;
        if (p_cam[2] <= 0.05) continue;
        cv::Vec3d n_cam = R * cv::Vec3d(mesh_normals[i]);
        if (n_cam.dot(p_cam) > 0.10 * cv::norm(p_cam)) continue;

        double u = p_cam[0] * fx / p_cam[2] + cx;
        double v = p_cam[1] * fy / p_cam[2] + cy;
        if (u >= -32.0 && u < img_w + 32.0 && v >= -32.0 && v < img_h + 32.0) {
            int gx = std::clamp(static_cast<int>(u / 8.0), 0, grid_w - 1);
            int gy = std::clamp(static_cast<int>(v / 8.0), 0, grid_h - 1);
            float zf = static_cast<float>(p_cam[2]);
            float& cell_z = z_buf[static_cast<size_t>(gy) * grid_w + gx];
            if (zf < cell_z) cell_z = zf;
            cands.push_back(CandPt{i, gx, gy, zf});
        }
    }

    // Chỉ giữ các điểm nằm gần lớp vỏ ngoài cùng nhìn thấy (trong phạm vi 12mm tính từ z_buf mặt trước)
    out_pts.reserve(cands.size());
    out_norms.reserve(cands.size());
    for (const auto& c : cands) {
        if (c.z <= z_buf[static_cast<size_t>(c.gy) * grid_w + c.gx] + 0.012f) {
            out_pts.emplace_back(mesh_pts[c.idx]);
            out_norms.emplace_back(mesh_normals[c.idx]);
        }
    }
}

struct RgbEdgeMaps {
    cv::Mat dt_vert;
    cv::Mat dt_horiz;
    bool empty() const { return dt_vert.empty() || dt_horiz.empty(); }
};

RgbEdgeMaps buildRgbCannyDistanceTransform(const cv::Mat& rgb, const cv::Mat& fg_mask_2d = cv::Mat())
{
    RgbEdgeMaps maps;
    if (rgb.empty()) return maps;
    cv::Mat gray, edges, gx, gy;
    if (rgb.channels() == 3) {
        cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = rgb;
    }
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0.8);
    cv::Canny(gray, edges, 45, 130);
    if (!fg_mask_2d.empty() && fg_mask_2d.size() == edges.size() && fg_mask_2d.type() == CV_8UC1) {
        cv::Mat mask_dil;
        cv::dilate(fg_mask_2d, mask_dil, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13)));
        cv::bitwise_and(edges, mask_dil, edges);
    }
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

    cv::Mat abs_gx = cv::abs(gx);
    cv::Mat abs_gy = cv::abs(gy);
    cv::Mat vert_mask = (abs_gx > 1.15f * abs_gy);
    cv::Mat horiz_mask = (abs_gy > 1.15f * abs_gx);

    cv::Mat vert_edges, horiz_edges;
    cv::bitwise_and(edges, vert_mask, vert_edges);
    cv::bitwise_and(edges, horiz_mask, horiz_edges);

    cv::Mat inv_vert, inv_horiz;
    cv::bitwise_not(vert_edges, inv_vert);
    cv::bitwise_not(horiz_edges, inv_horiz);
    cv::distanceTransform(inv_vert, maps.dt_vert, cv::DIST_L2, 3);
    cv::distanceTransform(inv_horiz, maps.dt_horiz, cv::DIST_L2, 3);
    return maps;
}

double computeRgbSilhouetteChamferCost(const cv::Matx33d& R,
                                       const cv::Vec3d& t,
                                       const std::vector<cv::Vec3f>& mesh_pts,
                                       const std::vector<cv::Vec3f>& mesh_normals,
                                       const RgbEdgeMaps& edge_maps,
                                       double fx, double fy, double cx, double cy)
{
    (void)mesh_normals;
    if (mesh_pts.empty() || edge_maps.empty()) return 50.0;

    const int w = edge_maps.dt_vert.cols;
    const int h = edge_maps.dt_vert.rows;

    constexpr int kBins = 24;
    int row_min_u[kBins], row_max_u[kBins], row_v[kBins];
    for (int b = 0; b < kBins; ++b) {
        row_min_u[b] = 100000;
        row_max_u[b] = -100000;
        row_v[b] = -1;
    }

    int min_v = 100000, max_v = -100000;
    int min_u_all = 100000, max_u_all = -100000;
    const size_t step = std::max<size_t>(1, mesh_pts.size() / 600);

    std::vector<cv::Point> proj_pts;
    proj_pts.reserve(mesh_pts.size() / step + 1);

    for (size_t i = 0; i < mesh_pts.size(); i += step) {
        const cv::Vec3d pc = R * cv::Vec3d(mesh_pts[i]) + t;
        if (pc[2] <= 0.12) continue;
        const int u = static_cast<int>(std::round(fx * pc[0] / pc[2] + cx));
        const int v = static_cast<int>(std::round(fy * pc[1] / pc[2] + cy));
        if (u < 2 || u >= w - 2 || v < 2 || v >= h - 2) return 50.0;
        proj_pts.emplace_back(u, v);
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        if (u < min_u_all) min_u_all = u;
        if (u > max_u_all) max_u_all = u;
    }

    if (proj_pts.size() < 20 || max_v - min_v < 12 || max_u_all - min_u_all < 10) return 50.0;

    // Tính trục chính 2D của hình chiếu để xoay 24 bin theo góc nghiêng thực tế của vật thể
    double mu_u = 0.0, mu_v = 0.0;
    for (const auto& pt : proj_pts) {
        mu_u += pt.x;
        mu_v += pt.y;
    }
    mu_u /= proj_pts.size();
    mu_v /= proj_pts.size();
    double cuu = 0.0, cvv = 0.0, cuv = 0.0;
    for (const auto& pt : proj_pts) {
        const double du = pt.x - mu_u;
        const double dv = pt.y - mu_v;
        cuu += du * du;
        cvv += dv * dv;
        cuv += du * dv;
    }
    const double tr = cuu + cvv;
    const double disc = std::sqrt(std::max(0.0, (cuu - cvv) * (cuu - cvv) + 4.0 * cuv * cuv));
    const double lam0 = 0.5 * (tr + disc);
    double ax_u = (std::abs(cuv) > 1e-6) ? (lam0 - cvv) : ((cuu >= cvv) ? 1.0 : 0.0);
    double ax_v = (std::abs(cuv) > 1e-6) ? cuv : ((cuu >= cvv) ? 0.0 : 1.0);
    const double ax_n = std::hypot(ax_u, ax_v);
    if (ax_n > 1e-6) {
        ax_u /= ax_n;
        ax_v /= ax_n;
    } else {
        ax_u = 0.0;
        ax_v = 1.0;
    }
    if (ax_v < 0.0) { ax_u = -ax_u; ax_v = -ax_v; }
    const double perp_u = ax_v;
    const double perp_v = -ax_u;

    double min_along = 1e9, max_along = -1e9;
    for (const auto& pt : proj_pts) {
        const double s = (pt.x - mu_u) * ax_u + (pt.y - mu_v) * ax_v;
        if (s < min_along) min_along = s;
        if (s > max_along) max_along = s;
    }
    const double inv_span_along = static_cast<double>(kBins - 1) / std::max(1.0, max_along - min_along);

    double row_min_perp[kBins], row_max_perp[kBins];
    int row_lu[kBins], row_lv[kBins], row_ru[kBins], row_rv[kBins];
    for (int b = 0; b < kBins; ++b) {
        row_min_perp[b] = 1e9;
        row_max_perp[b] = -1e9;
        row_lu[b] = row_lv[b] = row_ru[b] = row_rv[b] = -1;
    }
    for (const auto& pt : proj_pts) {
        const double s = (pt.x - mu_u) * ax_u + (pt.y - mu_v) * ax_v;
        const double p = (pt.x - mu_u) * perp_u + (pt.y - mu_v) * perp_v;
        const int b = std::clamp(static_cast<int>((s - min_along) * inv_span_along), 0, kBins - 1);
        if (p < row_min_perp[b]) {
            row_min_perp[b] = p;
            row_lu[b] = pt.x;
            row_lv[b] = pt.y;
        }
        if (p > row_max_perp[b]) {
            row_max_perp[b] = p;
            row_ru[b] = pt.x;
            row_rv[b] = pt.y;
        }
    }

    const float w_side_v = static_cast<float>(ax_v * ax_v);
    const float w_side_h = static_cast<float>(ax_u * ax_u);
    auto sampleSideDt = [&](int u, int v) -> double {
        const float dv = std::min(30.0f, edge_maps.dt_vert.at<float>(v, u));
        const float dh = std::min(30.0f, edge_maps.dt_horiz.at<float>(v, u));
        return static_cast<double>(w_side_v * dv + w_side_h * dh);
    };
    auto sampleEndDt = [&](int u, int v) -> double {
        const float dv = std::min(30.0f, edge_maps.dt_vert.at<float>(v, u));
        const float dh = std::min(30.0f, edge_maps.dt_horiz.at<float>(v, u));
        return static_cast<double>(w_side_v * dh + w_side_h * dv);
    };

    double sum_left = 0.0;
    double sum_right = 0.0;
    int cnt_side = 0;
    int first_b = -1, last_b = -1;
    for (int b = 0; b < kBins; ++b) {
        if (row_lu[b] < 0 || row_ru[b] < 0) continue;
        if (first_b < 0) first_b = b;
        last_b = b;
        // Bỏ qua 15% đáy dưới cùng (b >= kBins - 3) để giảm nhiễu từ ngón tay/chân kẹp trên bàn
        if (b >= kBins - 3) continue;
        sum_left += sampleSideDt(row_lu[b], row_lv[b]);
        sum_right += sampleSideDt(row_ru[b], row_rv[b]);
        ++cnt_side;
    }
    if (cnt_side < 4) return 50.0;

    double sum_horiz = 0.0;
    int cnt_horiz = 0;
    for (int b : {first_b, last_b}) {
        if (b < 0 || row_lu[b] < 0 || row_ru[b] < 0) continue;
        const int u_mid = (row_lu[b] + row_ru[b]) / 2;
        const int v_mid = (row_lv[b] + row_rv[b]) / 2;
        sum_horiz += sampleEndDt(u_mid, v_mid);
        ++cnt_horiz;
    }

    const double mean_left = sum_left / cnt_side;
    const double mean_right = sum_right / cnt_side;
    const double mean_vert = 0.5 * (mean_left + mean_right);
    const double bilat_max = std::max(mean_left, mean_right);
    const double mean_horiz = (cnt_horiz > 0) ? (sum_horiz / cnt_horiz) : mean_vert;
    return 0.45 * bilat_max + 0.33 * mean_vert + 0.22 * mean_horiz;
}

bool projectCadBounds2D(const cv::Matx33d& R,
                        const cv::Vec3d& t,
                        const cv::Vec3d& cad_min,
                        const cv::Vec3d& cad_max,
                        double fx, double fy, double cx, double cy,
                        int& out_u_min, int& out_u_max,
                        int& out_v_top, int& out_v_bot)
{
    if (t[2] <= 0.10) return false;
    double u0 = 1e9, u1 = -1e9, v0 = 1e9, v1 = -1e9;
    int valid = 0;
    for (int ix = 0; ix < 2; ++ix) {
        for (int iy = 0; iy < 2; ++iy) {
            for (int iz = 0; iz < 2; ++iz) {
                const cv::Vec3d corner(
                    ix ? cad_max[0] : cad_min[0],
                    iy ? cad_max[1] : cad_min[1],
                    iz ? cad_max[2] : cad_min[2]);
                const cv::Vec3d pc = R * corner + t;
                if (pc[2] <= 0.08) continue;
                const double u = fx * pc[0] / pc[2] + cx;
                const double v = fy * pc[1] / pc[2] + cy;
                u0 = std::min(u0, u);
                u1 = std::max(u1, u);
                v0 = std::min(v0, v);
                v1 = std::max(v1, v);
                ++valid;
            }
        }
    }
    if (valid < 4) return false;
    out_u_min = static_cast<int>(std::floor(u0));
    out_u_max = static_cast<int>(std::ceil(u1));
    out_v_top = static_cast<int>(std::floor(v0));
    out_v_bot = static_cast<int>(std::ceil(v1));
    return true;
}

// Tự động suy luận độ sâu Z thực tế của vật thể tại vùng ảnh (u_min..u_max, v_top..v_bot)
// từ các điểm Depth tiền cảnh còn sống ở nắp/viền/đế và giao tuyến mặt phẳng đỡ (Zero Hardcoded Depth).
double estimateLocalForegroundDepth(const cv::Mat& depth_z16,
                                    int u_min, int u_max,
                                    int v_top, int v_bot,
                                    const cv::Vec4d& floor_plane,
                                    bool has_floor,
                                    double fx, double fy, double cx, double cy,
                                    double fallback_z,
                                    double max_z)
{
    std::vector<double> valid_zs;
    valid_zs.reserve(128);

    if (!depth_z16.empty() && depth_z16.type() == CV_16UC1) {
        const int w = depth_z16.cols;
        const int h = depth_z16.rows;
        const int u0 = std::clamp(u_min - 6, 0, w - 1);
        const int u1 = std::clamp(u_max + 6, 0, w - 1);
        const int v0 = std::clamp(v_top - 4, 0, h - 1);
        const int v1 = std::clamp(v_bot + 2, 0, h - 1);

        const int step_v = std::max(1, (v1 - v0) / 16);
        const int step_u = std::max(1, (u1 - u0) / 12);

        for (int v = v0; v <= v1; v += step_v) {
            const uint16_t* row = depth_z16.ptr<uint16_t>(v);
            for (int u = u0; u <= u1; u += step_u) {
                if (row[u] > 150) { // Điểm có tín hiệu cảm biến (> 0.15m)
                    const double z_m = static_cast<double>(row[u]) * 0.001;
                    if (z_m <= max_z) {
                        valid_zs.push_back(z_m);
                    }
                }
            }
        }
    }

    // Nếu phát hiện mặt phẳng đỡ (bàn/sàn), tính thêm độ sâu giao tuyến tia nhìn tại đáy vật thể (v_bot)
    double z_plane = -1.0;
    if (has_floor && std::abs(floor_plane[1]) > 0.25) {
        const double u_mid = 0.5 * (u_min + u_max);
        const double ray_x = (u_mid - cx) / fx;
        const double ray_y = (static_cast<double>(v_bot) - cy) / fy;
        const double denom = floor_plane[0] * ray_x + floor_plane[1] * ray_y + floor_plane[2];
        if (std::abs(denom) > 1e-4) {
            const double zp = -floor_plane[3] / denom;
            if (zp > 0.15 && zp <= max_z) {
                z_plane = zp;
            }
        }
    }

    if (!valid_zs.empty()) {
        std::sort(valid_zs.begin(), valid_zs.end());
        // Lấy phân vị thứ 30 của các điểm trong vùng vật thể để tránh vài pixel tay/bàn phía trước kéo Z tụt về gần.
        const double z_fg = valid_zs[(valid_zs.size() * 3) / 10];
        if (z_plane > 0.15 && std::abs(z_fg - z_plane) <= 0.12) {
            return 0.65 * z_fg + 0.35 * z_plane;
        }
        return z_fg;
    }

    return (z_plane > 0.15) ? z_plane : fallback_z;
}

double computeProjectedZeroDepthFraction(const cv::Matx33d& R,
                                         const cv::Vec3d& t,
                                         const cv::Vec3d& cad_min,
                                         const cv::Vec3d& cad_max,
                                         const cv::Mat& depth_z16,
                                         double fx, double fy, double cx, double cy)
{
    if (depth_z16.empty() || depth_z16.type() != CV_16UC1) return 0.0;
    int u_min = 0, u_max = 0, v_top = 0, v_bot = 0;
    if (!projectCadBounds2D(R, t, cad_min, cad_max, fx, fy, cx, cy, u_min, u_max, v_top, v_bot)) {
        return 0.0;
    }
    const int w = depth_z16.cols;
    const int h = depth_z16.rows;
    const int bw = u_max - u_min;
    const int bh = v_bot - v_top;
    if (bw < 16 || bh < 16) return 0.0;

    const int u0 = std::clamp(u_min + bw / 5, 0, w - 1);
    const int u1 = std::clamp(u_max - bw / 5, 0, w - 1);
    const int v0 = std::clamp(v_top + bh / 5, 0, h - 1);
    const int v1 = std::clamp(v_bot - bh / 5, 0, h - 1);
    int zero_cnt = 0, tot_cnt = 0;
    const int step_v = std::max(1, (v1 - v0) / 14);
    const int step_u = std::max(1, (u1 - u0) / 10);
    for (int v = v0; v <= v1; v += step_v) {
        const uint16_t* row = depth_z16.ptr<uint16_t>(v);
        for (int u = u0; u <= u1; u += step_u) {
            ++tot_cnt;
            if (row[u] == 0) ++zero_cnt;
        }
    }
    return (tot_cnt > 0) ? (static_cast<double>(zero_cnt) / tot_cnt) : 0.0;
}

double refineTranslationOnRgbSilhouette(cv::Matx33d& R,
                                        cv::Vec3d& t,
                                        const cv::Vec3d& cluster_centroid,
                                        const cv::Vec3d& cad_min,
                                        const cv::Vec3d& cad_max,
                                        const cv::Vec4d& floor_plane,
                                        bool has_floor,
                                        const std::vector<cv::Vec3f>& mesh_pts,
                                        const std::vector<cv::Vec3f>& mesh_normals,
                                        const RgbEdgeMaps& edge_maps,
                                        const cv::Mat& depth_z16,
                                        double fx, double fy, double cx, double cy,
                                        double max_z,
                                        bool is_tracking_warm_start = false)
{
    double best_cost = computeRgbSilhouetteChamferCost(R, t, mesh_pts, mesh_normals, edge_maps, fx, fy, cx, cy);
    if (edge_maps.empty() || mesh_pts.empty()) return best_cost;

    // Tạo thêm ứng viên R_up / R_up_flip (vật thể đứng thẳng trên mặt bàn dọc theo pháp tuyến sàn/bàn)
    // để khôi phục hướng đứng khi thân vật thể bị đen Depth chỉ còn điểm ở chân đế làm ICP bị đổ ngang.
    cv::Vec3d up_cam(0.0, -1.0, 0.0);
    const cv::Vec3d fn(floor_plane[0], floor_plane[1], floor_plane[2]);
    if (has_floor && cv::norm(fn) > 1e-6) {
        up_cam = fn * (1.0 / cv::norm(fn));
        if (up_cam[1] > 0.0) up_cam = -up_cam;
    }

    const cv::Vec3d span = cad_max - cad_min;
    int long_ax = 2;
    if (span[0] >= span[1] && span[0] >= span[2]) long_ax = 0;
    else if (span[1] >= span[0] && span[1] >= span[2]) long_ax = 1;
    const int ax1 = (long_ax + 1) % 3;
    const int ax2 = (long_ax + 2) % 3;
    const int ax_wide = (span[ax1] >= span[ax2]) ? ax1 : ax2;
    const int ax_thin = (ax_wide == ax1) ? ax2 : ax1;

    const cv::Vec3d cad_mid = 0.5 * (cad_min + cad_max);
    const cv::Vec3d center_pose = R * cad_mid + t;
    const cv::Vec3d view_ray = (center_pose[2] > 0.10)
        ? (center_pose * (1.0 / std::max(1e-6, cv::norm(center_pose))))
        : cv::Vec3d(0.0, 0.0, 1.0);
    cv::Vec3d s1 = view_ray.cross(up_cam);
    if (cv::norm(s1) < 1e-4) s1 = cv::Vec3d(1.0, 0.0, 0.0);
    if (s1[0] < 0.0) s1 = -s1;
    s1 *= (1.0 / std::max(1e-9, cv::norm(s1)));
    cv::Vec3d s2 = up_cam.cross(s1);
    s2 *= (1.0 / std::max(1e-9, cv::norm(s2)));

    cv::Matx33d R_up = cv::Matx33d::zeros();
    for (int r = 0; r < 3; ++r) {
        R_up(r, long_ax) = up_cam[r];
        R_up(r, ax_wide) = s1[r];
        R_up(r, ax_thin) = s2[r];
    }
    if (cv::determinant(R_up) < 0.0) {
        for (int r = 0; r < 3; ++r) R_up(r, ax_thin) = -R_up(r, ax_thin);
    }
    cv::Matx33d R_up_flip = R_up;
    for (int r = 0; r < 3; ++r) {
        R_up_flip(r, long_ax) = -R_up(r, long_ax);
        R_up_flip(r, ax_wide) = -R_up(r, ax_wide);
    }

    const std::vector<cv::Matx33d> R_cands = is_tracking_warm_start
        ? std::vector<cv::Matx33d>{R}
        : std::vector<cv::Matx33d>{R, R_up, R_up_flip};
    const double surf_z_off = 0.42 * std::min({span[0], span[1], span[2]});
    const double diag_obj = std::max(0.05, cv::norm(span));
    std::vector<cv::Vec3d> anchor_centers = {center_pose};
    if (!is_tracking_warm_start && cluster_centroid[2] > 0.15) {
        const cv::Vec3d c_shifted(cluster_centroid[0], cluster_centroid[1], cluster_centroid[2] + surf_z_off);
        if (cv::norm(c_shifted - center_pose) > 0.015) {
            anchor_centers.push_back(c_shifted);
        }
    }
    cv::Matx33d best_R = R;
    cv::Vec3d best_t = t;

    auto resolveDynamicCandidateT = [&](const cv::Matx33d& R_cand, const cv::Vec3d& t_probe) -> cv::Vec3d {
        const double z0 = std::max(0.15, t_probe[2]);
        int u_min = 0, u_max = 0, v_top = 0, v_bot = 0;
        if (!projectCadBounds2D(R_cand, t_probe, cad_min, cad_max, fx, fy, cx, cy, u_min, u_max, v_top, v_bot)) {
            return t_probe;
        }
        const double z_fg = estimateLocalForegroundDepth(
            depth_z16, u_min, u_max, v_top, v_bot, floor_plane, has_floor, fx, fy, cx, cy, z0 - surf_z_off, max_z);
        double z_dyn = z_fg + surf_z_off;
        // Không đẩy cụm tiền cảnh lùi ra sau quá 2cm, và không kéo tụt về trước quá 15% diag (tránh tay/vật cản trước mặt)
        const double max_fwd_pull = std::clamp(0.15 * diag_obj, 0.025, 0.060);
        z_dyn = std::clamp(z_dyn, z0 - max_fwd_pull, z0 + 0.020);
        cv::Vec3d t_dyn(t_probe[0] * (z_dyn / z0), t_probe[1] * (z_dyn / z0), z_dyn);
        if (z0 - z_dyn > 0.04 &&
            projectCadBounds2D(R_cand, t_dyn, cad_min, cad_max, fx, fy, cx, cy, u_min, u_max, v_top, v_bot)) {
            const double z_fg2 = estimateLocalForegroundDepth(
                depth_z16, u_min, u_max, v_top, v_bot, floor_plane, has_floor, fx, fy, cx, cy, z_dyn - surf_z_off, max_z);
            double z_refined = std::clamp(z_fg2 + surf_z_off, z0 - max_fwd_pull, z0 + 0.020);
            t_dyn = cv::Vec3d(t_probe[0] * (z_refined / z0), t_probe[1] * (z_refined / z0), z_refined);
        }
        return t_dyn;
    };

    for (const auto& center_0 : anchor_centers) {
        for (const auto& R_try : R_cands) {
            const cv::Vec3d base_t = center_0 - R_try * cad_mid;
            const bool is_flipped_down = (R_try(1, long_ax) > 0.35);
            const double flip_pen = is_flipped_down ? 0.85 : 0.0;
            const double dy_min = is_tracking_warm_start ? -0.024 : -0.060;
            const double dy_max = is_tracking_warm_start ? 0.025 : 0.031;
            const double dx_lim = is_tracking_warm_start ? 0.024 : 0.030;
            const double d_step = is_tracking_warm_start ? 0.008 : 0.015;
            for (double dy = dy_min; dy <= dy_max; dy += d_step) {
                for (double dx = -dx_lim; dx <= dx_lim + 1e-4; dx += d_step) {
                    const cv::Vec3d t_probe(base_t[0] + dx, base_t[1] + dy, base_t[2]);
                    const cv::Vec3d t_try = is_tracking_warm_start ? t_probe : resolveDynamicCandidateT(R_try, t_probe);
                    const double drift_x = (!is_tracking_warm_start && cluster_centroid[2] > 0.15)
                        ? std::abs(t_try[0] - cluster_centroid[0])
                        : std::abs(dx);
                    const double reg = is_tracking_warm_start
                        ? (2.0 * (std::abs(dx) + std::abs(dy)))
                        : (3.0 * drift_x + 2.0 * std::max(0.0, dy) + flip_pen);
                    const double c = computeRgbSilhouetteChamferCost(R_try, t_try, mesh_pts, mesh_normals, edge_maps, fx, fy, cx, cy) + reg;
                    if (c < best_cost) {
                        best_cost = c;
                        best_R = R_try;
                        best_t = t_try;
                    }
                }
            }
        }
    }

    // Tinh chỉnh góc tự xoay (Yaw quanh trục dài long_ax)
    best_cost = computeRgbSilhouetteChamferCost(best_R, best_t, mesh_pts, mesh_normals, edge_maps, fx, fy, cx, cy);
    {
        const cv::Vec3d win_center = best_R * cad_mid + best_t;
        const cv::Matx33d R_base = best_R;
        double best_yaw_obj = best_cost;
        const int yaw_lim = is_tracking_warm_start ? 8 : 30;
        const int yaw_step = is_tracking_warm_start ? 2 : 5;
        const double yaw_reg = is_tracking_warm_start ? 0.085 : 0.015;
        for (int deg = -yaw_lim; deg <= yaw_lim; deg += yaw_step) {
            if (deg == 0) continue;
            const double rad = static_cast<double>(deg) * kPi / 180.0;
            const double cs = std::cos(rad), sn = std::sin(rad);
            cv::Matx33d R_local = cv::Matx33d::eye();
            R_local(ax1, ax1) = cs;  R_local(ax1, ax2) = -sn;
            R_local(ax2, ax1) = sn;  R_local(ax2, ax2) = cs;
            const cv::Matx33d R_yaw = R_base * R_local;
            const cv::Vec3d t_yaw = win_center - R_yaw * cad_mid;
            const double c_raw = computeRgbSilhouetteChamferCost(R_yaw, t_yaw, mesh_pts, mesh_normals, edge_maps, fx, fy, cx, cy);
            const double c_obj = c_raw + yaw_reg * std::abs(deg);
            if (c_obj < best_yaw_obj) {
                best_yaw_obj = c_obj;
                best_cost = c_raw;
                best_R = R_yaw;
                best_t = t_yaw;
            }
        }
    }

    // Tinh chỉnh mịn theo tỷ lệ pixel quang học (step_xy = 1.5*Z/fx) và kích thước CAD (step_z = 0.02*diag)
    const double z_ref = std::max(0.15, best_t[2]);
    const double step_xy = std::clamp(1.5 * z_ref / std::max(100.0, fx), 0.0015, 0.012);
    const double range_xy = 6.0 * step_xy;
    const double step_z = std::clamp(0.02 * std::max(0.05, cv::norm(span)), 0.004, 0.025);
    const double range_z = 3.0 * step_z;

    const cv::Vec3d mid_t = best_t;
    for (double dy = -range_xy; dy <= range_xy + 1e-6; dy += step_xy) {
        for (double dx = -range_xy; dx <= range_xy + 1e-6; dx += step_xy) {
            const cv::Vec3d t_try(mid_t[0] + dx, mid_t[1] + dy, mid_t[2]);
            const double c = computeRgbSilhouetteChamferCost(best_R, t_try, mesh_pts, mesh_normals, edge_maps, fx, fy, cx, cy);
            if (c < best_cost) {
                best_cost = c;
                best_t = t_try;
            }
        }
    }
    const cv::Vec3d fine_t = best_t;
    for (double dz = -range_z; dz <= range_z + 1e-6; dz += step_z) {
        if (std::abs(dz) < 1e-4) continue;
        const cv::Vec3d t_try(fine_t[0], fine_t[1], fine_t[2] + dz);
        const double c = computeRgbSilhouetteChamferCost(best_R, t_try, mesh_pts, mesh_normals, edge_maps, fx, fy, cx, cy);
        if (c < best_cost) {
            best_cost = c;
            best_t = t_try;
        }
    }

    R = best_R;
    t = best_t;
    return best_cost;
}

struct BidirectionalConsensus {
    double fwd_kernel{0.0};
    double bwd_kernel{0.0};
    double fwd_fit{0.0};
    double rmse_m{0.05};
    double bwd_rmse_m{0.08};
    double score{-1e9};
};

BidirectionalConsensus evaluateBidirectionalConsensus(const cv::Matx33d& R,
                                                     const cv::Vec3d& t,
                                                     double cad_diag,
                                                     const std::vector<cv::Vec3f>& mesh_pts,
                                                     const std::vector<cv::Vec3f>& mesh_normals,
                                                     const std::vector<cv::Vec3d>& scene_fg_pts,
                                                     double fx,
                                                     double fy,
                                                     double cx,
                                                     double cy,
                                                     int img_w,
                                                     int img_h,
                                                     bool fast_search = false)
{
    (void)fast_search;
    BidirectionalConsensus res;
    if (mesh_pts.empty() || scene_fg_pts.empty()) return res;

    const size_t vis_stride = std::max<size_t>(1, mesh_pts.size() / 1600);
    std::vector<cv::Vec3d> vis_pts, vis_norms;
    computeVisibleCadSubset(R, t, mesh_pts, mesh_normals, fx, fy, cx, cy, img_w, img_h, vis_stride, vis_pts, vis_norms);
    if (vis_pts.size() < 20) {
        // Retry without frustum margin clamp in case coarse estimate was shifted
        vis_pts.clear();
        for (size_t i = 0; i < mesh_pts.size(); i += vis_stride) {
            cv::Vec3d p_cam = R * cv::Vec3d(mesh_pts[i]) + t;
            cv::Vec3d n_cam = R * cv::Vec3d(mesh_normals[i]);
            if (p_cam[2] > 0.05 && n_cam.dot(p_cam) <= 0.15 * cv::norm(p_cam)) {
                vis_pts.emplace_back(mesh_pts[i]);
            }
        }
        if (vis_pts.size() < 20) return res;
    }

    const double tol = std::clamp(cad_diag * 0.055, 0.008, 0.020);
    const double tol_sq = tol * tol;
    const double sigma = std::clamp(cad_diag * 0.045, 0.007, 0.016);
    const double inv_2sig2 = 0.5 / (sigma * sigma);
    const double clamp_fwd = (3.0 * tol) * (3.0 * tol);
    const double clamp_bwd = (5.0 * tol) * (5.0 * tol);

    const cv::Matx33d R_inv = R.t();
    const size_t max_samp = 220;
    const size_t fg_step = std::max<size_t>(1, scene_fg_pts.size() / max_samp);
    double fwd_kern_sum = 0.0;
    double dsq_sum = 0.0;
    int fwd_inliers = 0;
    int fwd_total = 0;

    for (size_t i = 0; i < scene_fg_pts.size(); i += fg_step) {
        const cv::Vec3d p_cad = R_inv * (scene_fg_pts[i] - t);
        double min_dsq = 1e10;
        for (const auto& vp : vis_pts) {
            const cv::Vec3d d = p_cad - vp;
            const double dsq = d.dot(d);
            if (dsq < min_dsq) min_dsq = dsq;
        }
        fwd_kern_sum += std::exp(-min_dsq * inv_2sig2);
        dsq_sum += std::min(min_dsq, clamp_fwd);
        if (min_dsq <= tol_sq) ++fwd_inliers;
        ++fwd_total;
    }

    const size_t cad_step = std::max<size_t>(1, vis_pts.size() / max_samp);
    double bwd_kern_sum = 0.0;
    double bwd_dsq_sum = 0.0;
    int bwd_total = 0;
    for (size_t j = 0; j < vis_pts.size(); j += cad_step) {
        const cv::Vec3d p_cam = R * vis_pts[j] + t;
        double min_dsq = 1e10;
        for (size_t i = 0; i < scene_fg_pts.size(); i += fg_step) {
            const cv::Vec3d d = p_cam - scene_fg_pts[i];
            const double dsq = d.dot(d);
            if (dsq < min_dsq) min_dsq = dsq;
        }
        bwd_kern_sum += std::exp(-min_dsq * inv_2sig2);
        bwd_dsq_sum += std::min(min_dsq, clamp_bwd);
        ++bwd_total;
    }

    if (fwd_total > 0 && bwd_total > 0) {
        res.fwd_kernel = fwd_kern_sum / static_cast<double>(fwd_total);
        res.bwd_kernel = bwd_kern_sum / static_cast<double>(bwd_total);
        res.fwd_fit = static_cast<double>(fwd_inliers) / static_cast<double>(fwd_total);
        res.rmse_m = std::sqrt(dsq_sum / static_cast<double>(fwd_total));
        res.bwd_rmse_m = std::sqrt(bwd_dsq_sum / static_cast<double>(bwd_total));
        const double arith_mean = 0.5 * (res.fwd_kernel + res.bwd_kernel);
        const double geom_mean = std::sqrt(std::max(0.0, res.fwd_kernel) * std::max(0.0, res.bwd_kernel));
        res.score = 0.25 * arith_mean + 0.75 * geom_mean
                  - 1.5 * std::min(0.06, 0.5 * (res.rmse_m + res.bwd_rmse_m));
    }
    return res;
}

void runPointToPlaneIcpGeneric(const cv::Matx33d& R_init,
                               const cv::Vec3d& t_init,
                               const std::vector<cv::Vec3f>& mesh_pts,
                               const std::vector<cv::Vec3f>& mesh_normals,
                               const std::vector<cv::Vec3d>& fg_pts,
                               size_t fg_stride,
                               size_t mesh_stride,
                               const std::vector<double>& dist_scales,
                               int max_iters_per_scale,
                               double fx,
                               double fy,
                               double cx,
                               double cy,
                               int img_w,
                               int img_h,
                               cv::Matx33d& out_R,
                               cv::Vec3d& out_t)
{
    out_R = R_init;
    out_t = t_init;
    if (mesh_pts.empty() || fg_pts.size() < 18) return;

    fg_stride = std::max<size_t>(1, fg_stride);
    cv::Matx33d R_icp = R_init.t();
    cv::Vec3d t_icp = -R_icp * t_init;

    std::vector<cv::Vec3d> cad_vis_pts;
    std::vector<cv::Vec3d> cad_vis_norms;

    for (double max_corr : dist_scales) {
        max_corr = std::max(0.003, max_corr);
        cv::Matx33d R_cur = R_icp.t();
        cv::Vec3d t_cur = -R_cur * t_icp;
        computeVisibleCadSubset(R_cur, t_cur, mesh_pts, mesh_normals, fx, fy, cx, cy, img_w, img_h,
                                mesh_stride, cad_vis_pts, cad_vis_norms);
        if (cad_vis_pts.size() < 32) {
            cad_vis_pts.clear();
            cad_vis_norms.clear();
            const size_t fb_stride = std::max<size_t>(1, mesh_pts.size() / 1500);
            for (size_t i = 0; i < mesh_pts.size(); i += fb_stride) {
                cad_vis_pts.emplace_back(mesh_pts[i]);
                cad_vis_norms.emplace_back(mesh_normals[i]);
            }
        }

        const double tukey_k = std::max(0.0025, max_corr * 0.72);
        const double max_corr_sq = max_corr * max_corr;
        double prev_residual = 1e10;

        for (int iter = 0; iter < max_iters_per_scale; ++iter) {
            cv::Matx66d JtJ = cv::Matx66d::zeros();
            cv::Matx61d Jtb = cv::Matx61d::zeros();
            int n_corr = 0;
            double sum_res = 0.0;

            for (size_t fi = 0; fi < fg_pts.size(); fi += fg_stride) {
                const cv::Vec3d p_trans = R_icp * fg_pts[fi] + t_icp;
                double best_dsq = 1e10;
                int best_j = -1;
                for (size_t j = 0; j < cad_vis_pts.size(); ++j) {
                    const cv::Vec3d d = p_trans - cad_vis_pts[j];
                    const double dsq = d.dot(d);
                    if (dsq < best_dsq) {
                        best_dsq = dsq;
                        best_j = static_cast<int>(j);
                    }
                }
                if (best_dsq > max_corr_sq || best_j < 0) continue;

                cv::Vec3d n = cad_vis_norms[best_j];
                double n_norm = cv::norm(n);
                if (n_norm < 1e-6) continue;
                n *= (1.0 / n_norm);

                const cv::Vec3d res = p_trans - cad_vis_pts[best_j];
                const double r = n.dot(res);
                const double abs_r = std::abs(r);
                double w = 0.0;
                if (abs_r < tukey_k) {
                    const double rk = r / tukey_k;
                    w = (1.0 - rk * rk);
                    w = w * w;
                }
                if (w < 1e-8) continue;

                const cv::Vec3d cross_np = p_trans.cross(n);
                const double J[6] = {cross_np[0], cross_np[1], cross_np[2], n[0], n[1], n[2]};
                for (int a = 0; a < 6; ++a) {
                    for (int b = 0; b < 6; ++b) {
                        JtJ(a, b) += w * J[a] * J[b];
                    }
                    Jtb(a, 0) -= w * J[a] * r;
                }
                sum_res += w * r * r;
                ++n_corr;
            }

            if (n_corr < 14) break;
            for (int d = 0; d < 6; ++d) JtJ(d, d) += 1e-5;

            cv::Matx61d delta;
            if (!cv::solve(JtJ, Jtb, delta, cv::DECOMP_CHOLESKY)) {
                if (!cv::solve(JtJ, Jtb, delta, cv::DECOMP_SVD)) break;
            }

            const cv::Vec3d d_w(delta(0, 0), delta(1, 0), delta(2, 0));
            const cv::Vec3d d_t(delta(3, 0), delta(4, 0), delta(5, 0));
            const double d_theta = cv::norm(d_w);

            cv::Matx33d d_R = cv::Matx33d::eye();
            if (d_theta > 1e-8) {
                const double clamp_th = std::min(d_theta, 0.25);
                const cv::Vec3d axis = d_w * (1.0 / d_theta);
                const cv::Matx33d K(
                     0.0,     -axis[2],  axis[1],
                     axis[2],  0.0,     -axis[0],
                    -axis[1],  axis[0],  0.0);
                d_R = cv::Matx33d::eye() + std::sin(clamp_th) * K + (1.0 - std::cos(clamp_th)) * (K * K);
            }

            R_icp = d_R * R_icp;
            t_icp = d_R * t_icp + d_t;

            const double rmse = std::sqrt(sum_res / n_corr);
            if (cv::norm(d_t) < 1e-5 && d_theta < 1e-5) break;
            if (std::abs(rmse - prev_residual) / (prev_residual + 1e-10) < 1e-5) break;
            prev_residual = rmse;
        }
    }

    out_R = R_icp.t();
    out_t = -out_R * t_icp;
}

cv::Vec3d computeVisibleSurfaceOffset(const cv::Matx33d& R,
                                      const cv::Vec3d& c_target,
                                      const cv::Vec3d& cad_mid,
                                      const std::vector<cv::Vec3f>& mesh_pts,
                                      const std::vector<cv::Vec3f>& mesh_normals)
{
    cv::Vec3d t_guess = c_target - R * cad_mid;
    if (mesh_pts.empty()) return t_guess;

    const cv::Vec3d view_dir = c_target * (1.0 / std::max(1e-6, cv::norm(c_target)));
    const size_t step = std::max<size_t>(1, mesh_pts.size() / 800);

    // Tìm độ sâu mặt trước gần camera nhất (min_proj) của mô hình CAD ở hướng xoay R
    double min_proj = 1e9, max_proj = -1e9;
    for (size_t i = 0; i < mesh_pts.size(); i += step) {
        const cv::Vec3d p_rot = R * (cv::Vec3d(mesh_pts[i]) - cad_mid);
        const double pr = p_rot.dot(view_dir);
        if (pr < min_proj) min_proj = pr;
        if (pr > max_proj) max_proj = pr;
    }
    const double front_shell_cutoff = min_proj + 0.42 * (max_proj - min_proj);

    cv::Vec3d vis_sum(0.0, 0.0, 0.0);
    int vis_cnt = 0;
    for (size_t i = 0; i < mesh_pts.size(); i += step) {
        const cv::Vec3d p_local(mesh_pts[i]);
        const cv::Vec3d p_rot = R * (p_local - cad_mid);
        if (p_rot.dot(view_dir) > front_shell_cutoff) continue; // Loại bỏ mặt lưng & vách sâu phía sau
        const cv::Vec3d n_cam = R * cv::Vec3d(mesh_normals[i]);
        if (n_cam.dot(view_dir) < -0.05) {
            vis_sum += p_local;
            ++vis_cnt;
        }
    }
    if (vis_cnt >= 10) {
        const cv::Vec3d vis_mean_local = vis_sum * (1.0 / static_cast<double>(vis_cnt));
        return c_target - R * vis_mean_local;
    }
    return t_guess;
}

cv::Vec3d computePcaSigmas(const std::vector<cv::Vec3d>& pts)
{
    if (pts.size() < 6) return cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec3d mean(0.0, 0.0, 0.0);
    for (const auto& p : pts) mean += p;
    mean *= (1.0 / static_cast<double>(pts.size()));

    cv::Matx33d cov = cv::Matx33d::zeros();
    for (const auto& p : pts) {
        const cv::Vec3d d = p - mean;
        cov += d * d.t();
    }
    cv::Mat eig_vals;
    cv::eigen(cv::Mat(cov), eig_vals);
    const double inv_n = 1.0 / static_cast<double>(pts.size());
    double s0 = std::sqrt(std::max(0.0, eig_vals.at<double>(0, 0) * inv_n));
    double s1 = std::sqrt(std::max(0.0, eig_vals.at<double>(1, 0) * inv_n));
    double s2 = std::sqrt(std::max(0.0, eig_vals.at<double>(2, 0) * inv_n));
    if (s0 < s1) std::swap(s0, s1);
    if (s1 < s2) std::swap(s1, s2);
    if (s0 < s1) std::swap(s0, s1);
    return cv::Vec3d(s0, s1, s2);
}

void computeSilhouetteAndEmptyBehind(const cv::Matx33d& R,
                                     const cv::Vec3d& t,
                                     const std::vector<cv::Vec3f>& mesh_pts,
                                     const cv::Mat& depth_z16,
                                     double fx,
                                     double fy,
                                     double cx,
                                     double cy,
                                     double diag,
                                     double& out_empty_behind_frac,
                                     double& out_ring_jump_frac,
                                     double& out_wide_out_frac,
                                     double& out_wide_convex_ratio,
                                     int& out_ring_tot)
{
    out_empty_behind_frac = 0.0;
    out_ring_jump_frac = 1.0;
    out_wide_out_frac = 1.0;
    out_wide_convex_ratio = 1.0;
    out_ring_tot = 0;

    if (depth_z16.empty() || depth_z16.type() != CV_16UC1 ||
        depth_z16.cols < 64 || depth_z16.rows < 64 || mesh_pts.empty()) {
        return;
    }

    const int sw = depth_z16.cols / 2;
    const int sh = depth_z16.rows / 2;
    const double hfx = fx * 0.5;
    const double hfy = fy * 0.5;
    const double hcx = cx * 0.5;
    const double hcy = cy * 0.5;

    cv::Mat mask(sh, sw, CV_8UC1, cv::Scalar(0));
    cv::Mat z_pred(sh, sw, CV_32FC1, cv::Scalar(1e6f));

    int min_x = sw, max_x = -1, min_y = sh, max_y = -1;
    const size_t stride = std::max<size_t>(1, mesh_pts.size() / 5000);
    for (size_t j = 0; j < mesh_pts.size(); j += stride) {
        const cv::Vec3d p_cam = R * cv::Vec3d(mesh_pts[j]) + t;
        if (p_cam[2] <= 0.10) continue;
        const int sx = static_cast<int>(std::round(hfx * p_cam[0] / p_cam[2] + hcx));
        const int sy = static_cast<int>(std::round(hfy * p_cam[1] / p_cam[2] + hcy));
        if (sx >= 1 && sx < sw - 1 && sy >= 1 && sy < sh - 1) {
            mask.at<uint8_t>(sy, sx) = 255;
            const float zf = static_cast<float>(p_cam[2]);
            if (zf < z_pred.at<float>(sy, sx)) {
                z_pred.at<float>(sy, sx) = zf;
            }
            if (sx < min_x) min_x = sx;
            if (sx > max_x) max_x = sx;
            if (sy < min_y) min_y = sy;
            if (sy > max_y) max_y = sy;
        }
    }

    if (max_x <= min_x || max_y <= min_y) return;

    const int pad_px = 14;
    const int rx0 = std::max(0, min_x - pad_px);
    const int ry0 = std::max(0, min_y - pad_px);
    const int rx1 = std::min(sw, max_x + pad_px + 1);
    const int ry1 = std::min(sh, max_y + pad_px + 1);
    const cv::Rect roi(rx0, ry0, rx1 - rx0, ry1 - ry0);
    if (roi.width < 6 || roi.height < 6) return;

    cv::Mat mask_roi = mask(roi);
    cv::Mat zp_roi = z_pred(roi);
    cv::Mat closed_roi, dil_roi, dil5_roi, dil17_roi;
    cv::Mat k3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat k5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    const int k_ring = (depth_z16.cols >= 1200) ? 9 : 7;
    const int k_wide = (depth_z16.cols >= 1200) ? 17 : 13;
    cv::Mat kr = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k_ring, k_ring));
    cv::Mat kw = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k_wide, k_wide));
    cv::dilate(mask_roi, closed_roi, k3);
    cv::erode(closed_roi, closed_roi, k3);
    cv::dilate(closed_roi, dil_roi, kr);
    cv::dilate(closed_roi, dil5_roi, k5);
    cv::dilate(closed_roi, dil17_roi, kw);
    cv::Mat ring_roi = dil_roi - closed_roi;
    cv::Mat wide_roi = dil17_roi - dil5_roi;

    const float empty_tol = static_cast<float>(std::clamp(0.11 * diag, 0.016, 0.035));
    const float jump_tol = static_cast<float>(std::clamp(0.075 * diag, 0.010, 0.024));

    auto nearestMaskZ = [&](int ly, int lx, int rad) -> float {
        float zr = 1e6f;
        int best_d2 = 9999;
        const int ny0 = std::max(0, ly - rad);
        const int ny1 = std::min(roi.height - 1, ly + rad);
        const int nx0 = std::max(0, lx - rad);
        const int nx1 = std::min(roi.width - 1, lx + rad);
        for (int ny = ny0; ny <= ny1; ++ny) {
            const uint8_t* nm = mask_roi.ptr<uint8_t>(ny);
            const float* nzp = zp_roi.ptr<float>(ny);
            const int dy = ny - ly;
            for (int nx = nx0; nx <= nx1; ++nx) {
                if (nm[nx] > 0) {
                    const int dx = nx - lx;
                    const int d2 = dx * dx + dy * dy;
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        zr = nzp[nx];
                    }
                }
            }
        }
        return zr;
    };

    int z_tot = 0;
    int z_empty = 0;
    int r_tot = 0;
    int r_jump = 0;
    int w_tot = 0;
    int w_out = 0;
    int w_in = 0;

    for (int ly = 0; ly < roi.height; ++ly) {
        const int sy = ry0 + ly;
        const int py = sy * 2;
        if (py < 0 || py >= depth_z16.rows) continue;
        const uint16_t* dep_row = depth_z16.ptr<uint16_t>(py);
        const uint8_t* m_row = mask_roi.ptr<uint8_t>(ly);
        const uint8_t* r_row = ring_roi.ptr<uint8_t>(ly);
        const uint8_t* w_row = wide_roi.ptr<uint8_t>(ly);
        const float* zp_row = zp_roi.ptr<float>(ly);

        for (int lx = 0; lx < roi.width; ++lx) {
            const int sx = rx0 + lx;
            const int px = sx * 2;
            if (px < 0 || px >= depth_z16.cols) continue;
            const uint16_t d = dep_row[px];
            if (m_row[lx] > 0 && d > 0) {
                const float z_obs = static_cast<float>(d) * 0.001f;
                ++z_tot;
                if (z_obs > zp_row[lx] + empty_tol) {
                    ++z_empty;
                }
            } else {
                const bool in_near = (r_row[lx] > 0);
                const bool in_wide = (w_row[lx] > 0);
                if (!in_near && !in_wide) continue;
                const float zr = nearestMaskZ(ly, lx, in_wide ? 8 : 5);
                if (zr >= 1e5f) continue;

                if (in_near) {
                    ++r_tot;
                    if (d == 0 || std::abs(static_cast<float>(d) * 0.001f - zr) > jump_tol) {
                        ++r_jump;
                    }
                }
                if (in_wide) {
                    ++w_tot;
                    if (d == 0) {
                        ++w_out;
                    } else {
                        const float diff = static_cast<float>(d) * 0.001f - zr;
                        if (diff > jump_tol) ++w_out;
                        else if (diff < -jump_tol) ++w_in;
                    }
                }
            }
        }
    }

    if (z_tot >= 20) {
        out_empty_behind_frac = static_cast<double>(z_empty) / static_cast<double>(z_tot);
    }
    out_ring_tot = r_tot;
    if (r_tot > 0) {
        out_ring_jump_frac = static_cast<double>(r_jump) / static_cast<double>(r_tot);
    }
    if (w_tot >= 40) {
        out_wide_out_frac = static_cast<double>(w_out) / static_cast<double>(w_tot);
        const int w_noncop = w_out + w_in;
        out_wide_convex_ratio = (w_noncop >= 20) ? (static_cast<double>(w_out) / static_cast<double>(w_noncop)) : out_wide_out_frac;
    }
}
} // anonymous namespace

UniversalPoseEstimator::UniversalPoseEstimator()
    : _ort_env(ORT_LOGGING_LEVEL_WARNING, "TMCAD_UniversalPose"),
      _ort_mem_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    _symmetry_matrices.push_back(cv::Matx33d::eye());
}

UniversalPoseEstimator::~UniversalPoseEstimator()
{
    _ort_session.reset();
}

bool UniversalPoseEstimator::initialize(const std::string& onnx_or_model_dir)
{
    std::string req = onnx_or_model_dir;
    if (req.empty()) {
        const char* env_model = std::getenv("TMCAD_ONNX_MODEL");
        if (env_model && env_model[0] != '\0') {
            req = std::string(env_model);
        } else {
            req = "TABLE-DEMO";
        }
    }

    // 1. Dynamically discover any .onnx file in the provided path or models/ directory
    std::vector<std::string> candidates;
    if (!req.empty()) {
        if (fileExists(req) && req.size() > 5 && req.substr(req.size() - 5) == ".onnx") {
            candidates.push_back(req);
        } else {
            candidates.push_back(req + ".onnx");
            candidates.push_back("models/" + req + ".onnx");
            candidates.push_back("deploy_orange_pi_5b/" + req + "/" + req + ".onnx");
            candidates.push_back(req + "/model_obs_npu.onnx");
        }
    }

    // Scan models/ folder dynamically for any available .onnx if req was not explicitly found
    if (std::filesystem::exists("models") && std::filesystem::is_directory("models")) {
        for (const auto& entry : std::filesystem::directory_iterator("models")) {
            if (entry.path().extension() == ".onnx") {
                candidates.push_back(entry.path().string());
            }
        }
    }

    for (const auto& cand : candidates) {
        if (fileExists(cand) && loadModelOnnx(cand)) {
            return true;
        }
    }

    std::cerr << "[UniversalPoseEstimator] No valid .onnx model found for request: " << req << std::endl;
    return false;
}

static void filterCleanCadFeatureEdges(std::vector<FeatureEdge3D>& edges,
                                       const cv::Vec3d& cad_min,
                                       const cv::Vec3d& cad_max,
                                       bool is_floor_standing)
{
    if (edges.size() <= 4) return;
    const double model_diag = std::max(0.05, cv::norm(cad_max - cad_min));
    const float perp_tol = static_cast<float>(std::clamp(0.004 * model_diag, 0.0015, 0.006));
    const float gap_tol = static_cast<float>(std::clamp(0.012 * model_diag, 0.004, 0.020));
    const float min_keep = static_cast<float>(std::clamp(0.018 * model_diag, 0.005, 0.030));

    // 1. Collinear Line Welding (gộp các đoạn thẳng đồng tuyến bị cắt vụn bởi lưới STL)
    std::vector<FeatureEdge3D> segs = std::move(edges);
    std::sort(segs.begin(), segs.end(), [](const FeatureEdge3D& a, const FeatureEdge3D& b) {
        return a.length_m > b.length_m;
    });

    std::vector<bool> merged(segs.size(), false);
    std::vector<FeatureEdge3D> welded;
    welded.reserve(segs.size());

    const float main_min_len = is_floor_standing
        ? std::max(min_keep, static_cast<float>(model_diag * 0.065))
        : min_keep;
    const float foot_min_len = std::max(min_keep, static_cast<float>(model_diag * 0.028));
    const float foot_zone_z = static_cast<float>(cad_min[2] + (cad_max[2] - cad_min[2]) * 0.14);

    for (size_t i = 0; i < segs.size(); ++i) {
        if (merged[i]) continue;
        merged[i] = true;

        cv::Vec3f a = segs[i].p0;
        cv::Vec3f b = segs[i].p1;
        cv::Vec3f dir = b - a;
        float len = static_cast<float>(cv::norm(dir));
        if (len < 1e-4f) continue;
        dir *= (1.0f / len);

        bool grew = true;
        while (grew) {
            grew = false;
            for (size_t j = i + 1; j < segs.size(); ++j) {
                if (merged[j]) continue;
                cv::Vec3f dj = segs[j].p1 - segs[j].p0;
                const float lj = segs[j].length_m;
                if (lj < 1e-4f) continue;
                dj *= (1.0f / lj);
                if (std::abs(dir.dot(dj)) < 0.996f) continue;

                const cv::Vec3f diff = segs[j].p0 - a;
                const cv::Vec3f perp = diff - dir * diff.dot(dir);
                if (cv::norm(perp) > perp_tol) continue;

                float tj0 = (segs[j].p0 - a).dot(dir);
                float tj1 = (segs[j].p1 - a).dot(dir);
                if (tj0 > tj1) std::swap(tj0, tj1);

                if (tj1 >= -gap_tol && tj0 <= len + gap_tol) {
                    const float new_min = std::min(0.0f, tj0);
                    const float new_max = std::max(len, tj1);
                    b = a + dir * new_max;
                    a = a + dir * new_min;
                    len = new_max - new_min;
                    merged[j] = true;
                    grew = true;
                }
            }
        }

        const bool is_foot_edge = is_floor_standing && (std::min(a[2], b[2]) <= foot_zone_z);
        const float req_len = is_foot_edge ? foot_min_len : main_min_len;
        if (len >= req_len) {
            welded.push_back(FeatureEdge3D{a, b, segs[i].n0, segs[i].n1, len});
        }
    }

    if (!is_floor_standing) {
        edges = std::move(welded);
        return;
    }

    // 2. 2D Cross-Section Corner Hull (khử rãnh T-slot và giữ 4 cạnh biên ngoài cho kết cấu đứng sàn)
    std::sort(welded.begin(), welded.end(), [](const FeatureEdge3D& e1, const FeatureEdge3D& e2) {
        return e1.length_m > e2.length_m;
    });

    std::vector<bool> grouped(welded.size(), false);
    std::vector<FeatureEdge3D> clean_edges;
    clean_edges.reserve(welded.size());
    const float beam_cluster_radius = static_cast<float>(model_diag * 0.050);
    const float dup_perp_tol = static_cast<float>(std::clamp(0.006 * model_diag, 0.002, 0.010));
    const float corner_pad = static_cast<float>(0.003 * model_diag);

    for (size_t i = 0; i < welded.size(); ++i) {
        if (grouped[i]) continue;
        grouped[i] = true;

        const auto& seed = welded[i];
        cv::Vec3f dir = seed.p1 - seed.p0;
        dir *= (1.0f / std::max(1e-5f, seed.length_m));
        if (dir[0] + dir[1] + dir[2] < 0.0f) dir = -dir;

        const cv::Vec3f origin = 0.5f * (seed.p0 + seed.p1);
        std::vector<size_t> cluster = {i};

        for (size_t j = i + 1; j < welded.size(); ++j) {
            if (grouped[j]) continue;
            cv::Vec3f dj = welded[j].p1 - welded[j].p0;
            dj *= (1.0f / std::max(1e-5f, welded[j].length_m));
            if (std::abs(dir.dot(dj)) < 0.992f) continue;

            const cv::Vec3f mid_j = 0.5f * (welded[j].p0 + welded[j].p1);
            const cv::Vec3f diff = mid_j - origin;
            const float axial = std::abs(diff.dot(dir));
            const cv::Vec3f perp = diff - dir * diff.dot(dir);

            if (cv::norm(perp) <= beam_cluster_radius &&
                axial <= 0.60f * (seed.length_m + welded[j].length_m)) {
                grouped[j] = true;
                cluster.push_back(j);
            }
        }

        if (cluster.size() <= 2) {
            for (size_t idx : cluster) clean_edges.push_back(welded[idx]);
            continue;
        }

        const cv::Vec3f ref_up = (std::abs(dir[2]) < 0.85f) ? cv::Vec3f(0, 0, 1) : cv::Vec3f(1, 0, 0);
        cv::Vec3f u_axis = dir.cross(ref_up);
        u_axis *= (1.0f / std::max(1e-6f, static_cast<float>(cv::norm(u_axis))));
        cv::Vec3f v_axis = dir.cross(u_axis);
        v_axis *= (1.0f / std::max(1e-6f, static_cast<float>(cv::norm(v_axis))));

        float u_min = 1e9f, u_max = -1e9f, v_min = 1e9f, v_max = -1e9f;
        std::vector<cv::Vec2f> uv_coords(cluster.size());
        for (size_t c = 0; c < cluster.size(); ++c) {
            const auto& e = welded[cluster[c]];
            const cv::Vec3f mid = 0.5f * (e.p0 + e.p1) - origin;
            const float u = mid.dot(u_axis);
            const float v = mid.dot(v_axis);
            uv_coords[c] = cv::Vec2f(u, v);
            u_min = std::min(u_min, u);
            u_max = std::max(u_max, u);
            v_min = std::min(v_min, v);
            v_max = std::max(v_max, v);
        }

        const float span_u = u_max - u_min;
        const float span_v = v_max - v_min;
        const cv::Vec2f corners_2d[4] = {
            cv::Vec2f(u_min, v_min), cv::Vec2f(u_max, v_min),
            cv::Vec2f(u_max, v_max), cv::Vec2f(u_min, v_max)
        };

        std::vector<size_t> selected_in_cluster;
        for (int cor = 0; cor < 4; ++cor) {
            int best_c = -1;
            float best_score = 1e9f;
            for (size_t c = 0; c < cluster.size(); ++c) {
                const float dist_corner = static_cast<float>(cv::norm(uv_coords[c] - corners_2d[cor]));
                if (dist_corner > 0.32f * std::max(span_u, span_v) + corner_pad) continue;
                const float score = dist_corner - 0.015f * welded[cluster[c]].length_m;
                if (score < best_score) {
                    best_score = score;
                    best_c = static_cast<int>(c);
                }
            }
            if (best_c >= 0) {
                const size_t chosen_idx = cluster[best_c];
                bool dup = false;
                for (size_t prev_idx : selected_in_cluster) {
                    if (prev_idx == chosen_idx) { dup = true; break; }
                    const cv::Vec3f d_mid = 0.5f * (welded[chosen_idx].p0 + welded[chosen_idx].p1) -
                                            0.5f * (welded[prev_idx].p0 + welded[prev_idx].p1);
                    const cv::Vec3f perp = d_mid - dir * d_mid.dot(dir);
                    if (cv::norm(perp) < dup_perp_tol) { dup = true; break; }
                }
                if (!dup) selected_in_cluster.push_back(chosen_idx);
            }
        }

        for (size_t idx : selected_in_cluster) {
            clean_edges.push_back(welded[idx]);
        }
    }

    edges = std::move(clean_edges);
}

bool UniversalPoseEstimator::loadModelOnnx(const std::string& onnx_path)
{
    if (!fileExists(onnx_path)) return false;

    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        _ort_session = std::make_unique<Ort::Session>(_ort_env, onnx_path.c_str(), opts);
        _ort_loaded = true;
        _loaded_onnx_path = onnx_path;
        _model_name = extractStem(onnx_path);

        if (_model_name == "model_obs_npu" || _model_name == "model_obs_backbone") {
            auto parent = std::filesystem::path(onnx_path).parent_path().filename().string();
            if (!parent.empty() && parent != "models") _model_name = parent;
        }

        // Reset cached CAD buffers before unpacking new model
        _cad_descriptors.clear();
        _cad_centroids.clear();
        _mesh_pts.clear();
        _mesh_normals.clear();
        _cad_feature_edges.clear();
        _cad_top_polygon.clear();
        _cad_foot_centers.clear();
        _meta_unpacked = false;
        _symmetry_code = 0;
        _is_floor_standing = false;
        _enable_2d_frame_anchor = false;
        _initialized = false;
        _last_valid_depth_pts = 0;
        _last_floor_locked = false;
        _last_floor_pitch_deg = 0.0;
        _last_sparse_depth_fallback = false;
        _last_ai_latency_ms = 0.0;
        _last_solve_latency_ms = 0.0;
        _last_total_latency_ms = 0.0;
        _last_segmented_points.clear();
        _mask_matcher.clearTemplates();
        resetTemporalTrack();

        if (unpackBlackBoxOnnx()) {
            unpackBlackBoxOnnxMetadata();

            // Auto-load Shared 2D Vision Backbones (FastSAM + DINOv2) & Per-CAD Multi-View Templates
            {
                std::filesystem::path onnx_p(onnx_path);
                std::filesystem::path repo_root_cand = onnx_p.parent_path().parent_path();
                std::vector<std::filesystem::path> shared_dirs = {
                    onnx_p.parent_path() / "shared",
                    repo_root_cand / "models" / "shared",
                    std::filesystem::path("models") / "shared"
                };
                for (const auto& sd : shared_dirs) {
                    const std::string fs_p = (sd / "fastsam_s.onnx").string();
                    const std::string dn_p = (sd / "dinov2_vits14.onnx").string();
                    if (fileExists(fs_p) && fileExists(dn_p)) {
                        _mask_matcher.initBackbones(fs_p, dn_p);
                        break;
                    }
                }
                std::vector<std::string> tpl_cands = {
                    (onnx_p.parent_path() / (_model_name + "_templates.bin")).string(),
                    (repo_root_cand / "models" / (_model_name + "_templates.bin")).string(),
                    "models/" + _model_name + "_templates.bin"
                };
                for (const auto& tp : tpl_cands) {
                    if (fileExists(tp) && _mask_matcher.loadTemplates(tp)) {
                        break;
                    }
                }
            }

            // Auto-discover optional <CAD_NAME>_profile.json for adaptive solver hints
            std::filesystem::path onnx_p(onnx_path);
            std::filesystem::path repo_root_cand = onnx_p.parent_path().parent_path();
            std::vector<std::string> profile_cands = {
                (onnx_p.parent_path() / (_model_name + "_profile.json")).string(),
                "CAD_DATA/" + _model_name + "/" + _model_name + "_profile.json",
                (repo_root_cand / "CAD_DATA" / _model_name / (_model_name + "_profile.json")).string(),
                "configs/" + _model_name + "_profile.json",
                (repo_root_cand / "configs" / (_model_name + "_profile.json")).string()
            };
            for (const auto& prof : profile_cands) {
                if (fileExists(prof) && loadUniversalProfile(prof)) {
                    break;
                }
            }

            if (_symmetry_code == 0) {
                std::vector<std::string> sym_cands = {
                    (onnx_p.parent_path() / (_model_name + "_symmetry.json")).string(),
                    "CAD_DATA/" + _model_name + "/" + _model_name + "_symmetry.json",
                    (repo_root_cand / "CAD_DATA" / _model_name / (_model_name + "_symmetry.json")).string()
                };
                for (const auto& sym_p : sym_cands) {
                    if (fileExists(sym_p)) {
                        std::ifstream sifs(sym_p);
                        std::stringstream sss;
                        sss << sifs.rdbuf();
                        const std::string sc = sss.str();
                        if (sc.find("C4") != std::string::npos || sc.find("D2") != std::string::npos) {
                            _symmetry_code = 2;
                        } else if (sc.find("C2") != std::string::npos || sc.find("\"is_symmetric\": true") != std::string::npos) {
                            _symmetry_code = 1;
                        }
                        break;
                    }
                }
            }

            filterCleanCadFeatureEdges(_cad_feature_edges, _cad_min, _cad_max, _is_floor_standing);

            _initialized = true;
            std::cout << "[UniversalPoseEstimator] Ready with Black-Box ONNX (" << _model_name
                      << ") | Mode: " << (_is_floor_standing ? "FLOOR_STANDING" : "FREE_6DOF")
                      << " | Diag: " << _real_bounding_diag_m << " m"
                      << " | CAD Pts: " << _cad_centroids.size()
                      << " | Surf Pts: " << _mesh_pts.size()
                      << " | Edges: " << _cad_feature_edges.size() << std::endl;
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "[UniversalPoseEstimator] ONNX load error: " << e.what() << std::endl;
        _ort_loaded = false;
    }
    return false;
}

bool UniversalPoseEstimator::unpackBlackBoxOnnx()
{
    if (!_ort_loaded || !_ort_session) return false;
    const size_t num_outputs = _ort_session->GetOutputCount();
    if (num_outputs < 5) return false;

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name_ptr = _ort_session->GetInputNameAllocated(0, alloc);
    const char* in_names[] = {in_name_ptr.get()};

    // Map all available ONNX outputs by exact tensor name (schema-agnostic & future-proof)
    std::vector<Ort::AllocatedStringPtr> out_name_ptrs;
    std::vector<const char*> out_names;
    std::unordered_map<std::string, size_t> name_to_idx;
    out_name_ptrs.reserve(num_outputs);
    out_names.reserve(num_outputs);

    for (size_t i = 0; i < num_outputs; ++i) {
        out_name_ptrs.push_back(_ort_session->GetOutputNameAllocated(i, alloc));
        const char* name_cstr = out_name_ptrs.back().get();
        out_names.push_back(name_cstr);
        name_to_idx[std::string(name_cstr)] = i;
    }

    // Determine expected observation point count from ONNX input shape if fixed
    {
        auto in_shape = _ort_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (in_shape.size() == 3 && in_shape[1] > 0) {
            _num_obs_points = static_cast<size_t>(in_shape[1]);
        }
    }

    std::vector<float> dummy_in(_num_obs_points * 3, 0.0f);
    std::array<int64_t, 3> in_shape{1, static_cast<int64_t>(_num_obs_points), 3};
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        _ort_mem_info, dummy_in.data(), dummy_in.size(), in_shape.data(), in_shape.size());

    auto outs = _ort_session->Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1,
                                  out_names.data(), out_names.size());

    auto findOut = [&](const std::string& key) -> const Ort::Value* {
        auto it = name_to_idx.find(key);
        if (it != name_to_idx.end() && it->second < outs.size()) {
            return &outs[it->second];
        }
        return nullptr;
    };

    // 1. cad_descriptors: (M, D)
    if (const Ort::Value* v = findOut("cad_descriptors")) {
        auto shape = v->GetTensorTypeAndShapeInfo().GetShape();
        const float* ptr = v->GetTensorData<float>();
        if (shape.size() == 2 && shape[0] > 0 && shape[1] > 0 && ptr) {
            _num_cad_samples = static_cast<size_t>(shape[0]);
            _desc_dim = static_cast<size_t>(shape[1]);
            _cad_descriptors.assign(ptr, ptr + _num_cad_samples * _desc_dim);
        }
    }

    // 2. cad_centroids: (M, 3)
    if (const Ort::Value* v = findOut("cad_centroids")) {
        auto shape = v->GetTensorTypeAndShapeInfo().GetShape();
        const float* ptr = v->GetTensorData<float>();
        if (shape.size() == 2 && shape[1] == 3 && ptr) {
            size_t m = static_cast<size_t>(shape[0]);
            _cad_centroids.resize(m);
            for (size_t i = 0; i < m; ++i) {
                _cad_centroids[i] = cv::Vec3f(ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]);
            }
        }
    }

    // 3. cad_surface_pts & cad_surface_normals: (N_surf, 3)
    if (const Ort::Value* vp = findOut("cad_surface_pts")) {
        auto shape_p = vp->GetTensorTypeAndShapeInfo().GetShape();
        const float* p_ptr = vp->GetTensorData<float>();
        const Ort::Value* vn = findOut("cad_surface_normals");
        const float* n_ptr = vn ? vn->GetTensorData<float>() : nullptr;
        if (shape_p.size() == 2 && shape_p[1] == 3 && p_ptr) {
            size_t n_surf = static_cast<size_t>(shape_p[0]);
            _mesh_pts.resize(n_surf);
            _mesh_normals.resize(n_surf);
            for (size_t i = 0; i < n_surf; ++i) {
                _mesh_pts[i] = cv::Vec3f(p_ptr[i * 3 + 0], p_ptr[i * 3 + 1], p_ptr[i * 3 + 2]);
                if (n_ptr) {
                    _mesh_normals[i] = cv::Vec3f(n_ptr[i * 3 + 0], n_ptr[i * 3 + 1], n_ptr[i * 3 + 2]);
                } else {
                    _mesh_normals[i] = cv::Vec3f(0.0f, 0.0f, 1.0f);
                }
            }
        }
    }

    // 4. cad_feature_edges (E, 13) [p0(3), p1(3), n0(3), n1(3), len] or cad_wireframe_edges (E, 6)
    if (const Ort::Value* v = findOut("cad_feature_edges")) {
        auto shape = v->GetTensorTypeAndShapeInfo().GetShape();
        const float* e_ptr = v->GetTensorData<float>();
        if (shape.size() == 2 && e_ptr) {
            size_t n_edges = static_cast<size_t>(shape[0]);
            size_t cols = static_cast<size_t>(shape[1]);
            _cad_feature_edges.clear();
            _cad_feature_edges.reserve(n_edges);
            if (cols >= 13) {
                for (size_t i = 0; i < n_edges; ++i) {
                    const float* r = e_ptr + i * cols;
                    cv::Vec3f p0(r[0], r[1], r[2]);
                    cv::Vec3f p1(r[3], r[4], r[5]);
                    cv::Vec3f n0(r[6], r[7], r[8]);
                    cv::Vec3f n1(r[9], r[10], r[11]);
                    float len = r[12];
                    if (len <= 1e-5f) len = static_cast<float>(cv::norm(p1 - p0));
                    if (len > 1e-4f) {
                        _cad_feature_edges.push_back(FeatureEdge3D{p0, p1, n0, n1, len});
                    }
                }
            } else if (cols >= 6) {
                for (size_t i = 0; i < n_edges; ++i) {
                    const float* r = e_ptr + i * cols;
                    cv::Vec3f p0(r[0], r[1], r[2]);
                    cv::Vec3f p1(r[3], r[4], r[5]);
                    float len = static_cast<float>(cv::norm(p1 - p0));
                    if (len > 1e-4f) {
                        _cad_feature_edges.push_back(FeatureEdge3D{p0, p1, {0, 0, 1}, {0, 0, 1}, len});
                    }
                }
            }
        }
    } else if (const Ort::Value* vw = findOut("cad_wireframe_edges")) {
        auto shape = vw->GetTensorTypeAndShapeInfo().GetShape();
        const float* e_ptr = vw->GetTensorData<float>();
        if (shape.size() >= 2 && e_ptr) {
            size_t n_edges = static_cast<size_t>(shape[0]);
            _cad_feature_edges.clear();
            _cad_feature_edges.reserve(n_edges);
            for (size_t i = 0; i < n_edges; ++i) {
                cv::Vec3f p0(e_ptr[i * 6 + 0], e_ptr[i * 6 + 1], e_ptr[i * 6 + 2]);
                cv::Vec3f p1(e_ptr[i * 6 + 3], e_ptr[i * 6 + 4], e_ptr[i * 6 + 5]);
                float len = static_cast<float>(cv::norm(p1 - p0));
                if (len > 1e-4f) {
                    _cad_feature_edges.push_back(FeatureEdge3D{p0, p1, {0, 0, 1}, {0, 0, 1}, len});
                }
            }
        }
    }

    // 5. cad_top_poly & cad_foot_centers
    if (const Ort::Value* v = findOut("cad_top_poly")) {
        auto shape = v->GetTensorTypeAndShapeInfo().GetShape();
        const float* ptr = v->GetTensorData<float>();
        if (shape.size() == 2 && shape[1] == 3 && ptr) {
            size_t n = static_cast<size_t>(shape[0]);
            _cad_top_polygon.resize(n);
            for (size_t i = 0; i < n; ++i) {
                _cad_top_polygon[i] = cv::Vec3f(ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]);
            }
        }
    }
    if (const Ort::Value* v = findOut("cad_foot_centers")) {
        auto shape = v->GetTensorTypeAndShapeInfo().GetShape();
        const float* ptr = v->GetTensorData<float>();
        if (shape.size() == 2 && shape[1] == 3 && ptr) {
            size_t n = static_cast<size_t>(shape[0]);
            _cad_foot_centers.resize(n);
            for (size_t i = 0; i < n; ++i) {
                _cad_foot_centers[i] = cv::Vec3f(ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]);
            }
        }
    }

    // 6. cad_meta (16,) from export_hierarchy_onnx.py (canonical schema)
    if (const Ort::Value* v = findOut("cad_meta")) {
        auto count = v->GetTensorTypeAndShapeInfo().GetElementCount();
        const float* ptr = v->GetTensorData<float>();
        if (count >= 12 && ptr) {
            _cad_min = cv::Vec3d(ptr[0], ptr[1], ptr[2]);
            _cad_max = cv::Vec3d(ptr[3], ptr[4], ptr[5]);
            _half_w  = static_cast<double>(ptr[6]);
            _half_l  = static_cast<double>(ptr[7]);
            _top_z   = static_cast<double>(ptr[8]);
            _bot_z   = static_cast<double>(ptr[9]);
            _symmetry_code = static_cast<int>(std::round(ptr[10]));
            _is_floor_standing = (ptr[11] > 0.5f);
            _enable_2d_frame_anchor = _is_floor_standing;
            _meta_unpacked = true;
        }
    } else {
        // Optional fallback schema: cad_bbox_bounds + object_meta_flags
        if (const Ort::Value* vb = findOut("cad_bbox_bounds")) {
            auto count = vb->GetTensorTypeAndShapeInfo().GetElementCount();
            const float* b_ptr = vb->GetTensorData<float>();
            if (count >= 6 && b_ptr) {
                _cad_min = cv::Vec3d(b_ptr[0], b_ptr[1], b_ptr[2]);
                _cad_max = cv::Vec3d(b_ptr[3], b_ptr[4], b_ptr[5]);
                _meta_unpacked = true;
            }
        }
        if (const Ort::Value* vm = findOut("object_meta_flags")) {
            auto count = vm->GetTensorTypeAndShapeInfo().GetElementCount();
            const float* m_ptr = vm->GetTensorData<float>();
            if (count >= 2 && m_ptr) {
                _is_floor_standing = (m_ptr[0] > 0.5f);
                _real_bounding_diag_m = static_cast<double>(m_ptr[1]);
                _meta_unpacked = true;
            }
        }
    }

    return (!_cad_descriptors.empty() && !_cad_centroids.empty());
}

bool UniversalPoseEstimator::unpackBlackBoxOnnxMetadata()
{
    if (!_meta_unpacked) {
        std::string upper_name = _model_name;
        std::transform(upper_name.begin(), upper_name.end(), upper_name.begin(), ::toupper);

        if (upper_name.find("TABLE") != std::string::npos ||
            upper_name.find("PALLET") != std::string::npos ||
            upper_name.find("RACK") != std::string::npos) {
            _is_floor_standing = true;
        }
    }

    // Fallback bbox calculation from mesh/centroids if cad_meta wasn't present
    if (!_meta_unpacked && !_mesh_pts.empty()) {
        cv::Vec3d bmin(1e9, 1e9, 1e9), bmax(-1e9, -1e9, -1e9);
        for (const auto& p : _mesh_pts) {
            for (int k = 0; k < 3; ++k) {
                bmin[k] = std::min(bmin[k], static_cast<double>(p[k]));
                bmax[k] = std::max(bmax[k], static_cast<double>(p[k]));
            }
        }
        _cad_min = bmin;
        _cad_max = bmax;
    }

    _half_w = 0.5 * (_cad_max[0] - _cad_min[0]);
    _half_l = 0.5 * (_cad_max[1] - _cad_min[1]);
    _bot_z  = _cad_min[2];
    _top_z  = _cad_max[2];
    _real_bounding_diag_m = std::max(0.05, cv::norm(_cad_max - _cad_min));

    float length_m = static_cast<float>(std::max(_cad_max[0] - _cad_min[0], _cad_max[1] - _cad_min[1]));
    float width_m  = static_cast<float>(std::min(_cad_max[0] - _cad_min[0], _cad_max[1] - _cad_min[1]));
    float height_m = static_cast<float>(_cad_max[2] - _cad_min[2]);

    _topology_type = _is_floor_standing ? "FLOOR_STANDING" : "FREE_6DOF";
    _scene_segmenter.setExpectedObjectDimensions(length_m, width_m, height_m, _is_floor_standing);
    _scene_segmenter.setMultiInstanceMode(!_is_floor_standing && _multi_instance_enabled, _max_instances);
    return true;
}

bool UniversalPoseEstimator::loadUniversalProfile(const std::string& profile_json_path)
{
    if (!_scene_segmenter.loadProfileConfig(profile_json_path)) return false;
    std::ifstream ifs(profile_json_path);
    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string content = ss.str();
    _is_floor_standing = (content.find("\"placement_mode\": \"FLOOR_STANDING\"") != std::string::npos ||
                          content.find("FRAME_HOLLOW") != std::string::npos);
    _enable_2d_frame_anchor = (content.find("\"enable_2d_frame_anchor\": true") != std::string::npos);
    _topology_type = _is_floor_standing ? "FLOOR_STANDING" : "FREE_6DOF";
    _scene_segmenter.setMultiInstanceMode(!_is_floor_standing && _multi_instance_enabled, _max_instances);
    return true;
}

void UniversalPoseEstimator::setMultiInstanceMode(bool enable, int max_instances)
{
    _multi_instance_enabled = enable;
    _max_instances = std::max(1, max_instances);
    _scene_segmenter.setMultiInstanceMode(!_is_floor_standing && _multi_instance_enabled, _max_instances);
}

bool UniversalPoseEstimator::isInTargetWorkspace(const cv::Vec3d& t) const
{
    if (!std::isfinite(t[0]) || !std::isfinite(t[1]) || !std::isfinite(t[2])) return false;

    if (_is_floor_standing) {
        return (t[0] >= -1.35 && t[0] <= 1.35) &&
               (t[1] >= -0.95 && t[1] <= 0.65) &&
               (t[2] >= 0.45 && t[2] <= 2.95);
    }

    double max_z = std::clamp(_real_bounding_diag_m * 3.8 + 0.15, 0.95, 2.50);
    return (std::abs(t[0]) <= 1.20) &&
           (std::abs(t[1]) <= 1.00) &&
           (t[2] >= 0.15 && t[2] <= max_z);
}

bool UniversalPoseEstimator::isCandidateRotationValid(const cv::Matx33d& R_candidate) const
{
    if (_is_floor_standing) {
        // Hỗ trợ camera đặt từ thấp (0.12m) tới cao (1.85m, góc chúc tới ~78 độ):
        if (R_candidate(1, 2) > -0.18 ||
            std::abs(R_candidate(0, 2)) > 0.55 ||
            std::abs(R_candidate(2, 2)) > 0.98) {
            return false;
        }
    }
    return true;
}

bool UniversalPoseEstimator::runInference(const std::vector<cv::Vec3f>& obs_pts,
                                          std::vector<float>& out_obs_desc) const
{
    if (!_ort_loaded || !_ort_session || obs_pts.empty()) return false;

    std::vector<float> input_buf(_num_obs_points * 3, 0.0f);
    for (size_t i = 0; i < _num_obs_points; ++i) {
        const auto& p = obs_pts[i % obs_pts.size()];
        input_buf[i * 3 + 0] = p[0];
        input_buf[i * 3 + 1] = p[1];
        input_buf[i * 3 + 2] = p[2];
    }

    std::array<int64_t, 3> in_shape{1, static_cast<int64_t>(_num_obs_points), 3};
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        _ort_mem_info, input_buf.data(), input_buf.size(), in_shape.data(), in_shape.size());

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = _ort_session->GetInputNameAllocated(0, alloc);
    const char* in_names[] = {in_name.get()};
    const char* out_names[] = {"D_obs"};

    try {
        auto outs = _ort_session->Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
        if (outs.empty()) return false;
        const float* ptr = outs[0].GetTensorData<float>();
        if (!ptr) return false;
        out_obs_desc.assign(ptr, ptr + _num_obs_points * _desc_dim);
        return true;
    } catch (...) {
        auto out0_name = _ort_session->GetOutputNameAllocated(0, alloc);
        const char* fallback_names[] = {out0_name.get()};
        auto outs = _ort_session->Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, fallback_names, 1);
        if (outs.empty()) return false;
        const float* ptr = outs[0].GetTensorData<float>();
        if (!ptr) return false;
        out_obs_desc.assign(ptr, ptr + _num_obs_points * _desc_dim);
        return true;
    }
}

ObjectPose UniversalPoseEstimator::solveRansacKabsch(const std::vector<cv::Vec3f>& obs_pts,
                                                     const std::vector<float>& obs_desc,
                                                     const cv::Vec4d& floor_plane,
                                                     bool has_floor) const
{
    ObjectPose best_pose;
    best_pose.model_name = _model_name;
    if (obs_pts.size() < 16 || _cad_centroids.empty() || obs_desc.empty() || _cad_descriptors.empty()) {
        return best_pose;
    }

    const size_t N = std::min(obs_pts.size(), _num_obs_points);
    const size_t M = _cad_centroids.size();
    if (_cad_descriptors.size() < M * _desc_dim || obs_desc.size() < N * _desc_dim) {
        return best_pose;
    }

    // 1. Cosine similarity matching (obs_i -> cad_j)
    std::vector<int> matches(N, 0);
    std::vector<float> top_scores(N, 0.0f);
    float max_best_sim = -2.0f;
    for (size_t i = 0; i < N; ++i) {
        const float* o_vec = &obs_desc[i * _desc_dim];
        float best_sim = -2.0f;
        int best_j = 0;
        for (size_t j = 0; j < M; ++j) {
            const float* c_vec = &_cad_descriptors[j * _desc_dim];
            float dot = 0.0f;
            for (size_t d = 0; d < _desc_dim; ++d) dot += o_vec[d] * c_vec[d];
            if (dot > best_sim) {
                best_sim = dot;
                best_j = static_cast<int>(j);
            }
        }
        matches[i] = best_j;
        top_scores[i] = best_sim;
        if (best_sim > max_best_sim) max_best_sim = best_sim;
    }

    // 2. Cluster NMS: keep top 3 observation points per CAD centroid so symmetric modes stay balanced
    std::unordered_map<int, std::vector<int>> centroid_to_obs;
    centroid_to_obs.reserve(M);
    for (size_t i = 0; i < N; ++i) {
        centroid_to_obs[matches[i]].push_back(static_cast<int>(i));
    }

    std::vector<int> nms_indices;
    nms_indices.reserve(N);
    for (auto& kv : centroid_to_obs) {
        auto& indices = kv.second;
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return top_scores[a] > top_scores[b];
        });
        int take = std::min(3, static_cast<int>(indices.size()));
        for (int k = 0; k < take; ++k) {
            nms_indices.push_back(indices[k]);
        }
    }
    if (nms_indices.size() < 16) {
        nms_indices.resize(N);
        for (size_t i = 0; i < N; ++i) nms_indices[i] = static_cast<int>(i);
    }

    const double density_factor = (M <= 160) ? 0.085 : ((M <= 320) ? 0.062 : 0.048);
    const double inlier_thresh = _is_floor_standing
        ? std::clamp(_real_bounding_diag_m * 0.038, 0.008, 0.045)
        : std::clamp(_real_bounding_diag_m * density_factor, 0.015, 0.045);

    int best_inliers = 0;
    double best_err_sum = 0.0;
    std::vector<uint8_t> best_mask(N, 0);

    cv::Vec3d obs_center(0.0, 0.0, 0.0);
    for (size_t i = 0; i < N; ++i) obs_center += cv::Vec3d(obs_pts[i]);
    obs_center *= (1.0 / static_cast<double>(N));
    const cv::Vec3d cad_mid = 0.5 * (_cad_min + _cad_max);
    const double bbox_pad = std::max(0.020, 0.14 * _real_bounding_diag_m);
    const cv::Vec3d box_lo = _cad_min - cv::Vec3d(bbox_pad, bbox_pad, bbox_pad);
    const cv::Vec3d box_hi = _cad_max + cv::Vec3d(bbox_pad, bbox_pad, bbox_pad);

    std::vector<size_t> check_idx;
    if (!_is_floor_standing) {
        const size_t cstep = std::max<size_t>(1, N / 64);
        for (size_t i = 0; i < N; i += cstep) check_idx.push_back(i);
    }
    auto passesClusterContainment = [&](const cv::Matx33d& R_cand, const cv::Vec3d& t_cand) -> bool {
        if (_is_floor_standing) return true;
        const cv::Vec3d pred_mid = R_cand * cad_mid + t_cand;
        if (cv::norm(pred_mid - obs_center) > 0.65 * _real_bounding_diag_m + 0.035) return false;
        if (check_idx.empty()) return true;
        const cv::Matx33d Rt = R_cand.t();
        int inside = 0;
        for (size_t idx : check_idx) {
            const cv::Vec3d p_loc = Rt * (cv::Vec3d(obs_pts[idx]) - t_cand);
            if (p_loc[0] >= box_lo[0] && p_loc[0] <= box_hi[0] &&
                p_loc[1] >= box_lo[1] && p_loc[1] <= box_hi[1] &&
                p_loc[2] >= box_lo[2] && p_loc[2] <= box_hi[2]) {
                ++inside;
            }
        }
        return inside >= static_cast<int>(0.42 * check_idx.size());
    };

    cv::Vec3d n_fl(floor_plane[0], floor_plane[1], floor_plane[2]);
    double n_fl_norm = cv::norm(n_fl);
    bool valid_floor = has_floor && (n_fl_norm > 1e-6);
    if (valid_floor) {
        n_fl *= (1.0 / n_fl_norm);
        if (n_fl[1] > 0.0) n_fl = -n_fl;
    }

    // 3. Deterministic Floor-Aligned PCA Hypothesis (for FLOOR_STANDING frames like TABLE-DEMO)
    if (_is_floor_standing && valid_floor) {
        cv::Vec3d mu_obs_all(0.0, 0.0, 0.0);
        for (size_t i = 0; i < N; ++i) {
            mu_obs_all += cv::Vec3d(obs_pts[i]);
        }
        mu_obs_all *= (1.0 / static_cast<double>(N));

        // Project centered observations onto the floor plane before PCA
        cv::Matx33d cov = cv::Matx33d::zeros();
        for (size_t i = 0; i < N; ++i) {
            cv::Vec3d d = cv::Vec3d(obs_pts[i]) - mu_obs_all;
            d -= n_fl * d.dot(n_fl);
            cov += d * d.t();
        }
        cv::Matx33d U_cov, Vt_cov;
        cv::Vec3d W_cov;
        cv::SVD::compute(cov, W_cov, U_cov, Vt_cov);
        cv::Vec3d v_beam(Vt_cov(0, 0), Vt_cov(0, 1), Vt_cov(0, 2));
        if (v_beam[0] < 0.0) v_beam = -v_beam;

        cv::Vec3d cad_y = v_beam - v_beam.dot(n_fl) * n_fl;
        double y_norm = cv::norm(cad_y);
        if (y_norm > 1e-6) {
            cad_y *= (1.0 / y_norm);
            cv::Vec3d cad_z = n_fl;
            cv::Vec3d cad_x = cad_y.cross(cad_z);
            double x_norm = cv::norm(cad_x);
            if (x_norm > 1e-6) {
                cad_x *= (1.0 / x_norm);
                if (cad_x[2] > 0.0) { cad_x = -cad_x; cad_y = -cad_y; }

                cv::Matx33d R_fl(
                    cad_x[0], cad_y[0], cad_z[0],
                    cad_x[1], cad_y[1], cad_z[1],
                    cad_x[2], cad_y[2], cad_z[2]
                );
                const cv::Matx33d R_180(-1.0,  0.0, 0.0,
                                         0.0, -1.0, 0.0,
                                         0.0,  0.0, 1.0);
                const cv::Matx33d R_90( 0.0, -1.0, 0.0,
                                        1.0,  0.0, 0.0,
                                        0.0,  0.0, 1.0);
                const cv::Matx33d R_270( 0.0,  1.0, 0.0,
                                        -1.0,  0.0, 0.0,
                                         0.0,  0.0, 1.0);
                cv::Matx33d R_cands[4] = {R_fl, R_fl * R_180, R_fl * R_90, R_fl * R_270};

                const bool use_mesh_score = !_mesh_pts.empty();
                const size_t mesh_step = use_mesh_score ? std::max<size_t>(1, _mesh_pts.size() / 400) : 1;
                const double thresh_sq = inlier_thresh * inlier_thresh;

                for (int h = 0; h < 4; ++h) {
                    const auto& R_cand = R_cands[h];
                    if (!isCandidateRotationValid(R_cand)) continue;
                    cv::Vec3d t_cand = mu_obs_all - R_cand * cad_mid;
                    if (!isInTargetWorkspace(t_cand)) continue;

                    const cv::Matx33d Rt_cand = R_cand.t();
                    int inliers = 0;
                    double err_sum = 0.0;
                    std::vector<uint8_t> mask(N, 0);
                    for (size_t i = 0; i < N; ++i) {
                        if (use_mesh_score) {
                            const cv::Vec3d p_loc = Rt_cand * (cv::Vec3d(obs_pts[i]) - t_cand);
                            double min_dsq = 1e10;
                            for (size_t j = 0; j < _mesh_pts.size(); j += mesh_step) {
                                const cv::Vec3d d = p_loc - cv::Vec3d(_mesh_pts[j]);
                                const double dsq = d.dot(d);
                                if (dsq < min_dsq) min_dsq = dsq;
                            }
                            if (min_dsq <= thresh_sq) {
                                ++inliers;
                                err_sum += min_dsq;
                                mask[i] = 1;
                            }
                        } else {
                            cv::Vec3d pred = R_cand * cv::Vec3d(_cad_centroids[matches[i]]) + t_cand;
                            double e = cv::norm(pred - cv::Vec3d(obs_pts[i]));
                            if (e <= inlier_thresh) {
                                ++inliers;
                                err_sum += e * e;
                                mask[i] = 1;
                            }
                        }
                    }
                    if (inliers > best_inliers || (inliers == best_inliers && err_sum < best_err_sum)) {
                        best_inliers = inliers;
                        best_err_sum = err_sum;
                        best_mask = std::move(mask);
                        best_pose.valid = true;
                        best_pose.R = R_cand;
                        best_pose.t = t_cand;
                    }
                }
            }
        }
    }

    // 4. Confidence-Weighted 4-Point RANSAC Kabsch SVD
    std::mt19937 rng(1337);
    std::vector<float> sample_weights(nms_indices.size());
    for (size_t i = 0; i < nms_indices.size(); ++i) {
        const float sc = top_scores[nms_indices[i]];
        if (_is_floor_standing) {
            sample_weights[i] = std::pow(std::max(0.001f, sc), 1.5f);
        } else {
            sample_weights[i] = std::max(0.01f, std::exp((sc - max_best_sim) / 0.10f));
        }
    }
    std::discrete_distribution<int> dist(sample_weights.begin(), sample_weights.end());

    const int max_ransac_iters = _is_floor_standing ? 450 : 900;
    for (int iter = 0; iter < max_ransac_iters; ++iter) {
        int idxs[4];
        for (int s = 0; s < 4; ++s) {
            idxs[s] = nms_indices[dist(rng)];
        }
        if (idxs[0] == idxs[1] || idxs[1] == idxs[2] || idxs[0] == idxs[2] ||
            idxs[3] == idxs[0] || idxs[3] == idxs[1] || idxs[3] == idxs[2]) continue;

        cv::Vec3d p_mean(0.0, 0.0, 0.0), q_mean(0.0, 0.0, 0.0);
        for (int s = 0; s < 4; ++s) {
            p_mean += cv::Vec3d(obs_pts[idxs[s]]);
            q_mean += cv::Vec3d(_cad_centroids[matches[idxs[s]]]);
        }
        p_mean *= 0.25;
        q_mean *= 0.25;

        cv::Matx33d H = cv::Matx33d::zeros();
        for (int s = 0; s < 4; ++s) {
            cv::Vec3d dp = cv::Vec3d(obs_pts[idxs[s]]) - p_mean;
            cv::Vec3d dq = cv::Vec3d(_cad_centroids[matches[idxs[s]]]) - q_mean;
            H += dp * dq.t();
        }

        cv::Matx33d U, Vt;
        cv::Vec3d W;
        cv::SVD::compute(H, W, U, Vt);
        // Reject degenerate / collinear / single-centroid correspondence hypotheses
        if (W[0] < 1e-6 || W[1] < 1e-7) continue;
        cv::Matx33d R_cand = U * Vt;
        if (cv::determinant(R_cand) < 0.0) {
            cv::Matx33d fix = cv::Matx33d::eye();
            fix(2, 2) = -1.0;
            R_cand = U * fix * Vt;
        }

        if (!isCandidateRotationValid(R_cand)) continue;
        if (_is_floor_standing && valid_floor) {
            cv::Vec3d cad_z_cand(R_cand(0, 2), R_cand(1, 2), R_cand(2, 2));
            if (cad_z_cand.dot(n_fl) < 0.70) continue;
        }

        cv::Vec3d t_cand = p_mean - R_cand * q_mean;
        if (!isInTargetWorkspace(t_cand)) continue;
        if (!passesClusterContainment(R_cand, t_cand)) continue;

        int inliers = 0;
        double err_sum = 0.0;
        std::vector<uint8_t> mask(N, 0);
        for (size_t i = 0; i < N; ++i) {
            cv::Vec3d pred = R_cand * cv::Vec3d(_cad_centroids[matches[i]]) + t_cand;
            double e = cv::norm(pred - cv::Vec3d(obs_pts[i]));
            if (e <= inlier_thresh) {
                ++inliers;
                err_sum += e * e;
                mask[i] = 1;
            }
        }

        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_err_sum = err_sum;
            best_mask = std::move(mask);
            best_pose.valid = true;
            best_pose.R = R_cand;
            best_pose.t = t_cand;
            if (best_inliers >= static_cast<int>(0.55 * N)) {
                break;
            }
        }
    }

    // 5. Final Least-Squares Kabsch Refit on ALL RANSAC inliers
    if (best_pose.valid && best_inliers >= 6) {
        cv::Vec3d p_mean(0.0, 0.0, 0.0), q_mean(0.0, 0.0, 0.0);
        int cnt = 0;
        for (size_t i = 0; i < N; ++i) {
            if (!best_mask[i]) continue;
            p_mean += cv::Vec3d(obs_pts[i]);
            q_mean += cv::Vec3d(_cad_centroids[matches[i]]);
            ++cnt;
        }
        if (cnt >= 6) {
            p_mean *= (1.0 / cnt);
            q_mean *= (1.0 / cnt);
            cv::Matx33d H = cv::Matx33d::zeros();
            for (size_t i = 0; i < N; ++i) {
                if (!best_mask[i]) continue;
                cv::Vec3d dp = cv::Vec3d(obs_pts[i]) - p_mean;
                cv::Vec3d dq = cv::Vec3d(_cad_centroids[matches[i]]) - q_mean;
                H += dp * dq.t();
            }
            cv::Matx33d U, Vt;
            cv::Vec3d W;
            cv::SVD::compute(H, W, U, Vt);
            if (W[0] > 1e-6 && W[1] > 1e-7) {
                cv::Matx33d R_refit = U * Vt;
                if (cv::determinant(R_refit) < 0.0) {
                    cv::Matx33d fix = cv::Matx33d::eye();
                    fix(2, 2) = -1.0;
                    R_refit = U * fix * Vt;
                }
                cv::Vec3d t_refit = p_mean - R_refit * q_mean;
                if (isCandidateRotationValid(R_refit) && isInTargetWorkspace(t_refit) &&
                    passesClusterContainment(R_refit, t_refit)) {
                    best_pose.R = R_refit;
                    best_pose.t = t_refit;
                }
            }
        }
    }

    const int min_valid_inliers = std::max(8, static_cast<int>((_is_floor_standing ? 0.025 : 0.015) * N));
    if (best_pose.valid && best_inliers < min_valid_inliers) {
        best_pose.valid = false;
    }

    // Fallback geometric PCA seed for FREE_6DOF if descriptor RANSAC had heavy domain shift / noise
    if (!best_pose.valid && !_is_floor_standing && obs_pts.size() >= 24 && isInTargetWorkspace(obs_center)) {
        cv::Matx33d cov = cv::Matx33d::zeros();
        for (size_t i = 0; i < N; ++i) {
            const cv::Vec3d d = cv::Vec3d(obs_pts[i]) - obs_center;
            cov += d * d.t();
        }
        cv::Mat eig_vals, eig_vecs;
        cv::eigen(cv::Mat(cov), eig_vals, eig_vecs);
        cv::Vec3d v0(eig_vecs.at<double>(0, 0), eig_vecs.at<double>(0, 1), eig_vecs.at<double>(0, 2));
        cv::Vec3d v1(eig_vecs.at<double>(1, 0), eig_vecs.at<double>(1, 1), eig_vecs.at<double>(1, 2));
        v0 *= (1.0 / std::max(1e-9, cv::norm(v0)));
        v1 -= v0 * v0.dot(v1);
        v1 *= (1.0 / std::max(1e-9, cv::norm(v1)));
        cv::Vec3d v2 = v0.cross(v1);
        cv::Matx33d R_obs_pca(v0[0], v1[0], v2[0],
                              v0[1], v1[1], v2[1],
                              v0[2], v1[2], v2[2]);

        cv::Vec3d cad_ext = _cad_max - _cad_min;
        int ax[3] = {0, 1, 2};
        std::sort(ax, ax + 3, [&](int a, int b) { return cad_ext[a] > cad_ext[b]; });
        cv::Matx33d R_cad_pca = cv::Matx33d::zeros();
        R_cad_pca(ax[0], 0) = 1.0;
        R_cad_pca(ax[1], 1) = 1.0;
        cv::Vec3d c0(R_cad_pca(0, 0), R_cad_pca(1, 0), R_cad_pca(2, 0));
        cv::Vec3d c1(R_cad_pca(0, 1), R_cad_pca(1, 1), R_cad_pca(2, 1));
        cv::Vec3d c2 = c0.cross(c1);
        R_cad_pca(0, 2) = c2[0]; R_cad_pca(1, 2) = c2[1]; R_cad_pca(2, 2) = c2[2];

        cv::Matx33d R_seed = R_obs_pca * R_cad_pca.t();
        cv::Vec3d t_seed = obs_center - R_seed * cad_mid;
        if (isInTargetWorkspace(t_seed)) {
            best_pose.valid = true;
            best_pose.R = R_seed;
            best_pose.t = t_seed;
            best_pose.t_glass = t_seed;
            best_pose.inliers_count = 0;
            best_pose.inlier_ratio = 0.0;
            best_pose.rmse_m = 0.05;
            best_pose.confidence = 0.12f;
            best_pose.cluster_points_count = static_cast<int>(obs_pts.size());
            cv::Vec3d euler = rotationMatrixToEulerDeg(best_pose.R, false);
            best_pose.roll_deg = euler[0];
            best_pose.pitch_deg = euler[1];
            best_pose.yaw_deg = euler[2];
            best_pose.method = "pca_fallback_seed";
            return best_pose;
        }
    }

    if (!best_pose.valid) {
        return best_pose;
    }

    // 6. Floor upright gravity lock ONLY for FLOOR_STANDING archetypes (tables, racks, pallets)
    if (best_pose.valid && _is_floor_standing && valid_floor) {
        cv::Vec3d z_axis = n_fl;
        z_axis *= (1.0 / std::max(1e-6, cv::norm(z_axis)));
        cv::Vec3d y_axis(best_pose.R(0, 1), best_pose.R(1, 1), best_pose.R(2, 1));
        y_axis -= z_axis * z_axis.dot(y_axis);
        if (cv::norm(y_axis) > 1e-6) {
            y_axis *= (1.0 / cv::norm(y_axis));
            cv::Vec3d x_axis = y_axis.cross(z_axis);
            x_axis *= (1.0 / std::max(1e-6, cv::norm(x_axis)));

            cv::Vec3d cad_center(
                0.5 * (_cad_min[0] + _cad_max[0]),
                0.5 * (_cad_min[1] + _cad_max[1]),
                0.5 * (_cad_min[2] + _cad_max[2]));
            cv::Vec3d world_center = best_pose.R * cad_center + best_pose.t;

            best_pose.R = cv::Matx33d(
                x_axis[0], y_axis[0], z_axis[0],
                x_axis[1], y_axis[1], z_axis[1],
                x_axis[2], y_axis[2], z_axis[2]);
            best_pose.t = world_center - best_pose.R * cad_center;
        }
    }

    if (best_pose.valid) {
        best_pose.inliers_count = best_inliers;
        best_pose.inlier_ratio = static_cast<double>(best_inliers) / static_cast<double>(std::max<size_t>(1, N));
        best_pose.rmse_m = std::sqrt(best_err_sum / std::max(1, best_inliers));
        best_pose.confidence = std::clamp(static_cast<float>(best_inliers) / static_cast<float>(N), 0.10f, 0.99f);
        best_pose.cluster_points_count = static_cast<int>(obs_pts.size());
        best_pose.t_glass = best_pose.t;
        const bool fold_c2 = (_is_floor_standing && _symmetry_code >= 1);
        cv::Vec3d euler = rotationMatrixToEulerDeg(best_pose.R, fold_c2);
        best_pose.roll_deg = euler[0];
        best_pose.pitch_deg = euler[1];
        best_pose.yaw_deg = euler[2];
        best_pose.method = "universal_ransac_kabsch";
    }
    return best_pose;
}

void UniversalPoseEstimator::refineMultiScaleMeshICP(const std::vector<cv::Vec3f>& scene_pts,
                                                     ObjectPose& pose,
                                                     const CameraIntrinsics& intr) const
{
    if (!pose.valid || _mesh_pts.empty() || scene_pts.size() < 24) return;

    const double fx = (intr.fx > 100.0) ? intr.fx : 608.215;
    const double fy = (intr.fy > 100.0) ? intr.fy : 608.128;
    const double cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : 320.0);
    const double cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : 240.0);
    const int img_w = std::max(320, static_cast<int>(2.0 * cx + 0.5));
    const int img_h = std::max(240, static_cast<int>(2.0 * cy + 0.5));

    if (_is_floor_standing) {
        // FLOOR_STANDING branch preserved verbatim for TABLE-DEMO accuracy
        cv::Matx33d R_curr = pose.R;
        cv::Vec3d t_curr = pose.t;

        std::vector<cv::Vec3d> cad_vis_pts;
        std::vector<cv::Vec3d> cad_vis_norms;
        const size_t mesh_stride = (_mesh_pts.size() > 6000) ? 2 : 1;
        computeVisibleCadSubset(R_curr, t_curr, _mesh_pts, _mesh_normals,
                                fx, fy, cx, cy, img_w, img_h, mesh_stride,
                                cad_vis_pts, cad_vis_norms);
        if (cad_vis_pts.size() < 40) return;

        const double fg_thresh = std::clamp(_real_bounding_diag_m * 0.18, 0.04, 0.22);
        const double fg_thresh_sq = fg_thresh * fg_thresh;
        cv::Matx33d R_inv = R_curr.t();
        std::vector<cv::Vec3d> fg_pts;
        const size_t scene_step = std::max<size_t>(1, scene_pts.size() / 600);
        fg_pts.reserve(600);

        for (size_t i = 0; i < scene_pts.size(); i += scene_step) {
            cv::Vec3d p_scene(scene_pts[i]);
            cv::Vec3d p_cad = R_inv * (p_scene - t_curr);
            double min_dsq = 1e10;
            for (size_t j = 0; j < cad_vis_pts.size(); j += 4) {
                cv::Vec3d d = p_cad - cad_vis_pts[j];
                double dsq = d.dot(d);
                if (dsq < min_dsq) min_dsq = dsq;
            }
            if (min_dsq <= fg_thresh_sq) {
                fg_pts.push_back(p_scene);
            }
        }
        if (fg_pts.size() < 24) return;

        const double scale_k = std::clamp(_real_bounding_diag_m, 0.18, 1.40);
        const double dist_scales[4] = {0.032 * scale_k, 0.016 * scale_k, 0.007 * scale_k, 0.003 * scale_k};

        cv::Matx33d R_icp = R_inv;
        cv::Vec3d t_icp = -R_inv * t_curr;

        for (double max_corr : dist_scales) {
            max_corr = std::max(0.003, max_corr);
            double tukey_k = std::max(0.002, max_corr * 0.65);
            double max_corr_sq = max_corr * max_corr;
            double prev_residual = 1e10;

            for (int iter = 0; iter < 15; ++iter) {
                cv::Matx66d JtJ = cv::Matx66d::zeros();
                cv::Matx61d Jtb = cv::Matx61d::zeros();
                int n_corr = 0;
                double sum_res = 0.0;

                for (const auto& p_sc : fg_pts) {
                    cv::Vec3d p_trans = R_icp * p_sc + t_icp;
                    double best_dsq = 1e10;
                    int best_j = -1;
                    for (size_t j = 0; j < cad_vis_pts.size(); j += 2) {
                        cv::Vec3d d = p_trans - cad_vis_pts[j];
                        double dsq = d.dot(d);
                        if (dsq < best_dsq) {
                            best_dsq = dsq;
                            best_j = static_cast<int>(j);
                        }
                    }
                    if (best_dsq > max_corr_sq || best_j < 0) continue;

                    cv::Vec3d n = cad_vis_norms[best_j];
                    double n_norm = cv::norm(n);
                    if (n_norm < 1e-6) continue;
                    n *= (1.0 / n_norm);

                    cv::Vec3d res = p_trans - cad_vis_pts[best_j];
                    double r = n.dot(res);
                    double abs_r = std::abs(r);
                    double w = 0.0;
                    if (abs_r < tukey_k) {
                        double rk = r / tukey_k;
                        w = (1.0 - rk * rk);
                        w = w * w;
                    }
                    if (w < 1e-8) continue;

                    cv::Vec3d cross_np = p_trans.cross(n);
                    double J[6] = {cross_np[0], cross_np[1], cross_np[2], n[0], n[1], n[2]};
                    for (int a = 0; a < 6; ++a) {
                        for (int b = 0; b < 6; ++b) {
                            JtJ(a, b) += w * J[a] * J[b];
                        }
                        Jtb(a, 0) -= w * J[a] * r;
                    }
                    sum_res += w * r * r;
                    ++n_corr;
                }

                if (n_corr < 18) break;
                for (int d = 0; d < 6; ++d) JtJ(d, d) += 1e-5;

                cv::Matx61d delta;
                if (!cv::solve(JtJ, Jtb, delta, cv::DECOMP_CHOLESKY)) {
                    if (!cv::solve(JtJ, Jtb, delta, cv::DECOMP_SVD)) break;
                }

                cv::Vec3d d_w(delta(0, 0), delta(1, 0), delta(2, 0));
                cv::Vec3d d_t(delta(3, 0), delta(4, 0), delta(5, 0));
                double d_theta = cv::norm(d_w);

                cv::Matx33d d_R = cv::Matx33d::eye();
                if (d_theta > 1e-8) {
                    double clamp_th = std::min(d_theta, 0.25);
                    cv::Vec3d axis = d_w * (1.0 / d_theta);
                    cv::Matx33d K(
                         0.0,     -axis[2],  axis[1],
                         axis[2],  0.0,     -axis[0],
                        -axis[1],  axis[0],  0.0);
                    d_R = cv::Matx33d::eye() + std::sin(clamp_th) * K + (1.0 - std::cos(clamp_th)) * (K * K);
                }

                R_icp = d_R * R_icp;
                t_icp = d_R * t_icp + d_t;

                double rmse = std::sqrt(sum_res / n_corr);
                if (cv::norm(d_t) < 1e-5 && d_theta < 1e-5) break;
                if (std::abs(rmse - prev_residual) / (prev_residual + 1e-10) < 1e-5) break;
                prev_residual = rmse;
            }
        }

        cv::Matx33d R_final = R_icp.t();
        cv::Vec3d t_final = -R_final * t_icp;

        const double max_drift = std::clamp(_real_bounding_diag_m * 0.35, 0.08, 0.40);
        if (cv::norm(t_final - pose.t) > max_drift || !isInTargetWorkspace(t_final)) {
            return;
        }

        pose.R = R_final;
        pose.t = t_final;
        pose.t_glass = t_final;
        const bool fold_c2 = (_symmetry_code >= 1);
        cv::Vec3d euler = rotationMatrixToEulerDeg(pose.R, fold_c2);
        pose.roll_deg = euler[0];
        pose.pitch_deg = euler[1];
        pose.yaw_deg = euler[2];
        pose.method = "universal_mesh_icp";

        const cv::Matx33d R_check_inv = R_final.t();
        const double fit_thresh = std::clamp(_real_bounding_diag_m * 0.025, 0.006, 0.025);
        const double fit_thresh_sq = fit_thresh * fit_thresh;
        const size_t vis_step = (cad_vis_pts.size() > 1500) ? 2 : 1;
        int surf_inliers = 0;
        double inlier_dsq_sum = 0.0;
        for (const auto& p_sc : fg_pts) {
            const cv::Vec3d p_cad = R_check_inv * (p_sc - t_final);
            double min_dsq = 1e10;
            for (size_t j = 0; j < cad_vis_pts.size(); j += vis_step) {
                const cv::Vec3d d = p_cad - cad_vis_pts[j];
                const double dsq = d.dot(d);
                if (dsq < min_dsq) min_dsq = dsq;
            }
            if (min_dsq <= fit_thresh_sq) {
                ++surf_inliers;
                inlier_dsq_sum += min_dsq;
            }
        }
        if (surf_inliers >= 12) {
            const double icp_rmse = std::sqrt(inlier_dsq_sum / static_cast<double>(surf_inliers));
            const double icp_fitness = static_cast<double>(surf_inliers) / static_cast<double>(fg_pts.size());
            pose.rmse_m = icp_rmse;
            pose.inlier_ratio = icp_fitness;
            pose.inliers_count = std::max(pose.inliers_count, surf_inliers);
            pose.confidence = std::clamp(
                static_cast<float>(0.65 * icp_fitness + 0.35 * static_cast<double>(pose.confidence)),
                0.10f, 0.99f);
        }
        return;
    }

    // FREE_6DOF: Widened foreground recovery + Multi-Hypothesis Flip/PCA ICP + Bidirectional Consensus
    const double diag = std::max(0.05, _real_bounding_diag_m);
    const cv::Vec3d cad_mid = 0.5 * (_cad_min + _cad_max);

    std::vector<cv::Vec3d> scene_sub;
    const size_t scene_step = std::max<size_t>(1, scene_pts.size() / 600);
    scene_sub.reserve(scene_pts.size() / scene_step + 1);
    for (size_t i = 0; i < scene_pts.size(); i += scene_step) {
        scene_sub.emplace_back(scene_pts[i]);
    }
    if (scene_sub.size() < 24) return;

    // Extract foreground subset around coarse pose only if cluster is substantially oversized vs CAD;
    // never crop an object-sized cluster around a potentially noisy coarse RANSAC seed.
    std::vector<cv::Vec3d> fg_pts;
    bool cluster_is_object_sized = false;
    {
        std::vector<double> xs, ys, zs;
        xs.reserve(scene_sub.size());
        ys.reserve(scene_sub.size());
        zs.reserve(scene_sub.size());
        for (const auto& p : scene_sub) {
            xs.push_back(p[0]);
            ys.push_back(p[1]);
            zs.push_back(p[2]);
        }
        std::sort(xs.begin(), xs.end());
        std::sort(ys.begin(), ys.end());
        std::sort(zs.begin(), zs.end());
        const size_t nq = xs.size();
        const size_t lo = static_cast<size_t>(0.03 * nq);
        const size_t hi = std::min(nq - 1, static_cast<size_t>(0.97 * nq));
        const double q_diag = std::sqrt((xs[hi] - xs[lo]) * (xs[hi] - xs[lo]) +
                                        (ys[hi] - ys[lo]) * (ys[hi] - ys[lo]) +
                                        (zs[hi] - zs[lo]) * (zs[hi] - zs[lo]));
        cluster_is_object_sized = (q_diag <= 1.42 * diag);
    }

    if (cluster_is_object_sized) {
        fg_pts = scene_sub;
    } else {
        std::vector<cv::Vec3d> coarse_vis, coarse_norms;
        const size_t c_stride = std::max<size_t>(1, _mesh_pts.size() / 1200);
        computeVisibleCadSubset(pose.R, pose.t, _mesh_pts, _mesh_normals,
                                fx, fy, cx, cy, img_w, img_h, c_stride, coarse_vis, coarse_norms);
        if (coarse_vis.size() < 24) {
            for (size_t i = 0; i < _mesh_pts.size(); i += c_stride) coarse_vis.emplace_back(_mesh_pts[i]);
        }
        const double fg_thresh = std::clamp(diag * 0.16, 0.028, 0.060);
        const double fg_thresh_sq = fg_thresh * fg_thresh;
        const cv::Matx33d R_inv0 = pose.R.t();
        fg_pts.reserve(scene_sub.size());
        for (const auto& p_sc : scene_sub) {
            const cv::Vec3d p_cad = R_inv0 * (p_sc - pose.t);
            double min_dsq = 1e10;
            for (size_t j = 0; j < coarse_vis.size(); j += 2) {
                const cv::Vec3d d = p_cad - coarse_vis[j];
                const double dsq = d.dot(d);
                if (dsq < min_dsq) min_dsq = dsq;
            }
            if (min_dsq <= fg_thresh_sq) fg_pts.push_back(p_sc);
        }
        if (fg_pts.size() < 24 || fg_pts.size() < scene_sub.size() / 4) {
            fg_pts = scene_sub;
        }
    }

    cv::Vec3d c_fg(0.0, 0.0, 0.0);
    for (const auto& p : fg_pts) c_fg += p;
    c_fg *= (1.0 / static_cast<double>(fg_pts.size()));

    const std::vector<double> search_scales = {
        std::clamp(diag * 0.16, 0.028, 0.060),
        std::clamp(diag * 0.08, 0.014, 0.030),
        std::clamp(diag * 0.035, 0.007, 0.014)
    };
    const std::vector<double> polish_scales = {
        std::clamp(diag * 0.060, 0.012, 0.026),
        std::clamp(diag * 0.028, 0.006, 0.014),
        std::clamp(diag * 0.013, 0.0035, 0.007)
    };
    const size_t search_fg_stride = std::max<size_t>(1, fg_pts.size() / 220);
    const size_t search_mesh_stride = std::max<size_t>(1, _mesh_pts.size() / 520);
    const size_t polish_fg_stride = 1;
    const size_t polish_mesh_stride = (_mesh_pts.size() > 6000) ? 2 : 1;

    struct PoseSeed {
        cv::Matx33d R;
        cv::Vec3d t;
        double bonus;
    };
    std::vector<PoseSeed> seeds;

    // 1. Candidate 0: As-Is (run ICP first so flips pivot around the surface-settled axis!)
    cv::Matx33d R_cand0 = pose.R;
    cv::Vec3d t_cand0 = pose.t;
    const cv::Vec3d t_recentered0 = computeVisibleSurfaceOffset(pose.R, c_fg, cad_mid, _mesh_pts, _mesh_normals);
    if (cv::norm(t_recentered0 - pose.t) > 0.018) {
        // Test BOTH original pose.t and visible-centroid-aligned t_recentered0
        cv::Matx33d R_a, R_b;
        cv::Vec3d t_a, t_b;
        runPointToPlaneIcpGeneric(pose.R, pose.t, _mesh_pts, _mesh_normals, fg_pts,
                                  search_fg_stride, search_mesh_stride, search_scales, 8,
                                  fx, fy, cx, cy, img_w, img_h, R_a, t_a);
        runPointToPlaneIcpGeneric(pose.R, t_recentered0, _mesh_pts, _mesh_normals, fg_pts,
                                  search_fg_stride, search_mesh_stride, search_scales, 8,
                                  fx, fy, cx, cy, img_w, img_h, R_b, t_b);
        auto sa = evaluateBidirectionalConsensus(R_a, t_a, diag, _mesh_pts, _mesh_normals, fg_pts, fx, fy, cx, cy, img_w, img_h, true);
        auto sb = evaluateBidirectionalConsensus(R_b, t_b, diag, _mesh_pts, _mesh_normals, fg_pts, fx, fy, cx, cy, img_w, img_h, true);
        if (sb.score > sa.score) { R_cand0 = R_b; t_cand0 = t_b; }
        else { R_cand0 = R_a; t_cand0 = t_a; }
    } else {
        runPointToPlaneIcpGeneric(pose.R, pose.t, _mesh_pts, _mesh_normals, fg_pts,
                                  search_fg_stride, search_mesh_stride, search_scales, 8,
                                  fx, fy, cx, cy, img_w, img_h, R_cand0, t_cand0);
    }

    auto score_cand0 = evaluateBidirectionalConsensus(R_cand0, t_cand0, diag, _mesh_pts, _mesh_normals, fg_pts, fx, fy, cx, cy, img_w, img_h, true);

    double best_total = -1e9;
    cv::Matx33d best_R = pose.R;
    cv::Vec3d best_t = pose.t;
    BidirectionalConsensus best_cons;

    const double max_center_drift = std::max(0.065, 0.45 * diag);
    auto considerSolution = [&](const cv::Matx33d& R_sol, const cv::Vec3d& t_sol,
                                const BidirectionalConsensus& c_sol, double bonus) {
        if (!isInTargetWorkspace(t_sol)) return;
        const cv::Vec3d pred_mid = R_sol * cad_mid + t_sol;
        if (cv::norm(pred_mid - c_fg) > max_center_drift + 0.35 * diag) return;
        const double adj = c_sol.score + bonus;
        if (adj > best_total) {
            best_total = adj;
            best_R = R_sol;
            best_t = t_sol;
            best_cons = c_sol;
        }
    };
    considerSolution(R_cand0, t_cand0, score_cand0, 0.0);

    // 2. Transverse 180-deg Flips (Rx180/Ry180 compete purely on geometry to fix upside-down neck/cap;
    //    Rz180 carries a mild -0.008 hysteresis so Z-axis C2 objects keep AI front/back preference on ties)
    const cv::Matx33d Rx180(1.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0, -1.0);
    const cv::Matx33d Ry180(-1.0, 0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0, -1.0);
    const cv::Matx33d Rz180(-1.0, 0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  1.0);
    seeds.push_back(PoseSeed{R_cand0 * Rx180, computeVisibleSurfaceOffset(R_cand0 * Rx180, c_fg, cad_mid, _mesh_pts, _mesh_normals), 0.0});
    seeds.push_back(PoseSeed{R_cand0 * Ry180, computeVisibleSurfaceOffset(R_cand0 * Ry180, c_fg, cad_mid, _mesh_pts, _mesh_normals), 0.0});
    seeds.push_back(PoseSeed{R_cand0 * Rz180, computeVisibleSurfaceOffset(R_cand0 * Rz180, c_fg, cad_mid, _mesh_pts, _mesh_normals), -0.008});

    // 3. Pure 3D Free-Space SO(3) Multi-Hypothesis Search (Zero Table/Floor Dependency!)
    // Hoạt động cho vật thể ở BẤT KỲ tư thế nào trong không gian 3D (cầm trên tay, treo, nghiêng, hoặc trên bàn)
    // bằng cách kiểm tra toàn bộ 24 phép quay nhóm khối SO(3) (6 mặt x 4 góc 90 độ) + 36 góc xoay quanh 3 trục của R_cand0.
    if (score_cand0.fwd_fit < 0.92 || score_cand0.bwd_kernel < 0.85 || pose.method == "pca_fallback_seed") {
        cv::Matx33d cov = cv::Matx33d::zeros();
        for (const auto& p : fg_pts) {
            const cv::Vec3d d = p - c_fg;
            cov += d * d.t();
        }
        cv::Mat eig_vals, eig_vecs;
        cv::eigen(cv::Mat(cov), eig_vals, eig_vecs);
        cv::Vec3d v0(eig_vecs.at<double>(0, 0), eig_vecs.at<double>(0, 1), eig_vecs.at<double>(0, 2));
        cv::Vec3d v1(eig_vecs.at<double>(1, 0), eig_vecs.at<double>(1, 1), eig_vecs.at<double>(1, 2));
        v0 *= (1.0 / std::max(1e-9, cv::norm(v0)));
        v1 -= v0 * v0.dot(v1);
        v1 *= (1.0 / std::max(1e-9, cv::norm(v1)));
        cv::Vec3d v2 = v0.cross(v1);

        const cv::Matx33d R_obs_pca(
            v0[0], v1[0], v2[0],
            v0[1], v1[1], v2[1],
            v0[2], v1[2], v2[2]);

        // Tạo 24 ma trận hoán vị có dấu thuộc nhóm quay SO(3) (det = +1)
        // Bao phủ mọi trường hợp hoán vị trục (v0 <-> v1 <-> v2) khi vật thể xoay làm đổi tỷ lệ khung nhìn 2.5D.
        std::vector<cv::Matx33d> so3_cube_group;
        so3_cube_group.reserve(24);
        const int perms[6][3] = {
            {0, 1, 2}, {0, 2, 1}, {1, 0, 2},
            {1, 2, 0}, {2, 0, 1}, {2, 1, 0}
        };
        for (int p = 0; p < 6; ++p) {
            for (int sx : {-1, 1}) {
                for (int sy : {-1, 1}) {
                    for (int sz : {-1, 1}) {
                        cv::Matx33d P = cv::Matx33d::zeros();
                        P(0, perms[p][0]) = sx;
                        P(1, perms[p][1]) = sy;
                        P(2, perms[p][2]) = sz;
                        if (cv::determinant(P) > 0.5) {
                            so3_cube_group.push_back(P);
                        }
                    }
                }
            }
        }

        struct RankedSeed { double pre; cv::Matx33d R; cv::Vec3d t; };
        std::vector<RankedSeed> cand_pool;
        cand_pool.reserve(60);

        // (A) Quét toàn bộ 24 hướng SO(3) trên hệ trục PCA của đám mây điểm vật thể
        for (const auto& G : so3_cube_group) {
            cv::Matx33d R_try = R_obs_pca * G;
            cv::Vec3d t_try = computeVisibleSurfaceOffset(R_try, c_fg, cad_mid, _mesh_pts, _mesh_normals);
            auto sc = evaluateBidirectionalConsensus(
                R_try, t_try, diag, _mesh_pts, _mesh_normals,
                fg_pts, fx, fy, cx, cy, img_w, img_h, true);
            cand_pool.push_back(RankedSeed{sc.score, R_try, t_try});
        }

        // (B) Quét thêm các góc xoay mịn (mỗi 30 độ) quanh cả 3 trục X, Y, Z của R_cand0 (từ AI ONNX)
        for (int deg = 30; deg < 360; deg += 30) {
            const double rad = static_cast<double>(deg) * kPi / 180.0;
            const double cs = std::cos(rad), sn = std::sin(rad);
            const cv::Matx33d Rz(cs, -sn, 0.0,  sn, cs, 0.0,  0.0, 0.0, 1.0);
            const cv::Matx33d Ry(cs, 0.0,  sn, 0.0, 1.0, 0.0, -sn, 0.0,  cs);
            const cv::Matx33d Rx(1.0, 0.0, 0.0, 0.0, cs, -sn, 0.0,  sn,  cs);

            for (const auto& R_step : {Rz, Ry, Rx}) {
                cv::Matx33d R_try = R_cand0 * R_step;
                cv::Vec3d t_try = computeVisibleSurfaceOffset(R_try, c_fg, cad_mid, _mesh_pts, _mesh_normals);
                auto sc = evaluateBidirectionalConsensus(
                    R_try, t_try, diag, _mesh_pts, _mesh_normals,
                    fg_pts, fx, fy, cx, cy, img_w, img_h, true);
                cand_pool.push_back(RankedSeed{sc.score, R_try, t_try});
            }
        }

        std::sort(cand_pool.begin(), cand_pool.end(),
                  [](const RankedSeed& a, const RankedSeed& b) { return a.pre > b.pre; });

        const size_t take_top = std::min<size_t>(6, cand_pool.size());
        for (size_t k = 0; k < take_top; ++k) {
            seeds.push_back(PoseSeed{cand_pool[k].R, cand_pool[k].t, 0.0});
        }
    }

    for (const auto& sd : seeds) {
        cv::Matx33d R_out;
        cv::Vec3d t_out;
        runPointToPlaneIcpGeneric(sd.R, sd.t, _mesh_pts, _mesh_normals, fg_pts,
                                  search_fg_stride, search_mesh_stride, search_scales, 8,
                                  fx, fy, cx, cy, img_w, img_h, R_out, t_out);
        auto sc = evaluateBidirectionalConsensus(R_out, t_out, diag, _mesh_pts, _mesh_normals, fg_pts, fx, fy, cx, cy, img_w, img_h, true);
        considerSolution(R_out, t_out, sc, sd.bonus);
    }

    if (best_total <= -1e8) return;

    // 4. Final dense polish on winning hypothesis
    cv::Matx33d R_final = best_R;
    cv::Vec3d t_final = best_t;
    BidirectionalConsensus final_cons = best_cons;
    {
        cv::Matx33d R_pol;
        cv::Vec3d t_pol;
        runPointToPlaneIcpGeneric(best_R, best_t, _mesh_pts, _mesh_normals, fg_pts,
                                  polish_fg_stride, polish_mesh_stride, polish_scales, 15,
                                  fx, fy, cx, cy, img_w, img_h, R_pol, t_pol);
        if (isInTargetWorkspace(t_pol) && cv::norm((R_pol * cad_mid + t_pol) - c_fg) <= max_center_drift + 0.35 * diag) {
            auto pol_cons = evaluateBidirectionalConsensus(R_pol, t_pol, diag, _mesh_pts, _mesh_normals, fg_pts, fx, fy, cx, cy, img_w, img_h);
            if (pol_cons.score >= best_cons.score - 0.005) {
                R_final = R_pol;
                t_final = t_pol;
                final_cons = pol_cons;
            }
        }
    }

    const double max_drift = std::clamp(diag * 0.48, 0.085, 0.160);
    if ((cv::norm(t_final - pose.t) > max_drift && cv::norm((R_final * cad_mid + t_final) - c_fg) > max_center_drift) ||
        !isInTargetWorkspace(t_final)) {
        return;
    }

    // Chuẩn hóa góc tự xoay (Spin Stabilization) cho vật thể đối xứng trụ (như chai Chin-su):
    // Giữ nguyên hướng trục dọc của chai, khóa mặt bên luôn hướng ổn định về camera để hộp 3D không quay tít.
    {
        const cv::Vec3d cad_span = _cad_max - _cad_min;
        int long_ax = 2;
        if (cad_span[0] >= cad_span[1] && cad_span[0] >= cad_span[2]) long_ax = 0;
        else if (cad_span[1] >= cad_span[0] && cad_span[1] >= cad_span[2]) long_ax = 1;

        const int s_ax1 = (long_ax + 1) % 3;
        const int s_ax2 = (long_ax + 2) % 3;
        const double ratio_cross = std::max(cad_span[s_ax1], cad_span[s_ax2]) /
                                   std::max(1e-4, std::min(cad_span[s_ax1], cad_span[s_ax2]));

        if (!_is_floor_standing && ratio_cross < 1.35) {
            cv::Vec3d axis_long(R_final(0, long_ax), R_final(1, long_ax), R_final(2, long_ax));
            axis_long *= (1.0 / std::max(1e-9, cv::norm(axis_long)));

            cv::Vec3d cam_back(0.0, 0.0, -1.0);
            if (std::abs(axis_long.dot(cam_back)) > 0.92) cam_back = cv::Vec3d(1.0, 0.0, 0.0);

            cv::Vec3d axis_s1 = cam_back - axis_long * axis_long.dot(cam_back);
            axis_s1 *= (1.0 / std::max(1e-9, cv::norm(axis_s1)));
            cv::Vec3d axis_s2 = axis_long.cross(axis_s1);
            axis_s2 *= (1.0 / std::max(1e-9, cv::norm(axis_s2)));

            cv::Matx33d R_stab = cv::Matx33d::zeros();
            for (int r = 0; r < 3; ++r) {
                R_stab(r, long_ax) = axis_long[r];
                R_stab(r, s_ax1) = axis_s1[r];
                R_stab(r, s_ax2) = axis_s2[r];
            }
            if (cv::determinant(R_stab) < 0.0) {
                for (int r = 0; r < 3; ++r) R_stab(r, s_ax2) = -R_stab(r, s_ax2);
            }
            const cv::Vec3d world_mid = R_final * cad_mid + t_final;
            R_final = R_stab;
            t_final = world_mid - R_final * cad_mid;
        }
    }

    pose.R = R_final;
    pose.t = t_final;
    pose.t_glass = t_final;
    cv::Vec3d euler = rotationMatrixToEulerDeg(pose.R, false);
    pose.roll_deg = euler[0];
    pose.pitch_deg = euler[1];
    pose.yaw_deg = euler[2];
    pose.method = "universal_mesh_icp";

    const int surf_inliers = static_cast<int>(std::round(final_cons.fwd_fit * static_cast<double>(fg_pts.size())));
    pose.rmse_m = final_cons.rmse_m;
    pose.inlier_ratio = final_cons.fwd_fit;
    pose.inliers_count = std::max(pose.inliers_count, surf_inliers);
    const double bidir_conf = std::clamp(final_cons.score + 0.05, 0.05, 0.99);
    pose.confidence = std::clamp(
        static_cast<float>(0.88 * bidir_conf + 0.12 * static_cast<double>(pose.confidence)),
        0.05f, 0.99f);
}

void UniversalPoseEstimator::refinePoseUniversalCrease2D3D(ObjectPose& pose,
                                                           const cv::Mat& rgb,
                                                           const CameraIntrinsics& intr) const
{
    if (!pose.valid || rgb.empty() || _cad_feature_edges.empty()) return;
    // Skip OBB-box fallback edges (<=16 edges) on smooth/curved FREE_6DOF parts
    // so RGB Canny clutter does not pull the 3D pose away from the fitted depth surface.
    if (!_is_floor_standing && _cad_feature_edges.size() <= 16) return;

    cv::Mat gray, edges, dt;
    if (rgb.channels() == 3) cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    else gray = rgb;

    cv::Canny(gray, edges, 40, 120);
    cv::Mat inv_edges = 255 - edges;
    cv::distanceTransform(inv_edges, dt, cv::DIST_L2, 3);

    const double fx = (intr.fx > 100.0) ? intr.fx : 608.215;
    const double fy = (intr.fy > 100.0) ? intr.fy : 608.128;
    const double cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : (0.5 * rgb.cols));
    const double cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : (0.5 * rgb.rows));

    auto evalChamfer = [&](const cv::Matx33d& R_try, const cv::Vec3d& t_try) -> double {
        double sum = 0.0;
        int valid = 0;
        for (const auto& e : _cad_feature_edges) {
            for (float a : {0.2f, 0.5f, 0.8f}) {
                cv::Vec3f p_local = e.p0 * (1.0f - a) + e.p1 * a;
                cv::Vec3d p_cam = R_try * cv::Vec3d(p_local) + t_try;
                if (p_cam[2] <= 0.15) continue;
                int u = static_cast<int>(std::round(fx * p_cam[0] / p_cam[2] + cx));
                int v = static_cast<int>(std::round(fy * p_cam[1] / p_cam[2] + cy));
                if (u >= 0 && u < dt.cols && v >= 0 && v < dt.rows) {
                    sum += std::min(25.0f, dt.at<float>(v, u));
                    ++valid;
                }
            }
        }
        return (valid > 6) ? (sum / valid) : 1e6;
    };

    const double scale_k = std::clamp(_real_bounding_diag_m, 0.15, 1.15);
    double best_cost = evalChamfer(pose.R, pose.t);
    for (double dz_u : {-0.014, -0.006, 0.0, 0.006, 0.014}) {
        for (double dx_u : {-0.010, 0.0, 0.010}) {
            cv::Vec3d t_try = pose.t + cv::Vec3d(dx_u * scale_k, 0.0, dz_u * scale_k);
            double c = evalChamfer(pose.R, t_try);
            if (c < best_cost) {
                best_cost = c;
                pose.t = t_try;
                pose.t_glass = t_try;
            }
        }
    }
}

ObjectPose UniversalPoseEstimator::estimateSingleCluster(const std::vector<cv::Vec3f>& sampled_pts,
                                                         const std::vector<cv::Vec3f>& dense_pts,
                                                         const cv::Vec4d& floor_plane,
                                                         bool floor_detected,
                                                         double floor_pitch_deg,
                                                         const cv::Mat& rgb,
                                                         const CameraIntrinsics& intr,
                                                         int img_w,
                                                         int img_h,
                                                         double* out_ai_ms) const
{
    ObjectPose pose;
    pose.model_name = _model_name;
    if (sampled_pts.size() < 16) return pose;

    // Chuẩn hóa đám mây điểm quan sát về tâm (0,0,0) và chia cho _real_bounding_diag_m (FREE_6DOF)
    // khớp 100% với: pts_cam_norm = (pts_cam - centroid) / self.cad_diag trong dataset_hierarchy.py,
    // đồng thời giữ nguyên tọa độ camera-space (sampled_pts) cho Kabsch / ICP.
    std::vector<cv::Vec3f> ai_input_pts = sampled_pts;
    if (!ai_input_pts.empty()) {
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (const auto& p : ai_input_pts) {
            sx += p[0];
            sy += p[1];
            sz += p[2];
        }
        const float inv_n = 1.0f / static_cast<float>(ai_input_pts.size());
        const float cx_pt = static_cast<float>(sx) * inv_n;
        const float cy_pt = static_cast<float>(sy) * inv_n;
        const float cz_pt = static_cast<float>(sz) * inv_n;
        if (_is_floor_standing) {
            for (auto& p : ai_input_pts) {
                p[0] -= cx_pt;
                p[1] -= cy_pt;
                p[2] -= cz_pt;
            }
        } else {
            const float inv_diag = 1.0f / std::max(0.05f, static_cast<float>(_real_bounding_diag_m));
            for (auto& p : ai_input_pts) {
                p[0] = (p[0] - cx_pt) * inv_diag;
                p[1] = (p[1] - cy_pt) * inv_diag;
                p[2] = (p[2] - cz_pt) * inv_diag;
            }
        }
    }

    auto t_ai_0 = std::chrono::steady_clock::now();
    std::vector<float> obs_desc;
    if (!runInference(ai_input_pts, obs_desc)) {
        if (out_ai_ms) *out_ai_ms = 0.0;
        return pose;
    }
    auto t_ai_1 = std::chrono::steady_clock::now();
    if (out_ai_ms) {
        *out_ai_ms = std::chrono::duration<double, std::milli>(t_ai_1 - t_ai_0).count();
    }

    pose = solveRansacKabsch(sampled_pts, obs_desc, floor_plane, floor_detected);
    if (!pose.valid) return pose;

    const std::vector<cv::Vec3f>& icp_cloud = dense_pts.empty() ? sampled_pts : dense_pts;
    refineMultiScaleMeshICP(icp_cloud, pose, intr);

    // Kiểm tra phân cực Đầu nắp vs Đáy chai bằng biên dạng cạnh 2D trên ảnh RGB
    // khi vật thể có dạng đối xứng trụ (tỷ lệ thiết diện < 1.35) và phần cổ/nắp có thể bị mất Depth.
    if (!_is_floor_standing && !rgb.empty() && !_mesh_pts.empty()) {
        const cv::Vec3d cad_span = _cad_max - _cad_min;
        int long_ax = 2;
        if (cad_span[0] >= cad_span[1] && cad_span[0] >= cad_span[2]) long_ax = 0;
        else if (cad_span[1] >= cad_span[0] && cad_span[1] >= cad_span[2]) long_ax = 1;
        const int s_ax1 = (long_ax + 1) % 3;
        const int s_ax2 = (long_ax + 2) % 3;
        const double ratio_cross = std::max(cad_span[s_ax1], cad_span[s_ax2]) /
                                   std::max(1e-4, std::min(cad_span[s_ax1], cad_span[s_ax2]));
        if (ratio_cross < 1.35) {
            cv::Mat gray, edges, inv_edges, dt;
            if (rgb.channels() == 3) cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
            else gray = rgb;
            cv::Canny(gray, edges, 40, 120);
            cv::bitwise_not(edges, inv_edges);
            cv::distanceTransform(inv_edges, dt, cv::DIST_L2, 3);

            const double fx = (intr.fx > 100.0) ? intr.fx : 600.0;
            const double fy = (intr.fy > 100.0) ? intr.fy : fx;
            const double cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : 0.5 * img_w);
            const double cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : 0.5 * img_h);

            cv::Matx33d R_flip_local = cv::Matx33d::eye();
            R_flip_local(long_ax, long_ax) = -1.0;
            R_flip_local(s_ax2, s_ax2) = -1.0;
            const cv::Vec3d cad_mid = 0.5 * (_cad_min + _cad_max);
            const cv::Vec3d world_mid = pose.R * cad_mid + pose.t;
            const cv::Matx33d R_flip = pose.R * R_flip_local;
            const cv::Vec3d t_flip = world_mid - R_flip * cad_mid;

            auto eval2DEdgeDist = [&](const cv::Matx33d& R_test, const cv::Vec3d& t_test) -> double {
                std::vector<cv::Vec3d> vis_pts, vis_norms;
                const size_t vis_stride = std::max<size_t>(1, _mesh_pts.size() / 400);
                computeVisibleCadSubset(R_test, t_test, _mesh_pts, _mesh_normals,
                                        fx, fy, cx, cy, img_w, img_h, vis_stride, vis_pts, vis_norms);
                if (vis_pts.empty()) return 1e6;
                double sum_dt = 0.0;
                int cnt = 0;
                for (const auto& vp : vis_pts) {
                    const cv::Vec3d p_cam = R_test * vp + t_test;
                    if (p_cam[2] <= 0.05) continue;
                    const int u = static_cast<int>(std::round(p_cam[0] * fx / p_cam[2] + cx));
                    const int v = static_cast<int>(std::round(p_cam[1] * fy / p_cam[2] + cy));
                    if (u >= 0 && u < dt.cols && v >= 0 && v < dt.rows) {
                        sum_dt += std::min(12.0, static_cast<double>(dt.at<float>(v, u)));
                        ++cnt;
                    }
                }
                return (cnt >= 10) ? (sum_dt / cnt) : 1e6;
            };

            const double err_cur = eval2DEdgeDist(pose.R, pose.t);
            const double err_flip = eval2DEdgeDist(R_flip, t_flip);
            if (err_flip + 0.35 < err_cur) {
                pose.R = R_flip;
                pose.t = t_flip;
                pose.t_glass = t_flip;
                const cv::Vec3d euler = rotationMatrixToEulerDeg(pose.R, false);
                pose.roll_deg = euler[0];
                pose.pitch_deg = euler[1];
                pose.yaw_deg = euler[2];
            }
        }
    }

    if (_is_floor_standing || icp_cloud.size() < 250) {
        refinePoseUniversalCrease2D3D(pose, rgb, intr);
    }
    pose.bbox_2d = projectCadBoundingBox(pose.R, pose.t, intr, img_w, img_h, &pose.corners_2d);

    pose.cluster_points_count = static_cast<int>(icp_cloud.size());
    pose.valid_depth_pts = static_cast<int>(icp_cloud.size());
    pose.floor_locked = (_is_floor_standing && floor_detected);
    pose.floor_pitch_deg = floor_pitch_deg;
    pose.sparse_depth_fallback = (pose.valid_depth_pts < 250);
    pose.segmented_points = icp_cloud;
    return pose;
}

bool UniversalPoseEstimator::rescorePoseUnclipped(ObjectPose& pose,
                                                  const std::vector<cv::Vec3f>& full_fg_pts,
                                                  const cv::Mat& depth_z16,
                                                  const CameraIntrinsics& intr,
                                                  int img_w,
                                                  int img_h,
                                                  bool strict_clutter_gate) const
{
    if (_is_floor_standing || !pose.valid || full_fg_pts.empty() || _mesh_pts.empty()) return true;

    const double diag = std::max(0.05, _real_bounding_diag_m);
    const double pad = std::clamp(0.068 * diag, 0.008, 0.018);
    const cv::Matx33d R_inv = pose.R.t();

    // Collect unclipped points inside the tight rotated CAD OBB so adjacent objects do not corrupt fit/RMSE
    std::vector<cv::Vec3d> local_raw;
    local_raw.reserve(1024);
    for (const auto& p : full_fg_pts) {
        const cv::Vec3d pc = R_inv * (cv::Vec3d(p) - pose.t);
        if (pc[0] >= _cad_min[0] - pad && pc[0] <= _cad_max[0] + pad &&
            pc[1] >= _cad_min[1] - pad && pc[1] <= _cad_max[1] + pad &&
            pc[2] >= _cad_min[2] - pad && pc[2] <= _cad_max[2] + pad) {
            local_raw.emplace_back(p);
        }
    }
    if (local_raw.size() < 8) {
        pose.confidence = 0.05f;
        pose.inliers_count = 0;
        return false;
    }

    std::vector<cv::Vec3d> unclipped;
    unclipped.reserve(std::min<size_t>(local_raw.size(), 600));
    const size_t step = std::max<size_t>(1, local_raw.size() / 600);
    for (size_t i = 0; i < local_raw.size(); i += step) {
        unclipped.push_back(local_raw[i]);
    }

    const double fx = (intr.fx > 100.0) ? intr.fx : 608.215;
    const double fy = (intr.fy > 100.0) ? intr.fy : 608.128;
    const double cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : 0.5 * img_w);
    const double cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : 0.5 * img_h);

    auto cons = evaluateBidirectionalConsensus(pose.R, pose.t, diag,
                                             _mesh_pts, _mesh_normals, unclipped,
                                             fx, fy, cx, cy, img_w, img_h);

    if (cons.score > -100.0) {
        pose.confidence = std::clamp(static_cast<float>(cons.score + 0.05), 0.05f, 0.99f);
        pose.rmse_m = cons.rmse_m;
        pose.inlier_ratio = cons.fwd_fit;
        pose.inliers_count = std::max(pose.inliers_count, static_cast<int>(std::round(cons.fwd_fit * local_raw.size())));
    } else if (strict_clutter_gate) {
        pose.confidence = 0.05f;
        return false;
    }

    if (local_raw.size() < 36) {
        double empty_behind_frac = 0.0, ring_jump_frac = 1.0, wide_out_frac = 1.0, wide_convex_ratio = 1.0;
        int ring_tot = 0;
        computeSilhouetteAndEmptyBehind(pose.R, pose.t, _mesh_pts, depth_z16,
                                        fx, fy, cx, cy, diag,
                                        empty_behind_frac, ring_jump_frac,
                                        wide_out_frac, wide_convex_ratio, ring_tot);
        if (empty_behind_frac > 0.20) {
            pose.confidence = 0.05f;
            return false;
        }
        return !strict_clutter_gate;
    }

    const bool is_cluttered = (full_fg_pts.size() > std::max<size_t>(400, local_raw.size() * 3 / 2));
    if (!is_cluttered) {
        return true;
    }

    // 0. In-frame visibility guard in cluttered scenes (rejects partially off-screen wall/armrest crops)
    if (strict_clutter_gate && img_w > 0 && img_h > 0) {
        int tot_proj = 0;
        int in_frame = 0;
        const size_t p_stride = std::max<size_t>(1, _mesh_pts.size() / 400);
        for (size_t i = 0; i < _mesh_pts.size(); i += p_stride) {
            const cv::Vec3d pc = pose.R * cv::Vec3d(_mesh_pts[i]) + pose.t;
            if (pc[2] <= 0.10) continue;
            const double u = fx * pc[0] / pc[2] + cx;
            const double v = fy * pc[1] / pc[2] + cy;
            ++tot_proj;
            if (u >= 6.0 && u < img_w - 6.0 && v >= 6.0 && v < img_h - 6.0) {
                ++in_frame;
            }
        }
        if (tot_proj >= 20 && static_cast<double>(in_frame) < 0.90 * static_cast<double>(tot_proj)) {
            pose.confidence = 0.05f;
            return false;
        }
    }

    // 1. Physical Non-Planarity Check vs predicted visible CAD PCA spread (rejects 1D strips & flat plane patches)
    const size_t vis_stride = std::max<size_t>(1, _mesh_pts.size() / 800);
    std::vector<cv::Vec3d> vis_pts, vis_norms;
    computeVisibleCadSubset(pose.R, pose.t, _mesh_pts, _mesh_normals,
                            fx, fy, cx, cy, img_w, img_h, vis_stride, vis_pts, vis_norms);
    if (vis_pts.size() >= 20 && unclipped.size() >= 20) {
        const cv::Vec3d vis_sig = computePcaSigmas(vis_pts);
        const cv::Vec3d obs_sig = computePcaSigmas(unclipped);
        if (vis_sig[2] >= 0.0040 && (obs_sig[2] < 0.28 * vis_sig[2] || obs_sig[1] < 0.40 * vis_sig[1])) {
            pose.confidence = std::max(0.05f, pose.confidence * 0.35f);
            if (strict_clutter_gate) return false;
        }
    }

    // 2. Z-Buffer Free-Space & Outer Boundary Convexity/Discontinuity Verification
    double empty_behind_frac = 0.0;
    double ring_jump_frac = 1.0;
    double wide_out_frac = 1.0;
    double wide_convex_ratio = 1.0;
    int ring_tot = 0;
    computeSilhouetteAndEmptyBehind(pose.R, pose.t, _mesh_pts, depth_z16,
                                    fx, fy, cx, cy, diag,
                                    empty_behind_frac, ring_jump_frac,
                                    wide_out_frac, wide_convex_ratio, ring_tot);
    const bool concave_crease = (ring_tot >= 40 && (wide_out_frac < 0.14 || wide_convex_ratio < 0.24));
    const int min_ring_tot = strict_clutter_gate ? 24 : 40;
    const double min_ring_jump = strict_clutter_gate ? 0.45 : 0.36;
    if (empty_behind_frac > 0.16 || (ring_tot >= min_ring_tot && ring_jump_frac < min_ring_jump) || concave_crease) {
        pose.confidence = std::max(0.05f, pose.confidence * 0.40f);
        if (strict_clutter_gate) return false;
    }

    const double sil_adj = 0.08 * std::clamp((ring_jump_frac - 0.40) / 0.35, -0.5, 1.0)
                         - 0.18 * empty_behind_frac;
    pose.confidence = std::clamp(static_cast<float>(static_cast<double>(pose.confidence) + sil_adj), 0.05f, 0.99f);

    if (strict_clutter_gate) {
        const double max_clutter_rmse = std::clamp(0.082 * diag, 0.012, 0.028);
        if (pose.rmse_m > max_clutter_rmse || pose.inlier_ratio < 0.48) {
            pose.confidence = std::min(pose.confidence, 0.35f);
            return false;
        }
    }
    return true;
}

std::vector<ObjectPose> UniversalPoseEstimator::proposeDepthHoleSeeds(
    const cv::Mat& rgb,
    const cv::Mat& depth_z16,
    const CameraIntrinsics& intr,
    const cv::Vec4d& floor_plane,
    bool has_floor,
    const cv::Rect& roi_hint) const
{
    std::vector<ObjectPose> seeds;
    if (rgb.empty() || depth_z16.empty() || _is_floor_standing || _mesh_pts.empty()) return seeds;

    const double fx = (intr.fx > 100.0) ? intr.fx : 608.215;
    const double fy = (intr.fy > 100.0) ? intr.fy : 608.128;
    const double cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : 0.5 * rgb.cols);
    const double cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : 0.5 * rgb.rows);
    const double max_ws_z = std::clamp(_real_bounding_diag_m * 3.8 + 0.15, 0.95, 2.50);

    const cv::Vec3d span = _cad_max - _cad_min;
    int long_ax = 2;
    if (span[0] >= span[1] && span[0] >= span[2]) long_ax = 0;
    else if (span[1] >= span[0] && span[1] >= span[2]) long_ax = 1;
    const int ax1 = (long_ax + 1) % 3;
    const int ax2 = (long_ax + 2) % 3;
    const double cad_long = std::max(0.02, span[long_ax]);
    const double cad_short = std::max(0.02, std::max(span[ax1], span[ax2]));
    const double stand_aspect = cad_long / cad_short;
    const double lie_aspect = cad_short / cad_long;
    const cv::Vec3d cad_mid = 0.5 * (_cad_min + _cad_max);

    // Luôn quét toàn khung nếu roi_hint đang bị kéo lệch sát mép trái (< 0.20 * W)
    cv::Rect search_rect(12, 12, std::max(1, rgb.cols - 24), std::max(1, rgb.rows - 24));
    if (roi_hint.width >= 32 && roi_hint.height >= 32 && roi_hint.x >= static_cast<int>(0.20 * rgb.cols)) {
        search_rect &= roi_hint;
    }
    if (search_rect.width < 32 || search_rect.height < 32) return seeds;

    // 1. Zero-HSV: Tạo cả mặt nạ hố khuyết Depth nguyên vẹn lẫn mặt nạ tách theo biên Canny
    // (để không bị gộp vào màn hình tối phía sau, và tự khôi phục chiều cao thật từ z_rim dưới đáy khi nhãn cắt đôi).
    cv::Mat gray, canny, edge_dil;
    if (rgb.channels() == 3) cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    else gray = rgb;
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0.8);
    cv::Canny(gray, canny, 40, 120);
    cv::dilate(canny, edge_dil, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    cv::Mat hole_raw = cv::Mat::zeros(depth_z16.size(), CV_8UC1);
    cv::Mat hole_split = cv::Mat::zeros(depth_z16.size(), CV_8UC1);
    for (int v = search_rect.y; v < search_rect.br().y; ++v) {
        const uint16_t* d_row = depth_z16.ptr<uint16_t>(v);
        const uint8_t* e_row = edge_dil.ptr<uint8_t>(v);
        uint8_t* r_row = hole_raw.ptr<uint8_t>(v);
        uint8_t* s_row = hole_split.ptr<uint8_t>(v);
        for (int u = search_rect.x; u < search_rect.br().x; ++u) {
            if (d_row[u] == 0) {
                r_row[u] = 255;
                if (e_row[u] == 0) s_row[u] = 255;
            }
        }
    }
    cv::morphologyEx(hole_raw, hole_raw, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    cv::morphologyEx(hole_raw, hole_raw, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 19)));
    cv::morphologyEx(hole_split, hole_split, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    cv::morphologyEx(hole_split, hole_split, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 11)));

    std::vector<std::vector<cv::Point>> contours, contours_split;
    cv::findContours(hole_raw, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::findContours(hole_split, contours_split, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    contours.insert(contours.end(), contours_split.begin(), contours_split.end());

    struct ScoredSeed {
        double score{0.0};
        ObjectPose pose;
    };
    std::vector<ScoredSeed> cands;

    cv::Vec3d up_axis(0.0, -1.0, 0.0);
    if (has_floor && std::abs(floor_plane[1]) > 0.55) {
        up_axis = cv::Vec3d(floor_plane[0], floor_plane[1], floor_plane[2]);
        up_axis *= (1.0 / std::max(1e-9, cv::norm(up_axis)));
        if (up_axis[1] > 0.0) up_axis = -up_axis;
    }

    for (const auto& c : contours) {
        const cv::Rect box = cv::boundingRect(c);
        if (std::max(box.width, box.height) < 48 || std::min(box.width, box.height) < 18) continue;

        const double aspect = static_cast<double>(box.height) / std::max(1, box.width);
        const bool match_stand = (aspect >= 0.50 * stand_aspect && aspect <= 2.20 * stand_aspect);
        const bool match_lie = (aspect >= 0.45 * lie_aspect && aspect <= 2.00 * lie_aspect);
        if (!match_stand && !match_lie) continue;

        // Tỉ lệ depth == 0 bên trong hộp >= 0.30
        int zero_cnt = 0, tot_cnt = 0;
        const int v_in0 = box.y + box.height / 5;
        const int v_in1 = box.y + 4 * box.height / 5;
        const int u_in0 = box.x + box.width / 5;
        const int u_in1 = box.x + 4 * box.width / 5;
        for (int v = v_in0; v < v_in1; v += 2) {
            const uint16_t* d_row = depth_z16.ptr<uint16_t>(v);
            for (int u = u_in0; u < u_in1; u += 2) {
                ++tot_cnt;
                if (d_row[u] == 0) ++zero_cnt;
            }
        }
        const double zero_ratio = (tot_cnt > 0) ? (static_cast<double>(zero_cnt) / tot_cnt) : 0.0;
        if (zero_ratio < 0.30) continue;

        // Kiểm tra biên Canny bao quanh 2 phía đối xứng của hố
        int left_hits = 0, right_hits = 0, top_hits = 0, bot_hits = 0;
        for (int v = box.y; v < box.br().y; v += 3) {
            bool hl = false, hr = false;
            for (int du = -4; du <= 4; ++du) {
                const int ul = std::clamp(box.x + du, 0, canny.cols - 1);
                const int ur = std::clamp(box.br().x + du, 0, canny.cols - 1);
                if (canny.at<uint8_t>(v, ul) > 0) hl = true;
                if (canny.at<uint8_t>(v, ur) > 0) hr = true;
            }
            if (hl) ++left_hits;
            if (hr) ++right_hits;
        }
        for (int u = box.x; u < box.br().x; u += 3) {
            bool ht = false, hb = false;
            for (int dv = -4; dv <= 4; ++dv) {
                const int vt = std::clamp(box.y + dv, 0, canny.rows - 1);
                const int vb = std::clamp(box.br().y + dv, 0, canny.rows - 1);
                if (canny.at<uint8_t>(vt, u) > 0) ht = true;
                if (canny.at<uint8_t>(vb, u) > 0) hb = true;
            }
            if (ht) ++top_hits;
            if (hb) ++bot_hits;
        }
        const bool stand_edges = (left_hits >= 4 && right_hits >= 4);
        const bool lie_edges = (top_hits >= 4 && bot_hits >= 4);
        if (!stand_edges && !lie_edges) continue;

        const bool use_stand = match_stand && (stand_edges || !match_lie) && (box.height >= box.width);
        const double z_pinhole = use_stand
            ? (fy * cad_long) / static_cast<double>(box.height)
            : (fx * cad_long) / static_cast<double>(box.width);

        // Quét viền nắp/đế trong toàn dải tiền cảnh [0.15, max_ws_z] (không khóa theo z_pinhole bị cắt ngắn bởi nhãn)
        std::vector<double> rim_zs;
        const int band_u0 = std::max(0, box.x - 6);
        const int band_u1 = std::min(depth_z16.cols - 1, box.br().x + 6);
        const int bands[2][2] = {
            {std::max(0, box.y - 12), std::min(depth_z16.rows - 1, box.y + box.height / 4)},
            {std::max(0, box.y + 3 * box.height / 4), std::min(depth_z16.rows - 1, box.br().y + 18)}
        };
        for (int b = 0; b < 2; ++b) {
            for (int v = bands[b][0]; v <= bands[b][1]; v += 2) {
                const uint16_t* d_row = depth_z16.ptr<uint16_t>(v);
                for (int u = band_u0; u <= band_u1; u += 2) {
                    const double zm = static_cast<double>(d_row[u]) * 0.001;
                    if (zm >= 0.15 && zm <= max_ws_z) {
                        rim_zs.push_back(zm);
                    }
                }
            }
        }
        double z_seed = z_pinhole;
        double v_c = box.y + 0.5 * box.height;
        if (rim_zs.size() >= 6) {
            std::sort(rim_zs.begin(), rim_zs.end());
            const double z_rim = rim_zs[rim_zs.size() / 5];
            if (use_stand && z_rim < 0.72 * z_pinhole) {
                // Mảnh hố bị nhãn cắt ngang: lấy z_rim tại chân đế và dựng lại chiều cao Pinhole đầy đủ từ đáy hộp
                z_seed = z_rim;
                const double full_h = (fy * cad_long) / std::max(0.15, z_seed);
                v_c = static_cast<double>(box.br().y) - 0.5 * full_h;
            } else if (std::abs(z_rim - z_pinhole) <= 0.30 * z_pinhole) {
                z_seed = 0.50 * z_pinhole + 0.50 * z_rim;
            } else {
                z_seed = z_rim;
            }
        }
        if (z_seed < 0.15 || z_seed > max_ws_z) continue;

        cv::Vec3d axis_long = up_axis;
        if (!use_stand) {
            axis_long = cv::Vec3d(1.0, 0.0, 0.0);
        }
        const cv::Vec3d ref_x = (std::abs(axis_long[0]) < 0.85) ? cv::Vec3d(1.0, 0.0, 0.0) : cv::Vec3d(0.0, -1.0, 0.0);
        cv::Vec3d s1 = ref_x - axis_long * axis_long.dot(ref_x);
        s1 *= (1.0 / std::max(1e-9, cv::norm(s1)));
        cv::Vec3d s2 = axis_long.cross(s1);
        s2 *= (1.0 / std::max(1e-9, cv::norm(s2)));

        cv::Matx33d R_seed = cv::Matx33d::zeros();
        for (int r = 0; r < 3; ++r) {
            R_seed(r, long_ax) = -axis_long[r];
            R_seed(r, ax1) = s1[r];
            R_seed(r, ax2) = s2[r];
        }
        if (cv::determinant(R_seed) < 0.0) {
            for (int r = 0; r < 3; ++r) R_seed(r, ax2) = -R_seed(r, ax2);
        }

        const double u_c = box.x + 0.5 * box.width;
        const cv::Vec3d center_3d((u_c - cx) * z_seed / fx, (v_c - cy) * z_seed / fy, z_seed);
        const cv::Vec3d t_seed = center_3d - R_seed * cad_mid;
        if (!isInTargetWorkspace(t_seed)) continue;

        ObjectPose sp;
        sp.valid = true;
        sp.model_name = _model_name;
        sp.R = R_seed;
        sp.t = t_seed;
        sp.t_glass = t_seed;
        sp.method = "depth_hole_seed";
        const cv::Vec3d euler = rotationMatrixToEulerDeg(R_seed, false);
        sp.roll_deg = euler[0];
        sp.pitch_deg = euler[1];
        sp.yaw_deg = euler[2];
        sp.bbox_2d = projectCadBoundingBox(R_seed, t_seed, intr, rgb.cols, rgb.rows, &sp.corners_2d);

        const double target_asp = use_stand ? stand_aspect : lie_aspect;
        const double asp_sim = std::exp(-std::abs(aspect - target_asp) * 0.45);
        const double edge_score = use_stand ? std::min(left_hits, right_hits) : std::min(top_hits, bot_hits);
        const double sc = static_cast<double>(box.area()) * asp_sim * zero_ratio * std::min(12.0, edge_score);
        cands.push_back({sc, std::move(sp)});
    }

    std::sort(cands.begin(), cands.end(), [](const ScoredSeed& a, const ScoredSeed& b) {
        return a.score > b.score;
    });
    const size_t keep = std::min<size_t>(cands.size(), 3);
    seeds.reserve(keep);
    for (size_t i = 0; i < keep; ++i) {
        seeds.push_back(std::move(cands[i].pose));
    }
    return seeds;
}

bool UniversalPoseEstimator::trackFromWarmStart(const cv::Mat& rgb,
                                                const cv::Mat& depth_z16,
                                                const CameraIntrinsics& intr,
                                                ObjectPose& out_pose)
{
    if (!_track_active || !_track_pose.valid || depth_z16.empty()) return false;

    const int w = depth_z16.cols;
    const int h = depth_z16.rows;
    const double fx = (intr.fx > 100.0) ? intr.fx : 608.215;
    const double fy = (intr.fy > 100.0) ? intr.fy : 608.128;
    const double cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : 0.5 * w);
    const double cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : 0.5 * h);
    const double scale = (intr.depth_scale > 0.0) ? intr.depth_scale : 0.001;
    const double diag = std::max(0.05, _real_bounding_diag_m);
    const double max_ws_z = _is_floor_standing ? 2.95 : std::clamp(diag * 3.8 + 0.15, 0.95, 2.50);

    const cv::Rect2f& pb = _track_pose.bbox_2d;
    const int pad_x = std::max(48, static_cast<int>(std::round(0.35f * pb.width)));
    const int pad_y = std::max(48, static_cast<int>(std::round(0.35f * pb.height)));
    const int x0 = (pb.width > 8.0f) ? std::max(0, static_cast<int>(std::floor(pb.x)) - pad_x) : 0;
    const int y0 = (pb.height > 8.0f) ? std::max(0, static_cast<int>(std::floor(pb.y)) - pad_y) : 0;
    const int x1 = (pb.width > 8.0f) ? std::min(w, static_cast<int>(std::ceil(pb.x + pb.width)) + pad_x) : w;
    const int y1 = (pb.height > 8.0f) ? std::min(h, static_cast<int>(std::ceil(pb.y + pb.height)) + pad_y) : h;

    const cv::Matx33d R_inv = _track_pose.R.t();
    const double obb_pad = std::clamp(0.16 * diag, 0.022, 0.042);
    const cv::Vec3d obb_min = _cad_min - cv::Vec3d(obb_pad, obb_pad, obb_pad);
    const cv::Vec3d obb_max = _cad_max + cv::Vec3d(obb_pad, obb_pad, obb_pad);
    const double floor_norm = cv::norm(cv::Vec3d(_track_floor_plane[0], _track_floor_plane[1], _track_floor_plane[2]));
    const bool use_floor_filter = _track_has_floor && (floor_norm > 0.5);

    std::vector<cv::Vec3f> local_fg;
    local_fg.reserve(2048);
    const long long area = static_cast<long long>(x1 - x0) * static_cast<long long>(y1 - y0);
    const int stride = (area < 90000) ? 1 : 2;
    for (int v = y0; v < y1; v += stride) {
        const uint16_t* row = depth_z16.ptr<uint16_t>(v);
        for (int u = x0; u < x1; u += stride) {
            const uint16_t dr = row[u];
            if (dr == 0) continue;
            const double z = static_cast<double>(dr) * scale;
            if (z < 0.15 || z > max_ws_z) continue;
            const double x = (static_cast<double>(u) - cx) * z / fx;
            const double y = (static_cast<double>(v) - cy) * z / fy;
            if (use_floor_filter) {
                const double signed_d = _track_floor_plane[0] * x + _track_floor_plane[1] * y +
                                        _track_floor_plane[2] * z + _track_floor_plane[3];
                if (std::abs(signed_d) <= 0.018) continue;
                if (_track_floor_plane[1] < -0.5 && signed_d > 0.018) continue;
            }
            const cv::Vec3d pc = R_inv * (cv::Vec3d(x, y, z) - _track_pose.t);
            if (pc[0] >= obb_min[0] && pc[0] <= obb_max[0] &&
                pc[1] >= obb_min[1] && pc[1] <= obb_max[1] &&
                pc[2] >= obb_min[2] && pc[2] <= obb_max[2]) {
                local_fg.emplace_back(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
            }
        }
    }

    if (local_fg.size() < 16) return false;
    _last_valid_depth_pts = static_cast<int>(local_fg.size());
    _last_sparse_depth_fallback = (local_fg.size() < 250);
    _last_segmented_points = local_fg;

    ObjectPose cand = _track_pose;
    cand.valid = true;
    cand.method = "warm_start_icp";
    cand.valid_depth_pts = _last_valid_depth_pts;
    cand.sparse_depth_fallback = _last_sparse_depth_fallback;

    // Chặng A: 3D Point-to-Plane ICP cục bộ khởi tạo từ (_track_pose.R, _track_pose.t)
    const size_t icp_step = std::max<size_t>(1, local_fg.size() / 600);
    std::vector<cv::Vec3d> fg_icp;
    fg_icp.reserve(local_fg.size() / icp_step + 1);
    for (size_t i = 0; i < local_fg.size(); i += icp_step) {
        fg_icp.emplace_back(local_fg[i][0], local_fg[i][1], local_fg[i][2]);
    }
    if (fg_icp.size() >= 20) {
        cv::Matx33d R_icp = cand.R;
        cv::Vec3d t_icp = cand.t;
        const std::vector<double> tr_scales = {
            std::clamp(0.050 * diag, 0.010, 0.040),
            std::clamp(0.025 * diag, 0.006, 0.020)
        };
        runPointToPlaneIcpGeneric(cand.R, cand.t, _mesh_pts, _mesh_normals, fg_icp,
                                  1, std::max<size_t>(1, _mesh_pts.size() / 1600),
                                  tr_scales, 10, fx, fy, cx, cy, w, h, R_icp, t_icp);
        if (isInTargetWorkspace(t_icp) && cv::norm(t_icp - _track_pose.t) <= std::clamp(0.20 * diag, 0.035, 0.080)) {
            cand.R = R_icp;
            cand.t = t_icp;
        }
    }

    // Chặng B: Tinh chỉnh Yaw & dịch chuyển mịn quanh Warm-Start theo tỷ lệ pixel/CAD
    const cv::Vec3d cad_mid = 0.5 * (_cad_min + _cad_max);
    double sil_cost = 0.0;
    if (!_is_floor_standing && !rgb.empty()) {
        cv::Mat proj_mask = cv::Mat::zeros(rgb.rows, rgb.cols, CV_8UC1);
        const size_t m_step = std::max<size_t>(1, _mesh_pts.size() / 1200);
        for (size_t i = 0; i < _mesh_pts.size(); i += m_step) {
            const cv::Vec3d pc = cand.R * cv::Vec3d(_mesh_pts[i]) + cand.t;
            if (pc[2] <= 0.10) continue;
            const int u = static_cast<int>(std::round(fx * pc[0] / pc[2] + cx));
            const int v = static_cast<int>(std::round(fy * pc[1] / pc[2] + cy));
            if (u >= 0 && u < rgb.cols && v >= 0 && v < rgb.rows) {
                proj_mask.at<uint8_t>(v, u) = 255;
            }
        }
        cv::dilate(proj_mask, proj_mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(19, 19)));
        const RgbEdgeMaps dt_canny = buildRgbCannyDistanceTransform(rgb, proj_mask);
        if (!dt_canny.empty()) {
            sil_cost = refineTranslationOnRgbSilhouette(
                cand.R, cand.t, cand.R * cad_mid + cand.t, _cad_min, _cad_max,
                _track_floor_plane, _track_has_floor,
                _mesh_pts, _mesh_normals, dt_canny, depth_z16, fx, fy, cx, cy, max_ws_z, true);
        }
    }

    if (!isInTargetWorkspace(cand.t)) return false;

    // Kiểm tra liên tục vị trí & góc trục dài so với _track_pose
    const double jump_m = cv::norm(cand.t - _track_pose.t);
    const cv::Vec3d z_cur(cand.R(0, 2), cand.R(1, 2), cand.R(2, 2));
    const cv::Vec3d z_prv(_track_pose.R(0, 2), _track_pose.R(1, 2), _track_pose.R(2, 2));
    const double tilt_deg = std::acos(std::clamp(std::abs(z_cur.dot(z_prv)), 0.0, 1.0)) * 180.0 / kPi;
    if (jump_m > std::max(0.15, 0.75 * diag) || tilt_deg > 30.0) return false;

    cand.t_glass = cand.t;
    const cv::Vec3d euler = rotationMatrixToEulerDeg(cand.R, _is_floor_standing && _symmetry_code >= 1);
    cand.roll_deg = euler[0];
    cand.pitch_deg = euler[1];
    cand.yaw_deg = euler[2];
    cand.bbox_2d = projectCadBoundingBox(cand.R, cand.t, intr, w, h, &cand.corners_2d);

    // Chặng C: Hậu kiểm độc lập
    const bool strict_ok = rescorePoseUnclipped(cand, local_fg, depth_z16, intr, w, h, true);
    if (!strict_ok && !_is_floor_standing && sil_cost <= 8.0) {
        const double z_hole = computeProjectedZeroDepthFraction(
            cand.R, cand.t, _cad_min, _cad_max, depth_z16, fx, fy, cx, cy);
        if (z_hole >= 0.25 || cand.inliers_count >= 25) {
            rescorePoseUnclipped(cand, local_fg, depth_z16, intr, w, h, false);
            if (sil_cost <= 6.2 && z_hole >= 0.25) {
                cand.rgb_sil_strong = true;
                cand.confidence = std::clamp(cand.confidence + 0.18f, 0.05f, 0.99f);
            }
        }
    }
    if (!DetectionAcceptancePolicy::isAccepted(cand, _real_bounding_diag_m, _is_floor_standing)) {
        return false;
    }

    out_pose = std::move(cand);
    return true;
}

ObjectPose UniversalPoseEstimator::estimate(const cv::Mat& rgb,
                                            const cv::Mat& depth_z16,
                                            const CameraIntrinsics& intr,
                                            const cv::Rect& roi_hint)
{
    auto t_start = std::chrono::steady_clock::now();
    ObjectPose pose;
    pose.model_name = _model_name;
    _last_valid_depth_pts = 0;
    _last_floor_locked = false;
    _last_floor_pitch_deg = 0.0;
    _last_sparse_depth_fallback = false;
    _last_ai_latency_ms = 0.0;
    _last_solve_latency_ms = 0.0;
    _last_total_latency_ms = 0.0;
    _last_segmented_points.clear();

    if (!_initialized || depth_z16.empty()) return pose;

    // Fast-path Warm-Start Tracking khi đã khóa 2 frame liên tiếp và không bật ROI thủ công
    bool has_tracked_fallback = false;
    ObjectPose tracked_fallback;
    if (_temporal_tracking_enabled && !_multi_instance_enabled && roi_hint.empty() && _track_active) {
        ObjectPose tracked;
        if (trackFromWarmStart(rgb, depth_z16, intr, tracked)) {
            if (tracked.confidence >= 0.62f) {
                _track_pose = tracked;
                _track_miss_streak = 0;
                auto t_end = std::chrono::steady_clock::now();
                _last_ai_latency_ms = 0.0;
                _last_total_latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
                _last_solve_latency_ms = _last_total_latency_ms;
                return tracked;
            }
            // Adaptive Re-check: chỉ chạy Global nếu chất lượng track suy giảm (< 0.62)
            has_tracked_fallback = true;
            tracked_fallback = tracked;
            pose = std::move(tracked);
        }
    }

    cv::Mat fg_mask_2d;
    if (_mask_matcher.isReady() && !rgb.empty() && roi_hint.empty() && !_is_floor_standing) {
        auto mask_cands = _mask_matcher.match(rgb, 2);
        if (!mask_cands.empty() && mask_cands[0].score >= 0.45f) {
            fg_mask_2d = mask_cands[0].mask_u8;
        }
    }

    SegmentationResult seg_res = _scene_segmenter.segment(depth_z16, intr, roi_hint, _num_obs_points, fg_mask_2d);
    if (!fg_mask_2d.empty() && !seg_res.valid) {
        fg_mask_2d.release();
        seg_res = _scene_segmenter.segment(depth_z16, intr, roi_hint, _num_obs_points);
    }
    _last_valid_depth_pts = static_cast<int>(seg_res.dense_object_points.size());
    _last_floor_locked = (_is_floor_standing && seg_res.floor_detected);
    _last_floor_pitch_deg = seg_res.floor_pitch_deg;
    _last_sparse_depth_fallback = (_last_valid_depth_pts >= 20 && _last_valid_depth_pts < 250);
    _last_segmented_points = seg_res.dense_object_points;

    if (!seg_res.valid) {
        auto t_end = std::chrono::steady_clock::now();
        _last_total_latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        _last_solve_latency_ms = _last_total_latency_ms;
        return pose;
    }

    double ai_ms = 0.0;
    const RgbEdgeMaps dt_canny = (!_is_floor_standing && !rgb.empty()) ? buildRgbCannyDistanceTransform(rgb, fg_mask_2d) : RgbEdgeMaps{};
    const double sil_fx = (intr.fx > 100.0) ? intr.fx : 600.0;
    const double sil_fy = (intr.fy > 100.0) ? intr.fy : sil_fx;
    const double sil_cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : 0.5 * depth_z16.cols);
    const double sil_cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : 0.5 * depth_z16.rows);

    if (!_is_floor_standing && seg_res.multi_clusters.size() > 1) {
        double best_strict_rank = -1e9;
        double best_fb_rank = -1e9;
        ObjectPose best_fb_pose;
        const size_t n_try = std::min<size_t>(seg_res.multi_clusters.size(), 12);
        for (size_t i = 0; i < n_try; ++i) {
            const auto& cl = seg_res.multi_clusters[i];
            if (cl.points.size() < 20) continue;
            const std::vector<cv::Vec3f>& dense_cl = cl.dense_points.empty() ? cl.points : cl.dense_points;
            double c_ai = 0.0;
            ObjectPose cand = estimateSingleCluster(cl.points, dense_cl, seg_res.floor_plane, false,
                                                    seg_res.floor_pitch_deg, rgb, intr, depth_z16.cols, depth_z16.rows, &c_ai);
            ai_ms += c_ai;
            if (!cand.valid) continue;

            double sil_cost = 0.0;
            double z_hole_frac = 0.0;
            if (!dt_canny.empty()) {
                const double max_ws_z = std::clamp(_real_bounding_diag_m * 3.8 + 0.15, 0.95, 2.50);
                sil_cost = refineTranslationOnRgbSilhouette(
                    cand.R, cand.t, cv::Vec3d(cl.centroid), _cad_min, _cad_max, seg_res.floor_plane, seg_res.floor_detected,
                    _mesh_pts, _mesh_normals, dt_canny, depth_z16, sil_fx, sil_fy, sil_cx, sil_cy, max_ws_z);
                cand.t_glass = cand.t;
                const cv::Vec3d euler = rotationMatrixToEulerDeg(cand.R, false);
                cand.roll_deg = euler[0];
                cand.pitch_deg = euler[1];
                cand.yaw_deg = euler[2];
                cand.bbox_2d = projectCadBoundingBox(cand.R, cand.t, intr, depth_z16.cols, depth_z16.rows, &cand.corners_2d);
                if (!isInTargetWorkspace(cand.t)) continue;
                z_hole_frac = computeProjectedZeroDepthFraction(
                    cand.R, cand.t, _cad_min, _cad_max, depth_z16, sil_fx, sil_fy, sil_cx, sil_cy);
            }

            ObjectPose strict_cand = cand;
            const bool strong_rgb_sil = (!dt_canny.empty() && sil_cost <= 5.5 && z_hole_frac >= 0.35 &&
                                         cand.inliers_count >= 12 && cand.rmse_m <= 0.016);
            const bool acceptable_rgb_sil = (dt_canny.empty() || sil_cost <= 9.5);
            const bool passed_unclipped = acceptable_rgb_sil &&
                rescorePoseUnclipped(strict_cand, seg_res.foreground_points, depth_z16, intr, depth_z16.cols, depth_z16.rows, true);
            if (passed_unclipped || strong_rgb_sil) {
                if (passed_unclipped && !dt_canny.empty() && sil_cost <= 6.2) {
                    strict_cand.confidence = std::clamp(strict_cand.confidence + 0.12f, 0.05f, 0.99f);
                } else if (!passed_unclipped && strong_rgb_sil) {
                    strict_cand = cand;
                    strict_cand.rgb_sil_strong = true;
                    if (strict_cand.method == "pca_fallback_seed") {
                        strict_cand.method = "rgb_sil_refined";
                    }
                    strict_cand.confidence = std::clamp(strict_cand.confidence + 0.18f, 0.05f, 0.99f);
                }
                double rank = static_cast<double>(strict_cand.confidence)
                            - 2.0 * std::max(0.0, strict_cand.rmse_m - 0.015)
                            - 0.04 * sil_cost
                            - 0.22 * strict_cand.t[2];
                if (rank > best_strict_rank) {
                    best_strict_rank = rank;
                    pose = std::move(strict_cand);
                }
            } else if (dt_canny.empty() || sil_cost <= 6.5) {
                rescorePoseUnclipped(cand, seg_res.foreground_points, depth_z16, intr, depth_z16.cols, depth_z16.rows, false);
                double rank = static_cast<double>(cand.confidence) - 2.0 * std::max(0.0, cand.rmse_m - 0.015) - 0.06 * sil_cost;
                if (rank > best_fb_rank) {
                    best_fb_rank = rank;
                    best_fb_pose = std::move(cand);
                }
            }
        }
        if (!pose.valid && best_fb_pose.valid &&
            DetectionAcceptancePolicy::isAccepted(best_fb_pose, _real_bounding_diag_m, _is_floor_standing)) {
            pose = std::move(best_fb_pose);
        }
        if (pose.valid) {
            _last_valid_depth_pts = pose.valid_depth_pts;
            _last_sparse_depth_fallback = pose.sparse_depth_fallback;
            _last_segmented_points = pose.segmented_points;
        }
    } else {
        pose = estimateSingleCluster(seg_res.object_points,
                                     seg_res.dense_object_points,
                                     seg_res.floor_plane,
                                     seg_res.floor_detected,
                                     seg_res.floor_pitch_deg,
                                     rgb, intr, depth_z16.cols, depth_z16.rows, &ai_ms);
        if (pose.valid && !_is_floor_standing &&
            seg_res.foreground_points.size() > static_cast<size_t>(pose.valid_depth_pts * 1.18)) {
            rescorePoseUnclipped(pose, seg_res.foreground_points, depth_z16, intr, depth_z16.cols, depth_z16.rows, false);
        }
    }

    if (pose.valid && !DetectionAcceptancePolicy::isAccepted(pose, _real_bounding_diag_m, _is_floor_standing)) {
        pose.valid = false;
    }

    if (!pose.valid && !_is_floor_standing && !dt_canny.empty()) {
        const double max_ws_z = std::clamp(_real_bounding_diag_m * 3.8 + 0.15, 0.95, 2.50);
        auto hole_seeds = proposeDepthHoleSeeds(rgb, depth_z16, intr, seg_res.floor_plane, seg_res.floor_detected, roi_hint);
        double best_hole_rank = -1e9;
        for (auto& seed : hole_seeds) {
            // 1. Thử giải cụm 3D cục bộ (Nắp + Đế) trong khung ROI mở rộng quanh hố khuyết Depth
            if (seed.bbox_2d.width > 12.0f && seed.bbox_2d.height > 12.0f) {
                const int pad_x = std::max(24, static_cast<int>(std::round(0.30f * seed.bbox_2d.width)));
                const int pad_y = std::max(28, static_cast<int>(std::round(0.28f * seed.bbox_2d.height)));
                const int rx0 = std::max(0, static_cast<int>(std::floor(seed.bbox_2d.x)) - pad_x);
                const int ry0 = std::max(0, static_cast<int>(std::floor(seed.bbox_2d.y)) - pad_y);
                const int rx1 = std::min(depth_z16.cols, static_cast<int>(std::ceil(seed.bbox_2d.x + seed.bbox_2d.width)) + pad_x);
                const int ry1 = std::min(depth_z16.rows, static_cast<int>(std::ceil(seed.bbox_2d.y + seed.bbox_2d.height)) + pad_y);
                const cv::Rect hole_roi(rx0, ry0, std::max(1, rx1 - rx0), std::max(1, ry1 - ry0));

                SegmentationResult hole_seg = _scene_segmenter.segment(depth_z16, intr, hole_roi, _num_obs_points);
                if (hole_seg.valid && hole_seg.object_points.size() >= 20) {
                    double c_ai = 0.0;
                    ObjectPose roi_cand = estimateSingleCluster(
                        hole_seg.object_points, hole_seg.dense_object_points,
                        seg_res.floor_plane, false, seg_res.floor_pitch_deg,
                        rgb, intr, depth_z16.cols, depth_z16.rows, &c_ai);
                    ai_ms += c_ai;
                    if (roi_cand.valid) {
                        const double roi_sil = refineTranslationOnRgbSilhouette(
                            roi_cand.R, roi_cand.t, seed.t, _cad_min, _cad_max, seg_res.floor_plane, seg_res.floor_detected,
                            _mesh_pts, _mesh_normals, dt_canny, depth_z16, sil_fx, sil_fy, sil_cx, sil_cy, max_ws_z);
                        if (isInTargetWorkspace(roi_cand.t) && roi_sil <= 6.2) {
                            roi_cand.t_glass = roi_cand.t;
                            const cv::Vec3d eu = rotationMatrixToEulerDeg(roi_cand.R, false);
                            roi_cand.roll_deg = eu[0];
                            roi_cand.pitch_deg = eu[1];
                            roi_cand.yaw_deg = eu[2];
                            roi_cand.bbox_2d = projectCadBoundingBox(roi_cand.R, roi_cand.t, intr, depth_z16.cols, depth_z16.rows, &roi_cand.corners_2d);
                            rescorePoseUnclipped(roi_cand, seg_res.foreground_points, depth_z16, intr, depth_z16.cols, depth_z16.rows, false);
                            if (roi_sil <= 4.8) {
                                roi_cand.rgb_sil_strong = true;
                                roi_cand.confidence = std::clamp(roi_cand.confidence + 0.20f, 0.05f, 0.99f);
                            }
                            if (DetectionAcceptancePolicy::isAccepted(roi_cand, _real_bounding_diag_m, _is_floor_standing)) {
                                const double rnk = static_cast<double>(roi_cand.confidence) - 0.04 * roi_sil;
                                if (rnk > best_hole_rank) {
                                    best_hole_rank = rnk;
                                    pose = std::move(roi_cand);
                                }
                            }
                        }
                    }
                }
            }

            // 2. Nhánh tinh chỉnh trực tiếp hạt giống Pinhole + Rim Depth
            const double sil_cost = refineTranslationOnRgbSilhouette(
                seed.R, seed.t, seed.t, _cad_min, _cad_max, seg_res.floor_plane, seg_res.floor_detected,
                _mesh_pts, _mesh_normals, dt_canny, depth_z16, sil_fx, sil_fy, sil_cx, sil_cy, max_ws_z);
            if (!isInTargetWorkspace(seed.t)) continue;

            seed.t_glass = seed.t;
            const cv::Vec3d euler = rotationMatrixToEulerDeg(seed.R, false);
            seed.roll_deg = euler[0];
            seed.pitch_deg = euler[1];
            seed.yaw_deg = euler[2];
            seed.bbox_2d = projectCadBoundingBox(seed.R, seed.t, intr, depth_z16.cols, depth_z16.rows, &seed.corners_2d);

            double empty_behind_frac = 0.0, ring_jump_frac = 1.0, wide_out_frac = 1.0, wide_convex_ratio = 1.0;
            int ring_tot = 0;
            computeSilhouetteAndEmptyBehind(seed.R, seed.t, _mesh_pts, depth_z16,
                                            sil_fx, sil_fy, sil_cx, sil_cy, _real_bounding_diag_m,
                                            empty_behind_frac, ring_jump_frac,
                                            wide_out_frac, wide_convex_ratio, ring_tot);
            if (empty_behind_frac > 0.22 || (ring_tot >= 24 && ring_jump_frac < 0.30)) continue;

            rescorePoseUnclipped(seed, seg_res.foreground_points, depth_z16, intr, depth_z16.cols, depth_z16.rows, false);
            if (sil_cost <= 5.2) {
                seed.rgb_sil_strong = true;
                seed.method = "depth_hole_rgb_refined";
                seed.confidence = std::max(seed.confidence,
                                           static_cast<float>(std::clamp(0.92 - 0.05 * sil_cost, 0.66, 0.92)));
            }
            if (!DetectionAcceptancePolicy::isAccepted(seed, _real_bounding_diag_m, _is_floor_standing)) continue;

            const double rank = static_cast<double>(seed.confidence) - 0.04 * sil_cost - 0.25 * empty_behind_frac;
            if (rank > best_hole_rank) {
                best_hole_rank = rank;
                pose = std::move(seed);
            }
        }
    }

    // Chống nhảy cụm sai & lật góc đối xứng khi Global Detection chạy lại trong lúc đang track
    if (_temporal_tracking_enabled && !_multi_instance_enabled && roi_hint.empty() &&
        pose.valid && _track_pose.valid && _track_active) {
        const double g_jump = cv::norm(pose.t - _track_pose.t);
        const bool bad_jump = (g_jump > std::max(0.055, 0.24 * _real_bounding_diag_m));
        const bool worse_jump = (g_jump > 0.030 && pose.rmse_m > _track_pose.rmse_m + 0.0045);
        if (bad_jump || worse_jump) {
            pose = has_tracked_fallback ? std::move(tracked_fallback) : ObjectPose{};
        }
    }
    if (_temporal_tracking_enabled && !_multi_instance_enabled && roi_hint.empty() &&
        pose.valid && _track_pose.valid && !_is_floor_standing) {
        const cv::Vec3d span = _cad_max - _cad_min;
        int long_ax = 2;
        if (span[0] >= span[1] && span[0] >= span[2]) long_ax = 0;
        else if (span[1] >= span[0] && span[1] >= span[2]) long_ax = 1;
        const int ax1 = (long_ax + 1) % 3;
        const int ax2 = (long_ax + 2) % 3;
        const cv::Vec3d z_cur(pose.R(0, long_ax), pose.R(1, long_ax), pose.R(2, long_ax));
        const cv::Vec3d z_prv(_track_pose.R(0, long_ax), _track_pose.R(1, long_ax), _track_pose.R(2, long_ax));
        const cv::Vec3d x_cur(pose.R(0, ax1), pose.R(1, ax1), pose.R(2, ax1));
        const cv::Vec3d x_prv(_track_pose.R(0, ax1), _track_pose.R(1, ax1), _track_pose.R(2, ax1));
        if (z_cur.dot(z_prv) > 0.70 && x_cur.dot(x_prv) < -0.50) {
            const cv::Vec3d cad_mid = 0.5 * (_cad_min + _cad_max);
            const cv::Vec3d win_c = pose.R * cad_mid + pose.t;
            cv::Matx33d R_flip = pose.R;
            for (int r = 0; r < 3; ++r) {
                R_flip(r, ax1) = -pose.R(r, ax1);
                R_flip(r, ax2) = -pose.R(r, ax2);
            }
            ObjectPose unflipped = pose;
            unflipped.R = R_flip;
            unflipped.t = win_c - R_flip * cad_mid;
            unflipped.t_glass = unflipped.t;
            if (rescorePoseUnclipped(unflipped, seg_res.foreground_points, depth_z16, intr, depth_z16.cols, depth_z16.rows, false) &&
                DetectionAcceptancePolicy::isAccepted(unflipped, _real_bounding_diag_m, _is_floor_standing) &&
                unflipped.confidence >= pose.confidence - 0.06f) {
                const cv::Vec3d eu = rotationMatrixToEulerDeg(unflipped.R, false);
                unflipped.roll_deg = eu[0];
                unflipped.pitch_deg = eu[1];
                unflipped.yaw_deg = eu[2];
                unflipped.bbox_2d = projectCadBoundingBox(unflipped.R, unflipped.t, intr, depth_z16.cols, depth_z16.rows, &unflipped.corners_2d);
                pose = std::move(unflipped);
            } else {
                ObjectPose cont = pose;
                cont.R = _track_pose.R;
                cont.t = win_c - cont.R * cad_mid;
                cont.t_glass = cont.t;
                if (rescorePoseUnclipped(cont, seg_res.foreground_points, depth_z16, intr, depth_z16.cols, depth_z16.rows, false) &&
                    DetectionAcceptancePolicy::isAccepted(cont, _real_bounding_diag_m, _is_floor_standing) &&
                    cont.confidence >= pose.confidence - 0.12f) {
                    const cv::Vec3d eu = rotationMatrixToEulerDeg(cont.R, false);
                    cont.roll_deg = eu[0];
                    cont.pitch_deg = eu[1];
                    cont.yaw_deg = eu[2];
                    cont.bbox_2d = projectCadBoundingBox(cont.R, cont.t, intr, depth_z16.cols, depth_z16.rows, &cont.corners_2d);
                    pose = std::move(cont);
                }
            }
        } else if (z_cur.dot(z_prv) > 0.85 && x_cur.dot(x_prv) < 0.94) {
            // Nếu Global Re-check bị nhiễu xoay Yaw > 20 độ quanh trục dài, kiểm tra lại hướng _track_pose.R
            const cv::Vec3d cad_mid = 0.5 * (_cad_min + _cad_max);
            const cv::Vec3d win_c = pose.R * cad_mid + pose.t;
            ObjectPose cont = pose;
            cont.R = _track_pose.R;
            cont.t = win_c - cont.R * cad_mid;
            cont.t_glass = cont.t;
            if (rescorePoseUnclipped(cont, seg_res.foreground_points, depth_z16, intr, depth_z16.cols, depth_z16.rows, false) &&
                DetectionAcceptancePolicy::isAccepted(cont, _real_bounding_diag_m, _is_floor_standing) &&
                cont.confidence >= pose.confidence - 0.12f) {
                const cv::Vec3d eu = rotationMatrixToEulerDeg(cont.R, false);
                cont.roll_deg = eu[0];
                cont.pitch_deg = eu[1];
                cont.yaw_deg = eu[2];
                cont.bbox_2d = projectCadBoundingBox(cont.R, cont.t, intr, depth_z16.cols, depth_z16.rows, &cont.corners_2d);
                pose = std::move(cont);
            }
        }
    }

    // Cập nhật State Machine SEARCHING <-> TRACKING
    if (_temporal_tracking_enabled && !_multi_instance_enabled && roi_hint.empty()) {
        if (pose.valid) {
            if (_track_active) {
                _track_pose = pose;
                _track_miss_streak = 0;
            } else {
                bool consistent = false;
                if (_track_prev_global.valid) {
                    const double dt = cv::norm(pose.t - _track_prev_global.t);
                    const cv::Vec3d z_cur(pose.R(0, 2), pose.R(1, 2), pose.R(2, 2));
                    const cv::Vec3d z_prv(_track_prev_global.R(0, 2), _track_prev_global.R(1, 2), _track_prev_global.R(2, 2));
                    const double tilt = std::acos(std::clamp(std::abs(z_cur.dot(z_prv)), 0.0, 1.0)) * 180.0 / kPi;
                    consistent = (dt <= std::max(0.12, 0.6 * _real_bounding_diag_m) && tilt <= 25.0);
                }
                _track_confirm_streak = consistent ? (_track_confirm_streak + 1) : 1;
                _track_prev_global = pose;
                if (_track_confirm_streak >= 2) {
                    _track_active = true;
                    _track_pose = pose;
                    _track_miss_streak = 0;
                }
            }
            if (seg_res.floor_detected) {
                _track_has_floor = true;
                _track_floor_plane = seg_res.floor_plane;
            }
        } else if (_track_active) {
            ++_track_miss_streak;
            if (_track_miss_streak <= 2 && _track_pose.valid) {
                pose = _track_pose;
                pose.method = "TRACKED/HOLD";
            } else {
                resetTemporalTrack();
            }
        } else {
            _track_confirm_streak = 0;
            _track_prev_global = ObjectPose{};
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    _last_ai_latency_ms = ai_ms;
    _last_total_latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    _last_solve_latency_ms = std::max(0.0, _last_total_latency_ms - _last_ai_latency_ms);
    return pose;
}

std::vector<ObjectPose> UniversalPoseEstimator::estimateMultiObjects(const cv::Mat& rgb,
                                                                     const cv::Mat& depth_z16,
                                                                     const CameraIntrinsics& intr,
                                                                     const cv::Rect& roi_hint)
{
    auto t_start = std::chrono::steady_clock::now();
    std::vector<ObjectPose> results;
    _last_valid_depth_pts = 0;
    _last_floor_locked = false;
    _last_floor_pitch_deg = 0.0;
    _last_sparse_depth_fallback = false;
    _last_ai_latency_ms = 0.0;
    _last_solve_latency_ms = 0.0;
    _last_total_latency_ms = 0.0;
    _last_segmented_points.clear();

    if (!_initialized || depth_z16.empty()) return results;

    if (!_multi_instance_enabled) {
        ObjectPose single = estimate(rgb, depth_z16, intr, roi_hint);
        if (single.valid &&
            DetectionAcceptancePolicy::isAccepted(single, _real_bounding_diag_m, _is_floor_standing)) {
            results.push_back(std::move(single));
        }
        return results;
    }

    SegmentationResult seg_res = _scene_segmenter.segment(depth_z16, intr, roi_hint, _num_obs_points);
    _last_valid_depth_pts = static_cast<int>(seg_res.dense_object_points.size());
    _last_floor_locked = (_is_floor_standing && seg_res.floor_detected);
    _last_floor_pitch_deg = seg_res.floor_pitch_deg;
    _last_sparse_depth_fallback = (_last_valid_depth_pts >= 20 && _last_valid_depth_pts < 250);
    _last_segmented_points = seg_res.dense_object_points;

    if (!seg_res.valid) {
        auto t_end = std::chrono::steady_clock::now();
        _last_total_latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        _last_solve_latency_ms = _last_total_latency_ms;
        return results;
    }

    double total_ai_ms = 0.0;
    if (_is_floor_standing || seg_res.multi_clusters.size() <= 1) {
        ObjectPose single = estimateSingleCluster(seg_res.object_points,
                                                  seg_res.dense_object_points,
                                                  seg_res.floor_plane,
                                                  seg_res.floor_detected,
                                                  seg_res.floor_pitch_deg,
                                                  rgb, intr, depth_z16.cols, depth_z16.rows, &total_ai_ms);
        if (single.valid) {
            if (!_is_floor_standing &&
                seg_res.foreground_points.size() > static_cast<size_t>(single.valid_depth_pts * 1.18)) {
                rescorePoseUnclipped(single, seg_res.foreground_points, depth_z16, intr, depth_z16.cols, depth_z16.rows, false);
            }
            if (DetectionAcceptancePolicy::isAccepted(single, _real_bounding_diag_m, _is_floor_standing)) {
                results.push_back(std::move(single));
            }
        }
    } else {
        const RgbEdgeMaps dt_canny = (!_is_floor_standing && !rgb.empty()) ? buildRgbCannyDistanceTransform(rgb) : RgbEdgeMaps{};
        const double sil_fx = (intr.fx > 100.0) ? intr.fx : 600.0;
        const double sil_fy = (intr.fy > 100.0) ? intr.fy : sil_fx;
        const double sil_cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : 0.5 * depth_z16.cols);
        const double sil_cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : 0.5 * depth_z16.rows);

        const size_t n_try = std::min<size_t>(seg_res.multi_clusters.size(), 12);
        for (size_t i = 0; i < n_try; ++i) {
            const auto& cl = seg_res.multi_clusters[i];
            if (cl.points.size() < 20) continue;
            const std::vector<cv::Vec3f>& dense_cl = cl.dense_points.empty() ? cl.points : cl.dense_points;

            double ai_ms = 0.0;
            ObjectPose inst_pose = estimateSingleCluster(cl.points,
                                                         dense_cl,
                                                         seg_res.floor_plane,
                                                         false,
                                                         seg_res.floor_pitch_deg,
                                                         rgb, intr, depth_z16.cols, depth_z16.rows, &ai_ms);
            total_ai_ms += ai_ms;
            if (!inst_pose.valid) continue;

            double sil_cost = 0.0;
            double z_hole_frac = 0.0;
            if (!dt_canny.empty()) {
                const double max_ws_z = std::clamp(_real_bounding_diag_m * 3.8 + 0.15, 0.95, 2.50);
                sil_cost = refineTranslationOnRgbSilhouette(
                    inst_pose.R, inst_pose.t, cv::Vec3d(cl.centroid), _cad_min, _cad_max, seg_res.floor_plane, seg_res.floor_detected,
                    _mesh_pts, _mesh_normals, dt_canny, depth_z16, sil_fx, sil_fy, sil_cx, sil_cy, max_ws_z);
                inst_pose.t_glass = inst_pose.t;
                const cv::Vec3d euler = rotationMatrixToEulerDeg(inst_pose.R, false);
                inst_pose.roll_deg = euler[0];
                inst_pose.pitch_deg = euler[1];
                inst_pose.yaw_deg = euler[2];
                inst_pose.bbox_2d = projectCadBoundingBox(inst_pose.R, inst_pose.t, intr, depth_z16.cols, depth_z16.rows, &inst_pose.corners_2d);
                if (!isInTargetWorkspace(inst_pose.t)) continue;
                z_hole_frac = computeProjectedZeroDepthFraction(
                    inst_pose.R, inst_pose.t, _cad_min, _cad_max, depth_z16, sil_fx, sil_fy, sil_cx, sil_cy);
            }

            ObjectPose strict_pose = inst_pose;
            const bool strong_rgb_sil = (!dt_canny.empty() && sil_cost <= 5.5 && z_hole_frac >= 0.35 &&
                                         inst_pose.inliers_count >= 12 && inst_pose.rmse_m <= 0.016);
            const bool acceptable_rgb_sil = (dt_canny.empty() || sil_cost <= 9.5);
            const bool passed_unclipped = acceptable_rgb_sil &&
                rescorePoseUnclipped(strict_pose, seg_res.foreground_points, depth_z16, intr, depth_z16.cols, depth_z16.rows, true);
            if (passed_unclipped || strong_rgb_sil) {
                if (passed_unclipped && !dt_canny.empty() && sil_cost <= 6.2) {
                    strict_pose.confidence = std::clamp(strict_pose.confidence + 0.12f, 0.05f, 0.99f);
                } else if (!passed_unclipped && strong_rgb_sil) {
                    strict_pose = inst_pose;
                    strict_pose.rgb_sil_strong = true;
                    if (strict_pose.method == "pca_fallback_seed") {
                        strict_pose.method = "rgb_sil_refined";
                    }
                    strict_pose.confidence = std::clamp(strict_pose.confidence + 0.18f, 0.05f, 0.99f);
                }
                if (DetectionAcceptancePolicy::isAccepted(strict_pose, _real_bounding_diag_m, _is_floor_standing)) {
                    results.push_back(std::move(strict_pose));
                }
            }
        }

        if (!results.empty()) {
            std::sort(results.begin(), results.end(), [](const ObjectPose& a, const ObjectPose& b) {
                const double ra = static_cast<double>(a.confidence) - 2.0 * std::max(0.0, a.rmse_m - 0.015);
                const double rb = static_cast<double>(b.confidence) - 2.0 * std::max(0.0, b.rmse_m - 0.015);
                return ra > rb;
            });

            // 3D NMS to merge duplicate spherical proposals on the same physical object instance
            const double nms_min_sep = std::max(0.045, 0.35 * _real_bounding_diag_m);
            std::vector<ObjectPose> nms_kept;
            nms_kept.reserve(results.size());
            for (auto& cand : results) {
                bool dup = false;
                for (const auto& kept : nms_kept) {
                    if (cv::norm(cand.t - kept.t) < nms_min_sep) {
                        dup = true;
                        break;
                    }
                }
                if (!dup) {
                    nms_kept.push_back(std::move(cand));
                }
            }
            results = std::move(nms_kept);

            const float best_conf = results.front().confidence;
            if (best_conf >= 0.45f) {
                const float min_keep_conf = std::max(0.35f, 0.78f * best_conf);
                results.erase(std::remove_if(results.begin() + 1, results.end(),
                                             [min_keep_conf](const ObjectPose& p) { return p.confidence < min_keep_conf; }),
                              results.end());
            }
            if (static_cast<int>(results.size()) > _max_instances) {
                results.resize(static_cast<size_t>(_max_instances));
            }
            size_t total_cl_pts = 0;
            std::vector<cv::Vec3f> combined_seg_pts;
            for (size_t k = 0; k < results.size(); ++k) {
                results[k].model_name = _model_name + " #" + std::to_string(k + 1);
                total_cl_pts += results[k].segmented_points.size();
                combined_seg_pts.insert(combined_seg_pts.end(),
                                        results[k].segmented_points.begin(),
                                        results[k].segmented_points.end());
            }
            if (total_cl_pts > 0) {
                _last_valid_depth_pts = static_cast<int>(total_cl_pts);
                _last_sparse_depth_fallback = (_last_valid_depth_pts >= 20 && _last_valid_depth_pts < 250);
                _last_segmented_points = std::move(combined_seg_pts);
            }
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    _last_ai_latency_ms = total_ai_ms;
    _last_total_latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    _last_solve_latency_ms = std::max(0.0, _last_total_latency_ms - _last_ai_latency_ms);
    return results;
}

cv::Rect2f UniversalPoseEstimator::projectCadBoundingBox(const cv::Matx33d& R,
                                                         const cv::Vec3d& t,
                                                         const CameraIntrinsics& intr,
                                                         int img_w,
                                                         int img_h,
                                                         std::vector<cv::Point2f>* out_corners) const
{
    const double fx = (intr.fx > 100.0) ? intr.fx : 608.215;
    const double fy = (intr.fy > 100.0) ? intr.fy : 608.128;
    const double cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : 0.5 * img_w);
    const double cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : 0.5 * img_h);

    const double x0 = _cad_min[0], x1 = _cad_max[0];
    const double y0 = _cad_min[1], y1 = _cad_max[1];
    const double z0 = _cad_min[2], z1 = _cad_max[2];

    const cv::Vec3d corners_3d[8] = {
        {x0, y0, z0}, {x1, y0, z0}, {x1, y1, z0}, {x0, y1, z0},
        {x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1}
    };

    float min_u = 1e9f, min_v = 1e9f, max_u = -1e9f, max_v = -1e9f;
    if (out_corners) out_corners->clear();

    for (int i = 0; i < 8; ++i) {
        cv::Vec3d p = R * corners_3d[i] + t;
        double z = std::max(0.05, p[2]);
        float u = static_cast<float>(fx * p[0] / z + cx);
        float v = static_cast<float>(fy * p[1] / z + cy);
        if (out_corners) out_corners->emplace_back(u, v);
        min_u = std::min(min_u, u); max_u = std::max(max_u, u);
        min_v = std::min(min_v, v); max_v = std::max(max_v, v);
    }
    return cv::Rect2f(min_u, min_v, max_u - min_u, max_v - min_v);
}

void UniversalPoseEstimator::drawOverlay(cv::Mat& img,
                                         const ObjectPose& pose,
                                         const CameraIntrinsics& intr) const
{
    if (img.empty() || !pose.valid) return;

    const double fx = (intr.fx > 100.0) ? intr.fx : 608.215;
    const double fy = (intr.fy > 100.0) ? intr.fy : 608.128;
    const double cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : (0.5 * img.cols));
    const double cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : (0.5 * img.rows));

    auto proj_f = [&](double x, double y, double z) -> cv::Point2f {
        cv::Vec3d p = pose.R * cv::Vec3d(x, y, z) + pose.t;
        double pz = std::max(0.05, p[2]);
        return cv::Point2f(static_cast<float>(fx * p[0] / pz + cx),
                           static_cast<float>(fy * p[1] / pz + cy));
    };
    auto to_pt = [](const cv::Point2f& p) -> cv::Point {
        return cv::Point(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y)));
    };

    const cv::Scalar CYAN(255, 229, 0), GREEN(50, 235, 50), YELLOW(0, 235, 255), RED(40, 40, 255), BLUE(255, 120, 30), WHITE(245, 245, 245);

    // 1. Project 3D CAD feature/crease wireframe edges for ANY object
    for (const auto& e : _cad_feature_edges) {
        cv::Point2f a = proj_f(e.p0[0], e.p0[1], e.p0[2]);
        cv::Point2f b = proj_f(e.p1[0], e.p1[1], e.p1[2]);
        cv::line(img, to_pt(a), to_pt(b), CYAN, 2, cv::LINE_AA);
    }

    // 1b. Draw floor-standing foot contact ellipses from ONNX _cad_foot_centers (no hardcoded insets)
    if (_is_floor_standing && !_cad_foot_centers.empty()) {
        for (const auto& fc : _cad_foot_centers) {
            const cv::Vec3d p_cam = pose.R * cv::Vec3d(fc[0], fc[1], fc[2]) + pose.t;
            if (p_cam[2] <= 0.10) continue;
            const cv::Point2f bp = proj_f(fc[0], fc[1], fc[2]);
            const int rx = std::clamp(static_cast<int>(std::round(0.020 * fx / p_cam[2])), 5, 18);
            const int ry = std::max(3, rx * 2 / 5);
            cv::ellipse(img, to_pt(bp), cv::Size(rx, ry), 0, 0, 360, GREEN, 1, cv::LINE_AA);
        }
    }

    // 2. Draw 8-corner 3D Oriented Bounding Box
    std::vector<cv::Point2f> c2d = pose.corners_2d;
    if (c2d.size() != 8) {
        projectCadBoundingBox(pose.R, pose.t, intr, img.cols, img.rows, &c2d);
    }
    if (c2d.size() == 8) {
        for (int i = 0; i < 4; ++i) {
            cv::line(img, to_pt(c2d[i]), to_pt(c2d[(i + 1) % 4]), GREEN, 2, cv::LINE_AA);
            cv::line(img, to_pt(c2d[i + 4]), to_pt(c2d[((i + 1) % 4) + 4]), CYAN, 2, cv::LINE_AA);
            cv::line(img, to_pt(c2d[i]), to_pt(c2d[i + 4]), YELLOW, 2, cv::LINE_AA);
        }
    }

    // 3. Scale-adaptive 3D Coordinate Axes at object center
    const double axis_len = std::clamp(_real_bounding_diag_m * 0.16, 0.04, 0.16);
    cv::Point2f p_c = proj_f(0, 0, 0);
    cv::Point2f p_x = proj_f(axis_len, 0, 0);
    cv::Point2f p_y = proj_f(0, axis_len, 0);
    cv::Point2f p_z = proj_f(0, 0, axis_len);
    cv::arrowedLine(img, to_pt(p_c), to_pt(p_x), RED, 2, cv::LINE_AA, 0, 0.22);
    cv::arrowedLine(img, to_pt(p_c), to_pt(p_y), GREEN, 2, cv::LINE_AA, 0, 0.22);
    cv::arrowedLine(img, to_pt(p_c), to_pt(p_z), BLUE, 2, cv::LINE_AA, 0, 0.22);

    // 4. Universal Object Telemetry Label
    std::ostringstream oss;
    oss << "[" << pose.model_name << "] Z:" << std::fixed << std::setprecision(2) << pose.t[2]
        << "m | Yaw:" << std::setprecision(1) << pose.yaw_deg << " deg";
    cv::Point lbl_pt = (c2d.size() == 8) ? to_pt(c2d[4]) : to_pt(p_c);
    lbl_pt.x = std::clamp(lbl_pt.x, 10, std::max(10, img.cols - 260));
    lbl_pt.y = std::clamp(lbl_pt.y - 10, 24, std::max(24, img.rows - 10));
    cv::putText(img, oss.str(), lbl_pt, cv::FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1, cv::LINE_AA);
}

void UniversalPoseEstimator::drawMultiOverlay(cv::Mat& img,
                                              const std::vector<ObjectPose>& poses,
                                              const CameraIntrinsics& intr) const
{
    for (const auto& p : poses) {
        drawOverlay(img, p, intr);
    }
}

void UniversalPoseEstimator::transformPoseCameraFrame(ObjectPose& pose,
                                                      const cv::Matx33d& target_from_source_R,
                                                      const cv::Vec3d& target_from_source_t,
                                                      const CameraIntrinsics& target_intr,
                                                      int target_width,
                                                      int target_height) const
{
    pose.R = target_from_source_R * pose.R;
    pose.t = target_from_source_R * pose.t + target_from_source_t;
    pose.t_glass = pose.t;
    const bool fold_c2 = (_is_floor_standing && _symmetry_code >= 1);
    cv::Vec3d euler = rotationMatrixToEulerDeg(pose.R, fold_c2);
    pose.roll_deg = euler[0];
    pose.pitch_deg = euler[1];
    pose.yaw_deg = euler[2];
    pose.bbox_2d = projectCadBoundingBox(pose.R, pose.t, target_intr,
                                         target_width, target_height, &pose.corners_2d);
}

cv::Vec3d UniversalPoseEstimator::rotationMatrixToEulerDeg(const cv::Matx33d& R, bool fold_c2_yaw)
{
    cv::Vec3d v_rail(R(0, 1), R(1, 1), R(2, 1));
    cv::Vec3d v_up(R(0, 2), R(1, 2), R(2, 2));
    double roll = std::asin(std::clamp(v_up[0], -1.0, 1.0)) * 180.0 / kPi;
    double pitch = std::asin(std::clamp(v_up[2], -1.0, 1.0)) * 180.0 / kPi;
    double yaw = std::atan2(v_rail[2], -v_rail[0]) * 180.0 / kPi;
    if (fold_c2_yaw) {
        while (yaw > 90.0) yaw -= 180.0;
        while (yaw < -90.0) yaw += 180.0;
    }
    return cv::Vec3d(roll, pitch, yaw);
}

} // namespace tmcad
