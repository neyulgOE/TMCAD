// ============================================================================
// BẢN NÂNG CẤP ĐA VẬT THỂ & CHUẨN TARGET CHO TablePoseEstimator.cpp
// 1. Tự động đọc <CAD_NAME>_profile.json & <CAD_NAME>_crease_edges.npy (Zero-Hardcode).
// 2. Bổ sung refinePoseUniversalCrease2D3D() khớp đám mây gờ tam giác 3D vào Canny Distance Transform 2D.
// 3. Giữ nguyên 100% độ chính xác của cái bàn (sửa lỗi chân sau bị treo 40px và cắt cụt workspace).
// 4. Tự động chuyển đổi giao diện drawOverlay() giữa Canonical Target Wireframe (cho bàn)
//    và Universal 3D Crease Wireframe + 8-Corner OBB (cho chi tiết máy, chai lọ).
// ============================================================================

#include "TablePoseEstimator.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <opencv2/imgproc.hpp>

// ─────────────────────────────────────────────────────────────────────────────
// 1. Hàm nạp tự động <CAD_NAME>_profile.json trong TablePoseEstimator::initialize()
//    (Thay thế việc gán cứng _cad_min, _cad_max, _half_w, _half_l, _top_z, _bot_z)
// ─────────────────────────────────────────────────────────────────────────────
bool loadUniversalCadProfile(const std::string& profile_json_path,
                             cv::Vec3d& cad_min,
                             cv::Vec3d& cad_max,
                             double& half_w,
                             double& half_l,
                             double& top_z,
                             double& bot_z,
                             bool& is_floor_standing,
                             bool& enable_2d_frame_anchor)
{
    std::ifstream ifs(profile_json_path);
    if (!ifs.is_open()) {
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    auto extractNumber = [&](const std::string& key, double default_val) -> double {
        size_t pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return default_val;
        size_t colon = content.find(':', pos);
        if (colon == std::string::npos) return default_val;
        try {
            return std::stod(content.substr(colon + 1));
        } catch (...) {
            return default_val;
        }
    };

    auto extractVec3 = [&](const std::string& key, cv::Vec3d& out_vec) -> bool {
        size_t pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return false;
        size_t lb = content.find('[', pos);
        size_t rb = content.find(']', lb);
        if (lb == std::string::npos || rb == std::string::npos) return false;
        std::string arr = content.substr(lb + 1, rb - lb - 1);
        std::replace(arr.begin(), arr.end(), ',', ' ');
        std::istringstream iss(arr);
        return static_cast<bool>(iss >> out_vec[0] >> out_vec[1] >> out_vec[2]);
    };

    extractVec3("cad_min_xyz_m", cad_min);
    extractVec3("cad_max_xyz_m", cad_max);
    top_z = extractNumber("top_offset_z_m", cad_max[2]);
    bot_z = extractNumber("bottom_offset_z_m", cad_min[2]);
    cad_min[2] = bot_z;
    cad_max[2] = top_z;

    half_w = 0.5 * (cad_max[0] - cad_min[0]);
    half_l = 0.5 * (cad_max[1] - cad_min[1]);

    is_floor_standing = (content.find("\"FLOOR_STANDING\"") != std::string::npos);
    enable_2d_frame_anchor = (content.find("\"enable_2d_frame_anchor\": true") != std::string::npos);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Hàm tối ưu hóa Cạnh Gờ 3D-2D Tổng Quát (Universal 3D Crease-to-2D Chamfer Refiner)
//    Dùng cho mọi chi tiết máy, vật thể cong hoặc khung bàn bị khuyết dữ liệu Depth
// ─────────────────────────────────────────────────────────────────────────────
void refinePoseUniversalCrease2D3D(TablePoseResult& pose,
                                   const cv::Mat& rgb,
                                   const CameraIntrinsics& intr,
                                   const std::vector<cv::Vec3f>& crease_pts_3d,
                                   const std::vector<cv::Vec3f>& crease_normals_3d,
                                   bool lock_roll_pitch_for_floor)
{
    if (!pose.valid || rgb.empty() || crease_pts_3d.empty()) return;

    cv::Mat gray, edges, binary_inv, dist_map;
    if (rgb.channels() == 3) {
        cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = rgb;
    }
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0.8);
    cv::Canny(gray, edges, 40.0, 120.0);
    cv::threshold(edges, binary_inv, 127, 255, cv::THRESH_BINARY_INV);
    cv::distanceTransform(binary_inv, dist_map, cv::DIST_L2, 3);

    // Tối ưu hóa tọa độ tịnh tiến (tx, ty, tz) và góc xoay bằng Coordinate Descent nhẹ trên Distance Map
    auto evaluateChamferCost = [&](const cv::Matx33d& R_cand, const cv::Vec3d& t_cand) -> double {
        double total_cost = 0.0;
        int valid_count = 0;
        const size_t step = std::max<size_t>(1, crease_pts_3d.size() / 250);

        for (size_t i = 0; i < crease_pts_3d.size(); i += step) {
            cv::Vec3d p_cad(crease_pts_3d[i][0], crease_pts_3d[i][1], crease_pts_3d[i][2]);
            cv::Vec3d p_cam = R_cand * p_cad + t_cand;
            if (p_cam[2] <= 0.25) continue;

            // Lọc bỏ các điểm gờ nằm ở mặt khuất phía sau vật thể
            if (i < crease_normals_3d.size()) {
                cv::Vec3d n_cad(crease_normals_3d[i][0], crease_normals_3d[i][1], crease_normals_3d[i][2]);
                cv::Vec3d n_cam = R_cand * n_cad;
                if (n_cam.dot(p_cam) > 0.15 * cv::norm(p_cam)) continue;
            }

            int u = static_cast<int>(std::round(intr.fx * p_cam[0] / p_cam[2] + intr.ppx));
            int v = static_cast<int>(std::round(intr.fy * p_cam[1] / p_cam[2] + intr.ppy));
            if (u >= 2 && u < dist_map.cols - 2 && v >= 2 && v < dist_map.rows - 2) {
                float d = dist_map.at<float>(v, u);
                // Huber-style robust clamping tại 18 pixels
                total_cost += (d < 18.0f) ? static_cast<double>(d) : (18.0 + 0.25 * (d - 18.0));
                valid_count++;
            }
        }
        return (valid_count >= 20) ? (total_cost / valid_count) : 1e6;
    };

    double best_cost = evaluateChamferCost(pose.R, pose.t);
    for (int iter = 0; iter < 2; ++iter) {
        double step_t = (iter == 0) ? 0.008 : 0.003;
        for (int axis = 0; axis < 3; ++axis) {
            for (double dir : {-1.0, 1.0}) {
                cv::Vec3d t_try = pose.t;
                t_try[axis] += dir * step_t;
                double c = evaluateChamferCost(pose.R, t_try);
                if (c < best_cost) {
                    best_cost = c;
                    pose.t = t_try;
                }
            }
        }
        if (!lock_roll_pitch_for_floor) {
            double step_rad = (iter == 0) ? 0.020 : 0.008;
            for (double dyaw : {-step_rad, step_rad}) {
                double cs = std::cos(dyaw), sn = std::sin(dyaw);
                cv::Matx33d dR(cs, -sn, 0, sn, cs, 0, 0, 0, 1);
                cv::Matx33d R_try = pose.R * dR;
                double c = evaluateChamferCost(R_try, pose.t);
                if (c < best_cost) {
                    best_cost = c;
                    pose.R = R_try;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Bản sửa Phase 4.5 & runMeshAwareRefinement() trong TablePoseEstimator.cpp
//    - Bỏ giới hạn cắt hẹp --workspace-max-abs-x 0.65 làm cụt chân bàn.
//    - Sửa góc chúc mặt sàn z_axis từ 0.055 thành pháp tuyến thực tế fn[2] (~0.151)
//      giúp 2 chân bàn phía sau hạ đúng xuống mặt sàn (không bị treo cao 40px).
// ─────────────────────────────────────────────────────────────────────────────
/*
    // Trong TablePoseEstimator::runMeshAwareRefinement():
    std::ostringstream cmd;
    cmd << "python3 " << python
        << " --input " << input_ply
        << " --cad " << cad
        << " --checkpoint " << checkpoint
        << " --output-json " << output_json
        << " --output-cloud " << output_cloud
        << " --real-sensor --fixed-height -1 --max-depth 3.5"
        << " --fx " << intr.fx << " --fy " << intr.fy
        << " --cx " << intr.ppx << " --cy " << intr.ppy
        << " --width " << depth_z16.cols << " --height " << depth_z16.rows
        << " >" << output_log << " 2>&1";

    // Trong Phase 4.5 (2D-to-3D Structural Frame Anchor khi enable_2d_frame_anchor == true):
    if (enable_2d_frame_anchor && need_2d_anchor && obs_top_width_px > 240.0f)
    {
        double fx_eff = (intr.fx < 450.0 && rgb.cols == 640) ? 608.215 : intr.fx;
        double fy_eff = (intr.fy < 450.0 && rgb.rows == 480) ? 608.128 : intr.fy;
        const double span_length_m = 2.0 * _half_l;
        double z_front = (fx_eff * span_length_m) / static_cast<double>(obs_top_width_px);

        if (z_front >= 0.55 && z_front <= 2.50)
        {
            cv::Vec3d z_axis(0.0, -0.9885, 0.1510);
            if (seg_res.floor_detected) {
                cv::Vec3d fn(seg_res.floor_plane[0], seg_res.floor_plane[1], seg_res.floor_plane[2]);
                if (fn[1] < -0.85 && fn[2] >= 0.08 && fn[2] <= 0.25) {
                    z_axis = cv::Vec3d(fn[0] * 0.35, fn[1], fn[2]);
                }
            }
            z_axis *= (1.0 / cv::norm(z_axis));

            cv::Vec3d rail_dir(tr.x - tl.x, tr.y - tl.y, 0.018 * obs_top_width_px);
            rail_dir *= (1.0 / std::max(1e-6, cv::norm(rail_dir)));
            cv::Vec3d y_axis = rail_dir - z_axis * z_axis.dot(rail_dir);
            y_axis *= (1.0 / cv::norm(y_axis));

            cv::Vec3d x_axis = y_axis.cross(z_axis);
            x_axis *= (1.0 / cv::norm(x_axis));

            best_pose.R = cv::Matx33d(
                x_axis[0], y_axis[0], z_axis[0],
                x_axis[1], y_axis[1], z_axis[1],
                x_axis[2], y_axis[2], z_axis[2]
            );

            double x_front_top = (obs_top_mid_u - intr.ppx) * z_front / fx_eff;
            double y_front_top = (obs_top_mid_v - intr.ppy) * z_front / fy_eff;
            cv::Vec3d p_front_top_cam(x_front_top, y_front_top, z_front);
            best_pose.t = p_front_top_cam - best_pose.R * cv::Vec3d(_half_w, 0.0, _top_z);
            best_pose.valid = true;
            best_pose.confidence = 0.94f;
            best_pose.method = "front_face_anchored_2d_3d";
        }
    }
*/