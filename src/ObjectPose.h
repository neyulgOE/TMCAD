#pragma once

#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace tmcad {

struct FeatureEdge3D {
    cv::Vec3f p0{0.0f, 0.0f, 0.0f};
    cv::Vec3f p1{0.0f, 0.0f, 0.0f};
    cv::Vec3f n0{0.0f, 0.0f, 1.0f};
    cv::Vec3f n1{0.0f, 0.0f, 1.0f};
    float length_m{0.0f};
};

struct ObjectPose {
    bool valid{false};
    bool face_detected{false};
    float face_confidence{0.0f};
    float confidence{0.0f};
    int inliers_count{0};
    double inlier_ratio{0.0};
    int cluster_points_count{0};
    double rmse_m{0.0};

    // Diagnostic telemetry fields populated by UniversalPoseEstimator
    int valid_depth_pts{0};
    bool floor_locked{false};
    double floor_pitch_deg{0.0};
    bool sparse_depth_fallback{false};
    bool rgb_sil_strong{false};

    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d t{0.0, 0.0, 0.0};
    cv::Vec3d t_glass{0.0, 0.0, 0.0};

    double roll_deg{0.0};
    double pitch_deg{0.0};
    double yaw_deg{0.0};

    std::string model_name{"GENERIC_CAD"};
    std::string method{"none"};
    cv::Rect2f bbox_2d;
    std::vector<cv::Point2f> corners_2d;
    std::vector<cv::Point2f> face_corners_2d;
    std::vector<cv::Vec3f> segmented_points;
};

struct DetectionAcceptancePolicy {
    static bool isInWorkspace(const cv::Vec3d& t, double cad_diag, bool is_floor_standing) {
        if (!std::isfinite(t[0]) || !std::isfinite(t[1]) || !std::isfinite(t[2])) return false;
        if (is_floor_standing) {
            return (t[0] >= -1.35 && t[0] <= 1.35) &&
                   (t[1] >= -0.95 && t[1] <= 0.65) &&
                   (t[2] >= 0.45 && t[2] <= 2.95);
        }
        const double max_z = std::clamp(std::max(0.05, cad_diag) * 3.8 + 0.15, 0.95, 2.50);
        return (std::abs(t[0]) <= 1.20) &&
               (std::abs(t[1]) <= 1.00) &&
               (t[2] >= 0.15 && t[2] <= max_z);
    }

    static bool isAccepted(const ObjectPose& pose, double cad_diag, bool is_floor_standing) {
        if (!pose.valid) return false;
        if (pose.method == "pca_fallback_seed") return false;
        if (!isInWorkspace(pose.t, cad_diag, is_floor_standing)) return false;
        if (pose.bbox_2d.width > 0.0f && pose.bbox_2d.height > 0.0f) {
            if (std::max(pose.bbox_2d.width, pose.bbox_2d.height) < 24.0f ||
                std::min(pose.bbox_2d.width, pose.bbox_2d.height) < 12.0f) {
                return false;
            }
        }
        if (pose.confidence < 0.65f) return false;
        if (pose.rmse_m <= 1e-5 || pose.inliers_count < 8) return false;
        const double max_rmse = std::clamp(0.072 * std::max(0.05, cad_diag), 0.012, 0.025);
        if (pose.rmse_m > max_rmse) return false;
        if (!is_floor_standing && !pose.rgb_sil_strong && pose.inlier_ratio < 0.50) return false;
        return true;
    }
};

} // namespace tmcad

