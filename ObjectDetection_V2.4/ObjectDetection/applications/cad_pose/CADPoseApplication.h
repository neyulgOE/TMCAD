#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <opencv2/core.hpp>

#include "IApplication.h"
#include "RgbdFrame.h"
#include "CADTemplateDetector.h"
#include "CADSceneSegmenter.h"
#if __has_include(<rknn_api.h>)
#include <rknn_api.h>
#elif __has_include("rknn_api.h")
#include "rknn_api.h"
#else
typedef uint64_t rknn_context;
#endif

/**
 * @brief CADPoseApplication: Computes full 6DoF Pose of a known CAD model from RealSense RGB-D.
 *
 * Capabilities:
 * - 100% CAD-native: Runs NPU inference (rknn_api) directly on depth-derived point cloud.
 * - Solves full 6DoF rigid transformation (R, t) via RANSAC Kabsch SVD.
 * - Applies exact camera front glass centerline offset (+17.5 mm in X from Left IR).
 * - Extracts full 6DoF (X, Y, Z, Roll, Pitch, Yaw) with CAD vertical centerline origin.
 * - Reprojects 3D bounding box onto RGB image for visual verification without 2D detectors.
 */
class CADPoseApplication final : public IApplication
{
public:
    CADPoseApplication() = default;
    ~CADPoseApplication() override;

    std::string name() const override { return "cad_pose"; }
    bool requiresRgbd() const override { return true; }

    bool initialize(const DetectorConfig& config,
                    std::string& error) override;

    bool process(const RgbdFrame& frame,
                 DetectorOutput& output,
                 std::string& error) override;

    void release();

private:
    struct Pose6D
    {
        cv::Matx33d R = cv::Matx33d::eye();
        cv::Vec3d t{0.0, 0.0, 0.0};
        double roll_deg = 0.0;
        double pitch_deg = 0.0;
        double yaw_deg = 0.0;
        double inlier_ratio = 0.0;
        double rmse_m = 0.0;
        int inliers_count = 0;
        bool valid = false;
    };

    // Preprocessing: extracts & subsamples 1024 3D points from depth_z16
    std::vector<cv::Vec3f> extractObservationPoints(const RgbdFrame& frame) const;

    // V2.4: ROI-guided depth extraction (focuses 1024 points purely on the detected table)
    std::vector<cv::Vec3f> extractROIPoints(const RgbdFrame& frame, const cv::Rect2f& roi) const;

    // Feature preparation: builds 7D k-NN graph features (1, 7, 1024, 20)
    std::vector<float> buildGraphFeatures(const std::vector<cv::Vec3f>& pts, int k) const;

    // NPU forward execution: runs model_obs_rk3588.rknn
    bool runNpuInference(const std::vector<float>& graph_tensor,
                         std::vector<float>& out_descriptors,
                         std::string& error);

    // Matching & 6DoF solver: RANSAC Kabsch SVD
    Pose6D solveRansacKabsch(const std::vector<cv::Vec3f>& obs_points,
                             const std::vector<float>& obs_descriptors) const;

    // Fast Point-to-Plane Gauss-Newton refinement (8-10 iters, ~1.5ms)
    // Minimizes point-to-plane residual on tabletop surface down to < 2-5 mm and < 0.8 deg
    Pose6D refinePointToPlaneGaussNewton(const Pose6D& coarse_pose,
                                        const std::vector<cv::Vec3f>& obs_points,
                                        const std::vector<cv::Vec3f>& cad_interp_points,
                                        const std::vector<uint8_t>& inlier_mask) const;

    // Reproject 3D bounding box to 2D image coordinates (and optionally export 8 2D corners)
    cv::Rect2f projectCadBoundingBox(const cv::Matx33d& R,
                                     const cv::Vec3d& t,
                                     const CameraIntrinsics& intr,
                                     int img_w, int img_h,
                                     std::vector<cv::Point2f>* out_corners_2d = nullptr) const;

    // Helper functions
    static cv::Vec3d rotationMatrixToEulerDeg(const cv::Matx33d& R);
    static std::vector<uint8_t> readBinaryFile(const std::string& path);
    static bool parseNpyCentroids(const std::string& path, std::vector<cv::Vec3f>& centroids);

private:
    DetectorConfig _config;

    // NPU RKNN context
    rknn_context _rknn_ctx = 0;
    bool _rknn_loaded = false;
    uint32_t _n_input = 1;

    // Precomputed CAD descriptors & centroids
    std::vector<float> _cad_descriptors;     // (512 * 128)
    std::vector<cv::Vec3f> _cad_centroids;   // (512)
    int _num_cad_samples = 512;
    int _desc_dim = 128;

    // Observation parameters
    int _num_obs_points = 1024;
    int _obs_k = 20;
    double _max_depth_m = 3.50;              // Extended depth working range (up to 3.5m)

    // RANSAC parameters
    double _ransac_dist_threshold_m = 0.035; // 35 mm
    int _ransac_max_iterations = 250;
    double _min_inlier_ratio = 0.30;         // Detection acceptance threshold
    int _min_inliers_count = 25;             // Real-sensor minimum inlier count
    double _max_rmse_m = 0.025;              // Maximum allowable inlier RMSE (25 mm)

    // Mechanical camera offset (Intel RealSense D435 Figure 4-6)
    double _cam_offset_x_m = 0.0175;         // +17.5 mm from Left IR to Centerline 1/4-20
    double _cam_offset_y_m = 0.0;
    double _cam_offset_z_m = 0.0;            // Relative to front cover glass surface

    // CAD Dimensions for 3D Bounding Box (Tabletop: 0.44m x 0.83m)
    double _cad_half_width_m = 0.220;        // Width = 0.44 m (X)
    double _cad_half_length_m = 0.415;       // Length = 0.83 m (Y)
    double _cad_top_z_m = 0.1665;            // Top tabletop Z
    double _cad_bottom_z_m = -0.3355;        // Bottom feet Z

    // Preallocated auxiliary buffers for multi-input RKNN inference
    std::vector<uint8_t> _f2_buf;
    std::vector<uint8_t> _f3_buf;

    // V2.4: RGB Template Detector
    CADTemplateDetector _template_detector;
    bool _rgb_detection_enabled = true;
    double _rgb_min_score = 0.18;

    // V2.4+: 3D Point Cloud Scene Segmenter (RANSAC floor removal & clustering)
    CADSceneSegmenter _scene_segmenter;
    bool _enable_3d_segmentation = true;
    bool _enable_gn_refinement = true;
};
