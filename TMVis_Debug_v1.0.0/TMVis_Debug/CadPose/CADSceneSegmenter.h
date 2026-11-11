#pragma once

#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <opencv2/core.hpp>

#include "RgbdFrame.h"

/**
 * @brief CADSceneSegmenter: Fast 3D Point Cloud Segmentation for CAD table isolation.
 *
 * Implements:
 * 1. Fast sub-sampled 3D point cloud extraction from depth_z16 (~0.4 ms).
 * 2. RANSAC Floor Plane extraction (ax + by + cz + d = 0) with gravity-aligned normal (~1.0 ms).
 * 3. Elevation band filtering: isolates objects standing on the floor (0.04m <= h <= 1.25m).
 * 4. Fast 3D spatial clustering: extracts the discrete table cluster in front of the robot (~1.5 ms).
 * 5. Uniform subsampling to exactly N observation points (1024) for NPU EdgeConv.
 *
 * Latency on RK3588 Cortex-A76: ~3.0 - 4.5 ms total.
 * 100% independent of ambient lighting, shadows, or texture-less tabletop surfaces.
 */
class CADSceneSegmenter
{
public:
    struct SegmentResult
    {
        bool valid = false;
        std::vector<cv::Vec3f> table_points; // Exactly 1024 points for NPU inference
        std::vector<std::vector<cv::Vec3f>> cluster_candidates; // Candidate table clusters (1024 pts each)
        cv::Rect2f roi_2d;                   // Projected 2D bounding box on image
        cv::Vec4d floor_plane{0.0, -1.0, 0.0, 0.0}; // (a, b, c, d) with normal pointing UP
        double floor_height_m = 0.0;         // Estimated camera height above floor
        double floor_tilt_deg = 0.0;         // Angle between floor normal and vertical
        bool floor_detected = false;
        int raw_points_count = 0;
        int table_cluster_points_count = 0;
        std::string status;
    };

    CADSceneSegmenter() = default;
    ~CADSceneSegmenter() = default;

    /**
     * @brief Configure geometric dimensions of the target table.
     */
    void configure(double half_w, double half_l, double total_height,
                   double min_depth = 0.75, double max_depth = 3.50);

    /**
     * @brief Online execution: Segments the table from depth frame and extracts 1024 points.
     * When rgb_roi is provided, confines 3D point extraction strictly within the RGB bounding box
     * to eliminate all peripheral background noise, flying pixels, and unrelated objects.
     */
    SegmentResult segment(const RgbdFrame& frame,
                          int target_points = 1024,
                          const cv::Rect2f* rgb_roi = nullptr) const;

private:
    // Sub-sample depth pixels and unproject to 3D camera coordinates (optionally bounded by RGB ROI)
    std::vector<cv::Vec3f> extractDepthPoints(const RgbdFrame& frame,
                                              int step = 4,
                                              const cv::Rect2f* rgb_roi = nullptr) const;

    // RANSAC floor plane estimation
    bool fitFloorPlane(const std::vector<cv::Vec3f>& points,
                       cv::Vec4d& out_plane,
                       double& out_height,
                       double& out_tilt_deg) const;

    // Fast 3D grid voxel clustering to isolate table from walls/clutter
    std::vector<std::vector<cv::Vec3f>> clusterCentralTable(const std::vector<cv::Vec3f>& candidates,
                                                            cv::Rect2f& out_roi,
                                                            const CameraIntrinsics& intr,
                                                            int img_w, int img_h,
                                                            const cv::Vec4d& floor_plane = cv::Vec4d(0, -1, 0, 0),
                                                            bool floor_detected = false) const;

private:
    double _half_w = 0.220;         // Table half width (0.44m / 2)
    double _half_l = 0.415;         // Table half length (0.83m / 2)
    double _total_h = 0.5050;       // Table total height (0.505m)
    double _min_depth_m = 0.75;     // Min working range (table is at >=0.75m)
    double _max_depth_m = 3.50;     // Max working range

    // RANSAC floor parameters (aligned with real_sensor_pipeline.py)
    int _ransac_iterations = 200;
    double _ransac_distance_thresh_m = 0.030; // 30 mm
    double _min_floor_inlier_ratio = 0.10;
    double _min_cam_height_m = 0.15;
    double _max_cam_height_m = 1.80;
};
