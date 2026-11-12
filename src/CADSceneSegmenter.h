#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include "RgbdFrame.h"

namespace tmcad {

struct SegmentedCluster {
    std::vector<cv::Vec3f> points;
    std::vector<cv::Vec3f> dense_points;
    cv::Vec3f centroid{0.0f, 0.0f, 0.0f};
    cv::Vec3f bbox_min{0.0f, 0.0f, 0.0f};
    cv::Vec3f bbox_max{0.0f, 0.0f, 0.0f};
    double score{0.0};
};

struct SegmentationResult {
    bool valid{false};
    bool floor_detected{false};
    size_t raw_points_count{0};
    double floor_pitch_deg{0.0};
    cv::Vec4d floor_plane{0.0, -1.0, 0.0, 0.0}; // nx, ny, nz, d (nx*X + ny*Y + nz*Z + d = 0)
    std::vector<cv::Vec3f> object_points;       // Sampled 1024 points for primary/merged target
    std::vector<cv::Vec3f> dense_object_points; // Unsampled dense cluster points for ICP
    std::vector<SegmentedCluster> multi_clusters; // Top-K separated clusters for Multi-Instance (e.g. 2 BBQ Sauce bottles)
    std::vector<cv::Vec3f> foreground_points;   // Unclipped foreground points after support-plane removal
};

class CADSceneSegmenter {
public:
    CADSceneSegmenter() = default;
    ~CADSceneSegmenter() = default;

    // Load scale-adaptive segmentation parameters from <CAD_NAME>_profile.json
    bool loadProfileConfig(const std::string& profile_json_path);

    // Setters for manual or profile-driven configuration
    void setDepthRange(float min_depth_m, float max_depth_m);
    void setBorderMarginRatio(float left_margin, float right_margin);
    void setExpectedObjectDimensions(float length_m, float width_m, float height_m, bool floor_standing);
    void setMultiInstanceMode(bool enable, int max_instances = 4);

    // Main segmentation entry point
    SegmentationResult segment(const cv::Mat& depth_z16,
                               const CameraIntrinsics& intr,
                               const cv::Rect& roi_hint = cv::Rect(),
                               size_t target_num_points = 1024) const;

private:
    void extractDepthPoints(const cv::Mat& depth_z16,
                            const CameraIntrinsics& intr,
                            const cv::Rect& roi_hint,
                            std::vector<cv::Vec3f>& out_pts) const;

    bool fitSupportPlaneRANSAC(const std::vector<cv::Vec3f>& pts,
                               bool has_roi_hint,
                               cv::Vec4d& out_plane,
                               std::vector<cv::Vec3f>& out_above_floor) const;

    void clusterAndMergeTarget(const std::vector<cv::Vec3f>& fg_pts,
                               bool has_roi_hint,
                               std::vector<cv::Vec3f>& out_merged_dense,
                               std::vector<SegmentedCluster>& out_multi_clusters) const;

    std::vector<cv::Vec3f> samplePointsCurvatureAware(const std::vector<cv::Vec3f>& pts,
                                                      size_t target_count) const;

private:
    float _min_depth_m{0.35f};
    float _max_depth_m{2.10f};          // Tightened from 3.50m to 2.10m to reject background people/chairs under table
    float _left_margin_ratio{0.08f};    // Reject left-border aluminum post artifact (x < 0.08 * W)
    float _right_margin_ratio{0.04f};   // Reject right-border edge noise (x > 0.96 * W)

    float _expected_length_m{0.830f};   // Default TABLE-DEMO length (updated dynamically via profile)
    float _expected_width_m{0.440f};    // Default TABLE-DEMO width
    float _expected_height_m{0.505f};   // Default TABLE-DEMO height
    float _bounding_diag_m{1.066f};     // Updated from realBoundingDiag in <CAD_NAME>_profile.json

    bool _floor_standing_mode{true};    // True: co-box merging for hollow frames; False: free 6DoF multi-instance
    bool _enable_multi_instance{true};
    int _max_instances{4};
};

} // namespace tmcad

