#pragma once

#include <vector>
#include <string>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "RgbdFrame.h"

/**
 * @brief CADTemplateDetector: Multi-view CAD Template Detector for 2D RGB guidance.
 *
 * Implements:
 * - Offline / Init: Multi-angle CAD wireframe/surface rendering and ORB feature indexing.
 * - Online / Realtime: Fast ORB matching against CAD template library to locate table ROI.
 * - Robust Fallback: Geometric elevation clustering on depth frame when visual texture is low.
 */
class CADTemplateDetector
{
public:
    struct TemplateMatch
    {
        cv::Rect2f roi;                 // 2D Bounding Box on RGB image
        float match_score = 0.0f;       // Confidence score [0.0, 1.0]
        int template_index = -1;        // Best matching template angle index
        float estimated_yaw_deg = 0.0f; // Preliminary yaw angle from template
        int good_matches = 0;           // Number of filtered inlier feature matches
        bool valid = false;             // True if detection meets threshold
        std::string method;             // "rgb_orb_template" or "geometric_depth_cluster"
    };

    CADTemplateDetector() = default;
    ~CADTemplateDetector() = default;

    /**
     * @brief Offline/Initialization: Renders or loads pre-baked CAD multi-angle templates.
     */
    bool initialize(const std::string& cad_dir,
                    const std::vector<cv::Vec3f>& cad_centroids,
                    double half_w, double half_l,
                    double top_z, double bot_z,
                    std::string& error);

    /**
     * @brief Loads pre-baked binary templates (.cad_templates.bin) in ~1-2 ms without rendering.
     */
    bool loadBinaryTemplates(const std::string& bin_path, std::string& error);

    /**
     * @brief Online/Per-frame: Detects object in RGB image and returns candidate ROI.
     */
    TemplateMatch detect(const cv::Mat& rgb,
                         const cv::Mat& depth_z16,
                         double depth_scale,
                         const CameraIntrinsics& intr) const;

    void setMinMatchScore(float score) { _min_match_score = score; }
    float minMatchScore() const { return _min_match_score; }

    bool isInitialized() const { return _initialized; }
    size_t templateCount() const { return _templates.size(); }

private:
    struct CADTemplate
    {
        float yaw_deg = 0.0f;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cv::Rect2f bbox;
    };

    // Render wireframe/surface silhouette of CAD model at specified yaw angle
    void renderTemplate(float yaw_deg,
                        const std::vector<cv::Vec3f>& points_3d,
                        const cv::Matx33d& K,
                        CADTemplate& out_tmpl);

    // Fallback: Geometric elevation clustering on depth frame
    TemplateMatch detectByGeometricElevation(const cv::Mat& depth_z16,
                                             double depth_scale,
                                             const CameraIntrinsics& intr) const;

private:
    std::vector<CADTemplate> _templates;
    cv::Ptr<cv::ORB> _orb;
    cv::Ptr<cv::BFMatcher> _matcher;

    int _num_templates = 36;        // Every 10 degrees from 0 to 350
    float _min_match_score = 0.18f;  // Minimum score to accept detection
    int _min_good_matches = 12;      // Minimum inliers after ratio test
    bool _initialized = false;

    // CAD bounding box geometry (Tabletop: 0.44m x 0.83m)
    double _half_w = 0.220;
    double _half_l = 0.415;
    double _top_z = 0.1665;
    double _bot_z = -0.3355;
};
