#pragma once

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

/**
 * @brief Multi-view CAD Template Detector for 2D RGB Object Detection.
 * 
 * Supports:
 * - Loading pre-baked binary templates (.cad_templates.bin) for instant (~1ms) startup.
 * - Generating multi-angle templates (36 views) dynamically from any 3D CAD mesh vertices/centroids.
 * - Multi-scale ORB feature matching with CLAHE contrast enhancement.
 * - RANSAC geometric verification (Homography/Affine) to reject false positives.
 * - Pure 2D detection on any image format (PNG, JPG, BMP, WEBP, etc.) without depth requirement.
 */
class CADTemplateDetector
{
public:
    struct CADTemplate
    {
        float yaw_deg = 0.0f;
        cv::Rect2f bbox;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };

    struct TemplateMatch
    {
        cv::Rect2f roi;                          // 2D Bounding Box on RGB image [x, y, w, h]
        float match_score = 0.0f;                // Confidence score [0.0, 1.0]
        int template_index = -1;                 // Best matching template angle index
        float estimated_yaw_deg = 0.0f;          // Estimated object yaw angle in degrees
        int good_matches = 0;                    // Filtered inlier feature matches
        int ransac_inliers = 0;                  // Geometrically consistent inlier matches
        bool valid = false;                      // True if detection meets threshold
        std::string method;                      // Detection method name
        std::vector<cv::Point2f> matched_kpts;   // Inlier keypoints on the query image
        std::vector<cv::Point2f> projected_box;  // 4 transformed corners of the template box
        double latency_ms = 0.0;                 // Processing time in milliseconds
    };

    CADTemplateDetector();
    ~CADTemplateDetector() = default;

    /**
     * @brief Initialize detector for a CAD model.
     * Searches for pre-baked binary templates first; if not found, renders 36 templates from 3D points.
     */
    bool initialize(const std::string& cad_path_or_dir,
                    const std::string& cad_name,
                    const std::vector<cv::Vec3f>& cad_vertices_or_centroids,
                    double half_w, double half_l,
                    double top_z, double bot_z,
                    std::string& error);

    /**
     * @brief Loads pre-baked binary templates (.cad_templates.bin) in ~1 ms.
     */
    bool loadBinaryTemplates(const std::string& bin_path, std::string& error);

    /**
     * @brief Pure 2D image detection on any BGR or RGB image (PNG, JPG, BMP, WEBP, etc.).
     * No depth map and no camera intrinsics required.
     */
    TemplateMatch detect(const cv::Mat& image) const;

    /**
     * @brief Annotates the input image with detection bounding box, HUD telemetry, and keypoints.
     */
    static cv::Mat annotate(const cv::Mat& src,
                            const TemplateMatch& match,
                            const std::string& label,
                            bool show_keypoints = true,
                            bool show_hud = true);

    bool isInitialized() const { return _initialized; }
    size_t templateCount() const { return _templates.size(); }
    void setMinMatchScore(float score) { _min_match_score = score; }
    float minMatchScore() const { return _min_match_score; }
    void setMinInlierMatches(int count) { _min_good_matches = count; }
    int minInlierMatches() const { return _min_good_matches; }

private:
    void renderTemplate(float yaw_deg,
                        const std::vector<cv::Vec3f>& points_3d,
                        const cv::Matx33d& K,
                        CADTemplate& out_tmpl);

private:
    bool _initialized = false;
    int _num_templates = 36;
    float _min_match_score = 0.20f;
    int _min_good_matches = 6;

    double _half_w = 0.220;
    double _half_l = 0.415;
    double _top_z = 0.1665;
    double _bot_z = -0.3355;

    std::vector<CADTemplate> _templates;
    cv::Ptr<cv::ORB> _orb;
    cv::Ptr<cv::BFMatcher> _matcher;
};
