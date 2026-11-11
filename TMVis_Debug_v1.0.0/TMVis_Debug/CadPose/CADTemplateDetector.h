#pragma once

#include <vector>
#include <string>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "RgbdFrame.h"

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

    bool initialize(const std::string& cad_dir,
                    const std::vector<cv::Vec3f>& cad_centroids,
                    double half_w, double half_l,
                    double top_z, double bot_z,
                    std::string& error);

    bool loadBinaryTemplates(const std::string& bin_path, std::string& error);

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

    void renderTemplate(float yaw_deg,
                        const std::vector<cv::Vec3f>& points_3d,
                        const cv::Matx33d& K,
                        CADTemplate& out_tmpl);

    TemplateMatch detectByGeometricElevation(const cv::Mat& depth_z16,
                                             double depth_scale,
                                             const CameraIntrinsics& intr) const;

private:
    std::vector<CADTemplate> _templates;
    cv::Ptr<cv::ORB> _orb;
    cv::Ptr<cv::BFMatcher> _matcher;

    int _num_templates = 36;
    float _min_match_score = 0.15f;
    int _min_good_matches = 8;
    bool _initialized = false;

    double _half_w = 0.220;
    double _half_l = 0.415;
    double _top_z = 0.1665;
    double _bot_z = -0.3355;
};
