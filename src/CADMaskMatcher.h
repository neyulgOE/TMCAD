#pragma once

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>

namespace tmcad {

struct MatchedMaskCandidate {
    cv::Mat mask_u8;          // Full-resolution binary mask (CV_8UC1, 0 or 255)
    cv::Rect bbox;            // Tight 2D bounding box in image coordinates
    cv::Matx33d R_cw_prior{cv::Matx33d::eye()};
    float score{0.0f};
    float dino_sim{0.0f};
    float sil_iou{0.0f};
    int matched_view_idx{-1};
};

class CADMaskMatcher {
public:
    CADMaskMatcher() = default;
    ~CADMaskMatcher() = default;

    bool initBackbones(const std::string& fastsam_onnx_path,
                       const std::string& dinov2_onnx_path);
    bool loadTemplates(const std::string& templates_bin_path);
    void clearTemplates();

    bool isBackbonesReady() const { return _backbones_ready; }
    bool isReady() const { return _backbones_ready && _templates_ready; }

    std::vector<MatchedMaskCandidate> match(const cv::Mat& rgb_bgr, int top_k = 3) const;

private:
    struct TemplateEntry {
        cv::Matx33d R_cw{cv::Matx33d::eye()};
        std::vector<float> feat;       // L2-normalized [feat_dim]
        std::vector<uint8_t> mask_64;  // [mask_size * mask_size], 0 or 1
    };

    std::unique_ptr<Ort::Env> _ort_env;
    std::unique_ptr<Ort::Session> _fastsam_sess;
    std::unique_ptr<Ort::Session> _dinov2_sess;
    std::string _fastsam_in_name;
    std::vector<std::string> _fastsam_out_names;
    std::string _dinov2_in_name;
    std::string _dinov2_out_name;

    std::string _loaded_fastsam_path;
    std::string _loaded_dinov2_path;
    bool _backbones_ready{false};

    uint32_t _feat_dim{384};
    uint32_t _mask_size{64};
    std::vector<TemplateEntry> _templates;
    bool _templates_ready{false};
};

} // namespace tmcad

