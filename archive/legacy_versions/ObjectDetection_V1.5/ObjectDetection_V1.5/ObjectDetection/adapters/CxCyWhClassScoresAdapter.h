#pragma once

#include <string>
#include <vector>

#include "IModelAdapter.h"

class CxCyWhClassScoresAdapter : public IModelAdapter
{
public:
    explicit CxCyWhClassScoresAdapter(
        std::string adapter_name = "cxcywh_class_scores_v1");

    bool initialize(const ModelManifest& manifest,
                    const BackendModelInfo& backend_info,
                    const std::vector<std::string>& labels,
                    const AdapterRuntimeOptions& options,
                    std::string& error) override;

    bool preprocess(const cv::Mat& rgb,
                    BackendInput& input,
                    PreprocessContext& context,
                    std::string& error) override;

    bool postprocess(const std::vector<BackendOutput>& outputs,
                     const PreprocessContext& context,
                     DetectionResult& result,
                     std::string& error) override;

    std::string name() const override { return _adapter_name; }

private:
    cv::Mat letterbox(const cv::Mat& rgb, LetterboxInfo& info) const;
    bool buildInputBuffer(const cv::Mat& input_rgb,
                          BackendInput& input,
                          std::string& error) const;

    std::vector<RawDetection> decode(const BackendOutput& output,
                                     const LetterboxInfo& lb,
                                     std::string& error) const;
    std::vector<RawDetection> nms(const std::vector<RawDetection>& detections) const;

    float iou(const cv::Rect2f& a, const cv::Rect2f& b) const;
    bool isClassEnabled(int class_id) const;
    DetectionObject toDetectionObject(const RawDetection& raw) const;

private:
    std::string _adapter_name;
    ModelManifest _manifest;
    BackendModelInfo _backend_info;
    std::vector<std::string> _labels;
    AdapterRuntimeOptions _options;
    int _num_classes = 0;
};
