#pragma once

#include <string>
#include <vector>

#include "IModelAdapter.h"

class DflMultiScaleClassScoresAdapter : public IModelAdapter
{
public:
    explicit DflMultiScaleClassScoresAdapter(
        std::string adapter_name = "dfl_multiscale_class_scores_v1");

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
    struct StrideGroup
    {
        int stride = 0;
        int grid_width = 0;
        int grid_height = 0;
        int box_output_index = -1;
        int cls_output_index = -1;
        int aux_output_index = -1;
        TensorLayout box_layout = TensorLayout::Unknown;
        TensorLayout cls_layout = TensorLayout::Unknown;
    };

    cv::Mat letterbox(const cv::Mat& rgb, LetterboxInfo& info) const;
    bool buildInputBuffer(const cv::Mat& input_rgb,
                          BackendInput& input,
                          std::string& error) const;

    bool buildStrideGroups(std::string& error);
    bool tensorCHW(const TensorInfo& info,
                   int& channels,
                   int& height,
                   int& width) const;

    float elementAt(const BackendOutput& output,
                    TensorLayout layout,
                    int channels,
                    int height,
                    int width,
                    int channel,
                    int y,
                    int x) const;

    float decodeDflDistance(const BackendOutput& box_output,
                            TensorLayout layout,
                            int height,
                            int width,
                            int side,
                            int y,
                            int x,
                            bool& ok) const;

    std::vector<RawDetection> decodeAll(
        const std::vector<BackendOutput>& outputs,
        const LetterboxInfo& lb,
        std::string& error) const;

    std::vector<RawDetection> nms(
        const std::vector<RawDetection>& detections) const;

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
    int _reg_max = 0;
    std::vector<int> _strides;
    std::vector<StrideGroup> _groups;
};
