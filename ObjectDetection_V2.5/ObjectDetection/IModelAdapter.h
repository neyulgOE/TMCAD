#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "DetectionTypes.h"
#include "InferenceTypes.h"
#include "ModelManifest.h"

struct AdapterRuntimeOptions
{
    float conf_threshold = 0.45f;
    float nms_threshold = 0.45f;
    int max_detections = 50;
    std::vector<int> enabled_class_ids;
};

class IModelAdapter
{
public:
    virtual ~IModelAdapter() = default;

    virtual bool initialize(const ModelManifest& manifest,
                            const BackendModelInfo& backend_info,
                            const std::vector<std::string>& labels,
                            const AdapterRuntimeOptions& options,
                            std::string& error) = 0;

    virtual bool preprocess(const cv::Mat& rgb,
                            BackendInput& input,
                            PreprocessContext& context,
                            std::string& error) = 0;

    virtual bool postprocess(const std::vector<BackendOutput>& outputs,
                             const PreprocessContext& context,
                             DetectionResult& result,
                             std::string& error) = 0;

    virtual std::string name() const = 0;
};
