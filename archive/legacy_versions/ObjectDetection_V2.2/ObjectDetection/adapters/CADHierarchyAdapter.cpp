#include "CADHierarchyAdapter.h"

#include <fstream>
#include <cstring>

CADHierarchyAdapter::CADHierarchyAdapter(std::string adapter_name)
    : _adapter_name(std::move(adapter_name))
{
}

bool CADHierarchyAdapter::initialize(const ModelManifest& manifest,
                                     const BackendModelInfo& backend_info,
                                     const std::vector<std::string>& labels,
                                     const AdapterRuntimeOptions& options,
                                     std::string& error)
{
    _manifest = manifest;
    _backend_info = backend_info;
    _labels = labels;
    _options = options;
    error.clear();
    return true;
}

bool CADHierarchyAdapter::preprocess(const cv::Mat& /*rgb*/,
                                     BackendInput& input,
                                     PreprocessContext& /*context*/,
                                     std::string& error)
{
    error.clear();
    input.name = "f1";
    input.type = TensorDataType::Float32;
    input.layout = TensorLayout::NCHW;
    input.shape = {1, 7, 1024, 20};
    input.buffer.assign(1 * 7 * 1024 * 20 * sizeof(float), 0);
    return true;
}

bool CADHierarchyAdapter::postprocess(const std::vector<BackendOutput>& outputs,
                                      const PreprocessContext& /*context*/,
                                      DetectionResult& result,
                                      std::string& error)
{
    error.clear();
    result.objects.clear();

    if (outputs.empty() || outputs[0].data.empty())
    {
        return true;
    }

    // Default object placeholder populated; full 6DoF is finalized by CADPoseApplication
    DetectionObject obj;
    obj.class_id = 0;
    obj.label = _labels.empty() ? "table" : _labels[0];
    obj.confidence = 0.50f;
    result.objects.push_back(obj);

    return true;
}
