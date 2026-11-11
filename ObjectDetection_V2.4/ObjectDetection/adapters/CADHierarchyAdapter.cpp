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

bool CADHierarchyAdapter::postprocess(const std::vector<BackendOutput>& /*outputs*/,
                                      const PreprocessContext& /*context*/,
                                      DetectionResult& result,
                                      std::string& error)
{
    error.clear();
    result.objects.clear();
    // 3D point cloud detection and 6DoF pose estimation are executed
    // by CADPoseApplication. If the object is not verified by RANSAC,
    // result.objects remains empty to prevent false positives.
    return true;
}
