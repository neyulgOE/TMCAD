#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "IModelAdapter.h"
#include "ModelManifest.h"
#include "InferenceTypes.h"

/**
 * @brief CADHierarchyAdapter: Adapter for CAD-based geometric representation models.
 */
class CADHierarchyAdapter final : public IModelAdapter
{
public:
    explicit CADHierarchyAdapter(std::string adapter_name = "cad_hierarchy_v1");

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

    const std::vector<float>& cadDescriptors() const { return _cad_descriptors; }
    const std::vector<cv::Vec3f>& cadCentroids() const { return _cad_centroids; }

private:
    std::string _adapter_name;
    ModelManifest _manifest;
    BackendModelInfo _backend_info;
    std::vector<std::string> _labels;
    AdapterRuntimeOptions _options;

    std::vector<float> _cad_descriptors;
    std::vector<cv::Vec3f> _cad_centroids;
    int _num_cad_samples = 512;
    int _desc_dim = 128;
};
