#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "DetectionTypes.h"
#include "IInferenceBackend.h"
#include "IModelAdapter.h"
#include "ModelManifest.h"
#include "ObjectDetectionConfig.h"

class Detector
{
public:
    Detector() = default;
    ~Detector() = default;

    bool initialize(const std::string& model_root,
                    const std::string& expected_backend,
                    const std::string& platform_profile,
                    const DetectorConfig& config,
                    std::string& error);

    bool isDue(uint64_t now_ns) const;
    bool detect(const cv::Mat& rgb,
                uint64_t now_ns,
                DetectorOutput& output,
                std::string& error);

    bool isEnabled() const { return _config.enabled; }
    bool isReady() const { return _ready; }

    const DetectorConfig& config() const { return _config; }
    const ModelManifest& manifest() const { return _manifest; }

    const std::string& packagePath() const { return _package_path; }
    const std::string& artifactPath() const { return _artifact_path; }
    const std::string& resolvedVariant() const { return _resolved_variant; }
    const std::string& resolvedPlatformProfile() const { return _resolved_platform_profile; }
    const std::string& statusCode() const { return _status_code; }
    const std::string& statusDetail() const { return _status_detail; }
    const std::string& lastRuntimeError() const { return _last_runtime_error; }
    uint64_t failureCount() const { return _failure_count; }

private:
    static std::string joinPath(const std::string& a, const std::string& b);
    static bool fileExists(const std::string& path);
    static bool dirExists(const std::string& path);
    static std::vector<std::string> loadLabels(const std::string& path);
    static std::string lower(const std::string& value);

    void setInitError(const std::string& code, const std::string& detail);
    void setReady();

private:
    DetectorConfig _config;
    ModelManifest _manifest;

    std::unique_ptr<IInferenceBackend> _backend;
    std::unique_ptr<IModelAdapter> _adapter;

    std::string _package_path;
    std::string _manifest_path;
    std::string _artifact_path;
    std::string _resolved_variant;
    std::string _resolved_platform_profile;

    uint64_t _interval_ns = 0;
    uint64_t _last_run_ns = 0;

    bool _ready = false;
    std::string _status_code = "not_initialized";
    std::string _status_detail = "detector has not been initialized";
    std::string _last_runtime_error;
    uint64_t _failure_count = 0;
};
