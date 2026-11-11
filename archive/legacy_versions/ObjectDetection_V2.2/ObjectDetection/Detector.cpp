#include "Detector.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <sys/stat.h>

#include "BackendFactory.h"
#include "ModelAdapterFactory.h"

bool Detector::initialize(const std::string& model_root,
                          const std::string& expected_backend,
                          const std::string& platform_profile,
                          const DetectorConfig& config,
                          std::string& error)
{
    _config = config;
    _manifest = ModelManifest{};
    _backend.reset();
    _adapter.reset();
    _ready = false;
    _failure_count = 0;
    _last_runtime_error.clear();
    _last_run_ns = 0;
    _resolved_platform_profile.clear();
    _resolved_variant.clear();
    _interval_ns = static_cast<uint64_t>(std::max(0, _config.interval_ms)) * 1000000ULL;

    if (!_config.enabled)
    {
        _status_code = "disabled";
        _status_detail = "detector disabled by configuration";
        error.clear();
        return true;
    }

    if (_config.detector_id.empty())
    {
        error = "detector_id is empty";
        setInitError("invalid_config", error);
        return false;
    }
    if (_config.model_package.empty())
    {
        error = "model_package is empty for detector=" + _config.detector_id;
        setInitError("invalid_config", error);
        return false;
    }

    _package_path = joinPath(model_root, _config.model_package);
    _manifest_path = joinPath(_package_path, "model.json");

    if (!dirExists(_package_path))
    {
        error = "model package not found: " + _package_path;
        setInitError("package_not_found", error);
        return false;
    }
    if (!fileExists(_manifest_path))
    {
        error = "model manifest not found: " + _manifest_path;
        setInitError("manifest_not_found", error);
        return false;
    }

    if (!ModelManifestLoader::load(_manifest_path, _manifest, error))
    {
        setInitError("manifest_invalid", error);
        return false;
    }

    const std::string compiled_backend = BackendFactory::compiledBackendName();
    if (compiled_backend == "unknown")
    {
        error = "ObjectDetection backend was not selected at build time";
        setInitError("backend_not_built", error);
        return false;
    }
    if (!expected_backend.empty() && lower(expected_backend) != compiled_backend)
    {
        error = "configured backend=" + expected_backend +
                " does not match compiled backend=" + compiled_backend;
        setInitError("backend_mismatch", error);
        return false;
    }
    const std::string labels_path = joinPath(_package_path, _manifest.labels_file);
    const std::vector<std::string> labels = loadLabels(labels_path);
    if (labels.empty())
    {
        error = "labels file is missing or empty: " + labels_path;
        setInitError("labels_invalid", error);
        return false;
    }

    ResolvedModelArtifact resolved;
    if (!_manifest.resolveArtifact(platform_profile,
                                   compiled_backend,
                                   _config.model_variant,
                                   resolved,
                                   error))
    {
        setInitError("artifact_resolution_failed", error);
        return false;
    }

    _resolved_variant = resolved.variant;
    _resolved_platform_profile = resolved.profile_id;
    _manifest.input = resolved.input;

    _artifact_path = joinPath(_package_path, resolved.relative_path);
    if (!fileExists(_artifact_path))
    {
        error = "model artifact not found: " + _artifact_path;
        setInitError("artifact_file_not_found", error);
        return false;
    }

    _backend = BackendFactory::create(error);
    if (!_backend)
    {
        setInitError("backend_create_failed", error);
        return false;
    }
    if (!_backend->loadModel(_artifact_path, error))
    {
        setInitError("backend_load_failed", error);
        return false;
    }

    _adapter = ModelAdapterFactory::create(_manifest.adapter, error);
    if (!_adapter)
    {
        setInitError("adapter_create_failed", error);
        return false;
    }

    AdapterRuntimeOptions options;
    options.conf_threshold = _config.conf_threshold;
    options.nms_threshold = _config.nms_threshold;
    options.max_detections = _config.max_detections;

    if (!_config.enabled_classes.empty())
    {
        for (const std::string& requested : _config.enabled_classes)
        {
            const std::string wanted = lower(requested);
            bool found = false;
            for (size_t index = 0; index < labels.size(); ++index)
            {
                if (lower(labels[index]) == wanted)
                {
                    options.enabled_class_ids.push_back(static_cast<int>(index));
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                error = "enabled class label not found in model labels: " + requested;
                setInitError("class_filter_invalid", error);
                return false;
            }
        }
    }

    if (!_adapter->initialize(_manifest,
                              _backend->modelInfo(),
                              labels,
                              options,
                              error))
    {
        setInitError("adapter_init_failed", error);
        return false;
    }

    setReady();
    error.clear();
    return true;
}

bool Detector::isDue(uint64_t now_ns) const
{
    if (!_config.enabled || !_ready) return false;
    if (_last_run_ns == 0 || _interval_ns == 0) return true;
    if (now_ns < _last_run_ns) return true;
    return (now_ns - _last_run_ns) >= _interval_ns;
}

bool Detector::detect(const cv::Mat& rgb,
                      uint64_t now_ns,
                      DetectorOutput& output,
                      std::string& error)
{
    if (!isDue(now_ns))
    {
        error.clear();
        return false;
    }

    _last_run_ns = now_ns;
    error.clear();

    BackendInput input;
    PreprocessContext context;
    std::vector<BackendOutput> backend_outputs;
    DetectionResult result;

    const auto t0 = std::chrono::steady_clock::now();
    if (!_adapter->preprocess(rgb, input, context, error))
    {
        ++_failure_count;
        _last_runtime_error = error;
        return false;
    }
    const auto t1 = std::chrono::steady_clock::now();

    if (!_backend->infer(input, backend_outputs, error))
    {
        ++_failure_count;
        _last_runtime_error = error;
        return false;
    }
    const auto t2 = std::chrono::steady_clock::now();

    if (!_adapter->postprocess(backend_outputs, context, result, error))
    {
        ++_failure_count;
        _last_runtime_error = error;
        return false;
    }
    const auto t3 = std::chrono::steady_clock::now();

    result.preprocess_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.inference_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    result.postprocess_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    result.total_ms = std::chrono::duration<double, std::milli>(t3 - t0).count();

    output = DetectorOutput{};
    output.detector_id = _config.detector_id;
    output.object_type = _manifest.object_type;
    output.task_type = _manifest.task_type;
    output.platform_profile = _resolved_platform_profile;
    output.model_id = _manifest.model_id;
    output.model_name = _manifest.model_name;
    output.model_version = _manifest.model_version;
    output.backend = _backend->name();
    output.adapter = _adapter->name();
    output.model_variant = _resolved_variant;
    output.artifact_path = _artifact_path;
    output.result = std::move(result);

    _last_runtime_error.clear();
    return true;
}

std::string Detector::joinPath(const std::string& a, const std::string& b)
{
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (b.front() == '/') return b;
    return a.back() == '/' ? a + b : a + "/" + b;
}

bool Detector::fileExists(const std::string& path)
{
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool Detector::dirExists(const std::string& path)
{
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::vector<std::string> Detector::loadLabels(const std::string& path)
{
    std::vector<std::string> labels;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line))
    {
        while (!line.empty() &&
               (line.back() == '\r' || line.back() == '\n' || line.back() == ' ' ||
                line.back() == '\t'))
        {
            line.pop_back();
        }
        if (!line.empty()) labels.push_back(line);
    }
    return labels;
}

std::string Detector::lower(const std::string& value)
{
    std::string output = value;
    std::transform(output.begin(), output.end(), output.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return output;
}

void Detector::setInitError(const std::string& code, const std::string& detail)
{
    _ready = false;
    _status_code = code;
    _status_detail = detail;
}

void Detector::setReady()
{
    _ready = true;
    _status_code = "ready";
    _status_detail = "detector ready";
}
