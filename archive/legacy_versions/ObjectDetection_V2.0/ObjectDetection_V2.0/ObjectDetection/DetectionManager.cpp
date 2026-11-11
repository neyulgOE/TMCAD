#include "DetectionManager.h"

#include <iomanip>
#include <sstream>

bool DetectionManager::initialize(const ObjectDetectionConfig& common_config,
                                  const std::vector<DetectorConfig>& detector_configs)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _config = common_config;
    _detectors.clear();
    _detectors.reserve(detector_configs.size());

    for (const DetectorConfig& config : detector_configs)
    {
        std::unique_ptr<Detector> detector(new Detector());
        std::string error;
        detector->initialize(_config.model_root,
                             _config.backend,
                             _config.platform_profile,
                             config,
                             error);
        _detectors.push_back(std::move(detector));
    }

    return hasReadyDetectorUnlocked();
}

bool DetectionManager::process(const cv::Mat& rgb,
                               uint64_t now_ns,
                               std::vector<DetectorOutput>& outputs)
{
    std::lock_guard<std::mutex> lock(_mutex);
    outputs.clear();

    for (std::unique_ptr<Detector>& detector : _detectors)
    {
        if (!detector->isDue(now_ns)) continue;

        DetectorOutput output;
        std::string error;
        if (detector->detect(rgb, now_ns, output, error))
            outputs.push_back(std::move(output));
    }

    return true;
}

bool DetectionManager::hasReadyDetector() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return hasReadyDetectorUnlocked();
}

size_t DetectionManager::readyCount() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return readyCountUnlocked();
}

bool DetectionManager::hasReadyDetectorUnlocked() const
{
    for (const std::unique_ptr<Detector>& detector : _detectors)
    {
        if (detector->isReady()) return true;
    }
    return false;
}

size_t DetectionManager::readyCountUnlocked() const
{
    size_t count = 0;
    for (const std::unique_ptr<Detector>& detector : _detectors)
    {
        if (detector->isReady()) ++count;
    }
    return count;
}

std::string DetectionManager::statusJson() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    std::ostringstream ss;
    ss << '{';
    ss << "\"detector_count\":" << _detectors.size();
    ss << ",\"ready_count\":" << readyCountUnlocked();
    ss << ",\"detectors\":[";

    for (size_t index = 0; index < _detectors.size(); ++index)
    {
        const Detector& detector = *_detectors[index];
        if (index > 0) ss << ',';

        ss << '{';
        ss << "\"detector_id\":" << jsonEscape(detector.config().detector_id);
        ss << ",\"enabled\":" << (detector.isEnabled() ? "true" : "false");
        ss << ",\"ready\":" << (detector.isReady() ? "true" : "false");
        ss << ",\"status_code\":" << jsonEscape(detector.statusCode());
        ss << ",\"status_detail\":" << jsonEscape(detector.statusDetail());
        ss << ",\"last_runtime_error\":" << jsonEscape(detector.lastRuntimeError());
        ss << ",\"failure_count\":" << detector.failureCount();
        ss << ",\"model_package\":" << jsonEscape(detector.config().model_package);
        ss << ",\"package_path\":" << jsonEscape(detector.packagePath());
        ss << ",\"model_id\":" << jsonEscape(detector.manifest().model_id);
        ss << ",\"model_name\":" << jsonEscape(detector.manifest().model_name);
        ss << ",\"model_version\":" << jsonEscape(detector.manifest().model_version);
        ss << ",\"object_type\":" << jsonEscape(detector.manifest().object_type);
        ss << ",\"task_type\":" << jsonEscape(detector.manifest().task_type);
        ss << ",\"adapter\":" << jsonEscape(detector.manifest().adapter);
        ss << ",\"application\":" << jsonEscape(detector.config().application);
        ss << ",\"model_variant\":" << jsonEscape(detector.resolvedVariant());
        ss << ",\"artifact_path\":" << jsonEscape(detector.artifactPath());
        ss << ",\"interval_ms\":" << detector.config().interval_ms;
        ss << ",\"confidence_threshold\":" << detector.config().conf_threshold;
        ss << ",\"nms_threshold\":" << detector.config().nms_threshold;
        ss << ",\"max_detections\":" << detector.config().max_detections;
        ss << '}';
    }

    ss << "]}";
    return ss.str();
}

std::string DetectionManager::jsonEscape(const std::string& value)
{
    std::ostringstream ss;
    ss << '"';
    for (char c : value)
    {
        switch (c)
        {
        case '\\': ss << "\\\\"; break;
        case '"': ss << "\\\""; break;
        case '\n': ss << "\\n"; break;
        case '\r': ss << "\\r"; break;
        case '\t': ss << "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20)
            {
                ss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                   << static_cast<int>(static_cast<unsigned char>(c));
            }
            else
            {
                ss << c;
            }
        }
    }
    ss << '"';
    return ss.str();
}
