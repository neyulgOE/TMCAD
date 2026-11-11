#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "DetectionTypes.h"
#include "Detector.h"
#include "ObjectDetectionConfig.h"

class DetectionManager
{
public:
    bool initialize(const ObjectDetectionConfig& common_config,
                    const std::vector<DetectorConfig>& detector_configs);

    bool process(const cv::Mat& rgb,
                 uint64_t now_ns,
                 std::vector<DetectorOutput>& outputs);

    bool hasReadyDetector() const;
    size_t readyCount() const;
    size_t detectorCount() const { return _detectors.size(); }

    std::string statusJson() const;

private:
    bool hasReadyDetectorUnlocked() const;
    size_t readyCountUnlocked() const;
    static std::string jsonEscape(const std::string& value);

private:
    mutable std::mutex _mutex;
    ObjectDetectionConfig _config;
    std::vector<std::unique_ptr<Detector>> _detectors;
};
