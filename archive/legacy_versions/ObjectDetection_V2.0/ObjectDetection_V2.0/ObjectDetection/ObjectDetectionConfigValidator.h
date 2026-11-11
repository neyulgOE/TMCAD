#pragma once

#include <string>
#include <vector>

#include "ObjectDetectionConfig.h"

struct ObjectDetectionConfigIssue
{
    std::string detector_id;
    std::string code;
    std::string detail;
    bool fatal = true;
};

class ObjectDetectionConfigValidator
{
public:
    static bool validate(const ObjectDetectionConfig& common,
                         const std::vector<DetectorConfig>& detectors,
                         std::vector<ObjectDetectionConfigIssue>& issues);

private:
    static void add(std::vector<ObjectDetectionConfigIssue>& issues,
                    const std::string& detector_id,
                    const std::string& code,
                    const std::string& detail,
                    bool fatal = true);
};
