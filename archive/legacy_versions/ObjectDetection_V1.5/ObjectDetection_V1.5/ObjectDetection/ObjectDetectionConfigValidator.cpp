#include "ObjectDetectionConfigValidator.h"

#include <algorithm>
#include <set>
#include <utility>

bool ObjectDetectionConfigValidator::validate(
    const ObjectDetectionConfig& common,
    const std::vector<DetectorConfig>& detectors,
    std::vector<ObjectDetectionConfigIssue>& issues)
{
    issues.clear();

    if (!common.enabled) return true;

    if (common.backend.empty() || common.backend == "unknown")
        add(issues, "", "backend_invalid",
            "backend is empty or was not selected at build time");

    if (common.platform_profile.empty())
        add(issues, "", "platform_profile_missing",
            "ObjectDetection.platform_profile is required");

    if (common.model_root.empty())
        add(issues, "", "model_root_missing",
            "ObjectDetection.model_root is empty");

    if (common.source_name != "CameraRGBD")
        add(issues, "", "source_contract_invalid",
            "ObjectDetection input source must be CameraRGBD");

    if (common.image_encoding != "bgr8" && common.image_encoding != "rgb8")
        add(issues, "", "image_encoding_invalid",
            "image_encoding must be bgr8 or rgb8");

    if (detectors.empty())
        add(issues, "", "detectors_empty",
            "no detector instance is configured");

    std::set<std::string> ids;
    size_t enabled_count = 0;

    for (const DetectorConfig& detector : detectors)
    {
        if (detector.detector_id.empty())
        {
            add(issues, "", "detector_id_missing", "detector instance ID is empty");
            continue;
        }

        if (!ids.insert(detector.detector_id).second)
            add(issues, detector.detector_id, "detector_id_duplicate",
                "detector_id must be unique");

        if (!detector.enabled) continue;
        ++enabled_count;

        if (detector.model_package.empty())
            add(issues, detector.detector_id, "model_package_missing",
                "model_package is empty");

        if (detector.application.empty())
            add(issues, detector.detector_id, "application_invalid",
                "application must be none or a registered application ID");

    }

    if (enabled_count == 0)
        add(issues, "", "detectors_disabled",
            "all configured detector instances are disabled");

    return std::none_of(
        issues.begin(), issues.end(),
        [](const ObjectDetectionConfigIssue& issue) { return issue.fatal; });
}

void ObjectDetectionConfigValidator::add(
    std::vector<ObjectDetectionConfigIssue>& issues,
    const std::string& detector_id,
    const std::string& code,
    const std::string& detail,
    bool fatal)
{
    ObjectDetectionConfigIssue issue;
    issue.detector_id = detector_id;
    issue.code = code;
    issue.detail = detail;
    issue.fatal = fatal;
    issues.push_back(std::move(issue));
}
