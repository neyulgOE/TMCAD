#pragma once

#include <string>
#include <vector>

struct ObjectDetectionConfig
{
    bool enabled = true;
    bool enable_perf_log = true;
    int perf_log_every_n = 300;
    bool debug_publish_raw_objects = false;
    int loop_sleep_ms = 20;

    std::string backend;
    std::string platform_profile;
    bool strict_config_validation = true;

    std::string model_root = "./models";
    std::string detectors_text;
    std::vector<std::string> detector_ids;

    std::string source_name = "CameraRGBD";
    std::string image_encoding = "bgr8";

    int default_interval_ms = 70;
    float default_conf_threshold = 0.45f;
    float default_nms_threshold = 0.45f;
    int default_max_detections = 50;
};

struct DetectorConfig
{
    std::string detector_id;
    bool enabled = true;
    std::string model_package;
    std::string model_variant = "default";

    int interval_ms = 70;
    float conf_threshold = 0.45f;
    float nms_threshold = 0.45f;
    int max_detections = 50;

    std::string enabled_classes_text;
    std::vector<std::string> enabled_classes;
    std::string application = "none";
};
