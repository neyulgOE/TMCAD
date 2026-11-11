#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <robos/core/rbs_core.h>

#include "message_camera.pb.h"
#include "message_object_detection.pb.h"

#include "DetectionManager.h"
#include "IApplication.h"
#include "ObjectDetectionConfig.h"
#include "RgbdFrame.h"

using namespace rbs::core;

class ObjectDetection : public Plugin
{
public:
    explicit ObjectDetection(const std::string& name);
    ~ObjectDetection();

    void run();
    bool detection();

    bool isModelReady() const;
    bool isInitialized() const;
    std::string statusJson() const;

private:
    void loadParams();
    DetectorConfig loadDetectorParams(const std::string& detector_id);
    bool initializeManager();
    bool initializeApplications();

    void message_CameraRGBD(google::protobuf::Message* data);

    bool readLatestRgbd(protocol::Message_CameraRGBD& message,
                        cv::Mat& rgb,
                        RgbdFrame& frame,
                        uint64_t& data_ns,
                        std::string& error);

    void applyApplications(const RgbdFrame* frame,
                           std::vector<DetectorOutput>& outputs);

    void publishDetection(const protocol::Message_CameraRGBD& camera_message,
                          uint64_t data_ns,
                          const DetectorOutput& output);

    std::vector<std::string> parseList(const std::string& text) const;

    static std::string trim(const std::string& value);
    static std::string lower(const std::string& value);
    static std::string jsonEscape(const std::string& value);

    void setStatus(const std::string& code, const std::string& detail);
    bool shouldLogGuard(uint64_t now_ns);

private:
    std::atomic<bool> _running{false};
    std::atomic<bool> _has_new_rgbd{false};
    std::atomic<bool> _rgbd_received{false};
    std::atomic<bool> _initialized{false};
    std::atomic<bool> _model_ready{false};

    mutable std::mutex _status_mutex;
    std::mutex _detection_mutex;

    ObjectDetectionConfig _config;
    std::vector<DetectorConfig> _detector_configs;
    DetectionManager _manager;

    std::map<std::string, std::unique_ptr<IApplication>> _applications;
    std::map<std::string, std::string> _application_init_errors;
    bool _requires_rgbd = false;
    bool _logged_rgbd_metadata = false;

    uint64_t _last_processed_data_ns = 0;
    uint64_t _last_processed_color_frame_number = 0;
    uint64_t _last_guard_log_ns = 0;
    std::map<std::string, uint64_t> _detection_counters;

    std::string _status_code = "not_initialized";
    std::string _status_detail = "plugin has not been initialized";
};
