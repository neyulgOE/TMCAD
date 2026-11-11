#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

struct DetectionObject
{
    int class_id = -1;
    std::string label;
    float confidence = 0.0f;

    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;

    float center_x = 0.0f;
    float center_y = 0.0f;

    bool has_3d = false;
    float x_3d = 0.0f;
    float y_3d = 0.0f;
    float z_3d = 0.0f;

    float depth_m = 0.0f;
    float depth_valid_ratio = 0.0f;
    float depth_std_m = 0.0f;

    bool has_pose_6d = false;
    float roll_deg = 0.0f;
    float pitch_deg = 0.0f;
    float yaw_deg = 0.0f;

    float plane_normal_x = 0.0f;
    float plane_normal_y = 0.0f;
    float plane_normal_z = 0.0f;
    float plane_inlier_ratio = 0.0f;
    float plane_rmse_m = 0.0f;
    float pair_distance_m = 0.0f;
    std::string plane_source;
    std::string pose_frame_id;

    std::string spatial_status;
};

struct DetectionResult
{
    int image_width = 0;
    int image_height = 0;

    std::vector<DetectionObject> objects;

    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms = 0.0;
};

struct LetterboxInfo
{
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;

    int input_w = 0;
    int input_h = 0;

    int src_w = 0;
    int src_h = 0;
};

struct RawDetection
{
    int class_id = -1;
    float score = 0.0f;
    cv::Rect2f box;
};

struct DetectorOutput
{
    std::string detector_id;
    std::string object_type;
    std::string task_type;
    std::string platform_profile;

    std::string model_id;
    std::string model_name;
    std::string model_version;
    std::string backend;
    std::string adapter;
    std::string model_variant;
    std::string artifact_path;

    DetectionResult result;

    std::vector<DetectionObject> debug_raw_objects;

    bool application_requested = false;
    bool application_applied = false;
    std::string application;
    std::string application_status;
    double application_ms = 0.0;
    double depth_timestamp_ms = 0.0;
    int xyz_object_count = 0;
    int pose_object_count = 0;
};
