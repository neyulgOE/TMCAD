#pragma once

#include <cstdint>

#include <opencv2/opencv.hpp>

struct CameraIntrinsics
{
    double fx = 0.0;
    double fy = 0.0;
    double ppx = 0.0;
    double ppy = 0.0;

    bool valid() const
    {
        return fx > 1.0 && fy > 1.0;
    }
};

struct RgbdFrame
{
    cv::Mat depth_z16;
    CameraIntrinsics intrinsics;

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t color_stride_bytes = 0;
    uint32_t depth_stride_bytes = 0;

    double depth_scale = 0.0;
    double rgb_timestamp_ms = 0.0;
    double depth_timestamp_ms = 0.0;

    uint64_t color_frame_number = 0;
    uint64_t depth_frame_number = 0;

    int alignment = 0;
    int32_t color_timestamp_domain = 0;
    int32_t depth_timestamp_domain = 0;

    uint64_t data_ns = 0;

    bool valid() const
    {
        return width > 0 &&
               height > 0 &&
               !depth_z16.empty() &&
               depth_z16.type() == CV_16UC1 &&
               depth_z16.cols == static_cast<int>(width) &&
               depth_z16.rows == static_cast<int>(height) &&
               depth_scale > 0.0;
    }
};
