#pragma once

#include <cstdint>
#include <opencv2/opencv.hpp>

struct CameraIntrinsics
{
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double ppx = 0.0;
    double ppy = 0.0;
    double depth_scale = 0.001;

    bool valid() const
    {
        return fx > 1.0 && fy > 1.0;
    }
};

struct RgbdFrame
{
    cv::Mat depth_z16;
    cv::Mat rgb;
    CameraIntrinsics intrinsics;

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t color_stride_bytes = 0;
    uint32_t depth_stride_bytes = 0;

    double depth_scale = 0.001;
    double rgb_timestamp_ms = 0.0;
    double depth_timestamp_ms = 0.0;

    bool valid() const
    {
        return width > 0 &&
               height > 0 &&
               !depth_z16.empty() &&
               depth_z16.type() == CV_16UC1 &&
               depth_scale > 0.0;
    }
};
