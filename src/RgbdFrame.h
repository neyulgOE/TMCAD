#pragma once

#include "RGBDTypes.h"
#include <opencv2/core.hpp>

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
