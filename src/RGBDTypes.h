#pragma once
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <cstdint>
#include <memory>

// Pinhole camera intrinsics, in pixels. fx/fy/cx/cy match the convention
// used by Open3D's PinholeCameraIntrinsic and by both RealSense and Azure
// Kinect SDKs (after their own factory-calibration structs are converted).
struct CameraIntrinsics {
    int width = 0;
    int height = 0;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;

    bool valid() const { return width > 0 && height > 0 && fx > 0.0 && fy > 0.0; }

    open3d::camera::PinholeCameraIntrinsic toOpen3D() const {
        return open3d::camera::PinholeCameraIntrinsic(width, height, fx, fy, cx, cy);
    }
};

// One synchronized color+depth pair pulled from a live camera or a file.
// depth is expected in the SAME convention Open3D's RGBDImage wants: a
// single-channel image where raw_value / depthScale = meters (e.g. raw
// uint16 depth in millimeters -> depthScale = 1000.0).
struct RGBDFrame {
    std::shared_ptr<open3d::geometry::Image> color;
    std::shared_ptr<open3d::geometry::Image> depth;
    double depthScale = 1000.0;     // divide raw depth by this to get meters
    double depthTruncMeters = 3.0;  // ignore depth beyond this (sensor gets noisy far out)
    int64_t timestampUs = 0;

    bool valid() const {
        return color && depth && !color->IsEmpty() && !depth->IsEmpty();
    }
};
