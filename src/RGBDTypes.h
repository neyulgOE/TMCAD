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
    double ppx = 0.0;
    double ppy = 0.0;
    double depth_scale = 0.001;

    CameraIntrinsics() = default;
    CameraIntrinsics(int w, int h, double fx_, double fy_, double cx_, double cy_, double depth_scale_ = 0.001)
        : width(w), height(h), fx(fx_), fy(fy_), cx(cx_), cy(cy_), ppx(cx_), ppy(cy_), depth_scale(depth_scale_) {}

    bool valid() const { return width > 0 && height > 0 && fx > 0.0 && fy > 0.0; }

    open3d::camera::PinholeCameraIntrinsic toOpen3D() const {
        double px = (cx > 0.0) ? cx : ppx;
        double py = (cy > 0.0) ? cy : ppy;
        return open3d::camera::PinholeCameraIntrinsic(width, height, fx, fy, px, py);
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
