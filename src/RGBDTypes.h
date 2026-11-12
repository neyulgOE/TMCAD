#pragma once
#include "CameraIntrinsics.h"
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <cstdint>
#include <memory>

inline open3d::camera::PinholeCameraIntrinsic toOpen3DIntrinsic(const CameraIntrinsics& intr) {
    return open3d::camera::PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.principalX(), intr.principalY());
}

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
