#pragma once

// Unified camera intrinsics used across Open3D (CadViewer) and OpenCV (TMVis_Debug / RKNN).
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

    bool valid() const {
        return width > 0 && height > 0 && fx > 0.0 && fy > 0.0;
    }

    double principalX() const {
        return (cx > 0.0) ? cx : ppx;
    }

    double principalY() const {
        return (cy > 0.0) ? cy : ppy;
    }
};

