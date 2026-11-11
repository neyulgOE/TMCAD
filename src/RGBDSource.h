#pragma once
#include "RGBDTypes.h"
#include <string>

// Abstract live-camera source. Concrete backends (RealSense, Azure Kinect)
// each wrap their own vendor SDK behind this same interface, so
// ClassicalPoseEstimator / MainWindow never #include librealsense2 or k4a
// directly -- swapping camera hardware means swapping which concrete class
// gets constructed, nothing downstream changes.
//
// NOTE: this is a Phase-2 SKELETON. The concrete backends below are stubs
// with the vendor SDK calls sketched out as comments -- wire in the real
// SDK (and add it + its CMake package to the build) once hardware is on
// the bench. Until then, use FileRGBDSource to build/test everything else.
class IRGBDSource {
public:
    virtual ~IRGBDSource() = default;

    // Opens the device / starts streaming. Returns false on failure
    // (device not found, permissions, already in use, etc).
    virtual bool open() = 0;
    virtual void close() = 0;
    virtual bool isOpen() const = 0;

    // Intrinsics are only guaranteed valid AFTER a successful open() --
    // most SDKs only expose factory calibration once streaming starts.
    virtual CameraIntrinsics intrinsics() const = 0;

    // Blocking grab of the next synchronized color+depth pair. Returns
    // false on timeout/disconnect; caller should treat that as "skip this
    // frame", not necessarily fatal (matches how RealSense/Kinect SDKs
    // themselves report transient frame drops).
    virtual bool grabFrame(RGBDFrame& outFrame) = 0;

    virtual std::string deviceName() const = 0;
};

class RealSenseSource : public IRGBDSource {
public:
    RealSenseSource();
    ~RealSenseSource() override;

    bool open() override;
    void close() override;
    bool isOpen() const override;
    CameraIntrinsics intrinsics() const override;
    bool grabFrame(RGBDFrame& outFrame) override;
    std::string deviceName() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class RealSenseBagSource : public IRGBDSource {
public:
    explicit RealSenseBagSource(std::string bagPath);
    ~RealSenseBagSource() override;

    bool open() override;
    void close() override;
    bool isOpen() const override;
    CameraIntrinsics intrinsics() const override;
    bool grabFrame(RGBDFrame& outFrame) override;
    std::string deviceName() const override;
    int estimatedTotalFrames() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// --- Azure Kinect (DK) -------------------------------------------------------
// Real implementation needs k4a.h: k4a_device_open, k4a_device_start_cameras
// with K4A_IMAGE_FORMAT_COLOR_BGRA32 + depth in NFOV/WFOV mode, and
// k4a_transformation_depth_image_to_color_camera to align depth into the
// color frame (the Kinect equivalent of rs2::align above).
class AzureKinectSource : public IRGBDSource {
public:
    bool open() override {
        // TODO: k4a_device_open(K4A_DEVICE_DEFAULT, &device_);
        //   build k4a_device_configuration_t (color+depth formats/fps);
        //   k4a_device_start_cameras(device_, &config);
        //   k4a_device_get_calibration(...) -> intrinsics_;
        //   cache the k4a_transformation_t handle for depth->color alignment.
        opened_ = false;
        return opened_;
    }
    void close() override { opened_ = false; }
    bool isOpen() const override { return opened_; }
    CameraIntrinsics intrinsics() const override { return intrinsics_; }
    bool grabFrame(RGBDFrame& /*outFrame*/) override {
        // TODO: k4a_device_get_capture(device_, &capture, timeout_ms);
        //   k4a_capture_get_color_image / get_depth_image;
        //   k4a_transformation_depth_image_to_color_camera(...) to align;
        //   copy into open3d::geometry::Image.
        return false;
    }
    std::string deviceName() const override { return "Azure Kinect (stub, SDK not linked)"; }

private:
    bool opened_ = false;
    CameraIntrinsics intrinsics_;
};

// --- File-based source, for developing/testing the pipeline without any
// camera plugged in. Reads one pre-captured color+depth image pair (you
// supply the intrinsics, e.g. from the dataset's calibration file). Swap
// this in for RealSenseSource/AzureKinectSource anywhere a IRGBDSource is
// expected, until hardware is on the bench.
//
// Single-shot by design (one grabFrame() call serves the pair, then
// returns false) -- construct one instance per frame you want to replay,
// or extend this into a directory/sequence reader if you need to iterate
// a whole recorded run.
class FileRGBDSource : public IRGBDSource {
public:
    FileRGBDSource(std::string colorPath, std::string depthPath, CameraIntrinsics intr)
        : colorPath_(std::move(colorPath)), depthPath_(std::move(depthPath)), intrinsics_(intr) {}

    bool open() override { opened_ = true; served_ = false; return true; }
    void close() override { opened_ = false; }
    bool isOpen() const override { return opened_; }
    CameraIntrinsics intrinsics() const override { return intrinsics_; }
    bool grabFrame(RGBDFrame& outFrame) override;
    std::string deviceName() const override { return "File(" + colorPath_ + ")"; }

private:
    std::string colorPath_;
    std::string depthPath_;
    CameraIntrinsics intrinsics_;
    bool opened_ = false;
    bool served_ = false;
};
