#include "RGBDSource.h"
#include <iostream>

bool FileRGBDSource::grabFrame(RGBDFrame& outFrame) {
    if (!opened_ || served_) return false;

    auto color = std::make_shared<open3d::geometry::Image>();
    auto depth = std::make_shared<open3d::geometry::Image>();

    if (!open3d::io::ReadImage(colorPath_, *color) || color->IsEmpty()) {
        std::cerr << "FileRGBDSource: failed to read color image: " << colorPath_ << "\n";
        return false;
    }
    if (!open3d::io::ReadImage(depthPath_, *depth) || depth->IsEmpty()) {
        std::cerr << "FileRGBDSource: failed to read depth image: " << depthPath_ << "\n";
        return false;
    }

    outFrame.color = color;
    outFrame.depth = depth;
    // depthScale / depthTruncMeters are left at RGBDFrame's defaults --
    // override them on outFrame after this call returns if this dataset
    // uses a different convention (e.g. some datasets store depth in
    // 1/10 mm rather than mm).
    outFrame.timestampUs = 0;

    served_ = true; // single-shot: construct a new FileRGBDSource per pair
    return true;
}

// --- RealSenseSource Implementation ---

#include <librealsense2/rs.hpp>

struct RealSenseSource::Impl {
    rs2::pipeline pipe;
    rs2::config cfg;
    std::unique_ptr<rs2::align> align;
    bool opened = false;
    CameraIntrinsics intrinsics;
    double depth_scale = 1.0;
    int estimated_frames = 0;
};

RealSenseSource::RealSenseSource() : impl_(std::make_unique<Impl>()) {}
RealSenseSource::~RealSenseSource() { close(); }

bool RealSenseSource::open() {
    if (impl_->opened) return true;
    try {
        impl_->cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
        impl_->cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
        
        auto profile = impl_->pipe.start(impl_->cfg);
        impl_->align = std::make_unique<rs2::align>(RS2_STREAM_COLOR);
        
        auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
        impl_->depth_scale = depth_sensor.get_depth_scale();
        
        auto stream_profile = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        rs2_intrinsics intr = stream_profile.get_intrinsics();
        
        impl_->intrinsics.width = intr.width;
        impl_->intrinsics.height = intr.height;
        impl_->intrinsics.fx = intr.fx;
        impl_->intrinsics.fy = intr.fy;
        impl_->intrinsics.cx = intr.ppx;
        impl_->intrinsics.cy = intr.ppy;
        
        // Drop a few frames to let auto-exposure stabilize
        for (int i = 0; i < 15; ++i) {
            impl_->pipe.wait_for_frames();
        }
        
        impl_->opened = true;
        return true;
    } catch (const rs2::error& e) {
        std::cerr << "RealSenseSource open failed: " << e.what() << "\n";
        return false;
    } catch (const std::exception& e) {
        std::cerr << "RealSenseSource open failed: " << e.what() << "\n";
        return false;
    }
}

void RealSenseSource::close() {
    if (impl_->opened) {
        try {
            impl_->pipe.stop();
        } catch (...) {}
        impl_->opened = false;
    }
}

bool RealSenseSource::isOpen() const {
    return impl_->opened;
}

CameraIntrinsics RealSenseSource::intrinsics() const {
    return impl_->intrinsics;
}

bool RealSenseSource::grabFrame(RGBDFrame& outFrame) {
    if (!impl_->opened) return false;
    try {
        rs2::frameset frames = impl_->pipe.wait_for_frames(5000); // 5 seconds timeout
        if (impl_->align) {
            frames = impl_->align->process(frames);
        }
        
        rs2::video_frame color_frame = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();
        
        if (!color_frame || !depth_frame) return false;

        auto color = std::make_shared<open3d::geometry::Image>();
        color->Prepare(color_frame.get_width(), color_frame.get_height(), 3, 1);
        std::memcpy(color->data_.data(), color_frame.get_data(), color->data_.size());

        auto depth = std::make_shared<open3d::geometry::Image>();
        depth->Prepare(depth_frame.get_width(), depth_frame.get_height(), 1, 2); // Z16 = 2 bytes/pixel
        std::memcpy(depth->data_.data(), depth_frame.get_data(), depth->data_.size());

        outFrame.color = color;
        outFrame.depth = depth;
        outFrame.depthScale = 1.0 / impl_->depth_scale;
        outFrame.depthTruncMeters = 3.0; // Default truncation
        outFrame.timestampUs = static_cast<int64_t>(frames.get_timestamp() * 1000.0);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "RealSenseSource grabFrame failed: " << e.what() << "\n";
        return false;
    }
}

std::string RealSenseSource::deviceName() const {
    return "RealSense D400 (Live)";
}

// --- RealSenseBagSource Implementation ---

struct RealSenseBagSource::Impl {
    std::string bag_path;
    rs2::pipeline pipe;
    rs2::config cfg;
    std::unique_ptr<rs2::align> align;
    bool opened = false;
    CameraIntrinsics intrinsics;
    double depth_scale = 1.0;
    int estimated_frames = 0;
};

RealSenseBagSource::RealSenseBagSource(std::string bagPath)
    : impl_(std::make_unique<Impl>()) {
    impl_->bag_path = std::move(bagPath);
}

RealSenseBagSource::~RealSenseBagSource() { close(); }

bool RealSenseBagSource::open() {
    if (impl_->opened) return true;
    try {
        impl_->cfg.enable_device_from_file(impl_->bag_path, false);
        auto profile = impl_->pipe.start(impl_->cfg);
        impl_->align = std::make_unique<rs2::align>(RS2_STREAM_COLOR);

        auto device = profile.get_device();
        auto playback = device.as<rs2::playback>();
        if (playback) {
            playback.set_real_time(false);
            auto duration = playback.get_duration();
            long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
            if (duration_ms > 0) {
                impl_->estimated_frames = std::max(1, static_cast<int>(std::round(duration_ms * 30.0 / 1000.0)));
            }
        }

        try {
            auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
            impl_->depth_scale = depth_sensor.get_depth_scale();
        } catch (...) {
            impl_->depth_scale = 0.001;
        }

        auto stream_profile = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        rs2_intrinsics intr = stream_profile.get_intrinsics();

        impl_->intrinsics.width = intr.width;
        impl_->intrinsics.height = intr.height;
        impl_->intrinsics.fx = intr.fx;
        impl_->intrinsics.fy = intr.fy;
        impl_->intrinsics.cx = intr.ppx;
        impl_->intrinsics.cy = intr.ppy;

        impl_->opened = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "RealSenseBagSource open failed: " << e.what() << "\n";
        return false;
    }
}

void RealSenseBagSource::close() {
    if (impl_->opened) {
        try {
            impl_->pipe.stop();
        } catch (...) {}
        impl_->opened = false;
    }
}

bool RealSenseBagSource::isOpen() const {
    return impl_->opened;
}

CameraIntrinsics RealSenseBagSource::intrinsics() const {
    return impl_->intrinsics;
}

bool RealSenseBagSource::grabFrame(RGBDFrame& outFrame) {
    if (!impl_->opened) return false;
    try {
        rs2::frameset frames = impl_->pipe.wait_for_frames(3000);
        if (impl_->align) {
            frames = impl_->align->process(frames);
        }

        rs2::video_frame color_frame = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();

        if (!color_frame || !depth_frame) return false;

        auto color = std::make_shared<open3d::geometry::Image>();
        color->Prepare(color_frame.get_width(), color_frame.get_height(), 3, 1);
        std::memcpy(color->data_.data(), color_frame.get_data(), color->data_.size());

        auto depth = std::make_shared<open3d::geometry::Image>();
        depth->Prepare(depth_frame.get_width(), depth_frame.get_height(), 1, 2);
        std::memcpy(depth->data_.data(), depth_frame.get_data(), depth->data_.size());

        outFrame.color = color;
        outFrame.depth = depth;
        outFrame.depthScale = 1.0 / impl_->depth_scale;
        outFrame.depthTruncMeters = 3.0;
        outFrame.timestampUs = static_cast<int64_t>(frames.get_timestamp() * 1000.0);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "RealSenseBagSource grabFrame failed: " << e.what() << "\n";
        return false;
    }
}

std::string RealSenseBagSource::deviceName() const {
    return "RealSense (.bag: " + impl_->bag_path + ")";
}

int RealSenseBagSource::estimatedTotalFrames() const {
    return impl_ ? impl_->estimated_frames : 0;
}
