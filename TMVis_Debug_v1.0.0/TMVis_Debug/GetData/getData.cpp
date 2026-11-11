#include "getData.h"
#include "TablePoseEstimator.h"

#include <opencv2/opencv.hpp>

#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace
{
    CameraIntrinsics toCameraIntrinsics(const tmv::Intrinsics& intr, int w = 640, int h = 480)
    {
        CameraIntrinsics out;
        if (intr.fx > 1.0f && intr.fy > 1.0f) {
            out.fx = intr.fx;
            out.fy = intr.fy;
            out.ppx = intr.ppx;
            out.ppy = intr.ppy;
        } else {
            out.fx = 607.0 * (static_cast<double>(w) / 640.0);
            out.fy = 607.0 * (static_cast<double>(h) / 480.0);
            out.ppx = static_cast<double>(w) / 2.0;
            out.ppy = static_cast<double>(h) / 2.0;
        }
        return out;
    }

    bool samePixelGrid(const tmv::DepthFrame& depth, const tmv::ImageFrame& rgb)
    {
        // Identical calibrated intrinsics are the only safe implicit evidence
        // that the publisher has already registered depth onto colour.  A
        // matching 640x480 resolution alone is not registration.
        constexpr double kIntrinsicsTolerancePx = 0.25;
        if (depth.intr.fx <= 1.0f || depth.intr.fy <= 1.0f ||
            rgb.intr.fx <= 1.0f || rgb.intr.fy <= 1.0f) return false;
        return depth.width == rgb.width && depth.height == rgb.height &&
               std::abs(depth.intr.fx  - rgb.intr.fx)  <= kIntrinsicsTolerancePx &&
               std::abs(depth.intr.fy  - rgb.intr.fy)  <= kIntrinsicsTolerancePx &&
               std::abs(depth.intr.ppx - rgb.intr.ppx) <= kIntrinsicsTolerancePx &&
               std::abs(depth.intr.ppy - rgb.intr.ppy) <= kIntrinsicsTolerancePx;
    }

    cv::Matx33d depthToColorRotation(const tmv::DepthFrame& depth)
    {
        const auto& r = depth.depth_to_color_rotation;
        return cv::Matx33d(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]);
    }

    cv::Vec3d depthToColorTranslation(const tmv::DepthFrame& depth)
    {
        const auto& t = depth.depth_to_color_translation;
        return cv::Vec3d(t[0], t[1], t[2]);
    }

    cv::Mat toDepth16(const tmv::DepthFrame& frame)
    {
        if (!frame.valid || frame.width == 0 || frame.height == 0 || frame.depth_z16.empty()) {
            return {};
        }

        const int width = static_cast<int>(frame.width);
        const int height = static_cast<int>(frame.height);

        const size_t minCount = static_cast<size_t>(width) * static_cast<size_t>(height);
        if (frame.depth_z16.size() < minCount) {
            return {};
        }

        const size_t step = frame.stride_bytes > 0
            ? frame.stride_bytes
            : static_cast<uint32_t>(width * sizeof(uint16_t));

        cv::Mat raw16(height, width, CV_16UC1, const_cast<uint16_t*>(frame.depth_z16.data()), step);
        return raw16.clone();
    }

    cv::Mat toDepth16(const tmv::ImageFrame& frame)
    {
        if (!frame.valid || frame.width == 0 || frame.height == 0 || !frame.hasDepth()) return {};
        const int width = static_cast<int>(frame.width);
        const int height = static_cast<int>(frame.height);
        const size_t step = frame.depth_stride_bytes > 0
            ? frame.depth_stride_bytes
            : static_cast<uint32_t>(width * sizeof(uint16_t));
        const size_t required_bytes = static_cast<size_t>(step) * (height - 1) +
            static_cast<size_t>(width) * sizeof(uint16_t);
        if (frame.depth_z16.size() * sizeof(uint16_t) < required_bytes) return {};
        cv::Mat raw16(height, width, CV_16UC1,
                      const_cast<uint16_t*>(frame.depth_z16.data()), step);
        return raw16.clone();
    }
    cv::Mat toBGR(const tmv::ImageFrame& frame)
    {
        if (!frame.valid || frame.width == 0 || frame.height == 0 || frame.bgr8.empty()) {
            return {};
        }

        const int w = static_cast<int>(frame.width);
        const int h = static_cast<int>(frame.height);

        const size_t minSize = static_cast<size_t>(w) * static_cast<size_t>(h) * 3;
        if (frame.bgr8.size() < minSize) {
            return {};
        }

        const size_t step = frame.stride_bytes > 0 ? frame.stride_bytes : static_cast<uint32_t>(w * 3);
        cv::Mat raw(h, w, CV_8UC3, const_cast<uint8_t*>(frame.bgr8.data()), step);
        return raw.clone();
    }

    cv::Mat toDepthColor(const tmv::DepthFrame& frame, float min_m = 0.2f, float max_m = 3.0f)
    {
        if (!frame.valid || frame.width == 0 || frame.height == 0 || frame.depth_z16.empty()) {
            return {};
        }

        const int width = static_cast<int>(frame.width);
        const int height = static_cast<int>(frame.height);

        const size_t minCount = static_cast<size_t>(width) * static_cast<size_t>(height);
        if (frame.depth_z16.size() < minCount) {
            return {};
        }

        const size_t step = frame.stride_bytes > 0
            ? frame.stride_bytes
            : static_cast<uint32_t>(width * sizeof(uint16_t));

        cv::Mat raw16(height, width, CV_16UC1, const_cast<uint16_t*>(frame.depth_z16.data()), step);
        cv::Mat depth16 = raw16.clone();

        cv::Mat depth8(height, width, CV_8UC1, cv::Scalar(0));
        const float scale = frame.depth_scale > 0.0f ? frame.depth_scale : 0.001f;

        for (int y = 0; y < height; ++y) {
            const uint16_t* src = depth16.ptr<uint16_t>(y);
            uint8_t* dst = depth8.ptr<uint8_t>(y);

            for (int x = 0; x < width; ++x) {
                const uint16_t d = src[x];
                if (d == 0) {
                    dst[x] = 0;
                    continue;
                }

                const float z_m = static_cast<float>(d) * scale;
                if (z_m < min_m || z_m > max_m) {
                    dst[x] = 0;
                    continue;
                }

                float norm = (z_m - min_m) / (max_m - min_m);
                norm = std::clamp(norm, 0.0f, 1.0f);
                dst[x] = static_cast<uint8_t>((1.0f - norm) * 255.0f);
            }
        }

        cv::Mat color;
        cv::applyColorMap(depth8, color, cv::COLORMAP_JET);
        return color;
    }

    void drawMarkers(cv::Mat& img, const tmv::MarkerDetections& detections)
    {
        if (img.empty() || !detections.valid) return;

        for (const auto& m : detections.markers) {
            cv::Point center((int)m.center.u, (int)m.center.v);
            cv::Point c0((int)m.c0.u, (int)m.c0.v);
            cv::Point c1((int)m.c1.u, (int)m.c1.v);
            cv::Point c2((int)m.c2.u, (int)m.c2.v);
            cv::Point c3((int)m.c3.u, (int)m.c3.v);

            cv::line(img, c0, c1, cv::Scalar(0,255,0), 2, cv::LINE_AA);
            cv::line(img, c1, c2, cv::Scalar(0,255,0), 2, cv::LINE_AA);
            cv::line(img, c2, c3, cv::Scalar(0,255,0), 2, cv::LINE_AA);
            cv::line(img, c3, c0, cv::Scalar(0,255,0), 2, cv::LINE_AA);

            cv::circle(img, center, 5, cv::Scalar(0,0,255), -1, cv::LINE_AA);

            std::ostringstream oss;
            oss << "ID=" << m.id;
            if (m.has_xyz) {
                oss << std::fixed << std::setprecision(3)
                    << " X=" << m.x
                    << " Y=" << m.y
                    << " Z=" << m.z;
            }

            cv::putText(img, oss.str(), center + cv::Point(8, -8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0,255,255), 1, cv::LINE_AA);
        }
    }

    void drawSafety(cv::Mat& img, const tmv::SafetyStatus& safety)
    {
        if (img.empty() || !safety.valid) return;

        std::string text = "UNKNOWN";
        cv::Scalar color(255,255,255);

        switch (safety.level) {
            case tmv::SafetyLevel::Stop:
                text = "STOP";
                color = cv::Scalar(0,0,255);
                break;
            case tmv::SafetyLevel::Warning:
                text = "WARNING";
                color = cv::Scalar(0,255,255);
                break;
            case tmv::SafetyLevel::Safe:
                text = "SAFE";
                color = cv::Scalar(0,255,0);
                break;
            default:
                break;
        }

        cv::rectangle(img, cv::Rect(10, 10, 220, 42), cv::Scalar(30,30,30), -1);
        cv::rectangle(img, cv::Rect(10, 10, 220, 42), color, 2);

        cv::putText(img, text, cv::Point(20, 38),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv::LINE_AA);
    }

    void drawImageInfo(cv::Mat& img, const char* label, const tmv::ImageFrame& frame)
    {
        if (img.empty() || !frame.valid) return;

        std::ostringstream oss1;
        oss1 << label << ": " << frame.width << "x" << frame.height;

        std::ostringstream oss2;
        oss2 << std::fixed << std::setprecision(2)
             << "TS(ms): " << frame.rs_timestamp_ms;

        cv::putText(img, oss1.str(), cv::Point(10, 70),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255,255,255), 1, cv::LINE_AA);

        cv::putText(img, oss2.str(), cv::Point(10, 94),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255,255,255), 1, cv::LINE_AA);
    }

    void drawDepthInfo(cv::Mat& img, const tmv::DepthFrame& frame)
    {
        if (img.empty() || !frame.valid) return;

        std::ostringstream oss1;
        oss1 << "Depth: " << frame.width << "x" << frame.height;

        std::ostringstream oss2;
        oss2 << std::fixed << std::setprecision(6)
             << "Scale: " << frame.depth_scale;

        std::ostringstream oss3;
        oss3 << std::fixed << std::setprecision(2)
             << "TS(ms): " << frame.rs_timestamp_ms;

        cv::rectangle(img, cv::Rect(10, 10, 260, 90), cv::Scalar(30,30,30), -1);

        cv::putText(img, oss1.str(), cv::Point(20, 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255,255,255), 1, cv::LINE_AA);

        cv::putText(img, oss2.str(), cv::Point(20, 59),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255,255,255), 1, cv::LINE_AA);

        cv::putText(img, oss3.str(), cv::Point(20, 83),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255,255,255), 1, cv::LINE_AA);
    }
}

GetData::GetData(const std::string &name) : Plugin(name) {}
GetData::~GetData() {}

void GetData::run()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    cv::namedWindow("RGB",   cv::WINDOW_NORMAL);
    cv::namedWindow("RGBD",  cv::WINDOW_NORMAL);
    cv::namedWindow("Depth", cv::WINDOW_NORMAL);

    cv::resizeWindow("RGB",   960, 540);
    cv::resizeWindow("RGBD",  960, 540);
    cv::resizeWindow("Depth", 960, 540);

    // Initialize V2.5 CAD Pose Table Estimator
    TablePoseEstimator estimator;
    estimator.initialize();

    cv::Mat latestDepth16;
    float latestDepthScale = 0.001f;
    CameraIntrinsics latestIntr;
    cv::Mat latestRgbdBgr;
    cv::Mat latestRgbdDepth16;
    float latestRgbdDepthScale = 0.001f;
    CameraIntrinsics latestRgbdIntr;
    double latestRgbdTimestampMs = -1.0;

    double lastDepthTs  = -1.0;
    double lastRgbTs    = -1.0;
    double lastRgbdTs   = -1.0;
    uint64_t lastMarkNs = 0;

    // Track stream states
    bool hasSeenDepth = false;
    bool hasSeenRgb   = false;
    bool hasSeenRgbd  = false;
    int stableFrameCount = 0;

    // One-shot Snapshot state
    bool snapshotTaken = false;
    cv::Mat snapshotDisplayRgb;
    cv::Mat snapshotRgbForRefine;
    cv::Mat snapshotDepthForRefine;
    CameraIntrinsics snapshotIntrForRefine;
    float snapshotDepthScaleForRefine = 0.001f;
    TablePose snapshotPose;

    std::cout << "\n======================================================\n"
              << "🎥 [TMVis_Debug] Cho khoi dong cac luong cam bien...\n"
              << "   Buoc 1: Doi 3 khung hinh RGB, RGBD va Depth cung hien len.\n"
              << "   Buoc 2: Chup 1 khung hinh tinh (One-Shot Snapshot).\n"
              << "   Buoc 3: Chay giai thuat 6DoF va khoa luong kiem tra tinh.\n"
              << "======================================================\n" << std::endl;

    while (true) {
        tmv::DepthFrame      depthFrame;
        tmv::ImageFrame      rgbFrame;
        tmv::ImageFrame      rgbdFrame;
        tmv::MarkerDetections markers;
        tmv::SafetyStatus    safety;

        getSubscriberData(depthFrame);
        getSubscriberData(rgbFrame);
        getSubscriberRGBD(rgbdFrame);
        getSubscriberData(markers);
        getSubscriberData(safety);

        const bool newDepth = depthFrame.valid && depthFrame.rs_timestamp_ms != lastDepthTs;
        const bool newRgb   = rgbFrame.valid && rgbFrame.rs_timestamp_ms   != lastRgbTs;
        const bool newRgbd  = rgbdFrame.valid && rgbdFrame.rs_timestamp_ms  != lastRgbdTs;
        const bool newMark  = markers.valid && markers.header.data_nsec   != lastMarkNs;

        if (newDepth) {
            lastDepthTs = depthFrame.rs_timestamp_ms;
            cv::Mat d16 = toDepth16(depthFrame);
            if (!d16.empty()) {
                latestDepth16 = d16;
                latestDepthScale = depthFrame.depth_scale > 0.0f ? depthFrame.depth_scale : 0.001f;
                latestIntr = toCameraIntrinsics(depthFrame.intr, depthFrame.width, depthFrame.height);
                if (cv::countNonZero(latestDepth16) > 40000) {
                    hasSeenDepth = true;
                }
            }
            cv::Mat depthColor = toDepthColor(depthFrame, 0.2f, 3.0f);
            if (!depthColor.empty()) {
                drawDepthInfo(depthColor, depthFrame);
                cv::imshow("Depth", depthColor);
            }
        }

        if (newRgbd) {
            lastRgbdTs = rgbdFrame.rs_timestamp_ms;
            cv::Mat rgbd = toBGR(rgbdFrame);
            cv::Mat rgbdDepth = toDepth16(rgbdFrame);
            if (!rgbd.empty()) {
                hasSeenRgbd = true;
                // CameraRGBD is the V2.5 contract: colour, depth and its
                // intrinsics describe one registered camera frame.  Never
                // replace this depth with the independent 11000 stream for
                // pose estimation or RGB overlay.
                if (!rgbdDepth.empty()) {
                    latestRgbdBgr = rgbd.clone();
                    latestRgbdDepth16 = rgbdDepth;
                    latestRgbdDepthScale = rgbdFrame.depth_scale;
                    latestRgbdIntr = toCameraIntrinsics(rgbdFrame.intr, rgbdFrame.width, rgbdFrame.height);
                    latestRgbdTimestampMs = rgbdFrame.rs_timestamp_ms;
                }
                drawMarkers(rgbd, markers);
                drawSafety(rgbd, safety);
                drawImageInfo(rgbd, "RGBD", rgbdFrame);
                cv::imshow("RGBD", rgbd);
            }
        }

        if (newMark) lastMarkNs = markers.header.data_nsec;

        if (newRgb) {
            lastRgbTs = rgbFrame.rs_timestamp_ms;
            cv::Mat rgb = toBGR(rgbFrame);
            if (!rgb.empty()) {
                hasSeenRgb = true;
                const bool atomicRgbdReady = !latestRgbdBgr.empty() && !latestRgbdDepth16.empty() &&
                                             latestRgbdIntr.valid() && latestRgbdDepthScale > 0.0f;
                const CameraIntrinsics rgbIntr = toCameraIntrinsics(rgbFrame.intr, rgbFrame.width, rgbFrame.height);
                const double rawPairDeltaMs = depthFrame.valid
                    ? std::abs(depthFrame.rs_timestamp_ms - rgbFrame.rs_timestamp_ms)
                    : std::numeric_limits<double>::infinity();
                // Compatibility only: old publishers that have no atomic
                // CameraRGBD depth may proceed *only* when they explicitly
                // mark depth registered, or expose an identical calibrated
                // pixel grid, and when timestamps are within one frame.
                const bool safeLegacyPair = !latestDepth16.empty() &&
                    rawPairDeltaMs <= 33.0 &&
                    (depthFrame.registered_to_color || samePixelGrid(depthFrame, rgbFrame));

                if (!snapshotTaken) {
                    const bool allStreamsReady = hasSeenRgb && hasSeenRgbd && hasSeenDepth &&
                                                 (atomicRgbdReady || safeLegacyPair);

                    cv::Mat display = rgb.clone();
                    drawSafety(display, safety);
                    drawImageInfo(display, "RGB", rgbFrame);

                    if (!allStreamsReady) {
                        cv::Rect banner(10, display.rows - 50, display.cols - 20, 38);
                        cv::rectangle(display, banner, cv::Scalar(20, 20, 20), -1);
                        cv::rectangle(display, banner, cv::Scalar(0, 165, 255), 2);
                        std::string status = "Cho cap RGBD nguyen tu (BGR+Z16+intr): RGB [";
                        status += (hasSeenRgb ? "OK" : "..");
                        status += "] | RGBD [";
                        status += (hasSeenRgbd ? "OK" : "..");
                        status += "] | Depth [";
                        status += (hasSeenDepth ? "OK" : "..");
                        status += "]";
                        if (!atomicRgbdReady && !safeLegacyPair) {
                            status += " | RGBD chua co Z16/intr; legacy depth chua registered/sync";
                        }
                        cv::putText(display, status, cv::Point(20, display.rows - 25),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
                    } else {
                        stableFrameCount++;
                        cv::Rect banner(10, display.rows - 50, display.cols - 20, 38);
                        cv::rectangle(display, banner, cv::Scalar(20, 20, 20), -1);
                        cv::rectangle(display, banner, cv::Scalar(0, 230, 118), 2);
                        std::string status = "3 LUONG DA DONG BO (On-screen) | Bam [SPACE] hoac doi on dinh de chup";
                        cv::putText(display, status, cv::Point(20, display.rows - 25),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(0, 255, 118), 1, cv::LINE_AA);
                    }
                    cv::imshow("RGB", display);

                    // Auto snapshot after all 3 streams stable on screen for ~45 frames (or SPACE key)
                    if (allStreamsReady && stableFrameCount >= 45) {
                        std::cout << "\n======================================================\n"
                                  << "🎯 [SNAPSHOT TRIGGER] 3 luong da hien len dong thoi va on dinh!\n"
                                  << "   Cap RGB/Depth da dong bo timestamp va calibration.\n"
                                  << "======================================================\n" << std::endl;

                        const cv::Mat& captureRgb = atomicRgbdReady ? latestRgbdBgr : rgb;
                        const cv::Mat& captureDepth = atomicRgbdReady ? latestRgbdDepth16 : latestDepth16;
                        const CameraIntrinsics& captureIntr = atomicRgbdReady ? latestRgbdIntr : rgbIntr;
                        const float captureScale = atomicRgbdReady ? latestRgbdDepthScale : latestDepthScale;
                        const double captureTimestamp = atomicRgbdReady ? latestRgbdTimestampMs : rgbFrame.rs_timestamp_ms;
                        std::cout << "[TMVis_RGBD] Snapshot source="
                                  << (atomicRgbdReady ? "CameraRGBD atomic" : "legacy verified pair")
                                  << " ts=" << captureTimestamp << "ms intr=(fx=" << captureIntr.fx
                                  << ", fy=" << captureIntr.fy << ", cx=" << captureIntr.ppx
                                  << ", cy=" << captureIntr.ppy << ")" << std::endl;
                        cv::imwrite("/home/huyen1905/TMCAD/dump_rgb.png", captureRgb);
                        cv::imwrite("/home/huyen1905/TMCAD/dump_depth.png", captureDepth);
                        std::ofstream f("/home/huyen1905/TMCAD/dump_meta.txt");
                        f << "scale=" << captureScale << "\n"
                          << "fx=" << captureIntr.fx << "\n"
                          << "fy=" << captureIntr.fy << "\n"
                          << "ppx=" << captureIntr.ppx << "\n"
                          << "ppy=" << captureIntr.ppy << "\n"
                          << "timestamp_ms=" << captureTimestamp << "\n"
                          << "registered_pair=" << (atomicRgbdReady ? "atomic_rgbd" : "legacy_verified") << "\n"
                          << "w=" << captureRgb.cols << "\n"
                          << "h=" << captureRgb.rows << "\n";
                        f.close();

                        snapshotPose = estimator.estimateSnapshot(captureRgb, captureDepth,
                                                                   captureScale, captureIntr);
                        snapshotRgbForRefine = captureRgb.clone();
                        snapshotDepthForRefine = captureDepth.clone();
                        snapshotIntrForRefine = captureIntr;
                        snapshotDepthScaleForRefine = captureScale;
                        estimator.transformPoseCameraFrame(snapshotPose,
                                                           cv::Matx33d::eye(), cv::Vec3d(0.0, 0.0, 0.0),
                                                           captureIntr, captureRgb.cols, captureRgb.rows);
                        snapshotDisplayRgb = captureRgb.clone();
                        estimator.drawOverlay(snapshotDisplayRgb, snapshotPose, captureIntr);

                        // Status Tag on snapshot
                        cv::Rect snapTag(10, snapshotDisplayRgb.rows - 46, snapshotDisplayRgb.cols - 20, 36);
                        cv::rectangle(snapshotDisplayRgb, snapTag, cv::Scalar(20, 20, 20), -1);
                        cv::rectangle(snapshotDisplayRgb, snapTag, cv::Scalar(255, 229, 0), 2);
                        cv::putText(snapshotDisplayRgb, "TINH HOA SNAPSHOT HOAN TAT | Bam [SPACE] hoac [R] de chup lai | ESC: Thoat",
                                    cv::Point(20, snapshotDisplayRgb.rows - 23),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(255, 229, 0), 1, cv::LINE_AA);

                        cv::imshow("RGB", snapshotDisplayRgb);
                        cv::imwrite("/home/huyen1905/TMCAD/debug/snapshot_result.png", snapshotDisplayRgb);
                        snapshotTaken = true;
                    }
                } else {
                    // Keep the captured RGB/depth pair fixed, but allow the
                    // asynchronous STL/hierarchy worker to replace the first
                    // neural overlay once its quality gate is satisfied.
                    const cv::Vec3d previousT = snapshotPose.t;
                    const cv::Matx33d previousR = snapshotPose.R;
                    estimator.pollAndScheduleMeshRefinement(snapshotDepthForRefine,
                                                             snapshotDepthScaleForRefine,
                                                             snapshotIntrForRefine,
                                                             snapshotPose);
                    const bool poseChanged =
                        cv::norm(snapshotPose.t - previousT) > 1e-5 ||
                        cv::norm(snapshotPose.R - previousR) > 1e-5;
                    if (poseChanged && !snapshotRgbForRefine.empty()) {
                        estimator.transformPoseCameraFrame(snapshotPose,
                                                           cv::Matx33d::eye(), cv::Vec3d(0.0, 0.0, 0.0),
                                                           snapshotIntrForRefine,
                                                           snapshotRgbForRefine.cols,
                                                           snapshotRgbForRefine.rows);
                        snapshotDisplayRgb = snapshotRgbForRefine.clone();
                        estimator.drawOverlay(snapshotDisplayRgb, snapshotPose, snapshotIntrForRefine);
                        cv::Rect snapTag(10, snapshotDisplayRgb.rows - 46,
                                         snapshotDisplayRgb.cols - 20, 36);
                        cv::rectangle(snapshotDisplayRgb, snapTag, cv::Scalar(20, 20, 20), -1);
                        cv::rectangle(snapshotDisplayRgb, snapTag, cv::Scalar(255, 229, 0), 2);
                        cv::putText(snapshotDisplayRgb,
                                    "STL/HIERARCHY DA CAP NHAT | Bam [R] de chup lai | ESC: Thoat",
                                    cv::Point(20, snapshotDisplayRgb.rows - 23),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.46,
                                    cv::Scalar(255, 229, 0), 1, cv::LINE_AA);
                        cv::imwrite("/home/huyen1905/TMCAD/debug/snapshot_result.png",
                                    snapshotDisplayRgb);
                        std::cout << "[TMVis_RGBD] Snapshot overlay updated from cached STL hierarchy pose"
                                  << std::endl;
                    }
                    cv::imshow("RGB", snapshotDisplayRgb);
                }
            }
        }

        int key = cv::waitKey(16);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        } else if (key == 32 || key == 'r' || key == 'R' || key == 's' || key == 'S') {
            if (snapshotTaken) {
                std::cout << "[TMVis_Debug] Reset snapshot. Preparing to capture new snapshot..." << std::endl;
                snapshotTaken = false;
                stableFrameCount = 0;
            } else if (hasSeenRgb && hasSeenRgbd && hasSeenDepth) {
                stableFrameCount = 1000;
            }
        }
    }

    cv::destroyAllWindows();
}
