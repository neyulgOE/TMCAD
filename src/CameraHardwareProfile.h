#pragma once

#include "RGBDTypes.h"
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

struct StreamProfileEntry {
    int profileId = 0;
    std::string label;       // vd: "1280x720 (720p HD)"
    int width = 0;
    int height = 0;
    double fx = 0.0;
    double fy = 0.0;
    double ppx = 0.0;        // cx (tâm quang học X)
    double ppy = 0.0;        // cy (tâm quang học Y)
    double fovYDeg = 0.0;    // Góc mở dọc: 2 * atan(height / (2 * fy)) * 180 / M_PI

    bool isValid() const {
        return width > 0 && height > 0 && fx > 0.0 && fy > 0.0;
    }

    CameraIntrinsics toCameraIntrinsics() const {
        CameraIntrinsics intr;
        intr.width = width;
        intr.height = height;
        intr.fx = fx;
        intr.fy = fy;
        intr.cx = ppx;
        intr.cy = ppy;
        return intr;
    }
};

enum class CameraStreamAlignment {
    AlignedToColor,   // Depth được nắn khớp sang góc nhìn và độ phân giải của RGB (Mặc định trong TMCAD & RealSense SDK)
    RawStereoDepth    // Depth thô theo góc nhìn gốc của cảm biến Stereo Hồng ngoại IR (Góc nhìn rộng hơn)
};

class CameraHardwareProfile {
public:
    std::string profileName = "Intel RealSense D435if";
    std::string sourceFilePath;
    double baselineMeters = 0.0500813; // Khoảng cách giữa 2 mắt hồng ngoại Stereo IR Left-Right (m)
    double depthScale = 0.001;         // 1 đơn vị depth = 1 mm (0.001 m)
    double zNear = 0.15;               // Khoảng cách đo tối thiểu (m)
    double zFar = 3.50;                // Khoảng cách đo tối đa (m)

    // Khoảng cách vật lý giữa Cảm biến Depth và Cảm biến Color (thường lệch ~15mm theo trục X)
    Eigen::Vector3d depthToColorTranslation = Eigen::Vector3d(0.015, 0.0, 0.0);
    Eigen::Matrix3d depthToColorRotation = Eigen::Matrix3d::Identity();

    // Chế độ căn chỉnh luồng dữ liệu
    CameraStreamAlignment alignmentMode = CameraStreamAlignment::AlignedToColor;

    // 1. Hồ sơ Cảm biến Stereo Depth (Hồng ngoại IR - thường từ file D435if-dataset)
    std::map<int, StreamProfileEntry> depthProfiles;

    // 2. Hồ sơ Cảm biến Color RGB (Mắt màu - góc mở chuẩn D435if: FOV 69.4° x 42.5°)
    std::map<int, StreamProfileEntry> colorProfiles;

    // Ma trận xoay ngoại tại IMU / World
    Eigen::Matrix3d world2LeftRot = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d world2RightRot = Eigen::Matrix3d::Identity();

    CameraHardwareProfile() {
        initDefaultColorProfiles();
    }

    bool isValid() const {
        return (!depthProfiles.empty() || !colorProfiles.empty()) && baselineMeters > 0.0;
    }

    // Khởi tạo các profile màu RGB chuẩn của dòng D435if (FOV ngang ~69.4°, dọc ~42.5° ở tỉ lệ 16:9)
    void initDefaultColorProfiles() {
        // 1080p (1920x1080)
        StreamProfileEntry c0;
        c0.profileId = 0; c0.width = 1920; c0.height = 1080;
        c0.fx = 1386.4; c0.fy = 1386.4; c0.ppx = 960.0; c0.ppy = 540.0;
        c0.fovYDeg = 42.57; c0.label = "1920x1080 (Color RGB Full HD, FOV=42.6°)";
        colorProfiles[0] = c0;

        // 720p (1280x720)
        StreamProfileEntry c1;
        c1.profileId = 1; c1.width = 1280; c1.height = 720;
        c1.fx = 924.27; c1.fy = 924.27; c1.ppx = 640.0; c1.ppy = 360.0;
        c1.fovYDeg = 42.57; c1.label = "1280x720 (Color RGB HD, FOV=42.6°)";
        colorProfiles[1] = c1;

        // VGA (640x480 - 4:3)
        StreamProfileEntry c2;
        c2.profileId = 2; c2.width = 640; c2.height = 480;
        c2.fx = 616.18; c2.fy = 616.18; c2.ppx = 320.0; c2.ppy = 240.0;
        c2.fovYDeg = 42.57; c2.label = "640x480 (Color RGB VGA)";
        colorProfiles[2] = c2;
    }

    std::vector<StreamProfileEntry> listValidProfiles() const {
        std::vector<StreamProfileEntry> list;
        for (const auto& pair : depthProfiles) {
            if (pair.second.isValid()) list.push_back(pair.second);
        }
        for (const auto& pair : colorProfiles) {
            if (pair.second.isValid()) list.push_back(pair.second);
        }
        return list;
    }

    std::vector<StreamProfileEntry> listValidDepthProfiles() const {
        std::vector<StreamProfileEntry> list;
        for (const auto& pair : depthProfiles) {
            if (pair.second.isValid()) list.push_back(pair.second);
        }
        return list;
    }

    std::vector<StreamProfileEntry> listValidColorProfiles() const {
        std::vector<StreamProfileEntry> list;
        for (const auto& pair : colorProfiles) {
            if (pair.second.isValid()) list.push_back(pair.second);
        }
        return list;
    }

    // Tự động nội suy ma trận K thích hợp theo chế độ đang chọn (Color vs Depth)
    CameraIntrinsics interpolateIntrinsics(int targetW, int targetH, bool useColor = true) const {
        CameraIntrinsics res;
        res.width = targetW;
        res.height = targetH;
        const auto& targetMap = (useColor && !colorProfiles.empty()) ? colorProfiles : depthProfiles;
        if (targetW <= 0 || targetH <= 0 || targetMap.empty()) return res;

        double targetAspect = static_cast<double>(targetW) / static_cast<double>(targetH);
        const StreamProfileEntry* bestMatch = nullptr;
        double minAspectDiff = 1e9;

        for (const auto& pair : targetMap) {
            const auto& p = pair.second;
            if (!p.isValid()) continue;
            double pAspect = static_cast<double>(p.width) / static_cast<double>(p.height);
            double diff = std::abs(pAspect - targetAspect);
            if (diff < minAspectDiff) {
                minAspectDiff = diff;
                bestMatch = &p;
            }
        }

        if (bestMatch) {
            double scaleX = static_cast<double>(targetW) / static_cast<double>(bestMatch->width);
            double scaleY = static_cast<double>(targetH) / static_cast<double>(bestMatch->height);
            res.fx = bestMatch->fx * scaleX;
            res.fy = bestMatch->fy * scaleY;
            res.cx = bestMatch->ppx * scaleX;
            res.cy = bestMatch->ppy * scaleY;
        }
        return res;
    }
};
