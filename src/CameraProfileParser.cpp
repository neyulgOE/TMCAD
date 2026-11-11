#include "CameraProfileParser.h"
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonValue>
#include <QStringList>
#include <QDebug>
#include <cmath>
#include <iostream>

static double getJsonDouble(const QJsonValue& val, double defaultVal = 0.0) {
    if (val.isDouble()) return val.toDouble();
    if (val.isString()) {
        bool ok = false;
        double d = val.toString().trimmed().toDouble(&ok);
        if (ok) return d;
    }
    return defaultVal;
}

static int getJsonInt(const QJsonValue& val, int defaultVal = 0) {
    if (val.isDouble()) return val.toInt();
    if (val.isString()) {
        bool ok = false;
        int i = val.toString().trimmed().toInt(&ok);
        if (ok) return i;
    }
    return defaultVal;
}

bool CameraProfileParser::loadProfile(const std::string& filePath, CameraHardwareProfile& outProfile) {
    QFile file(QString::fromStdString(filePath));
    if (!file.open(QIODevice::ReadOnly)) {
        std::cerr << "CameraProfileParser: Cannot open file " << filePath << std::endl;
        return false;
    }
    QByteArray data = file.readAll();
    file.close();

    QFileInfo fi(QString::fromStdString(filePath));
    std::string baseName = fi.baseName().toStdString();

    bool ok = loadFromString(data.toStdString(), outProfile, baseName);
    if (ok) {
        outProfile.sourceFilePath = filePath;
    }
    return ok;
}

bool CameraProfileParser::loadFromString(const std::string& jsonContent,
                                         CameraHardwareProfile& outProfile,
                                         const std::string& sourceName) {
    QJsonParseError parseErr;
    QJsonDocument doc = QJsonDocument::fromJson(QByteArray::fromStdString(jsonContent), &parseErr);
    if (doc.isNull() || !doc.isObject()) {
        std::cerr << "CameraProfileParser: JSON parse error: "
                  << parseErr.errorString().toStdString() << std::endl;
        return false;
    }

    QJsonObject root = doc.object();
    outProfile = CameraHardwareProfile();
    outProfile.profileName = sourceName.empty() ? "Imported Camera Profile" : sourceName;

    // 1. Thử nhận diện định dạng Intel RealSense Calibration Dump (chứa "baseline" và "rectified.")
    if (root.contains("baseline") || root.contains("rectified.0.fx") || root.contains("rectified.1.fx")) {
        if (parseRealSenseDump(root, outProfile)) {
            return true;
        }
    }

    // 2. Thử nhận diện định dạng BOP scene_camera.json (chứa "cam_K")
    if (root.contains("cam_K") || root.contains("0")) {
        if (parseBopCamera(root, outProfile)) {
            return true;
        }
    }

    // 3. Thử nhận diện định dạng Generic Camera JSON (chứa "fx", "fy", "cx", "cy")
    if (root.contains("fx") && root.contains("fy")) {
        if (parseGenericCamera(root, outProfile)) {
            return true;
        }
    }

    std::cerr << "CameraProfileParser: Unrecognized camera profile format in " << sourceName << std::endl;
    return false;
}

bool CameraProfileParser::parseRealSenseDump(const QJsonObject& root, CameraHardwareProfile& outProfile) {
    // 1. Baseline
    if (root.contains("baseline")) {
        double rawBaselineMm = std::abs(getJsonDouble(root["baseline"], 50.0813));
        outProfile.baselineMeters = rawBaselineMm / 1000.0;
    }

    // 2. Các stream profile đã rectified
    // File có cấu trúc: "rectified.<id>.<param>" (ví dụ "rectified.1.fx")
    for (auto it = root.begin(); it != root.end(); ++it) {
        QString key = it.key();
        if (!key.startsWith("rectified.")) continue;

        QStringList parts = key.split('.');
        if (parts.size() < 3) continue;

        bool okId = false;
        int pId = parts[1].toInt(&okId);
        if (!okId) continue;

        QString param = parts[2].toLower();
        StreamProfileEntry& entry = outProfile.depthProfiles[pId];
        entry.profileId = pId;

        if (param == "width") {
            entry.width = getJsonInt(it.value());
        } else if (param == "height") {
            entry.height = getJsonInt(it.value());
        } else if (param == "fx") {
            entry.fx = getJsonDouble(it.value());
        } else if (param == "fy") {
            entry.fy = getJsonDouble(it.value());
        } else if (param == "ppx") {
            entry.ppx = getJsonDouble(it.value());
        } else if (param == "ppy") {
            entry.ppy = getJsonDouble(it.value());
        }
    }

    // Tính toán góc FOV và gán nhãn cho các profile Depth hợp lệ
    for (auto& pair : outProfile.depthProfiles) {
        StreamProfileEntry& p = pair.second;
        if (p.isValid()) {
            p.fovYDeg = 2.0 * std::atan(static_cast<double>(p.height) / (2.0 * p.fy)) * 180.0 / M_PI;
            p.label = std::to_string(p.width) + "x" + std::to_string(p.height) +
                      " (Depth IR Stereo - Profile " + std::to_string(p.profileId) +
                      ", fx=" + std::to_string(static_cast<int>(p.fx + 0.5)) +
                      ", FOV=" + std::to_string(static_cast<int>(p.fovYDeg + 0.5)) + "°)";
        }
    }

    // 3. Ma trận quay IMU ngoại tại (nếu có)
    if (root.contains("world2left_rot.x.x")) {
        outProfile.world2LeftRot(0, 0) = getJsonDouble(root["world2left_rot.x.x"], 1.0);
        outProfile.world2LeftRot(0, 1) = getJsonDouble(root["world2left_rot.x.y"], 0.0);
        outProfile.world2LeftRot(0, 2) = getJsonDouble(root["world2left_rot.x.z"], 0.0);
        outProfile.world2LeftRot(1, 0) = getJsonDouble(root["world2left_rot.y.x"], 0.0);
        outProfile.world2LeftRot(1, 1) = getJsonDouble(root["world2left_rot.y.y"], 1.0);
        outProfile.world2LeftRot(1, 2) = getJsonDouble(root["world2left_rot.y.z"], 0.0);
        outProfile.world2LeftRot(2, 0) = getJsonDouble(root["world2left_rot.z.x"], 0.0);
        outProfile.world2LeftRot(2, 1) = getJsonDouble(root["world2left_rot.z.y"], 0.0);
        outProfile.world2LeftRot(2, 2) = getJsonDouble(root["world2left_rot.z.z"], 1.0);
    }

    return outProfile.isValid();
}

bool CameraProfileParser::parseBopCamera(const QJsonObject& root, CameraHardwareProfile& outProfile) {
    QJsonObject camObj = root;
    if (root.contains("0") && root["0"].isObject()) {
        camObj = root["0"].toObject();
    }

    if (!camObj.contains("cam_K")) return false;
    QJsonArray kArr = camObj["cam_K"].toArray();
    if (kArr.size() < 9) return false;

    StreamProfileEntry entry;
    entry.profileId = 0;
    entry.fx = getJsonDouble(kArr[0]);
    entry.fy = getJsonDouble(kArr[4]);
    entry.ppx = getJsonDouble(kArr[2]);
    entry.ppy = getJsonDouble(kArr[5]);

    if (camObj.contains("depth_scale")) {
        outProfile.depthScale = getJsonDouble(camObj["depth_scale"], 0.001);
    }

    // Nếu không có width/height thì lấy mặc định từ tâm quang học (x2)
    entry.width = static_cast<int>(entry.ppx * 2.0 + 0.5);
    entry.height = static_cast<int>(entry.ppy * 2.0 + 0.5);
    if (entry.isValid()) {
        entry.fovYDeg = 2.0 * std::atan(static_cast<double>(entry.height) / (2.0 * entry.fy)) * 180.0 / M_PI;
        entry.label = std::to_string(entry.width) + "x" + std::to_string(entry.height) + " (BOP Benchmark)";
        outProfile.colorProfiles[0] = entry;
        outProfile.depthProfiles[0] = entry;
        return true;
    }
    return false;
}

bool CameraProfileParser::parseGenericCamera(const QJsonObject& root, CameraHardwareProfile& outProfile) {
    StreamProfileEntry entry;
    entry.profileId = 0;
    entry.fx = getJsonDouble(root["fx"]);
    entry.fy = getJsonDouble(root["fy"]);
    entry.ppx = getJsonDouble(root.contains("cx") ? root["cx"] : root["ppx"]);
    entry.ppy = getJsonDouble(root.contains("cy") ? root["cy"] : root["ppy"]);
    entry.width = getJsonInt(root["width"], static_cast<int>(entry.ppx * 2.0 + 0.5));
    entry.height = getJsonInt(root["height"], static_cast<int>(entry.ppy * 2.0 + 0.5));

    if (root.contains("baseline")) {
        double b = getJsonDouble(root["baseline"], 50.0);
        outProfile.baselineMeters = (b > 1.0) ? (b / 1000.0) : b; // Tự động xử lý nếu đơn vị là mm hay m
    }
    if (root.contains("depth_scale")) {
        outProfile.depthScale = getJsonDouble(root["depth_scale"], 0.001);
    }

    if (entry.isValid()) {
        entry.fovYDeg = 2.0 * std::atan(static_cast<double>(entry.height) / (2.0 * entry.fy)) * 180.0 / M_PI;
        entry.label = std::to_string(entry.width) + "x" + std::to_string(entry.height) + " (Generic)";
        outProfile.colorProfiles[0] = entry;
        outProfile.depthProfiles[0] = entry;
        return true;
    }
    return false;
}

