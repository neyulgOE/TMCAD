#pragma once

#include "CameraHardwareProfile.h"
#include <string>

class CameraProfileParser {
public:
    // Tự động nhận diện định dạng file (RealSense Dump JSON, BOP JSON, Generic Camera JSON)
    static bool loadProfile(const std::string& filePath, CameraHardwareProfile& outProfile);

    // Nạp từ chuỗi JSON trong bộ nhớ
    static bool loadFromString(const std::string& jsonContent,
                               CameraHardwareProfile& outProfile,
                               const std::string& sourceName = "Memory");

private:
    static bool parseRealSenseDump(const class QJsonObject& root, CameraHardwareProfile& outProfile);
    static bool parseBopCamera(const class QJsonObject& root, CameraHardwareProfile& outProfile);
    static bool parseGenericCamera(const class QJsonObject& root, CameraHardwareProfile& outProfile);
};

