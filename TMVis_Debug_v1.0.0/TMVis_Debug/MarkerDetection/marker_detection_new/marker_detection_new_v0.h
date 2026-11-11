#pragma once

#include <atomic>
#include <vector>
#include <cstdint>
#include <string>

#include <map>
#include <deque>
#include <numeric>
#include <sstream>
#include <chrono>


#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#include "ClientAPI/Plugin.h"
#include "common/DataModel.h"

class ArucoDetectorNew : public Plugin
{
// Thêm khai báo này vào bên trong class ArucoDetectorNew
float getPlaneDepthRobust(const tmv::DepthFrame& dmsg, const std::vector<cv::Point2f>& corners) const;
public:
    ArucoDetectorNew(const std::string& name);
    ~ArucoDetectorNew();

    void run() override;

private:
    void loadParams();
    bool tryBuildDetection(tmv::MarkerDetections& out);

    static inline int clampi(int v, int lo, int hi) {
        return (v < lo) ? lo : (v > hi) ? lo : (v > hi ? hi : v);
    }

    float depthMedianAt(const tmv::DepthFrame& dmsg, int u, int v, int half_win) const;

    bool deprojectXYZ(float& X, float& Y, float& Z,
                      float u, float v,
                      float depth_m,
                      float fx, float fy, float ppx, float ppy) const;

private:
    std::atomic<bool> _running{false};

    int _dictId = static_cast<int>(cv::aruco::DICT_6X6_100);
    cv::Ptr<cv::aruco::Dictionary> _dict;
    cv::Ptr<cv::aruco::DetectorParameters> _params;

    uint64_t _lastProcKey = 0;
    int _maxSyncDiffMs = 50;
    uint64_t _maxSyncDiffNs = 50ULL * 1000ULL * 1000ULL;
    int _loopSleepMs = 100;

    int _roiHalfWin = 3;
    bool _enableXYZ = true;
    bool _enableStdout = true;
};