#include "src/CameraProfileParser.h"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    std::string path = "/home/huyen1905/TMCAD/D435if-dataset";
    if (argc > 1) {
        path = argv[1];
    }

    CameraHardwareProfile profile;
    bool ok = CameraProfileParser::loadProfile(path, profile);
    if (!ok) {
        std::cerr << "FAIL: Could not load profile from " << path << std::endl;
        return 1;
    }

    std::cout << "SUCCESS! Loaded camera profile: " << profile.profileName << std::endl;
    std::cout << "Baseline: " << (profile.baselineMeters * 1000.0) << " mm (" 
              << profile.baselineMeters << " m)" << std::endl;
    std::cout << "Total stream profiles: " << profile.streamProfiles.size() << std::endl;
    
    std::cout << "\n--- VALID PROFILES ---" << std::endl;
    auto validList = profile.listValidProfiles();
    for (const auto& p : validList) {
        std::cout << " [ID " << std::setw(2) << p.profileId << "] "
                  << std::setw(4) << p.width << "x" << std::setw(4) << p.height
                  << " | fx: " << std::fixed << std::setprecision(2) << p.fx
                  << " | cx: " << p.ppx << " cy: " << p.ppy
                  << " | fovY: " << p.fovYDeg << " deg"
                  << " -> " << p.label << std::endl;
    }

    std::cout << "\n--- TEST INTERPOLATION TO 1024x576 (16:9) ---" << std::endl;
    auto intr1024 = profile.interpolateIntrinsics(1024, 576);
    std::cout << " 1024x576 -> fx: " << intr1024.fx 
              << " fy: " << intr1024.fy 
              << " cx: " << intr1024.cx 
              << " cy: " << intr1024.cy 
              << " (valid: " << (intr1024.valid() ? "YES" : "NO") << ")" << std::endl;

    return 0;
}

