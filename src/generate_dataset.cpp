#include "SceneManager.h"
#include "SyntheticDatasetGenerator.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <QFile>

int main(int argc, char** argv) {
    SceneManager manager;
    
    int numSamples = (argc > 1) ? std::stoi(argv[1]) : 500;
    std::string outputDir = (argc > 2) ? argv[2] : "/home/huyen1905/TMCAD/dataset/TABLE-DEMO_audit_v1";
    bool cleanMode = (argc > 3) ? (std::stoi(argv[3]) == 1) : true;
    std::string cadPath = (argc > 4) ? argv[4] : "/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO/TABLE-DEMO.stl";
    if (!QFile::exists(QString::fromStdString(cadPath))) {
        cadPath = "/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO.stl";
    }

    bool isTable = (cadPath.find("TABLE") != std::string::npos || cadPath.find("table") != std::string::npos);

    // Dataset generation configuration
    DatasetGenConfig cfg;
    cfg.numSamples = numSamples;
    cfg.outputDir = outputDir;
    cfg.minDistanceMeters = isTable ? 0.75 : 0.45;
    cfg.maxDistanceMeters = isTable ? 2.20 : 1.20;

    // Calibrated AMR Aligned RGB-D intrinsics (registered_pair=atomic_rgbd)
    cfg.intrinsics = CameraIntrinsics{640, 480, 608.215, 608.128, 341.473, 228.438};

    cfg.amrMode = isTable;
    cfg.amrMinCameraHeightMeters = 0.18; // 18cm - 115cm above floor (covers AMR + Tripod/MuJoCo)
    cfg.amrMaxCameraHeightMeters = 1.15;
    cfg.amrPitchRangeDeg = 12.0;        // [-12 deg, +12 deg] pitch perturbation around object center
    cfg.full360Rotation = true;         // 360 deg yaw around object

    if (cleanMode) {
        // Clean Audit Mode: unoccluded raycast, zero artificial noise
        cfg.visibleFractionLevels = {1.0};
        cfg.useRealisticNoiseModel = false;
        std::cout << ">>> Running in CLEAN AUDIT MODE (No noise, unoccluded)" << std::endl;
    } else {
        // Full Curriculum Mode: 4 occlusion levels + realistic disparity noise
        cfg.visibleFractionLevels = {1.0, 0.75, 0.5, 0.25};
        cfg.useRealisticNoiseModel = true;
        cfg.baselineMeters = 0.05;
        cfg.subpixelDisparityErrorPx = 0.08;
        cfg.maxIncidenceAngleDeg = 85.0;
        std::cout << ">>> Running in REALISTIC CURRICULUM MODE (Noise + 4 Occlusion levels)" << std::endl;
    }

    // Vital for InfoNCE correspondence training
    cfg.captureCorrespondences = true;

    // Seed for exact reproducibility
    cfg.rngSeed = 42;

    std::cout << "Generating dataset for TABLE-DEMO:\n"
              << "  Output Dir:    " << cfg.outputDir << "\n"
              << "  Samples:       " << cfg.numSamples << "\n"
              << "  Height (Floor): 0.15m - 0.40m\n"
              << "  Pitch:         [-10, +10] deg\n"
              << "  Distance:      0.9m - 2.5m\n"
              << "  Intrinsics:    fx=" << cfg.intrinsics.fx << " fy=" << cfg.intrinsics.fy
              << " cx=" << cfg.intrinsics.cx << " cy=" << cfg.intrinsics.cy << "\n"
              << std::endl;

    const CADAsset& cadTable = manager.getOrProcessCADAsset(cadPath);

    auto progress = [](int current, int total) -> bool {
        if (current % 50 == 0 || current == total) {
            std::cout << "[DATASET GEN] Progress: " << current << " / " << total
                      << " (" << std::fixed << std::setprecision(1) << (current * 100.0 / total) << "%)" << std::endl;
        }
        return true;
    };

    int nTable = SyntheticDatasetGenerator::generate(cadTable, cfg, progress);
    std::cout << "Successfully generated " << nTable << " samples in: " << cfg.outputDir << std::endl;

    return 0;
}
