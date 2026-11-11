#include "SyntheticDatasetGenerator.h"
#include <open3d/Open3D.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cstdio>
#include <random>

namespace fs = std::filesystem;

namespace {
std::string matrixToJsonArray(const Eigen::Matrix4d& m) {
    std::ostringstream oss;
    oss << "[";
    for (int r = 0; r < 4; ++r) {
        oss << "[";
        for (int c = 0; c < 4; ++c) {
            oss << std::setprecision(17) << m(r, c);
            if (c < 3) oss << ", ";
        }
        oss << "]";
        if (r < 3) oss << ", ";
    }
    oss << "]";
    return oss.str();
}
}  // namespace

int SyntheticDatasetGenerator::generate(const CADAsset& cadAsset, const DatasetGenConfig& cfg,
                                        std::function<bool(int current, int total)> progressCallback) {
    if (!cadAsset.valid()) {
        std::cerr << "SyntheticDatasetGenerator: CADAsset invalid, aborting.\n";
        return 0;
    }
    if (!cfg.intrinsics.valid()) {
        std::cerr << "SyntheticDatasetGenerator: intrinsics invalid "
                     "(set cfg.intrinsics before calling generate()), aborting.\n";
        return 0;
    }
    if (cfg.visibleFractionLevels.empty()) {
        std::cerr << "SyntheticDatasetGenerator: visibleFractionLevels is empty, aborting.\n";
        return 0;
    }
    if (cfg.numSamples <= 0) {
        std::cerr << "SyntheticDatasetGenerator: numSamples <= 0, nothing to do.\n";
        return 0;
    }

    std::error_code ec;
    fs::create_directories(cfg.outputDir, ec);
    if (ec) {
        std::cerr << "SyntheticDatasetGenerator: failed to create output dir "
                  << cfg.outputDir << ": " << ec.message() << "\n";
        return 0;
    }

    // Built once from the cached CADAsset -- every sample reuses this same
    // metric mesh, only the pose/occlusion config changes per sample.
    auto metricMesh = SyntheticRGBDSource::meshFromCADAsset(cadAsset);
    if (!metricMesh || metricMesh->vertices_.empty()) {
        std::cerr << "SyntheticDatasetGenerator: meshFromCADAsset() produced an empty mesh "
                     "-- is cadAsset.normalizedMesh populated?\n";
        return 0;
    }

    // Write manifest to document generation configuration
    {
        std::ofstream manifestOut(fs::path(cfg.outputDir) / "dataset_manifest.json");
        manifestOut << "{\n"
                    << "  \"numSamples\": " << cfg.numSamples << ",\n"
                    << "  \"minDistanceMeters\": " << cfg.minDistanceMeters << ",\n"
                    << "  \"maxDistanceMeters\": " << cfg.maxDistanceMeters << ",\n"
                    << "  \"amrMode\": " << (cfg.amrMode ? "true" : "false") << ",\n"
                    << "  \"amrMinCameraHeightMeters\": " << cfg.amrMinCameraHeightMeters << ",\n"
                    << "  \"amrMaxCameraHeightMeters\": " << cfg.amrMaxCameraHeightMeters << ",\n"
                    << "  \"amrPitchRangeDeg\": " << cfg.amrPitchRangeDeg << ",\n"
                    << "  \"full360Rotation\": " << (cfg.full360Rotation ? "true" : "false") << ",\n"
                    << "  \"floorZ\": " << metricMesh->GetMinBound().z() << ",\n"
                    << "  \"useRealisticNoiseModel\": " << (cfg.useRealisticNoiseModel ? "true" : "false") << ",\n"
                    << "  \"captureCorrespondences\": " << (cfg.captureCorrespondences ? "true" : "false") << ",\n"
                    << "  \"rngSeed\": " << cfg.rngSeed << ",\n"
                    << "  \"intrinsics\": {"
                    << "\"width\": " << cfg.intrinsics.width
                    << ", \"height\": " << cfg.intrinsics.height
                    << ", \"fx\": " << cfg.intrinsics.fx
                    << ", \"fy\": " << cfg.intrinsics.fy
                    << ", \"cx\": " << cfg.intrinsics.cx
                    << ", \"cy\": " << cfg.intrinsics.cy << "}\n"
                    << "}\n";
    }

    std::mt19937 rng(cfg.rngSeed != 0 ? cfg.rngSeed : std::random_device{}());
    std::uniform_real_distribution<double> distDist(cfg.minDistanceMeters, cfg.maxDistanceMeters);
    const double rotRange = cfg.full360Rotation ? 180.0 : cfg.rotRangeDegIfNotFull360;
    std::uniform_real_distribution<double> rotDist(-rotRange, rotRange);
    std::uniform_real_distribution<double> azimuthDist(-180.0, 180.0);
    std::uniform_real_distribution<double> pitchDist(-cfg.amrPitchRangeDeg, cfg.amrPitchRangeDeg);
    std::uniform_real_distribution<double> heightDist(
        std::min(cfg.amrMinCameraHeightMeters, cfg.amrMaxCameraHeightMeters),
        std::max(cfg.amrMinCameraHeightMeters, cfg.amrMaxCameraHeightMeters));
    std::uniform_int_distribution<size_t> levelDist(0, cfg.visibleFractionLevels.size() - 1);

    int written = 0;
    for (int i = 0; i < cfg.numSamples; ++i) {
        const double distance = distDist(rng);
        Eigen::Matrix4d gtPose;
        double rx = 0.0, ry = 0.0, rz = 0.0;

        if (cfg.amrMode) {
            // World Coordinate System (WCS): Z is Up, X is Right, Y is Forward.
            // Object is at the origin (0,0,0) with Z up.
            // Camera position in WCS:
            double azimuth = azimuthDist(rng) * M_PI / 180.0;
            // Camera starts on -X axis looking at +X (when azimuth=0)
            double camX = -distance * std::cos(azimuth);
            double camY = -distance * std::sin(azimuth);
            // Camera height is defined relative to the physical floor (bottom of CAD model)
            // rather than CAD centroid origin (0,0,0):
            const double floorZ = metricMesh->GetMinBound().z();
            double camZ = floorZ + heightDist(rng);

            // Camera looking parallel to the ground (same height)
            // so Pitch = 0 means perfectly horizontal.
            Eigen::Vector3d camPos(camX, camY, camZ);
            Eigen::Vector3d lookTarget(0.0, 0.0, camZ);
            Eigen::Vector3d forward = (lookTarget - camPos).normalized();
            Eigen::Vector3d upWorld(0, 0, 1);
            Eigen::Vector3d right = forward.cross(upWorld).normalized();
            Eigen::Vector3d upCam = right.cross(forward).normalized();

            // T_world_cam:
            Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
            Twc.block<3,1>(0,0) = right;
            Twc.block<3,1>(0,1) = -upCam; // OpenCV camera Y is down
            Twc.block<3,1>(0,2) = forward;
            Twc.block<3,1>(0,3) = camPos;

            // Apply randomized pitch (tilt up/down) in camera local frame
            double pitchDeg = pitchDist(rng);
            Eigen::Matrix3d Rpitch = Eigen::AngleAxisd(pitchDeg * M_PI / 180.0, Eigen::Vector3d::UnitX()).toRotationMatrix();
            Twc.block<3,3>(0,0) = Twc.block<3,3>(0,0) * Rpitch;

            // Store metadata (re-using rot variables for logging context)
            rx = pitchDeg;
            ry = azimuth * 180.0 / M_PI;
            rz = 0.0; // Roll is fixed in AMR mode

            // T_cam_obj = Twc^-1 * T_world_obj. Since T_world_obj is Identity:
            gtPose = Twc.inverse();
        } else {
            rx = rotDist(rng);
            ry = rotDist(rng);
            rz = rotDist(rng);
            gtPose = SyntheticRGBDSource::makeFrontFacingPose(distance, rx, ry, rz);
        }
        
        const double targetVisible = cfg.visibleFractionLevels[levelDist(rng)];

        SyntheticSceneConfig sceneCfg;
        sceneCfg.cameraFromObject = gtPose;
        sceneCfg.useRealisticNoiseModel = cfg.useRealisticNoiseModel;
        sceneCfg.baselineMeters = cfg.baselineMeters;
        sceneCfg.subpixelDisparityErrorPx = cfg.subpixelDisparityErrorPx;
        sceneCfg.maxIncidenceAngleDeg = cfg.maxIncidenceAngleDeg;
        sceneCfg.minRangeMeters = cfg.minRangeMeters;
        sceneCfg.maxRangeMeters = cfg.maxRangeMeters;
        sceneCfg.addBackgroundPlane = cfg.addBackgroundPlane;
        sceneCfg.backgroundPlaneDistanceMeters = cfg.backgroundPlaneDistanceMeters;
        sceneCfg.backgroundPlaneSizeMeters = cfg.backgroundPlaneSizeMeters;
        sceneCfg.occlusionGridDims = cfg.occlusionGridDims;
        sceneCfg.targetVisibleFraction = targetVisible;
        // Give each sample its own deterministic-if-seeded sub-seed so a
        // fixed cfg.rngSeed reproduces the exact same cell selections too,
        // not just the same poses.
        sceneCfg.occlusionSeed = cfg.rngSeed != 0
                                      ? static_cast<unsigned>(cfg.rngSeed + static_cast<unsigned>(i) + 1)
                                      : 0;
        sceneCfg.captureCorrespondences = cfg.captureCorrespondences;

        char nameBuf[32];
        std::snprintf(nameBuf, sizeof(nameBuf), "sample_%05d", i);
        fs::path sampleDir = fs::path(cfg.outputDir) / nameBuf;
        bool sampleComplete = fs::exists(sampleDir / "pose.json") &&
                              fs::exists(sampleDir / "color.png") &&
                              fs::exists(sampleDir / "depth.png");
        if (cfg.captureCorrespondences) {
            sampleComplete = sampleComplete && fs::exists(sampleDir / "correspondences.csv");
        }
        if (sampleComplete) {
            ++written;
            if (progressCallback && (written % 50 == 0 || written == cfg.numSamples)) {
                progressCallback(written, cfg.numSamples);
            }
            continue;
        }

        SyntheticRGBDSource src(metricMesh, cfg.intrinsics, sceneCfg);
        if (!src.open()) {
            std::cerr << "SyntheticDatasetGenerator: sample " << i
                      << " failed to open source, skipping.\n";
            continue;
        }
        RGBDFrame frame;
        if (!src.grabFrame(frame)) {
            std::cerr << "SyntheticDatasetGenerator: sample " << i
                      << " ray cast failed, skipping.\n";
            src.close();
            continue;
        }

        fs::create_directories(sampleDir, ec);
        if (ec) {
            std::cerr << "SyntheticDatasetGenerator: could not create " << sampleDir.string()
                      << ", skipping sample " << i << ".\n";
            src.close();
            continue;
        }

        bool ok = true;
        ok = ok && open3d::io::WriteImage((sampleDir / "color.png").string(), *frame.color);
        ok = ok && open3d::io::WriteImage((sampleDir / "depth.png").string(), *frame.depth);
        if (!ok) {
            std::cerr << "SyntheticDatasetGenerator: sample " << i
                      << " failed to write color/depth PNG, skipping.\n";
            src.close();
            continue;
        }

        // pose.json: everything needed to reload/interpret this sample
        // without re-deriving anything from the original CAD asset --
        // GT pose, requested vs. ACHIEVED occlusion (cells are discrete,
        // see SyntheticRGBDSource::lastVisibleFractionAchieved()), and the
        // intrinsics/depth convention used to render it.
        std::ofstream poseOut(sampleDir / "pose.json");
        poseOut << "{\n"
                << "  \"cameraFromObject\": " << matrixToJsonArray(gtPose) << ",\n"
                << "  \"distanceMeters\": " << distance << ",\n"
                << "  \"rotXDeg\": " << rx << ", \"rotYDeg\": " << ry << ", \"rotZDeg\": " << rz << ",\n"
                << "  \"targetVisibleFraction\": " << targetVisible << ",\n"
                << "  \"achievedVisibleFraction\": " << src.lastVisibleFractionAchieved() << ",\n"
                << "  \"depthScale\": " << frame.depthScale << ",\n"
                << "  \"depthTruncMeters\": " << frame.depthTruncMeters << ",\n"
                << "  \"intrinsics\": {"
                << "\"width\": " << cfg.intrinsics.width
                << ", \"height\": " << cfg.intrinsics.height
                << ", \"fx\": " << cfg.intrinsics.fx
                << ", \"fy\": " << cfg.intrinsics.fy
                << ", \"cx\": " << cfg.intrinsics.cx
                << ", \"cy\": " << cfg.intrinsics.cy << "}\n"
                << "}\n";
        poseOut.close();

        if (cfg.captureCorrespondences) {
            std::ofstream corrOut(sampleDir / "correspondences.csv");
            corrOut << "u,v,triangle_id,cad_x,cad_y,cad_z,cam_x,cam_y,cam_z\n";
            for (const auto& c : src.lastCorrespondences()) {
                corrOut << c.u << "," << c.v << "," << c.triangleId << ","
                        << c.cadPointObjectFrame.x() << "," << c.cadPointObjectFrame.y() << ","
                        << c.cadPointObjectFrame.z() << ","
                        << c.cameraPointMetric.x() << "," << c.cameraPointMetric.y() << ","
                        << c.cameraPointMetric.z() << "\n";
            }
        }

        src.close();
        ++written;

        if (progressCallback) {
            if (!progressCallback(written, cfg.numSamples)) {
                std::cerr << "SyntheticDatasetGenerator: cancelled by user at sample " << written << "\n";
                break;
            }
        }

        if (written % 100 == 0) {
            std::cerr << "SyntheticDatasetGenerator: " << written << " / " << cfg.numSamples
                      << " samples written.\n";
        }
    }

    // Automatically generate train.txt and val.txt
    if (written > 0) {
        std::vector<std::string> sampleNames;
        sampleNames.reserve(written);
        for (int i = 0; i < written; ++i) {
            char nameBuf[32];
            std::snprintf(nameBuf, sizeof(nameBuf), "sample_%05d", i);
            sampleNames.push_back(nameBuf);
        }
        std::mt19937 g(1337);
        std::shuffle(sampleNames.begin(), sampleNames.end(), g);
        int trainCount = static_cast<int>(sampleNames.size() * 0.85);
        if (trainCount == 0) trainCount = 1;

        std::ofstream trainOut(fs::path(cfg.outputDir) / "train.txt");
        for (int i = 0; i < trainCount; ++i) {
            trainOut << sampleNames[i] << "\n";
        }
        trainOut.close();

        std::ofstream valOut(fs::path(cfg.outputDir) / "val.txt");
        for (size_t i = trainCount; i < sampleNames.size(); ++i) {
            valOut << sampleNames[i] << "\n";
        }
        valOut.close();
        std::cerr << "SyntheticDatasetGenerator: generated train.txt (" << trainCount
                  << " samples) and val.txt (" << (sampleNames.size() - trainCount) << " samples).\n";
    }

    std::cerr << "SyntheticDatasetGenerator: done, " << written << " / " << cfg.numSamples
              << " samples written to " << cfg.outputDir << "\n";
    return written;
}
