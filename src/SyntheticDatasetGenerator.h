#pragma once
#include "SyntheticRGBDSource.h"
#include "CADAsset.h"
#include <string>
#include <vector>

// Batch generator for Phase 3 (deep representation training) and Phase 5
// (occlusion benchmark): sweeps random camera-relative poses -- and,
// optionally, Phase 5's occlusion levels -- over ONE CAD asset and writes
// each sample to disk as a self-contained folder:
//
//   <outputDir>/sample_00000/color.png
//   <outputDir>/sample_00000/depth.png           (16-bit PNG, mm, depthScale=1000)
//   <outputDir>/sample_00000/pose.json            (GT pose + occlusion + intrinsics)
//   <outputDir>/sample_00000/correspondences.csv  (only if captureCorrespondences)
//
// This is deliberately a SEPARATE path from MainWindow::onEstimatePoseSynthetic()
// / SyntheticSceneDialog: that UI-driven, single-shot flow is Phase 2's tool
// for interactively sweeping crop/ROI settings against one reproducible
// scene. This class is the offline, no-GUI Phase 3/5 tool: point it at a
// CAD file's already-processed CADAsset and a sample count, and it produces
// a training/benchmark dataset unattended (e.g. from a CLI wrapper or a
// script, run overnight).
struct DatasetGenConfig {
    std::string outputDir;
    int numSamples = 1000;

    // Camera-relative placement range
    double minDistanceMeters = 0.9;
    double maxDistanceMeters = 2.5;

    // Rotation is sampled independently per axis if amrMode = false.
    bool full360Rotation = true;
    double rotRangeDegIfNotFull360 = 45.0;

    // --- AMR Constrained Mode ---
    bool amrMode = true; // Use floor-constrained WCS
    double amrMinCameraHeightMeters = 0.1; // 10cm - 40cm uniform random height above floor
    double amrMaxCameraHeightMeters = 0.4;
    double amrCameraHeightMeters = 0.25;   // Fallback nominal height
    double amrPitchRangeDeg = 10.0;        // Camera tilts up/down [-10 deg, +10 deg]
    // Roll is fixed to 0 in AMR mode.
    // Distance uses [minDistanceMeters, maxDistanceMeters] on the floor (0.8m - 3.0m).
    // Yaw (azimuth) sweeps 360 degrees around the object.

    // Occlusion levels to sweep -- one is picked uniformly at random per
    // sample. Defaults match roadmap Phase 5's benchmark rows exactly
    // (100/75/50/25% visible surface area).
    std::vector<double> visibleFractionLevels = {1.0, 0.75, 0.5, 0.25};
    Eigen::Vector3i occlusionGridDims{4, 4, 4};

    // Noise/clutter settings applied identically to every sample in this
    // call; run generate() again with different values if you need a
    // separate noise-level sweep (e.g. clean vs. noisy dataset variants).
    bool useRealisticNoiseModel = true;
    double baselineMeters = 0.05;
    double subpixelDisparityErrorPx = 0.08;
    double maxIncidenceAngleDeg = 85.0;
    double minRangeMeters = 0.2;
    double maxRangeMeters = 10.0;
    bool addBackgroundPlane = true;
    double backgroundPlaneDistanceMeters = 3.0;
    double backgroundPlaneSizeMeters = 6.0;

    CameraIntrinsics intrinsics;  // default-constructed = invalid(); caller MUST set this

    // Phase 3's contrastive training needs these; a Phase-5-only run that
    // just scores ClassicalPoseEstimator/a trained model's pose accuracy
    // per occlusion level can turn this off to save disk space and time.
    bool captureCorrespondences = true;

    unsigned rngSeed = 0;  // 0 = std::random_device (non-reproducible run)
};

#include <functional>

class SyntheticDatasetGenerator {
public:
    // Renders cfg.numSamples random-pose/random-occlusion frames of
    // cadAsset and writes them under cfg.outputDir. Returns the number of
    // samples actually written (<= cfg.numSamples): a single failed
    // ray-cast or file write is logged to stderr and skipped rather than
    // aborting the whole run, so a bad sample can't cost an overnight batch.
    //
    // Optional progressCallback(written, total): return false to abort early.
    static int generate(const CADAsset& cadAsset, const DatasetGenConfig& cfg,
                        std::function<bool(int current, int total)> progressCallback = nullptr);
};
