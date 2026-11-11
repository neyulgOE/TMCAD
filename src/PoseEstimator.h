#pragma once
#include "CADAsset.h"
#include "RGBDTypes.h"
#include <open3d/Open3D.h>
#include <Eigen/Dense>

// Tunable knobs for the classical pipeline.
//
// Rule of thumb for scaling these to a new object size D (in meters):
//   sceneVoxelSize          ≈ D * 0.012  (enough to keep ~5k pts in scene crop)
//   normalRadius            ≈ D * 0.045
//   fpfhRadius              ≈ D * 0.090  (must be > normalRadius)
//   ransacDistanceThreshold ≈ D * 0.018
//   icpMaxCorrespondenceDistance ≈ D * 0.045
//
// Example for D=1.1m (table): voxel=0.013, normalR=0.05, fpfhR=0.10,
//   ransacDist=0.02, icpDist=0.05.
// Example for D=0.15m (hand-tool): voxel=0.002, normalR=0.007, fpfhR=0.014,
//   ransacDist=0.003, icpDist=0.007.
//
// These defaults are calibrated for TABLE-DEMO (realBoundingDiag ≈ 1.089 m).
// Previous defaults were for a ~1.65m diagonal Rack — DO NOT revert those.
// Sweep them against the actual target before trusting fitness numbers if the
// target size changes significantly.
struct PoseEstimatorConfig {
    double sceneVoxelSize = 0.013;   // meters; D*0.012; downsample real cloud before FPFH
    int outlierNeighbors = 20;
    double outlierStdRatio = 2.0;

    // Expressed directly in meters (unlike CAD-side unitless features)
    // because both radii below run on the METRIC scene/CAD clouds -- see
    // rescaleCadCloudToMetric(). Forgetting that rescale is the single
    // most likely bug when wiring this up.
    // fpfhRadius MUST be strictly larger than normalRadius -- Open3D's
    // ComputeFPFHFeature will silently produce degenerate descriptors if
    // these are swapped.
    double normalRadius = 0.049;  // D*0.045; was 0.074 (Rack D=1.65m)
    double fpfhRadius   = 0.098;  // D*0.090; was 0.149 (Rack D=1.65m) — must be > normalRadius

    double ransacDistanceThreshold = 0.020;  // D*0.018; was 0.030
    int ransacMaxIterations = 100000;
    double ransacConfidence = 0.999;

    bool refineWithICP = true;
    double icpMaxCorrespondenceDistance = 0.049;  // D*0.045; was 0.074

    // Crop the raw scene cloud to a camera-frame bounding box before
    // voxel downsampling. Dramatically reduces clutter when the object
    // occupies only a small fraction of the scene.
    // Units: meters, in camera frame (Z = forward, Y = down, X = right).
    // Typical starting point for a table-height object ~1m in front:
    //   cropMin = {-1.0, -0.5, 0.3},  cropMax = {1.0, 1.5, 2.5}
    bool cropEnabled = false;
    Eigen::Vector3d cropMin{-1e9, -1e9, -1e9};
    Eigen::Vector3d cropMax{ 1e9,  1e9,  1e9};
};

struct Pose6DoFResult {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    double fitness = 0.0;       // fraction of inlier correspondences (RANSAC, or ICP if it ran)
    double inlierRmse = 0.0;
    int numRansacInliers = 0;
    bool valid = false;         // false = "pose không xác định" (Phase 0/5 policy) --
                                 // caller must NOT act on R/t when this is false.

    // --- Diagnostics for offline logging (Phase 2 baseline sweeps) -----
    // Populated regardless of pass/fail (each field is filled as soon as
    // its corresponding pipeline stage in estimate() produces a cloud),
    // so a REJECTED pose (valid=false) still carries enough information
    // in the CSV log to tell "scene crop didn't fire" apart from
    // "CAD/scene scale mismatch" apart from "just not enough inliers" --
    // instead of every failure collapsing into the same fitness=0 row.
    size_t numRawScenePoints = 0;    // straight out of pointCloudFromRGBD(), before crop/voxel/outlier
    size_t numSceneCloudPoints = 0;  // after crop + voxel downsample + outlier removal
    size_t numCadCloudPoints = 0;    // rescaled-to-metric CAD sample cloud
    double cadCloudDiag = 0.0;       // meters
    double sceneCloudDiag = 0.0;     // meters -- compare against cadCloudDiag for the
                                      // >20x unit-scale sanity check estimate() already logs

    Eigen::Matrix4d asMatrix() const {
        Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
        m.block<3, 3>(0, 0) = R;
        m.block<3, 1>(0, 3) = t;
        return m;
    }
};

// Phase 2 classical baseline:
//
//   RGBD -> real point cloud -> preprocess -> FPFH  \
//                                                      RANSAC correspondence
//   CADAsset.sampledFeatures -> rescale to metric -> FPFH  /   -> Kabsch/SVD (R,t)
//                                                                -> optional ICP refine
//
// Deliberately uses only classical, non-learned features (Open3D's FPFH)
// -- see roadmap Phase 2, "không dùng learned feature ở bước này". Phase 4
// later swaps the correspondence SOURCE for a learned descriptor while
// reusing the same RANSAC+solver call here (ransacGlobalRegistration /
// refineWithICP), so keep those two methods free of FPFH-specific
// assumptions beyond the Feature type itself.
class ClassicalPoseEstimator {
public:
    explicit ClassicalPoseEstimator(PoseEstimatorConfig config = {});

    // Main entry point. cadAsset must satisfy CADAsset::valid(). frame /
    // intrinsics come from an IRGBDSource::grabFrame() call. Never throws;
    // returns a result with valid=false if any stage fails or the
    // correspondence quality falls below kMinAcceptableFitness -- per
    // Phase 5 policy that must mean "no pose", not a fallback guess.
    Pose6DoFResult estimate(const CADAsset& cadAsset,
                             const RGBDFrame& frame,
                             const CameraIntrinsics& intrinsics) const;

    // --- Individual stages, exposed for unit testing and for the Phase 5
    // occlusion-benchmark harness (which needs to swap pieces in/out per
    // occlusion level without going through a live camera). -------------

    // RGBD -> raw metric point cloud in camera frame (no filtering yet).
    static std::shared_ptr<open3d::geometry::PointCloud> pointCloudFromRGBD(
        const RGBDFrame& frame, const CameraIntrinsics& intrinsics);

    // Voxel downsample + statistical outlier removal + normal estimation
    // (normals oriented towards the camera origin, for point-to-plane ICP).
    std::shared_ptr<open3d::geometry::PointCloud> preprocessSceneCloud(
        const std::shared_ptr<open3d::geometry::PointCloud>& raw) const;

    // CADAsset::sampledFeatures live in unit-normalized space (diag == 1,
    // see CADAsset.h). This rescales positions by realBoundingDiag so the
    // CAD cloud and the scene cloud share the same physical units before
    // FPFH/RANSAC ever compare them.
    static std::shared_ptr<open3d::geometry::PointCloud> rescaleCadCloudToMetric(
        const CADAsset& cadAsset);

    std::shared_ptr<open3d::pipelines::registration::Feature> computeFPFH(
        const open3d::geometry::PointCloud& cloud, double normalRadius, double fpfhRadius) const;

    // RANSAC-based global registration on FPFH correspondences
    // (Open3D built-in; Kabsch/SVD is what
    // TransformationEstimationPointToPoint solves internally per
    // iteration, so it isn't hand-rolled separately here).
    //
    // source = cadCloud, target = sceneCloud -- Open3D returns the
    // transform mapping source INTO target, i.e. exactly T_camera<-object
    // per Phase 0's definition. Swapping source/target silently produces
    // the inverse pose; don't "simplify" this call without re-checking.
    open3d::pipelines::registration::RegistrationResult ransacGlobalRegistration(
        const open3d::geometry::PointCloud& cadCloud,
        const open3d::geometry::PointCloud& sceneCloud,
        const open3d::pipelines::registration::Feature& cadFeatures,
        const open3d::pipelines::registration::Feature& sceneFeatures) const;

    // Point-to-plane ICP starting from initialGuess; sceneCloud must
    // already have normals (preprocessSceneCloud provides this).
    open3d::pipelines::registration::RegistrationResult refineWithICP(
        const open3d::geometry::PointCloud& cadCloud,
        const open3d::geometry::PointCloud& sceneCloud,
        const Eigen::Matrix4d& initialGuess) const;

    // Minimum RANSAC fitness below which the result is considered
    // unreliable and estimate() returns valid=false instead of a guessed
    // pose. Placeholder -- Phase 0's accuracy target and Phase 5's
    // Calibrated baseline threshold: poses with fitness below this are rejected.
    static constexpr double kMinAcceptableFitness = 0.15;

private:
    PoseEstimatorConfig config_;
};