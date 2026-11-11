#include "PoseEstimator.h"
#include <iostream>

ClassicalPoseEstimator::ClassicalPoseEstimator(PoseEstimatorConfig config)
    : config_(config) {}

std::shared_ptr<open3d::geometry::PointCloud> ClassicalPoseEstimator::pointCloudFromRGBD(
    const RGBDFrame& frame, const CameraIntrinsics& intrinsics) {
    if (!frame.valid() || !intrinsics.valid()) {
        std::cerr << "pointCloudFromRGBD: invalid frame or intrinsics.\n";
        return nullptr;
    }

    // convert_rgb_to_intensity=false: keep color channels around, in case
    // a later stage wants color-aided filtering, even though the
    // classical pipeline below only ever uses geometry (FPFH).
    auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
        *frame.color, *frame.depth, frame.depthScale, frame.depthTruncMeters,
        /*convert_rgb_to_intensity=*/false);

    auto cloud = open3d::geometry::PointCloud::CreateFromRGBDImage(*rgbd, intrinsics.toOpen3D());
    if (!cloud || cloud->points_.empty()) {
        std::cerr << "pointCloudFromRGBD: resulting cloud is empty.\n";
        return nullptr;
    }
    return cloud;
}

std::shared_ptr<open3d::geometry::PointCloud> ClassicalPoseEstimator::preprocessSceneCloud(
    const std::shared_ptr<open3d::geometry::PointCloud>& raw) const {
    if (!raw || raw->points_.empty()) return nullptr;

    std::cerr << "[PoseEstimator] rawCloud bounds:"
              << " X[" << raw->GetMinBound().x() << " ~ " << raw->GetMaxBound().x() << "]"
              << " Y[" << raw->GetMinBound().y() << " ~ " << raw->GetMaxBound().y() << "]"
              << " Z[" << raw->GetMinBound().z() << " ~ " << raw->GetMaxBound().z() << "]\n";

    std::shared_ptr<open3d::geometry::PointCloud> roi = raw;
    if (config_.cropEnabled) {
        open3d::geometry::AxisAlignedBoundingBox box(config_.cropMin, config_.cropMax);
        roi = raw->Crop(box);
        std::cerr << "[PoseEstimator] after ROI crop: " << roi->points_.size()
                  << " points (from " << raw->points_.size() << ")\n";
        if (roi->points_.empty()) {
            std::cerr << "preprocessSceneCloud: empty after ROI crop -- "
                         "cropMin/cropMax likely don't overlap the scene cloud.\n";
            return nullptr;
        }
    }

    auto downsampled = roi->VoxelDownSample(config_.sceneVoxelSize);
    if (!downsampled || downsampled->points_.empty()) {
        std::cerr << "preprocessSceneCloud: empty after voxel downsample -- "
                     "sceneVoxelSize may be too large for this scene.\n";
        return nullptr;
    }

    auto [filtered, keptIdx] = downsampled->RemoveStatisticalOutliers(
        config_.outlierNeighbors, config_.outlierStdRatio);
    if (!filtered || filtered->points_.empty()) {
        std::cerr << "preprocessSceneCloud: empty after outlier removal.\n";
        return nullptr;
    }

    filtered->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(config_.normalRadius, 30));
    
    // Open3D's EstimateNormals will output (NaN, NaN, NaN) for isolated points
    // that have 0 neighbors within normalRadius. ComputeFPFHFeature segfaults
    // on NaN normals, so we MUST sanitize them out.
    std::vector<Eigen::Vector3d> cleanPoints;
    std::vector<Eigen::Vector3d> cleanNormals;
    cleanPoints.reserve(filtered->points_.size());
    cleanNormals.reserve(filtered->normals_.size());
    for (size_t i = 0; i < filtered->points_.size(); ++i) {
        if (filtered->normals_[i].hasNaN()) continue;
        cleanPoints.push_back(filtered->points_[i]);
        cleanNormals.push_back(filtered->normals_[i]);
    }
    filtered->points_ = std::move(cleanPoints);
    filtered->normals_ = std::move(cleanNormals);

    if (filtered->points_.empty()) {
        std::cerr << "preprocessSceneCloud: empty after NaN normal sanitization.\n";
        return nullptr;
    }

    // CreateFromRGBDImage points are already in camera frame, so "towards
    // the origin" is a well-defined outward-from-surface convention --
    // needed by point-to-plane ICP later.
    filtered->OrientNormalsTowardsCameraLocation(Eigen::Vector3d::Zero());

    return filtered;
}

std::shared_ptr<open3d::geometry::PointCloud> ClassicalPoseEstimator::rescaleCadCloudToMetric(
    const CADAsset& cadAsset) {
    auto cloud = std::make_shared<open3d::geometry::PointCloud>();
    if (!cadAsset.valid()) {
        std::cerr << "rescaleCadCloudToMetric: CADAsset is not valid().\n";
        return cloud;
    }

    cloud->points_.reserve(cadAsset.sampledFeatures.size());
    cloud->normals_.reserve(cadAsset.sampledFeatures.size());
    for (const auto& f : cadAsset.sampledFeatures) {
        if (f.normal.hasNaN()) continue; // Skip points with invalid normals to prevent FPFH segfault
        // sampledFeatures live in unit-normalized space (diag == 1, see
        // CADAsset.h) -- multiply by realBoundingDiag to get back to the
        // metric units the RGB-D scene cloud is in. Normal direction
        // itself is scale-invariant, so it's copied as-is.
        cloud->points_.push_back(f.position * cadAsset.realBoundingDiag);
        cloud->normals_.push_back(f.normal);
    }
    return cloud;
}

std::shared_ptr<open3d::pipelines::registration::Feature> ClassicalPoseEstimator::computeFPFH(
    const open3d::geometry::PointCloud& cloud, double normalRadius, double fpfhRadius) const {
    if (cloud.points_.empty()) return nullptr;

    // FPFH needs normals; both callers below (rescaled CAD cloud,
    // preprocessed scene cloud) should already have them, but guard here
    // since ComputeFPFHFeature silently produces garbage on a cloud with
    // missing/mismatched normals rather than failing loudly.
    open3d::geometry::PointCloud withNormals = cloud;
    if (withNormals.normals_.size() != withNormals.points_.size()) {
        withNormals.EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(normalRadius, 30));
    }

    return open3d::pipelines::registration::ComputeFPFHFeature(
        withNormals, open3d::geometry::KDTreeSearchParamHybrid(fpfhRadius, 100));
}

open3d::pipelines::registration::RegistrationResult ClassicalPoseEstimator::ransacGlobalRegistration(
    const open3d::geometry::PointCloud& cadCloud,
    const open3d::geometry::PointCloud& sceneCloud,
    const open3d::pipelines::registration::Feature& cadFeatures,
    const open3d::pipelines::registration::Feature& sceneFeatures) const {

    // Loosened from 0.9 -> 0.8: on noisy/low-texture correspondence sets,
    // the stricter ratio rejects true-positive triplets almost as often as
    // false ones, so it wasn't buying much precision at the cost of a lot
    // of recall. Revert to 0.9 once the object/scene crop is dialed in and
    // correspondence quality improves.
    open3d::pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength edgeChecker(0.8);
    open3d::pipelines::registration::CorrespondenceCheckerBasedOnDistance distChecker(
        config_.ransacDistanceThreshold);
    std::vector<std::reference_wrapper<const open3d::pipelines::registration::CorrespondenceChecker>>
        checkers{edgeChecker, distChecker};

    // source = cadCloud, target = sceneCloud: see the header comment on
    // this method for why that direction matters.
    //
    // mutual_filter=false: on low-texture/planar geometry, FPFH descriptors
    // are nearly identical across the whole surface, so requiring a mutual
    // (A->B AND B->A) nearest-feature match throws away almost every
    // correspondence. Disabling it keeps a larger (noisier) correspondence
    // pool and leans on RANSAC's iteration budget + the checkers above to
    // find a consistent inlier set instead.
    return open3d::pipelines::registration::RegistrationRANSACBasedOnFeatureMatching(
        cadCloud, sceneCloud, cadFeatures, sceneFeatures,
        /*mutual_filter=*/false,
        config_.ransacDistanceThreshold,
        open3d::pipelines::registration::TransformationEstimationPointToPoint(false),
        /*ransac_n=*/3,
        checkers,
        open3d::pipelines::registration::RANSACConvergenceCriteria(
            config_.ransacMaxIterations, config_.ransacConfidence));
}

open3d::pipelines::registration::RegistrationResult ClassicalPoseEstimator::refineWithICP(
    const open3d::geometry::PointCloud& cadCloud,
    const open3d::geometry::PointCloud& sceneCloud,
    const Eigen::Matrix4d& initialGuess) const {

    return open3d::pipelines::registration::RegistrationICP(
        cadCloud, sceneCloud, config_.icpMaxCorrespondenceDistance, initialGuess,
        open3d::pipelines::registration::TransformationEstimationPointToPlane());
}

Pose6DoFResult ClassicalPoseEstimator::estimate(const CADAsset& cadAsset,
                                                 const RGBDFrame& frame,
                                                 const CameraIntrinsics& intrinsics) const {
    Pose6DoFResult result;

    if (!cadAsset.valid()) {
        std::cerr << "ClassicalPoseEstimator::estimate: CADAsset invalid.\n";
        return result; // valid=false
    }

    auto rawScene = pointCloudFromRGBD(frame, intrinsics);
    if (!rawScene) return result;
    result.numRawScenePoints = rawScene->points_.size();
    std::cerr << "[PoseEstimator] rawScene points: " << rawScene->points_.size() << "\n";

    auto sceneCloud = preprocessSceneCloud(rawScene);
    if (!sceneCloud) return result;
    result.numSceneCloudPoints = sceneCloud->points_.size();
    std::cerr << "[PoseEstimator] sceneCloud points after voxel+outlier filter: "
              << sceneCloud->points_.size() << "\n";

    auto cadCloud = rescaleCadCloudToMetric(cadAsset);
    if (cadCloud->points_.empty()) return result;
    result.numCadCloudPoints = cadCloud->points_.size();
    std::cerr << "[PoseEstimator] cadCloud points (rescaled to metric, realBoundingDiag="
              << cadAsset.realBoundingDiag << "): " << cadCloud->points_.size() << "\n";

    // Quick, cheap sanity check before spending time on FPFH: if the two
    // clouds' bounding diagonals are wildly different orders of magnitude,
    // it's almost always a unit-scale bug (CADAsset.sourceUnitScale) rather
    // than a real geometry mismatch -- catch it here instead of reading it
    // off a silent RANSAC fitness of 0 later.
    double cadDiag = (cadCloud->GetMaxBound() - cadCloud->GetMinBound()).norm();
    double sceneDiag = (sceneCloud->GetMaxBound() - sceneCloud->GetMinBound()).norm();
    result.cadCloudDiag = cadDiag;
    result.sceneCloudDiag = sceneDiag;
    std::cerr << "[PoseEstimator] cadCloud bbox diag: " << cadDiag
              << " m, sceneCloud bbox diag: " << sceneDiag << " m\n";
    if (cadDiag > 0.0 && (cadDiag / sceneDiag > 20.0 || sceneDiag / cadDiag > 20.0)) {
        std::cerr << "[PoseEstimator] WARNING: cadCloud and sceneCloud bounding "
                     "diagonals differ by >20x -- check the CAD file's .unit "
                     "sidecar before trusting the fitness number below.\n";
    }

    // Debug dump: load these two into the app's own viewer (File > Add,
    // .ply is supported) to SEE where the object actually sits relative to
    // the cropped scene, instead of guessing cropMin/cropMax numerically.
#ifdef DEBUG_POSE_ESTIMATOR
    open3d::io::WritePointCloud("/home/huyen1905/TMCAD/debug/debug_scene_cropped.ply", *sceneCloud);
    open3d::io::WritePointCloud("/home/huyen1905/TMCAD/debug/debug_cad_metric.ply", *cadCloud);
    std::cerr << "[PoseEstimator] wrote /home/huyen1905/TMCAD/debug/debug_scene_cropped.ply and "
                 "/home/huyen1905/TMCAD/debug/debug_cad_metric.ply for visual inspection\n";
#endif

    auto cadFeatures = computeFPFH(*cadCloud, config_.normalRadius, config_.fpfhRadius);
    auto sceneFeatures = computeFPFH(*sceneCloud, config_.normalRadius, config_.fpfhRadius);
    if (!cadFeatures || !sceneFeatures) return result;

    auto ransacResult = ransacGlobalRegistration(*cadCloud, *sceneCloud, *cadFeatures, *sceneFeatures);

    result.numRansacInliers = static_cast<int>(ransacResult.correspondence_set_.size());
    result.fitness = ransacResult.fitness_;
    result.inlierRmse = ransacResult.inlier_rmse_;

    if (ransacResult.fitness_ < kMinAcceptableFitness || ransacResult.correspondence_set_.empty()) {
        // Per Phase 0/5 policy: below the minimum-correspondence threshold,
        // report "pose không xác định" rather than trusting a low-confidence
        // guess -- R/t are NOT populated from ransacResult in this branch.
        std::cerr << "ClassicalPoseEstimator::estimate: RANSAC fitness " << ransacResult.fitness_
                  << " below threshold " << kMinAcceptableFitness << " -- rejecting pose.\n";
        result.valid = false;
        return result;
    }

    Eigen::Matrix4d finalTransform = ransacResult.transformation_;

    if (config_.refineWithICP) {
        auto icpResult = refineWithICP(*cadCloud, *sceneCloud, finalTransform);
        if (icpResult.fitness_ > 0.0) {
            finalTransform = icpResult.transformation_;
            result.fitness = icpResult.fitness_;
            result.inlierRmse = icpResult.inlier_rmse_;
        }
        // If ICP fitness is 0 (diverged / no correspondences at this
        // threshold), keep the RANSAC transform rather than overwrite it
        // with a worse-than-nothing ICP result.
    }

    result.R = finalTransform.block<3, 3>(0, 0);
    result.t = finalTransform.block<3, 1>(0, 3);
    result.valid = true;
    return result;
}