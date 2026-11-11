#pragma once
#include "RGBDSource.h"
#include "RGBDTypes.h"
#include "CADAsset.h"
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <memory>
#include <random>
#include <vector>

// One CAD<->observation correspondence recovered directly from ray casting:
// pixel (u,v) hit CAD triangle `triangleId` at `cadPointObjectFrame` (in the
// object's own local frame -- i.e. the SAME frame as CADAsset::normalizedMesh
// BEFORE meshFromCADAsset()'s metric rescale, so this lines up directly with
// CADAsset::sampledFeatures positions once you divide by realBoundingDiag)
// / `cameraPointMetric` (camera frame, meters, matches what
// ClassicalPoseEstimator::pointCloudFromRGBD() would produce for this pixel).
//
// This is the ground-truth pixel<->CAD pairing that roadmap Phase 3's
// contrastive/InfoNCE training between the CAD encoder and the point-cloud
// encoder needs -- see the "GT correspondence" discussion that motivated
// this class. Only populated when SyntheticSceneConfig::captureCorrespondences
// is true, and only for pixels that hit a VISIBLE (non-hidden-cell,
// non-background-plane) object triangle.
struct CadCorrespondence {
    int u = 0;
    int v = 0;
    int triangleId = -1;
    Eigen::Vector3d cadPointObjectFrame{0, 0, 0};
    Eigen::Vector3d cameraPointMetric{0, 0, 0};
};

// Knobs for one synthetic scene. Kept separate from CameraIntrinsics /
// RGBDFrame so a single config can be reused across grabFrame() calls if
// SyntheticRGBDSource is ever extended to a multi-frame sequence source.
struct SyntheticSceneConfig {
    // T_camera<-object: transforms a point in the OBJECT's own local frame
    // into the CAMERA frame (Z=forward, Y=down, X=right, per RGBDTypes.h's
    // convention). This is exactly the ground truth that
    // ClassicalPoseEstimator::estimate() is trying to recover -- see
    // groundTruthCameraFromObject() below for how to compare against it.
    Eigen::Matrix4d cameraFromObject = Eigen::Matrix4d::Identity();

    // --- Realistic active-stereo depth model (D435-style) --------------
    // When true, depth noise and dropout are derived from sensor physics
    // (distance + surface angle) instead of the flat toy model below.
    // Prefer this ON for anything meant to represent a real RealSense
    // frame; the flat model is mainly useful for isolating "does crop
    // help at all" from "does crop survive realistic noise".
    bool useRealisticNoiseModel = true;

    // Stereo baseline (meters) between the two IR imagers. Intel lists
    // ~0.05m (50mm) for the D435 family -- check your specific unit's
    // datasheet and adjust if it differs (D415 is ~0.055m, for instance).
    double baselineMeters = 0.05;

    // Sub-pixel disparity matching error (pixels), the dominant term in
    // active-stereo depth noise. ~0.08px is a commonly cited approximate
    // figure in the stereo-depth literature, NOT a guaranteed factory
    // spec for this exact unit -- treat this as a starting point to tune
    // against your own measured RMS error at a known distance, not as an
    // authoritative datasheet number.
    double subpixelDisparityErrorPx = 0.08;

    // Surfaces whose normal makes an angle with the viewing ray greater
    // than this are treated as unable to return valid depth -- mirrors
    // real stereo matching failing at grazing incidence (this is exactly
    // the kind of surface TABLE-DEMO's thin legs present at many viewing
    // angles, per the roadmap's symmetry/geometry diagnosis).
    double maxIncidenceAngleDeg = 85.0;

    // Depth range clipping (meters) -- pixels whose true depth falls
    // outside [minRangeMeters, maxRangeMeters] are reported invalid, same
    // as a real sensor's near/far cutoff. D435-family near limits are
    // typically ~0.2m depending on visual preset/resolution; check your
    // own configuration rather than trusting this as exact.
    double minRangeMeters = 0.2;
    double maxRangeMeters = 10.0;

    // --- Flat/toy noise model (used only when useRealisticNoiseModel
    // is false) -----------------------------------------------------
    // Gaussian noise stddev (meters) added to each valid depth pixel,
    // constant regardless of distance or angle -- simpler to reason
    // about when you specifically want to isolate one variable, but not
    // representative of how a real depth sensor actually degrades.
    double depthNoiseStdMeters = 0.0;

    // Fraction of otherwise-valid depth pixels randomly zeroed out on
    // TOP OF whatever the noise model above already invalidates, to
    // approximate additional sensor dropout / specular holes. 0..1, 0.0
    // = disabled. NOTE: this is unstructured per-pixel dropout -- for
    // Phase 5's structured "hide this fraction of the object surface"
    // benchmark, use occlusionGridDims / targetVisibleFraction below
    // instead. The two stack: dropout can further thin an already
    // partially-occluded frame.
    double randomDropoutFraction = 0.0;

    // Adds a large flat quad behind the object (in camera frame, centered
    // on the camera's forward axis) purely to give crop/ROI something
    // real to remove -- simulates the "whole room" clutter problem
    // documented against TABLE-DEMO.stl (scene ~3.4m vs object ~1.09m).
    bool addBackgroundPlane = false;
    double backgroundPlaneDistanceMeters = 3.0; // along camera +Z from origin
    double backgroundPlaneSizeMeters = 6.0;     // edge length of the (square) plane

    // --- Structured occlusion (Phase 5 visibility grid) -----------------
    // Divides the object's OWN local bounding box into
    // occlusionGridDims.x * .y * .z cells (computed BEFORE the object is
    // placed/rotated into the camera frame, so the grid is stable across
    // different test poses of the same mesh) and randomly hides whole
    // cells until the visible *surface area* fraction is as close as
    // possible to targetVisibleFraction (1.0 = fully visible, i.e. Phase
    // 5's "100%" row; 0.5 = roughly half the object's surface area
    // hidden, i.e. the "50%" row).
    //
    // This is deliberately a SEPARATE knob from randomDropoutFraction
    // above: that one is unstructured per-pixel sensor noise, this is
    // "hide this fraction of the object surface" -- the piece this class
    // historically did not implement (see prior version's comments).
    Eigen::Vector3i occlusionGridDims{1, 1, 1};   // {1,1,1} = disabled (fully visible)
    double targetVisibleFraction = 1.0;           // 1.0 = disabled (no cells hidden)
    unsigned occlusionSeed = 0;                   // 0 = derive from this source's own rng_
                                                   // (non-reproducible cell choice);
                                                   // nonzero = reproducible cell selection

    // When true, grabFrame() also fills lastCorrespondences() with one
    // entry per pixel that hit a VISIBLE (non-hidden, non-background)
    // object triangle -- the GT pixel<->CAD-triangle pairing Phase 3
    // needs. Off by default: it adds one extra tensor readback and a
    // per-pixel push_back, which Phase 2's classical-baseline sweeps
    // don't need.
    bool captureCorrespondences = false;
};

// IRGBDSource backend that never touches real hardware or disk images:
// it ray-casts a CAD mesh, placed at a caller-chosen pose relative to the
// camera, into a depth+color RGBDFrame using the exact same intrinsics
// convention (RGBDTypes.h) and depth encoding (uint16 mm, depthScale to
// divide back to meters) the rest of the pipeline already expects.
//
// Why this exists: crop/ROI testing (and Phase 5 occlusion testing) needs
// a scene where the TRUE pose is known exactly and reproducible across
// repeated runs while cropMin/cropMax, object choice, or occlusion level
// is swept -- a live RealSense frame can't offer any of that. See
// groundTruthCameraFromObject() to score ClassicalPoseEstimator's output
// against this class's own input, using Phase 5's e_t / e_R formulas, and
// see lastCorrespondences() / SyntheticDatasetGenerator for Phase 3's
// bulk-training-data needs.
//
// Single-shot like FileRGBDSource: one instance serves one grabFrame()
// call, then returns false. Construct a new instance (with a new
// SyntheticSceneConfig) per test case -- SyntheticDatasetGenerator does
// exactly this in a loop to produce a dataset.
class SyntheticRGBDSource : public IRGBDSource {
public:
    // objectMeshMetric: CAD mesh in METRIC units (meters), in the object's
    // own local frame -- i.e. the physical-scale equivalent of
    // ClassicalPoseEstimator::rescaleCadCloudToMetric()'s output, but as a
    // mesh (for ray casting) instead of a sampled point cloud. Use
    // meshFromCADAsset() below to build this from a CADAsset you already
    // have cached via SceneManager::getOrProcessCADAsset().
    SyntheticRGBDSource(std::shared_ptr<open3d::geometry::TriangleMesh> objectMeshMetric,
                         CameraIntrinsics intrinsics,
                         SyntheticSceneConfig config);

    bool open() override;
    void close() override;
    bool isOpen() const override;
    CameraIntrinsics intrinsics() const override;
    bool grabFrame(RGBDFrame& outFrame) override;
    std::string deviceName() const override;

    // Ground truth T_camera<-object used to render the frame just served.
    // NOT part of IRGBDSource -- a real camera has no idea where the
    // object is, this only exists because we placed it there ourselves.
    // Compare this against ClassicalPoseEstimator's Pose6DoFResult::asMatrix()
    // using Phase 5's e_t = ||t_pred - t_gt|| and
    // e_R = acos((trace(R_pred * R_gt^T) - 1) / 2) formulas.
    const Eigen::Matrix4d& groundTruthCameraFromObject() const { return config_.cameraFromObject; }

    // Achieved visible-surface-area fraction from the last grabFrame()
    // call. Only meaningful when occlusionGridDims has >1 total cell or
    // targetVisibleFraction < 1.0 -- otherwise always 1.0. Won't exactly
    // equal targetVisibleFraction since cells are discrete (a greedy
    // cell-by-cell selection stops at the closest achievable value), so
    // log THIS value, not the requested one, alongside each dataset
    // sample (SyntheticDatasetGenerator does this in pose.json).
    double lastVisibleFractionAchieved() const { return lastVisibleFractionAchieved_; }

    // Valid only if config_.captureCorrespondences was true for the last
    // grabFrame() call. One entry per pixel that hit a visible object
    // triangle; background-plane hits, occluded/hidden-cell hits, and
    // range/incidence/dropout-invalidated hits are all excluded, so every
    // entry here is a pixel that also has a valid depth value in the
    // frame just returned.
    const std::vector<CadCorrespondence>& lastCorrespondences() const { return lastCorrespondences_; }

    // --- Convenience builders -----------------------------------------

    // Copies cadAsset.normalizedMesh (unit-normalized, diag == 1) and
    // scales it by realBoundingDiag to get a mesh in metric units, ready
    // to pass to this class's constructor. Mirrors
    // ClassicalPoseEstimator::rescaleCadCloudToMetric() but on the full
    // mesh rather than the Poisson-sampled feature cloud.
    static std::shared_ptr<open3d::geometry::TriangleMesh> meshFromCADAsset(
        const CADAsset& cadAsset);

    // Builds a cameraFromObject placing the object's own origin directly
    // on the camera's forward (+Z) axis at distanceMeters, after applying
    // an optional X-then-Y-then-Z rotation (degrees) to the object first
    // -- so repeated test runs can look at different faces of an
    // otherwise-identical object without hand-deriving a matrix each time.
    static Eigen::Matrix4d makeFrontFacingPose(double distanceMeters,
                                                double rotXDeg = 0.0,
                                                double rotYDeg = 0.0,
                                                double rotZDeg = 0.0);

    // Horizontal/vertical field of view (degrees) implied by a pinhole
    // model's intrinsics -- FOV isn't an independent knob, it falls out
    // of fx/fy/width/height (FOV = 2*atan(dim / (2*f))), so this exists
    // purely so a caller/UI can display "what FOV does this camera
    // config actually have" and sanity-check it against the real
    // sensor's spec (D435-family depth FOV is roughly 87 deg H x 58 deg
    // V) rather than trusting fx/fy numbers blindly.
    static double horizontalFovDeg(const CameraIntrinsics& intr);
    static double verticalFovDeg(const CameraIntrinsics& intr);

private:
    // Builds triangleCellId_ (once per instance, from objectMesh_'s own
    // local-frame bounding box) and (re)derives cellHidden_ from the
    // current config_.targetVisibleFraction / occlusionSeed. Re-running
    // the cell-hiding step (but not the triangle->cell assignment) on
    // every grabFrame() call is intentional: a caller that mutates
    // config_ between calls on the same instance (not the usual pattern,
    // but not forbidden) sees the new occlusion level take effect.
    void ensureOcclusionGrid();
    int cellIndexForPoint(const Eigen::Vector3d& localPoint) const;

    std::shared_ptr<open3d::geometry::TriangleMesh> objectMesh_;
    CameraIntrinsics intrinsics_;
    SyntheticSceneConfig config_;
    bool opened_ = false;
    bool served_ = false;
    std::mt19937 rng_{std::random_device{}()};

    // Occlusion-grid bookkeeping.
    bool occlusionGridBuilt_ = false;
    std::vector<int> triangleCellId_;     // size == objectMesh_->triangles_.size()
    std::vector<char> cellHidden_;        // size == gridDims.x*y*z; 1 = hidden this frame
    Eigen::Vector3d occlusionGridMin_{0, 0, 0};
    Eigen::Vector3d occlusionGridStep_{1, 1, 1};
    double lastVisibleFractionAchieved_ = 1.0;

    std::vector<CadCorrespondence> lastCorrespondences_;
};