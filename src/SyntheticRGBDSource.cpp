#include "SyntheticRGBDSource.h"
#include <open3d/t/geometry/RaycastingScene.h>
#include <open3d/t/geometry/TriangleMesh.h>
#include <open3d/core/Tensor.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {
// Flat quad, authored directly in CAMERA frame (not object frame -- unlike
// the object mesh, this never gets multiplied by cameraFromObject), facing
// back towards the camera along -Z. Purely clutter for crop/ROI to remove.
std::shared_ptr<open3d::geometry::TriangleMesh> makeBackgroundPlane(double distance, double size) {
    auto plane = std::make_shared<open3d::geometry::TriangleMesh>();
    double h = size * 0.5;
    plane->vertices_ = {
        Eigen::Vector3d(-h, -h, distance),
        Eigen::Vector3d(h, -h, distance),
        Eigen::Vector3d(h, h, distance),
        Eigen::Vector3d(-h, h, distance),
    };
    plane->triangles_ = {
        Eigen::Vector3i(0, 1, 2),
        Eigen::Vector3i(0, 2, 3),
    };
    plane->ComputeVertexNormals();
    return plane;
}
}  // namespace

SyntheticRGBDSource::SyntheticRGBDSource(std::shared_ptr<open3d::geometry::TriangleMesh> objectMeshMetric,
                                          CameraIntrinsics intrinsics,
                                          SyntheticSceneConfig config)
    : objectMesh_(std::move(objectMeshMetric)),
      intrinsics_(intrinsics),
      config_(std::move(config)) {}

bool SyntheticRGBDSource::open() {
    if (!objectMesh_ || objectMesh_->vertices_.empty()) {
        std::cerr << "SyntheticRGBDSource: object mesh is empty -- did you forget "
                     "to rescale it to metric via meshFromCADAsset()?\n";
        return false;
    }
    if (!intrinsics_.valid()) {
        std::cerr << "SyntheticRGBDSource: intrinsics invalid (width/height/fx/fy must be > 0).\n";
        return false;
    }
    opened_ = true;
    served_ = false;
    return true;
}

void SyntheticRGBDSource::close() { opened_ = false; }
bool SyntheticRGBDSource::isOpen() const { return opened_; }
CameraIntrinsics SyntheticRGBDSource::intrinsics() const { return intrinsics_; }
std::string SyntheticRGBDSource::deviceName() const { return "Synthetic (ray-cast, no hardware)"; }

// ---------------------------------------------------------------------------
// Structured occlusion grid (Phase 5 visibility benchmark)
// ---------------------------------------------------------------------------

int SyntheticRGBDSource::cellIndexForPoint(const Eigen::Vector3d& localPoint) const {
    const auto& dims = config_.occlusionGridDims;
    auto clampIdx = [](int i, int n) { return std::clamp(i, 0, std::max(0, n - 1)); };
    Eigen::Vector3d rel = localPoint - occlusionGridMin_;
    int ix = clampIdx(static_cast<int>(rel.x() / std::max(1e-12, occlusionGridStep_.x())), dims.x());
    int iy = clampIdx(static_cast<int>(rel.y() / std::max(1e-12, occlusionGridStep_.y())), dims.y());
    int iz = clampIdx(static_cast<int>(rel.z() / std::max(1e-12, occlusionGridStep_.z())), dims.z());
    return (iz * std::max(1, dims.y()) + iy) * std::max(1, dims.x()) + ix;
}

void SyntheticRGBDSource::ensureOcclusionGrid() {
    const auto dims = config_.occlusionGridDims;
    const int totalCells = std::max(1, dims.x()) * std::max(1, dims.y()) * std::max(1, dims.z());

    // Triangle -> cell assignment only needs computing once per instance
    // (the mesh's own local-frame geometry never changes across calls).
    if (!occlusionGridBuilt_) {
        triangleCellId_.assign(objectMesh_->triangles_.size(), 0);

        if (totalCells > 1 && !objectMesh_->vertices_.empty()) {
            Eigen::Vector3d mn = objectMesh_->GetMinBound();
            Eigen::Vector3d mx = objectMesh_->GetMaxBound();
            Eigen::Vector3d extent = (mx - mn).cwiseMax(Eigen::Vector3d(1e-9, 1e-9, 1e-9));
            occlusionGridMin_ = mn;
            occlusionGridStep_ = Eigen::Vector3d(
                extent.x() / std::max(1, dims.x()),
                extent.y() / std::max(1, dims.y()),
                extent.z() / std::max(1, dims.z()));

            for (size_t i = 0; i < objectMesh_->triangles_.size(); ++i) {
                const auto& tri = objectMesh_->triangles_[i];
                Eigen::Vector3d centroid = (objectMesh_->vertices_[tri(0)] +
                                             objectMesh_->vertices_[tri(1)] +
                                             objectMesh_->vertices_[tri(2)]) / 3.0;
                triangleCellId_[i] = cellIndexForPoint(centroid);
            }
        }
        occlusionGridBuilt_ = true;
    }

    // Per-cell hidden mask is re-derived every call (cheap -- proportional
    // to triangle count, not pixel count) so a caller sweeping
    // targetVisibleFraction across Phase 5's 100/75/50/25% rows on the
    // SAME cached mesh (as SyntheticDatasetGenerator does across separate
    // instances, or a caller reusing one instance across calls) always
    // gets the level it just asked for.
    cellHidden_.assign(totalCells, 0);
    lastVisibleFractionAchieved_ = 1.0;

    if (totalCells <= 1 || config_.targetVisibleFraction >= 1.0 - 1e-9) {
        return;  // nothing hidden -- fully visible, identical to pre-occlusion behavior
    }
    if (config_.targetVisibleFraction <= 1e-9) {
        std::fill(cellHidden_.begin(), cellHidden_.end(), 1);
        lastVisibleFractionAchieved_ = 0.0;
        return;
    }

    // Per-cell surface area (sum of triangle areas assigned to that cell),
    // so hiding a cell removes an area-proportional chunk of surface
    // rather than just a triangle count -- matters when triangle density
    // varies across the mesh (e.g. fine detail on one face).
    std::vector<double> cellArea(totalCells, 0.0);
    double totalArea = 0.0;
    for (size_t i = 0; i < objectMesh_->triangles_.size(); ++i) {
        const auto& tri = objectMesh_->triangles_[i];
        const Eigen::Vector3d& a = objectMesh_->vertices_[tri(0)];
        const Eigen::Vector3d& b = objectMesh_->vertices_[tri(1)];
        const Eigen::Vector3d& c = objectMesh_->vertices_[tri(2)];
        double area = 0.5 * (b - a).cross(c - a).norm();
        cellArea[triangleCellId_[i]] += area;
        totalArea += area;
    }
    if (totalArea < 1e-12) return;

    // Randomly order cells, then hide them one at a time until hiding the
    // next one would overshoot the requested occluded-area fraction.
    // Greedy, not exact -- adequate for a coarse benchmark grid (Phase 5
    // only needs four rough buckets), and lastVisibleFractionAchieved_
    // reports the actual value so downstream logging isn't misled.
    std::vector<int> order(totalCells);
    for (int i = 0; i < totalCells; ++i) order[i] = i;
    std::mt19937 gridRng(config_.occlusionSeed != 0
                              ? config_.occlusionSeed
                              : static_cast<unsigned>(rng_()));
    std::shuffle(order.begin(), order.end(), gridRng);

    const double targetOccludedArea = (1.0 - config_.targetVisibleFraction) * totalArea;
    double occludedSoFar = 0.0;
    for (int cell : order) {
        if (cellArea[cell] <= 0.0) continue;
        if (occludedSoFar > 0.0 && occludedSoFar + cellArea[cell] > targetOccludedArea) {
            // Stop before overshooting, but only once at least one cell is
            // already hidden -- otherwise a >0% occlusion request could
            // silently hide nothing just because the very first candidate
            // cell alone already exceeds the target, which is a worse
            // failure mode than landing slightly past the target.
            break;
        }
        cellHidden_[cell] = 1;
        occludedSoFar += cellArea[cell];
        if (occludedSoFar >= targetOccludedArea) break;
    }
    lastVisibleFractionAchieved_ = 1.0 - (occludedSoFar / totalArea);
}

// ---------------------------------------------------------------------------
// grabFrame
// ---------------------------------------------------------------------------

bool SyntheticRGBDSource::grabFrame(RGBDFrame& outFrame) {
    if (!opened_ || served_) return false;

    ensureOcclusionGrid();

    // --- Assemble the scene directly in CAMERA frame: object mesh is
    // transformed by cameraFromObject; the background plane (if any) is
    // already authored in camera frame by makeBackgroundPlane(), so no
    // further transform is applied to it.
    const size_t numObjectTriangles = objectMesh_->triangles_.size();

    open3d::geometry::TriangleMesh sceneMesh = *objectMesh_;
    sceneMesh.Transform(config_.cameraFromObject);

    if (config_.addBackgroundPlane) {
        auto bg = makeBackgroundPlane(config_.backgroundPlaneDistanceMeters,
                                       config_.backgroundPlaneSizeMeters);
        sceneMesh += *bg;  // TriangleMesh::operator+= merges vertices/triangles,
                            // appending bg's triangles AFTER the object's own --
                            // so a hit primitive id < numObjectTriangles means
                            // "this pixel saw the object", not the background.
    }

    // --- Ray cast via Open3D's tensor-based RaycastingScene (open3d::t::
    // namespace -- distinct from the legacy open3d::geometry:: API used
    // everywhere else in this codebase; FromLegacy() bridges the two).
    open3d::t::geometry::RaycastingScene raycastScene;
    auto meshT = open3d::t::geometry::TriangleMesh::FromLegacy(sceneMesh);
    raycastScene.AddTriangles(meshT);

    // Identity extrinsic: sceneMesh above was already placed in camera
    // frame, so rays are cast straight out along each pixel from the
    // camera origin with no additional extrinsic transform needed.
    open3d::core::Tensor intrinsicT = open3d::core::Tensor::Init<double>(
        {{intrinsics_.fx, 0.0, intrinsics_.cx},
         {0.0, intrinsics_.fy, intrinsics_.cy},
         {0.0, 0.0, 1.0}});
    open3d::core::Tensor extrinsicT =
        open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0"));

    const int W = intrinsics_.width;
    const int H = intrinsics_.height;

    auto rays = open3d::t::geometry::RaycastingScene::CreateRaysPinhole(
        intrinsicT, extrinsicT, W, H);
    auto castResult = raycastScene.CastRays(rays);

    // t_hit is the ray parameter along each pixel's (possibly non-unit)
    // direction vector, NOT already a Z-depth -- rather than trust an
    // assumed normalization convention, recover the hit point's own Z
    // (and, for correspondence capture, the full 3D hit point) directly
    // from the same rays tensor: with origin at (0,0,0) here,
    // hitPoint = t_hit * direction, hitZ = hitPoint.z(). This is correct
    // regardless of whether CreateRaysPinhole's directions happen to be
    // unit length or not.
    //
    // primitive_normals: flat FACE normal of whichever triangle was hit
    // (not an interpolated vertex normal) -- used below to reject
    // grazing-incidence pixels the way a real active-stereo sensor would.
    // primitive_ids: which triangle of sceneMesh was hit -- used both for
    // occlusion-grid masking and for GT correspondence capture.
    const float* raysPtr = rays.GetDataPtr<float>();          // shape (H, W, 6): [ox,oy,oz,dx,dy,dz]
    const float* hitPtr = castResult["t_hit"].GetDataPtr<float>();  // shape (H, W)
    const float* normalPtr = castResult["primitive_normals"].GetDataPtr<float>();  // shape (H, W, 3)
    const uint32_t* primIdPtr = castResult["primitive_ids"].GetDataPtr<uint32_t>();  // shape (H, W)

    auto depth = std::make_shared<open3d::geometry::Image>();
    depth->Prepare(W, H, 1, 2);  // single channel, 2 bytes/px == Z16, matches RealSenseSource's own format
    auto color = std::make_shared<open3d::geometry::Image>();
    color->Prepare(W, H, 3, 1);  // RGB8

    std::uniform_real_distribution<double> dropoutDist(0.0, 1.0);
    const double favg = 0.5 * (intrinsics_.fx + intrinsics_.fy);

    uint16_t* depthPx = reinterpret_cast<uint16_t*>(depth->data_.data());
    uint8_t* colorPx = color->data_.data();

    // Only needed if capturing correspondences, but cheap to compute
    // unconditionally: maps a camera-frame point back into the object's
    // own local frame (same frame CADAsset::normalizedMesh /
    // sampledFeatures live in, before the realBoundingDiag rescale).
    const Eigen::Matrix4d objectFromCamera = config_.cameraFromObject.inverse();

    lastCorrespondences_.clear();
    if (config_.captureCorrespondences) {
        lastCorrespondences_.reserve(static_cast<size_t>(W) * static_cast<size_t>(H) / 4);
    }

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int idx = y * W + x;
            const float t = hitPtr[idx];

            uint16_t depthMm = 0;              // 0 = invalid, same convention real RGBD sensors use
            uint8_t r = 30, g = 30, b = 30;     // dark = "no return" pixel

            if (std::isfinite(t)) {
                const Eigen::Vector3d rawDir(raysPtr[idx * 6 + 3], raysPtr[idx * 6 + 4], raysPtr[idx * 6 + 5]);
                const uint32_t primId = primIdPtr[idx];
                const bool isObjectHit = primId < numObjectTriangles;

                Eigen::Vector3d dir = rawDir;
                const double dirNorm = dir.norm();
                if (dirNorm > 1e-12) dir /= dirNorm;

                double zMeters = static_cast<double>(t) * static_cast<double>(raysPtr[idx * 6 + 5]);

                bool invalid = false;

                // Range clipping -- matches a real sensor's near/far cutoff.
                if (zMeters < config_.minRangeMeters || zMeters > config_.maxRangeMeters) {
                    invalid = true;
                }

                // Grazing-incidence rejection -- surfaces nearly edge-on to
                // the viewing ray don't return valid depth on a real
                // active-stereo sensor (this is exactly the failure mode
                // TABLE-DEMO's thin legs hit at many viewing angles).
                if (!invalid && config_.useRealisticNoiseModel) {
                    Eigen::Vector3d n(normalPtr[idx * 3 + 0], normalPtr[idx * 3 + 1], normalPtr[idx * 3 + 2]);
                    const double nNorm = n.norm();
                    if (nNorm > 1e-12) n /= nNorm;
                    const double cosTheta = std::clamp(std::fabs(dir.dot(n)), 0.0, 1.0);
                    const double angleDeg = std::acos(cosTheta) * 180.0 / M_PI;
                    if (angleDeg > config_.maxIncidenceAngleDeg) invalid = true;
                }

                // Structured occlusion (Phase 5 visibility grid) -- a hit
                // on the object whose triangle falls in a "hidden" grid
                // cell reports no depth, exactly like a real occluder
                // physically blocking that part of the surface. Hits on
                // the background plane are never subject to this (it has
                // no occlusion grid of its own).
                if (!invalid && isObjectHit && !cellHidden_.empty()) {
                    const int cell = triangleCellId_[primId];
                    if (cell >= 0 && cell < static_cast<int>(cellHidden_.size()) && cellHidden_[cell]) {
                        invalid = true;
                    }
                }

                // Extra unstructured dropout, stacked on top of whatever
                // the physical model + occlusion grid above already
                // rejected.
                if (!invalid && config_.randomDropoutFraction > 0.0 &&
                    dropoutDist(rng_) < config_.randomDropoutFraction) {
                    invalid = true;
                }

                if (!invalid && zMeters > 0.0) {
                    double sigma = 0.0;
                    if (config_.useRealisticNoiseModel) {
                        // Active-stereo depth error grows with the SQUARE
                        // of distance -- standard result for
                        // disparity-based depth (error propagates through
                        // Z = baseline*fx/disparity, so d(Z)/d(disparity)
                        // scales as Z^2). Flat-stddev noise is not a
                        // realistic stand-in for this.
                        sigma = (zMeters * zMeters * config_.subpixelDisparityErrorPx) /
                                std::max(1e-9, config_.baselineMeters * favg);
                    } else {
                        sigma = config_.depthNoiseStdMeters;
                    }
                    if (sigma > 0.0) {
                        std::normal_distribution<double> noiseDist(0.0, sigma);
                        zMeters += noiseDist(rng_);
                    }

                    depthMm = static_cast<uint16_t>(
                        std::clamp(zMeters * 1000.0, 0.0, 65535.0));
                    // Flat mid-gray -- color is never used by the classical
                    // pipeline (FPFH runs on geometry only, see
                    // ClassicalPoseEstimator::pointCloudFromRGBD's
                    // convert_rgb_to_intensity=false comment), this just
                    // keeps the color image non-degenerate for anything
                    // that displays it.
                    r = g = b = 180;

                    // GT correspondence: only recorded for pixels that
                    // ended up with a valid depth value AND hit the
                    // object (not the background plane) -- keeps
                    // lastCorrespondences() consistent with "what this
                    // frame's depth image actually shows".
                    if (config_.captureCorrespondences && isObjectHit) {
                        const Eigen::Vector3d camPoint = static_cast<double>(t) * rawDir;
                        const Eigen::Vector4d camPointH(camPoint.x(), camPoint.y(), camPoint.z(), 1.0);
                        const Eigen::Vector4d localH = objectFromCamera * camPointH;

                        CadCorrespondence corr;
                        corr.u = x;
                        corr.v = y;
                        corr.triangleId = static_cast<int>(primId);
                        corr.cadPointObjectFrame = localH.head<3>() / localH.w();
                        corr.cameraPointMetric = camPoint;
                        lastCorrespondences_.push_back(corr);
                    }
                }
            }

            depthPx[idx] = depthMm;
            colorPx[idx * 3 + 0] = r;
            colorPx[idx * 3 + 1] = g;
            colorPx[idx * 3 + 2] = b;
        }
    }

    outFrame.color = color;
    outFrame.depth = depth;
    outFrame.depthScale = 1000.0;       // matches the mm encoding written above
    outFrame.depthTruncMeters = 10.0;   // generous -- synthetic depth has no real sensor range limit to truncate
    outFrame.timestampUs = 0;

    served_ = true;
    return true;
}

std::shared_ptr<open3d::geometry::TriangleMesh> SyntheticRGBDSource::meshFromCADAsset(
    const CADAsset& cadAsset) {
    if (!cadAsset.valid() || !cadAsset.normalizedMesh) {
        std::cerr << "SyntheticRGBDSource::meshFromCADAsset: CADAsset is not valid().\n";
        return std::make_shared<open3d::geometry::TriangleMesh>();
    }
    // Copy (not share) -- Scale() below mutates in place, and callers
    // should not be surprised to see cadAsset's own normalizedMesh
    // rescaled out from under them.
    auto mesh = std::make_shared<open3d::geometry::TriangleMesh>(*cadAsset.normalizedMesh);
    mesh->Scale(cadAsset.realBoundingDiag, Eigen::Vector3d::Zero());
    return mesh;
}

Eigen::Matrix4d SyntheticRGBDSource::makeFrontFacingPose(double distanceMeters,
                                                          double rotXDeg,
                                                          double rotYDeg,
                                                          double rotZDeg) {
    const double d2r = M_PI / 180.0;
    // Same X-then-Y-then-Z convention as SceneObject::transform() and the
    // .orientation sidecar (see SceneManager.cpp's readOrientationFix), so
    // rotation numbers you already know from the viewer behave the same way here.
    Eigen::Matrix3d Rx = Eigen::AngleAxisd(rotXDeg * d2r, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::Matrix3d Ry = Eigen::AngleAxisd(rotYDeg * d2r, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Matrix3d Rz = Eigen::AngleAxisd(rotZDeg * d2r, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rx * Ry * Rz;
    T.block<3, 1>(0, 3) = Eigen::Vector3d(0.0, 0.0, distanceMeters);
    return T;
}

double SyntheticRGBDSource::horizontalFovDeg(const CameraIntrinsics& intr) {
    if (intr.fx <= 0.0) return 0.0;
    return 2.0 * std::atan(static_cast<double>(intr.width) / (2.0 * intr.fx)) * 180.0 / M_PI;
}

double SyntheticRGBDSource::verticalFovDeg(const CameraIntrinsics& intr) {
    if (intr.fy <= 0.0) return 0.0;
    return 2.0 * std::atan(static_cast<double>(intr.height) / (2.0 * intr.fy)) * 180.0 / M_PI;
}