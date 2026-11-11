#include "PalletApplication.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <limits>

namespace
{
constexpr int kDepthPatchRadius = 3;
constexpr double kDepthMinM = 0.15;
constexpr double kDepthMaxM = 4.0;
constexpr int kMaxBlocks = 3;
constexpr int kMaxYDeviationPx = 40;

constexpr int kRansacIterations = 600;
constexpr double kRansacDistanceM = 0.008;
constexpr double kRansacMinInlierRatio = 0.50;
constexpr int kRansacMinPoints = 30;

// [MARKER-STYLE] Loc outlier Z bang MAD (Median Absolute Deviation) TRUOC
// khi chay RANSAC -- port tu ArucoDetector::getPlaneDepthRobust. Giup RANSAC
// khong bi keo lech boi vai diem depth-spike (canh vat the, phan xa, hole).
constexpr double kMadZOutlierSigmaMultiplier = 3.0;
constexpr double kMadZOutlierFloorM = 0.005;   // 5 mm

// [MARKER-STYLE] Trong vong lap RANSAC, loai bo ngay ung vien mat phang
// nghieng qua 60 do so voi truc quang camera (|normal.z| < 0.5) -- port tu
// ArucoDetector::fitPlaneRansacZ. Tranh RANSAC vo tinh chon trung 1 mat
// ben/mep vat the thay vi mat truoc that su.
constexpr double kPlaneTiltRejectCosZ = 0.5;

constexpr double kRoiInsetRatio = 0.15;
constexpr int kRoiMaxPoints = 900;
constexpr int kEdgeRadiusPx = 2;
constexpr double kEdgeMaxAbsDiffM = 0.015;
constexpr double kEdgeMaxRelDiff = 0.03;
constexpr double kRmseWeightEpsMm = 0.1;

constexpr double kRmseMaxQualityM = 0.006;
constexpr double kPalletWidthM = 0.830;
// If the plane-normal-derived yaw and the block-position-derived yaw
// disagree by more than this, the block plane fit is likely corrupted
// (foreshortening / side-face contamination at oblique viewing angles) --
// fall back to the yaw-only pose instead of trusting the normal.
constexpr double kMaxYawNormalDisagreementDeg = 8.0;
constexpr double kRadToDeg = 180.0 / CV_PI;
constexpr const char* kPoseFrame = "rgbd_aligned_optical_frame";

float structuralConfidence(const std::vector<const DetectionObject*>& objects)
{
    if (objects.empty()) return 0.0f;
    float value = 1.0f;
    for (const DetectionObject* object : objects)
        value = std::min(value, object ? object->confidence : 0.0f);
    return value;
}

[[maybe_unused]] static std::optional<double> computeSimpleYawDeg(
    double depth_left_m,
    double depth_right_m,
    double pallet_width_m = kPalletWidthM)
{
    if (pallet_width_m <= 0.0) {
        return std::nullopt;
    }
    const double depth_diff_m = (depth_right_m - depth_left_m);
    const double ratio = std::clamp(depth_diff_m / pallet_width_m, -1.0, 1.0);
    return std::asin(ratio) * kRadToDeg;
}

// [MARKER-STYLE] Loc outlier theo Z bang MAD, port tu
// ArucoDetector::getPlaneDepthRobust (doan "MAD pre-filter on Z"). Dung
// truoc khi dua diem vao RANSAC de RANSAC khong bi vai diem depth-spike
// (canh vat the, lo, phan xa) keo lech ket qua vote.
std::vector<cv::Vec3d> madFilterByZ(const std::vector<cv::Vec3d>& points)
{
    if (points.size() < 3) return points;

    std::vector<double> zv;
    zv.reserve(points.size());
    for (const auto& p : points) zv.push_back(p[2]);

    std::vector<double> sorted = zv;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    const double med = sorted[sorted.size() / 2];

    std::vector<double> absdev;
    absdev.reserve(zv.size());
    for (double z : zv) absdev.push_back(std::abs(z - med));
    std::nth_element(absdev.begin(), absdev.begin() + absdev.size() / 2, absdev.end());
    const double mad = absdev[absdev.size() / 2];

    const double sigma = 1.4826 * mad;
    const double threshold =
        std::max(kMadZOutlierSigmaMultiplier * sigma, kMadZOutlierFloorM);

    std::vector<cv::Vec3d> filtered;
    filtered.reserve(points.size());
    for (const auto& p : points)
    {
        if (std::abs(p[2] - med) <= threshold) filtered.push_back(p);
    }
    return filtered;
}

}

bool PalletApplication::initialize(const DetectorConfig& config,
                                   std::string& error)
{
    _config = config;
    error.clear();
    return true;
}

bool PalletApplication::process(const RgbdFrame& frame,
                                DetectorOutput& output,
                                std::string& error)
{
    const auto start = std::chrono::steady_clock::now();

    output.application_requested = true;
    output.application_applied = false;
    output.application = name();
    output.application_status.clear();
    output.application_ms = 0.0;
    output.depth_timestamp_ms = frame.depth_timestamp_ms;
    output.xyz_object_count = 0;
    output.pose_object_count = 0;

    auto finish = [&](const std::string& status, bool applied) {
        const auto end = std::chrono::steady_clock::now();
        output.application_status = status;
        output.application_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        output.result.total_ms += output.application_ms;
        output.application_applied = applied;
        return true;
    };

    if (!frame.valid())
    {
        error = "RGBD depth frame is invalid";
        return finish("rgbd_invalid", false);
    }
    if (!frame.intrinsics.valid())
    {
        error = "RGBD camera intrinsics are invalid";
        return finish("intrinsics_invalid", false);
    }
    if (lower(output.object_type) != "pallet")
    {
        error = "pallet application requires model object_type=pallet";
        return finish("object_type_mismatch", false);
    }

    std::vector<const DetectionObject*> blocks;
    std::vector<const DetectionObject*> fronts;
    blocks.reserve(output.result.objects.size());
    fronts.reserve(output.result.objects.size());

    for (const DetectionObject& object : output.result.objects)
    {
        const std::string label = lower(object.label);
        if (label == "block") blocks.push_back(&object);
        else if (label == "front") fronts.push_back(&object);
    }

    if (blocks.size() > static_cast<size_t>(kMaxBlocks))
    {
        std::sort(blocks.begin(), blocks.end(),
                  [](const DetectionObject* a, const DetectionObject* b) {
                      return a->confidence > b->confidence;
                  });
        blocks.resize(kMaxBlocks);
    }

    if (blocks.size() != 3 || fronts.size() != 1)
    {
        output.result.objects.clear();
        error.clear();
        return finish("structure_not_found", true);
    }

    std::sort(blocks.begin(), blocks.end(),
              [](const DetectionObject* a, const DetectionObject* b) {
                  return a->center_x < b->center_x;
              });

    const DetectionObject& left = *blocks[0];
    const DetectionObject& middle = *blocks[1];
    const DetectionObject& right = *blocks[2];
    const DetectionObject& front = *fronts[0];

    const float min_y = std::min({left.center_y, middle.center_y, right.center_y});
    const float max_y = std::max({left.center_y, middle.center_y, right.center_y});
    if ((max_y - min_y) > static_cast<float>(kMaxYDeviationPx))
    {
        output.result.objects.clear();
        error.clear();
        return finish("vertical_deviation_rejected", true);
    }

    const auto depth_left = patchDepth(frame,
                                       static_cast<int>(std::lround(left.center_x)),
                                       static_cast<int>(std::lround(left.center_y)));
    const auto depth_middle = patchDepth(frame,
                                         static_cast<int>(std::lround(middle.center_x)),
                                         static_cast<int>(std::lround(middle.center_y)));
    const auto depth_right = patchDepth(frame,
                                        static_cast<int>(std::lround(right.center_x)),
                                        static_cast<int>(std::lround(right.center_y)));
    if (!depth_left || !depth_middle || !depth_right)
    {
        output.result.objects.clear();
        error.clear();
        return finish("block_depth_invalid", true);
    }

    // Fallback points (single-pixel patch depth) -- used only when the
    // per-block RANSAC plane fit fails for that specific block.
    const cv::Vec3d p_left = deproject(frame.intrinsics,
                                       left.center_x,
                                       left.center_y,
                                       *depth_left);
    const cv::Vec3d p_middle = deproject(frame.intrinsics,
                                         middle.center_x,
                                         middle.center_y,
                                         *depth_middle);
    const cv::Vec3d p_right = deproject(frame.intrinsics,
                                        right.center_x,
                                        right.center_y,
                                        *depth_right);

    const std::vector<Box> block_boxes{
        toBox(left, frame.depth_z16.cols, frame.depth_z16.rows),
        toBox(middle, frame.depth_z16.cols, frame.depth_z16.rows),
        toBox(right, frame.depth_z16.cols, frame.depth_z16.rows)};

    const PlaneSummary planes = fitBlockPlanes(frame, block_boxes);

    // [MARKER-STYLE] Uu tien giao diem tia-mat phang (planeDepthAtPixel) tai
    // DUNG pixel tam bbox 2D -- port tu ArucoDetector::fitPlaneRansacZ. Day
    // la cach chinh xac nhat: hoi "mat phang da fit giao voi tia camera qua
    // pixel nay o dau", khong phu thuoc phan bo/mat can xung cua cac diem
    // da sample (khac voi centroid, van co the lech neu block bi khuat 1
    // phan khi pallet xoay). Fallback: centroid inlier -> diem tam bbox tho.
    auto refinedPoint = [&](const DetectionObject& block,
                            const BlockFitInfo* info) -> cv::Vec3d
    {
        if (info && info->ok && info->has_plane_coeffs)
        {
            if (const auto hit = planeDepthAtPixel(
                    frame.intrinsics, block.center_x, block.center_y,
                    info->plane_a, info->plane_b, info->plane_c,
                    kDepthMinM, kDepthMaxM))
            {
                return *hit;
            }
        }
        if (info && info->ok) return info->centroid;
        return deproject(frame.intrinsics, block.center_x, block.center_y, 0.0);
    };

    const BlockFitInfo* left_info = planes.blocks.size() > 0 ? &planes.blocks[0] : nullptr;
    const BlockFitInfo* middle_info = planes.blocks.size() > 1 ? &planes.blocks[1] : nullptr;
    const BlockFitInfo* right_info = planes.blocks.size() > 2 ? &planes.blocks[2] : nullptr;

    const cv::Vec3d p_left_geom =
        (left_info && left_info->ok) ? refinedPoint(left, left_info) : p_left;
    const cv::Vec3d p_middle_geom =
        (middle_info && middle_info->ok) ? refinedPoint(middle, middle_info) : p_middle;
    const cv::Vec3d p_right_geom =
        (right_info && right_info->ok) ? refinedPoint(right, right_info) : p_right;

    // [STATELESS] Khong con EMA/settle: origin la gia tri hinh hoc truc
    // tiep cua FRAME NAY. Viec gom nhieu frame de lam muot/khu nhieu la
    // trach nhiem cua Client (moi request la mot pipeline doc lap, xem
    // ghi chu dau file).
    const cv::Vec3d hole1 = (p_left_geom + p_middle_geom) * 0.5;
    const cv::Vec3d hole2 = (p_middle_geom + p_right_geom) * 0.5;
    const cv::Vec3d origin = (hole1 + hole2) * 0.5;

    std::optional<PoseAxes> pose;
    bool fallback = false;
    cv::Vec3d pose_normal{0.0, 0.0, 1.0};

    double plane_inlier_ratio = 0.0;
    double plane_rmse_m = 0.0;
    if (planes.ok)
    {
        double rmse_sum = 0.0;
        int rmse_count = 0;
        for (const BlockFitInfo& info : planes.blocks)
        {
            if (!info.ok) continue;
            rmse_sum += info.rmse_m;
            ++rmse_count;
        }
        if (rmse_count > 0)
            plane_rmse_m = rmse_sum / static_cast<double>(rmse_count);

        plane_inlier_ratio = static_cast<double>(planes.total_inliers) /
                             static_cast<double>(std::max(planes.total_points, 1));

        cv::Vec3d normal = planes.normal;
        if (const auto n = normalized(normal)) normal = *n;

        pose = buildPoseFromNormal(p_left_geom, p_right_geom, normal);
        pose_normal = pose ? pose->z : -normal;
    }
    else
    {
        pose = buildYawOnlyPose(p_left_geom, p_right_geom);
        fallback = true;
        if (pose) pose_normal = pose->z;
    }

    if (!pose)
    {
        output.result.objects.clear();
        error.clear();
        return finish("degenerate_3d_geometry", true);
    }

    // [STATELESS] Yaw hinh hoc thuan tuy tu vi tri 2 block trai/phai cua
    // FRAME NAY -- khong con jump-limit/EMA (vo nghia khi khong co state
    // song giua cac request).
    const double yaw_deg = geometricYawDeg(p_left_geom, p_right_geom);

    cv::Vec3d rpy = rotationToEulerDeg(pose->rotation);

    // Consistency check: rpy[2] here is the yaw implied by the plane
    // normal alone (see rotationToEulerDeg). If it disagrees strongly with
    // the position-based yaw_deg, the block plane fit is likely corrupted
    // (foreshortened/occluded block faces at oblique angles feeding biased
    // points into RANSAC) -- its roll/pitch cannot be trusted either, so
    // drop to the yaw-only pose instead of reporting a skewed roll/pitch.
    if (!fallback &&
        std::abs(std::remainder(rpy[2] - yaw_deg, 360.0)) > kMaxYawNormalDisagreementDeg)
    {
        if (auto fallback_pose = buildYawOnlyPose(p_left_geom, p_right_geom))
        {
            pose = fallback_pose;
            fallback = true;
            pose_normal = pose->z;
            rpy = rotationToEulerDeg(pose->rotation);
        }
    }

    rpy[2] = yaw_deg;

    const int x1 = std::max(0, static_cast<int>(std::floor(
        std::min({left.x, middle.x, right.x, front.x}))));
    const int y1 = std::max(0, static_cast<int>(std::floor(
        std::min({left.y, middle.y, right.y, front.y}))));
    const int x2 = std::min(frame.depth_z16.cols, static_cast<int>(std::ceil(
        std::max({left.x + left.width,
                  middle.x + middle.width,
                  right.x + right.width,
                  front.x + front.width}))));
    const int y2 = std::min(frame.depth_z16.rows, static_cast<int>(std::ceil(
        std::max({left.y + left.height,
                  middle.y + middle.height,
                  right.y + right.height,
                  front.y + front.height}))));

    DetectionObject pallet;
    pallet.class_id = 0;
    pallet.label = "pallet";
    pallet.confidence = structuralConfidence({&left, &middle, &right, &front});
    pallet.x = static_cast<float>(x1);
    pallet.y = static_cast<float>(y1);
    pallet.width = static_cast<float>(std::max(0, x2 - x1));
    pallet.height = static_cast<float>(std::max(0, y2 - y1));
    const float bbox_center_x = pallet.x + pallet.width * 0.5f;
    const float bbox_center_y = pallet.y + pallet.height * 0.5f;

    if (origin[2] > std::numeric_limits<double>::epsilon() &&
        std::isfinite(origin[0]) &&
        std::isfinite(origin[1]) &&
        std::isfinite(origin[2]))
    {
        pallet.center_x = static_cast<float>(
            frame.intrinsics.fx * origin[0] / origin[2] + frame.intrinsics.ppx);
        pallet.center_y = static_cast<float>(
            frame.intrinsics.fy * origin[1] / origin[2] + frame.intrinsics.ppy);
    }
    else
    {
        pallet.center_x = bbox_center_x;
        pallet.center_y = bbox_center_y;
    }

    pallet.has_3d = true;
    pallet.x_3d = static_cast<float>(origin[0]);
    pallet.y_3d = static_cast<float>(origin[1]);
    pallet.z_3d = static_cast<float>(origin[2]);
    pallet.depth_m = pallet.z_3d;
    pallet.depth_valid_ratio = 1.0f;

    pallet.has_pose_6d = true;
    pallet.roll_deg = static_cast<float>(rpy[0]);
    pallet.pitch_deg = static_cast<float>(rpy[1]);
    pallet.yaw_deg = static_cast<float>(rpy[2]);
    pallet.plane_normal_x = static_cast<float>(pose_normal[0]);
    pallet.plane_normal_y = static_cast<float>(pose_normal[1]);
    pallet.plane_normal_z = static_cast<float>(pose_normal[2]);
    pallet.plane_inlier_ratio = static_cast<float>(plane_inlier_ratio);
    pallet.plane_rmse_m = static_cast<float>(plane_rmse_m);
    pallet.pair_distance_m = static_cast<float>(cv::norm(p_right_geom - p_left_geom));
    pallet.plane_source = fallback ? "yaw_only_fallback" : "three_block_planes";
    pallet.pose_frame_id = kPoseFrame;
    pallet.spatial_status = fallback ? "yaw_only_fallback" : "ok";

    output.result.objects.clear();
    output.result.objects.push_back(std::move(pallet));
    output.xyz_object_count = 1;
    output.pose_object_count = 1;

    error.clear();
    return finish(fallback ? "yaw_only_fallback" : "ok", true);
}

std::string PalletApplication::lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::optional<cv::Vec3d> PalletApplication::normalized(const cv::Vec3d& value)
{
    const double length = cv::norm(value);
    if (length < 1e-12 || !std::isfinite(length)) return std::nullopt;
    return value * (1.0 / length);
}

cv::Vec3d PalletApplication::deproject(const CameraIntrinsics& intr,
                                       double px,
                                       double py,
                                       double depth_m)
{
    return cv::Vec3d((px - intr.ppx) * depth_m / intr.fx,
                     (py - intr.ppy) * depth_m / intr.fy,
                     depth_m);
}

std::optional<cv::Vec3d> PalletApplication::planeDepthAtPixel(
    const CameraIntrinsics& intr,
    double px,
    double py,
    double plane_a,
    double plane_b,
    double plane_c,
    double min_depth_m,
    double max_depth_m)
{
    // [MARKER-STYLE] Giai giao diem tia-mat phang: voi X=lx*Z, Y=ly*Z (quan
    // he deproject nguoc) va mat phang Z=a*X+b*Y+c, ta co:
    //   Z = a*(lx*Z) + b*(ly*Z) + c  =>  Z*(1 - a*lx - b*ly) = c
    // Port dung cong thuc tu ArucoDetector::fitPlaneRansacZ (doan
    // "refined_z = c / denom"). Cho ra depth CHINH XAC tai pixel (px,py)
    // theo mat phang da fit, khong phu thuoc phan bo diem da sample (khac
    // voi centroid, von co the bi lech neu diem sample khong deu quanh
    // pixel do).
    if (!(intr.fx > 0.0) || !(intr.fy > 0.0)) return std::nullopt;

    const double lx = (px - intr.ppx) / intr.fx;
    const double ly = (py - intr.ppy) / intr.fy;
    const double denom = 1.0 - plane_a * lx - plane_b * ly;
    if (std::abs(denom) < 1e-9) return std::nullopt;

    const double z = plane_c / denom;
    if (!std::isfinite(z) || z < min_depth_m || z > max_depth_m) return std::nullopt;

    return deproject(intr, px, py, z);
}

PalletApplication::Box PalletApplication::toBox(const DetectionObject& object,
                                                 int width,
                                                 int height)
{
    Box box;
    box.x1 = std::clamp(static_cast<int>(std::floor(object.x)), 0, width);
    box.y1 = std::clamp(static_cast<int>(std::floor(object.y)), 0, height);
    box.x2 = std::clamp(static_cast<int>(std::ceil(object.x + object.width)), 0, width);
    box.y2 = std::clamp(static_cast<int>(std::ceil(object.y + object.height)), 0, height);
    return box;
}

PalletApplication::Box PalletApplication::insetBox(const Box& box,
                                                    double ratio)
{
    const int dx = static_cast<int>((box.x2 - box.x1) * ratio);
    const int dy = static_cast<int>((box.y2 - box.y1) * ratio);
    Box inset{box.x1 + dx, box.y1 + dy, box.x2 - dx, box.y2 - dy};
    if (inset.x2 - inset.x1 < 4 || inset.y2 - inset.y1 < 4) return box;
    return inset;
}

std::optional<double> PalletApplication::patchDepth(const RgbdFrame& frame,
                                                    int px,
                                                    int py) const
{
    std::vector<double> samples;
    for (int dy = -kDepthPatchRadius; dy <= kDepthPatchRadius; ++dy)
    {
        const int y = py + dy;
        if (y < 0 || y >= frame.depth_z16.rows) continue;
        const uint16_t* row = frame.depth_z16.ptr<uint16_t>(y);
        for (int dx = -kDepthPatchRadius; dx <= kDepthPatchRadius; ++dx)
        {
            const int x = px + dx;
            if (x < 0 || x >= frame.depth_z16.cols) continue;
            if (row[x] == 0) continue;
            const double depth = static_cast<double>(row[x]) * frame.depth_scale;
            if (depth > kDepthMinM && depth < kDepthMaxM)
                samples.push_back(depth);
        }
    }
    if (samples.empty()) return std::nullopt;
    std::sort(samples.begin(), samples.end());
    const size_t middle = samples.size() / 2;
    if (samples.size() % 2 != 0) return samples[middle];
    return (samples[middle - 1] + samples[middle]) * 0.5;
}

bool PalletApplication::locallyConsistent(const RgbdFrame& frame,
                                          int px,
                                          int py,
                                          double depth_m) const
{
    const double threshold = std::max(kEdgeMaxAbsDiffM,
                                      depth_m * kEdgeMaxRelDiff);
    const std::array<cv::Point, 4> offsets{
        cv::Point(-kEdgeRadiusPx, 0), cv::Point(kEdgeRadiusPx, 0),
        cv::Point(0, -kEdgeRadiusPx), cv::Point(0, kEdgeRadiusPx)};

    for (const cv::Point& offset : offsets)
    {
        const int x = px + offset.x;
        const int y = py + offset.y;
        if (x < 0 || x >= frame.depth_z16.cols ||
            y < 0 || y >= frame.depth_z16.rows)
            continue;

        const uint16_t raw = frame.depth_z16.at<uint16_t>(y, x);
        if (raw == 0) return false;
        const double neighbor = static_cast<double>(raw) * frame.depth_scale;
        if (std::abs(neighbor - depth_m) > threshold) return false;
    }
    return true;
}

std::vector<cv::Vec3d> PalletApplication::collectBlockPoints(
    const RgbdFrame& frame,
    const Box& source_box) const
{
    const Box box = insetBox(source_box, kRoiInsetRatio);
    const int box_w = std::max(1, box.x2 - box.x1);
    const int box_h = std::max(1, box.y2 - box.y1);
    const double raw_step = std::sqrt(
        static_cast<double>(box_w * box_h) / static_cast<double>(kRoiMaxPoints));
    const int step = std::max(1, static_cast<int>(std::ceil(raw_step)));

    std::vector<cv::Vec3d> points;
    points.reserve(kRoiMaxPoints);

    for (int y = box.y1; y < box.y2; y += step)
    {
        const uint16_t* row = frame.depth_z16.ptr<uint16_t>(y);
        for (int x = box.x1; x < box.x2; x += step)
        {
            if (row[x] == 0) continue;
            const double depth = static_cast<double>(row[x]) * frame.depth_scale;
            if (!(depth > kDepthMinM && depth < kDepthMaxM)) continue;
            if (!locallyConsistent(frame, x, y, depth)) continue;
            points.push_back(deproject(frame.intrinsics, x, y, depth));
        }
    }
    return points;
}

std::optional<PalletApplication::PlaneModel> PalletApplication::ransacPlane(
    const std::vector<cv::Vec3d>& raw_points)
{
    // [MARKER-STYLE] Loc outlier Z bang MAD truoc khi RANSAC (port tu
    // ArucoDetector::getPlaneDepthRobust).
    const std::vector<cv::Vec3d> points = madFilterByZ(raw_points);

    const int count = static_cast<int>(points.size());
    if (count < kRansacMinPoints) return std::nullopt;

    std::uniform_int_distribution<int> distribution(0, count - 1);
    int best_count = -1;
    std::vector<uint8_t> best_mask;
    cv::Vec3d best_normal{0.0, 0.0, 0.0};
    double best_d = 0.0;

    for (int iteration = 0; iteration < kRansacIterations; ++iteration)
    {
        int i0 = distribution(_rng);
        int i1 = distribution(_rng);
        while (i1 == i0) i1 = distribution(_rng);
        int i2 = distribution(_rng);
        while (i2 == i0 || i2 == i1) i2 = distribution(_rng);

        const cv::Vec3d v1 = points[i1] - points[i0];
        const cv::Vec3d v2 = points[i2] - points[i0];
        const auto normal_opt = normalized(v1.cross(v2));
        if (!normal_opt) continue;

        const cv::Vec3d normal = *normal_opt;

        // [MARKER-STYLE] Loai ung vien nghieng qua 60 do so voi truc quang
        // camera (Z) -- port tu ArucoDetector::fitPlaneRansacZ. Block front
        // face phai gan nhu doi dien camera; ung vien qua nghieng thuong la
        // trung mep/canh vat the chu khong phai mat truoc.
        if (std::abs(normal[2]) < kPlaneTiltRejectCosZ) continue;

        const double d = -normal.dot(points[i0]);
        std::vector<uint8_t> mask(static_cast<size_t>(count), 0);
        int inliers = 0;
        for (int index = 0; index < count; ++index)
        {
            if (std::abs(points[index].dot(normal) + d) < kRansacDistanceM)
            {
                mask[static_cast<size_t>(index)] = 1;
                ++inliers;
            }
        }
        if (inliers > best_count)
        {
            best_count = inliers;
            best_mask = std::move(mask);
            best_normal = normal;
            best_d = d;
        }
    }

    if (best_mask.empty() ||
        static_cast<double>(best_count) < kRansacMinInlierRatio * count)
        return std::nullopt;

    cv::Vec3d centroid{0.0, 0.0, 0.0};
    std::vector<cv::Vec3d> inlier_points;
    inlier_points.reserve(static_cast<size_t>(best_count));
    for (int index = 0; index < count; ++index)
    {
        if (!best_mask[static_cast<size_t>(index)]) continue;
        inlier_points.push_back(points[index]);
        centroid += points[index];
    }
    centroid *= 1.0 / static_cast<double>(inlier_points.size());

    cv::Mat centered(static_cast<int>(inlier_points.size()), 3, CV_64F);
    for (int row = 0; row < centered.rows; ++row)
    {
        const cv::Vec3d delta = inlier_points[static_cast<size_t>(row)] - centroid;
        centered.at<double>(row, 0) = delta[0];
        centered.at<double>(row, 1) = delta[1];
        centered.at<double>(row, 2) = delta[2];
    }

    cv::SVD svd(centered);
    cv::Vec3d normal(svd.vt.at<double>(2, 0),
                     svd.vt.at<double>(2, 1),
                     svd.vt.at<double>(2, 2));
    const auto refined = normalized(normal);
    if (!refined) return std::nullopt;

    PlaneModel model;
    model.normal = *refined;
    model.d = -model.normal.dot(centroid);
    model.centroid = centroid;
    model.inlier_count = static_cast<int>(inlier_points.size());

    // RMSE cua cac inlier point so voi mat phang SVD-refined (nhat quan voi
    // tap diem da MAD-filter/RANSAC-loc dung trong ham nay).
    {
        double squared_error = 0.0;
        for (const auto& p : inlier_points)
        {
            const double residual = p.dot(model.normal) + model.d;
            squared_error += residual * residual;
        }
        model.rmse_m = inlier_points.empty()
            ? 0.0
            : std::sqrt(squared_error / static_cast<double>(inlier_points.size()));
    }

    // [MARKER-STYLE] Weighted least-squares refit Z = a*X + b*Y + c tren cac
    // inlier (trong so Gaussian theo khoang cach toi mat phang RANSAC ban
    // dau) -- port tu ArucoDetector::fitPlaneRansacZ. He so (a,b,c) nay cho
    // phep suy depth chinh xac tai BAT KY pixel 2D nao bang giao diem
    // tia-mat phang, thay vi chi dung centroid cac diem da sample.
    {
        const double sigma_w = std::max(0.001, kRansacDistanceM * 0.5);
        std::vector<double> weights;
        weights.reserve(inlier_points.size());
        for (const auto& p : inlier_points)
        {
            const double err = std::abs(p.dot(best_normal) + best_d);
            weights.push_back(std::exp(-err / sigma_w));
        }

        double sw = 0.0, swx = 0.0, swy = 0.0, swz = 0.0;
        double swxx = 0.0, swxy = 0.0, swxz = 0.0, swyy = 0.0, swyz = 0.0;
        for (size_t i = 0; i < inlier_points.size(); ++i)
        {
            const double wi = weights[i];
            const double xi = inlier_points[i][0];
            const double yi = inlier_points[i][1];
            const double zi = inlier_points[i][2];
            sw += wi; swx += wi * xi; swy += wi * yi; swz += wi * zi;
            swxx += wi * xi * xi; swxy += wi * xi * yi; swxz += wi * xi * zi;
            swyy += wi * yi * yi; swyz += wi * yi * zi;
        }

        if (sw > 1e-12)
        {
            cv::Mat A = (cv::Mat_<double>(3, 3) <<
                swxx, swxy, swx,
                swxy, swyy, swy,
                swx,  swy,  sw);
            cv::Mat B = (cv::Mat_<double>(3, 1) << swxz, swyz, swz);
            cv::Mat coeff;
            if (cv::solve(A, B, coeff, cv::DECOMP_SVD))
            {
                model.plane_a = coeff.at<double>(0, 0);
                model.plane_b = coeff.at<double>(1, 0);
                model.plane_c = coeff.at<double>(2, 0);
                model.has_plane_coeffs = true;
            }
        }
    }

    return model;
}

PalletApplication::PlaneSummary PalletApplication::fitBlockPlanes(
    const RgbdFrame& frame,
    const std::vector<Box>& boxes)
{
    PlaneSummary summary;
    summary.blocks.reserve(boxes.size());
    cv::Vec3d normal_sum{0.0, 0.0, 0.0};

    for (const Box& box : boxes)
    {
        const std::vector<cv::Vec3d> points = collectBlockPoints(frame, box);
        summary.total_points += static_cast<int>(points.size());

        const auto plane_opt = ransacPlane(points);
        if (!plane_opt)
        {
            summary.blocks.push_back(BlockFitInfo{
                static_cast<int>(points.size()), 0, 0.0, false});
            continue;
        }

        PlaneModel plane = *plane_opt;
        if (plane.normal[2] > 0.0)
        {
            plane.normal = -plane.normal;
            plane.d = -plane.d;
        }

        if (plane.inlier_count <= 0)
        {
            summary.blocks.push_back(BlockFitInfo{
                static_cast<int>(points.size()), 0, 0.0, false});
            continue;
        }

        const double rmse_m = plane.rmse_m;
        const double weight = 1.0 / (rmse_m * 1000.0 + kRmseWeightEpsMm);
        normal_sum += plane.normal * weight;
        summary.total_inliers += plane.inlier_count;

        BlockFitInfo info{
            static_cast<int>(points.size()), plane.inlier_count, rmse_m, true,
            plane.centroid};
        info.plane_a = plane.plane_a;
        info.plane_b = plane.plane_b;
        info.plane_c = plane.plane_c;
        info.has_plane_coeffs = plane.has_plane_coeffs;
        summary.blocks.push_back(std::move(info));
    }

    const auto combined = normalized(normal_sum);
    if (!combined) return summary;

    summary.normal = *combined;
    if (summary.normal[2] > 0.0) summary.normal = -summary.normal;
    summary.ok = true;
    return summary;
}

std::optional<PalletApplication::PoseAxes> PalletApplication::buildPoseFromNormal(
    const cv::Vec3d& left,
    const cv::Vec3d& right,
    const cv::Vec3d& normal_towards_camera)
{
    auto x = normalized(right - left);
    auto z = normalized(-normal_towards_camera);
    if (!x || !z) return std::nullopt;
    auto y = normalized(z->cross(*x));
    if (!y) return std::nullopt;
    x = normalized(y->cross(*z));
    if (!x) return std::nullopt;

    PoseAxes pose;
    pose.x = *x;
    pose.y = *y;
    pose.z = *z;
    pose.rotation = cv::Matx33d(
        pose.x[0], pose.y[0], pose.z[0],
        pose.x[1], pose.y[1], pose.z[1],
        pose.x[2], pose.y[2], pose.z[2]);
    return pose;
}

std::optional<PalletApplication::PoseAxes> PalletApplication::buildYawOnlyPose(
    const cv::Vec3d& left,
    const cv::Vec3d& right)
{
    auto x = normalized(right - left);
    if (!x) return std::nullopt;

    const cv::Vec3d down{0.0, 1.0, 0.0};
    cv::Vec3d horizontal = *x - down * x->dot(down);
    x = normalized(horizontal);
    if (!x) return std::nullopt;

    auto z = normalized(x->cross(down));
    if (!z) return std::nullopt;
    cv::Vec3d y = down;
    if ((*z)[2] < 0.0)
    {
        *z = -*z;
        const auto corrected_y = normalized(z->cross(*x));
        if (!corrected_y) return std::nullopt;
        y = *corrected_y;
    }

    x = normalized(y.cross(*z));
    if (!x) return std::nullopt;

    PoseAxes pose;
    pose.x = *x;
    pose.y = y;
    pose.z = *z;
    pose.rotation = cv::Matx33d(
        pose.x[0], pose.y[0], pose.z[0],
        pose.x[1], pose.y[1], pose.z[1],
        pose.x[2], pose.y[2], pose.z[2]);
    return pose;
}

cv::Vec3d PalletApplication::rotationToEulerDeg(const cv::Matx33d& r)
{
    // `r`'s columns are the pallet's local axes expressed in the camera
    // optical frame (X-right, Y-down, Z-forward/depth):
    //   col0 = pose.x (left->right block axis)
    //   col2 = pose.z (plane normal, pointing away from camera)
    //
    // Y is the vertical axis in this frame (not Z), so the formulas below
    // use the physically correct axis for each angle in THIS frame instead
    // of a generic aircraft-style Rz*Ry*Rx decomposition:
    //   roll  (tilt about the viewing/Z axis, in-image rotation)
    //         -> angle of pose.x projected onto the image plane (X,Y)
    //   pitch (nod about the horizontal/X axis)
    //         -> vertical (Y) component of the plane normal (pose.z)
    //   yaw   (turn about the vertical/Y axis, from the normal only;
    //          the caller overwrites this with the more accurate
    //          position-based geometricYawDeg() before using it)
    //         -> horizontal (X vs Z) component of the plane normal (pose.z)
    const double roll_deg = std::atan2(r(1, 0), r(0, 0)) * kRadToDeg;
    const double pitch_deg = std::asin(std::clamp(-r(1, 2), -1.0, 1.0)) * kRadToDeg;
    const double yaw_from_normal_deg = std::atan2(r(0, 2), r(2, 2)) * kRadToDeg;
    return cv::Vec3d(roll_deg, pitch_deg, yaw_from_normal_deg);
}

double PalletApplication::geometricYawDeg(const cv::Vec3d& left,
                                          const cv::Vec3d& right)
{
    const double dx = right[0] - left[0];
    const double dz = right[2] - left[2];
    if (std::abs(dx) < 1e-12 && std::abs(dz) < 1e-12) return 0.0;
    return std::atan2(dz, dx) * kRadToDeg;
}