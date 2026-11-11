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
 
constexpr double kRoiInsetRatio = 0.15;
constexpr int kRoiMaxPoints = 900;
constexpr int kEdgeRadiusPx = 2;
constexpr double kEdgeMaxAbsDiffM = 0.015;
constexpr double kEdgeMaxRelDiff = 0.03;
constexpr double kRmseWeightEpsMm = 0.1;
 
constexpr double kPositionAlpha = 0.60;
constexpr double kNormalAlpha = 0.35;
constexpr double kMaxNormalJumpDeg = 4.0;
constexpr int kNormalMedianWindow = 3;
constexpr double kRmseMaxQualityM = 0.006;
constexpr double kPalletWidthM = 0.830;
constexpr double kYawAlpha = 0.30;
constexpr double kMaxYawJumpDeg = 5.0;
// If the plane-normal-derived yaw and the block-position-derived yaw
// disagree by more than this, the block plane fit is likely corrupted
// (foreshortening / side-face contamination at oblique viewing angles) --
// fall back to the yaw-only pose instead of trusting the normal.
constexpr double kMaxYawNormalDisagreementDeg = 8.0;
constexpr double kRadToDeg = 180.0 / CV_PI;
constexpr const char* kPoseFrame = "rgbd_aligned_optical_frame";

// [AUTO-RESET] Nguong cho RawJumpDetector: mot mau THO duoc coi la "vi tri
// moi" (khong phai nhieu) neu no lech khoi trung binh cua so gan day qua
// max(san, he_so * do_lech_chuan_do_duoc). San (kPositionJumpFloorM /
// kYawJumpFloorDeg) chan truong hop cua so qua "phang" (std ~ 0, camera
// dung rat yen) khien nguong gan bang 0 va bat nhay nham tu nhieu nho nhat.
constexpr double kPositionJumpStdMultiplier = 6.0;
constexpr double kPositionJumpFloorM = 0.01;    // 1 cm
constexpr double kYawJumpStdMultiplier = 6.0;
constexpr double kYawJumpFloorDeg = 2.0;

// [SETTLE] Nguong "dung yen" phai CHAT hon nguong "jump" o tren -- muc dich
// khac nhau: jump = phat hien buoc nhay lon (vi tri moi); settled = xac
// nhan da thuc su dung yen de tin tuong trung binh cua so thay EMA/jump
// limiter. Gia tri khoi diem, nen tinh chinh theo do nhieu thuc do duoc tu
// setup camera/khoang cach lam viec thuc te.
constexpr double kPositionSettleStdM = 0.003;   // 3 mm
constexpr double kYawSettleStdDeg = 0.30;       // 0.3 do
constexpr std::size_t kSettleMinSamples = 6;    // trong cua so 8 mau
 
float structuralConfidence(const std::vector<const DetectionObject*>& objects)
{
    if (objects.empty()) return 0.0f;
    float value = 1.0f;
    for (const DetectionObject* object : objects)
        value = std::min(value, object ? object->confidence : 0.0f);
    return value;
}
 
std::optional<double> computeSimpleYawDeg(
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
 
}
 
bool PalletApplication::initialize(const DetectorConfig& config,
                                   std::string& error)
{
    _config = config;
    resetTemporalState();
    _rawJumpDetector.clear();
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
        resetTemporalState();
        _rawJumpDetector.clear();
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

    // [CENTROID-FIX] Uu tien centroid cua inlier point (mat phang block da
    // RANSAC-loc, trung binh tren hang chuc/hang tram diem) thay cho depth
    // tai 1 pixel tam bbox (p_left/p_middle/p_right o tren) -- centroid it
    // bias hon vi khong bi lech theo huong foreshortening khi pallet xoay,
    // va it nhieu hon vi la trung binh nhieu diem da loc bien/outlier. Neu
    // fit that bai cho 1 block cu the, fallback ve diem tam bbox nhu cu.
    const cv::Vec3d p_left_geom =
        (planes.blocks.size() > 0 && planes.blocks[0].ok)
            ? planes.blocks[0].centroid : p_left;
    const cv::Vec3d p_middle_geom =
        (planes.blocks.size() > 1 && planes.blocks[1].ok)
            ? planes.blocks[1].centroid : p_middle;
    const cv::Vec3d p_right_geom =
        (planes.blocks.size() > 2 && planes.blocks[2].ok)
            ? planes.blocks[2].centroid : p_right;

    const cv::Vec3d hole1_raw = (p_left_geom + p_middle_geom) * 0.5;
    const cv::Vec3d hole2_raw = (p_middle_geom + p_right_geom) * 0.5;
    const cv::Vec3d origin_raw = (hole1_raw + hole2_raw) * 0.5;
    const double yaw_raw_check = geometricYawDeg(p_left_geom, p_right_geom);

    // [AUTO-RESET] So mau tho frame nay voi lich su gan day; neu la buoc
    // nhay bat thuong (vi tri moi) thi xoa EMA/jump-limiter truoc khi
    // smoothing, de frame nay tra ve dung nhu no la.
    if (_rawJumpDetector.checkAndUpdate(origin_raw, yaw_raw_check))
    {
        resetTemporalState();
    }

    const cv::Vec3d hole1 = smoothVector(_hole1, hole1_raw, kPositionAlpha);
    const cv::Vec3d hole2 = smoothVector(_hole2, hole2_raw, kPositionAlpha);
    cv::Vec3d origin = (hole1 + hole2) * 0.5;

    // [SETTLE] Da du mau on dinh (std thap) trong cua so gan day -> bo qua
    // EMA/jump-limit va dung thang trung binh cua so (khong lag) cho vi tri.
    // 'settled' duoc tinh lai o day (sau checkAndUpdate ben tren da nap mau
    // hien tai vao cua so) va dung chung cho ca origin, normal va yaw phia
    // duoi.
    const bool settled = _rawJumpDetector.isSettled();
    if (settled)
    {
        origin = _rawJumpDetector.meanPosition();
    }
 
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
        
        const double rmse_quality = std::clamp(1.0 - (plane_rmse_m / kRmseMaxQualityM), 0.0, 1.0);
        const double confidence = std::clamp(plane_inlier_ratio * rmse_quality, 0.0, 1.0);
        
        _normalWindowAverager.update(planes.normal);

        const cv::Vec3d normal_median = _normalMedianFilter.update(planes.normal);
        const double max_jump = std::max(1.0, kMaxNormalJumpDeg * confidence);
        const cv::Vec3d limited = limitNormalJump(normal_median, max_jump);
        cv::Vec3d normal = smoothVector(_normal,
                                        limited,
                                        kNormalAlpha,
                                        confidence);
        if (const auto n = normalized(normal)) normal = *n;

        // [SETTLE] Dung yen -> bo EMA/median/jump-limit o tren, dung thang
        // trung binh cua so cua normal tho (khong lag).
        if (settled)
        {
            if (const auto settled_normal = _normalWindowAverager.mean())
                normal = *settled_normal;
        }
 
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
 
    const double yaw_raw = geometricYawDeg(p_left_geom, p_right_geom);
    const double yaw_limited = limitYawJump(yaw_raw, kMaxYawJumpDeg);
    double yaw_deg = smoothScalar(_yaw, yaw_limited, kYawAlpha);

    // [SETTLE] Dung yen -> bo EMA/jump-limit, dung thang trung binh cua so
    // cua yaw tho (khong lag).
    if (settled)
    {
        yaw_deg = _rawJumpDetector.meanYaw();
    }
 
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
    const std::vector<cv::Vec3d>& points)
{
    const int count = static_cast<int>(points.size());
    if (count < kRansacMinPoints) return std::nullopt;
 
    std::uniform_int_distribution<int> distribution(0, count - 1);
    int best_count = -1;
    std::vector<uint8_t> best_mask;
 
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
    model.inliers = std::move(best_mask);
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
 
        double squared_error = 0.0;
        int inliers = 0;
        for (size_t index = 0; index < points.size(); ++index)
        {
            if (!plane.inliers[index]) continue;
            const double residual = points[index].dot(plane.normal) + plane.d;
            squared_error += residual * residual;
            ++inliers;
        }
 
        if (inliers <= 0)
        {
            summary.blocks.push_back(BlockFitInfo{
                static_cast<int>(points.size()), 0, 0.0, false});
            continue;
        }
 
        const double rmse_m =
            std::sqrt(squared_error / static_cast<double>(inliers));
        const double weight = 1.0 / (rmse_m * 1000.0 + kRmseWeightEpsMm);
        normal_sum += plane.normal * weight;
        summary.total_inliers += inliers;
        summary.blocks.push_back(BlockFitInfo{
            static_cast<int>(points.size()), inliers, rmse_m, true,
            plane.centroid});
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
    // NOTE (fix): the previous implementation assumed the generic
    // aircraft-style Euler order R = Rz(yaw)*Ry(pitch)*Rx(roll), which is
    // only correct when Z is the vertical axis. Here Y is vertical, so that
    // formula silently reported the normal's horizontal (yaw-induced) tilt
    // as "pitch" -- confirmed against test logs where pitch_deg matched
    // asin(NormX) to within 0.15 deg on every frame, i.e. it was measuring
    // yaw, not pitch. The formulas below use the physically correct axis
    // for each angle in THIS frame instead of a generic matrix decomposition:
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
 
cv::Vec3d PalletApplication::smoothVector(std::optional<cv::Vec3d>& state,
                                          const cv::Vec3d& value,
                                          double alpha,
                                          double weight) const
{
    if (!state)
    {
        state = value;
        return value;
    }
    const double effective_alpha = std::clamp(alpha * weight, 0.0, 1.0);
    *state = value * effective_alpha + *state * (1.0 - effective_alpha);
    return *state;
}
 
double PalletApplication::smoothScalar(std::optional<double>& state,
                                       double value,
                                       double alpha) const
{
    if (!state)
    {
        state = value;
        return value;
    }
    *state = alpha * value + (1.0 - alpha) * *state;
    return *state;
}
 
cv::Vec3d PalletApplication::limitNormalJump(const cv::Vec3d& value,
                                             double max_deg) const
{
    if (!_normal) return value;
    const auto previous = normalized(*_normal);
    const auto current = normalized(value);
    if (!previous || !current) return value;
 
    const double dot = std::clamp(previous->dot(*current), -1.0, 1.0);
    const double angle = std::acos(dot);
    const double angle_deg = angle * kRadToDeg;
    if (angle_deg <= max_deg || angle_deg < 1e-6) return *current;
 
    const double t = max_deg / angle_deg;
    const double sin_angle = std::sin(angle);
    cv::Vec3d limited;
    if (std::abs(sin_angle) < 1e-9)
        limited = *previous * (1.0 - t) + *current * t;
    else
        limited = *previous * (std::sin((1.0 - t) * angle) / sin_angle) +
                  *current * (std::sin(t * angle) / sin_angle);
 
    const auto result = normalized(limited);
    return result ? *result : *previous;
}
 
double PalletApplication::limitYawJump(double value, double max_deg) const
{
    if (!_yaw) return value;
    const double diff = value - *_yaw;
    if (std::abs(diff) <= max_deg) return value;
    return *_yaw + std::copysign(max_deg, diff);
}
 
void PalletApplication::resetTemporalState()
{
    _hole1.reset();
    _hole2.reset();
    _normal.reset();
    _yaw.reset();
    _normalMedianFilter = VectorMedianFilter{static_cast<std::size_t>(kNormalMedianWindow)};
    _normalWindowAverager.clear();
}
 
cv::Vec3d PalletApplication::VectorMedianFilter::update(const cv::Vec3d& vec)
{
    _buffer.push_back(vec);
    if (_buffer.size() > _window) {
        _buffer.pop_front();
    }
 
    cv::Vec3d med{0.0, 0.0, 0.0};
    for (int axis = 0; axis < 3; ++axis) {
        std::vector<double> vals;
        vals.reserve(_buffer.size());
        for (const auto& v : _buffer) {
            vals.push_back(v[axis]);
        }
        std::sort(vals.begin(), vals.end());
        const std::size_t n = vals.size();
        med[axis] = (n % 2 == 1) ? vals[n / 2] : (vals[n / 2 - 1] + vals[n / 2]) / 2.0;
    }
    return med;
}

// [AUTO-RESET] So mau tho hien tai voi trung binh + do lech chuan (std) cua
// cua so gan day. Nguong = max(san, he_so * std) -- tu thich nghi theo muc
// nhieu THUC TE dang do duoc, khong dung so co dinh, nen bat duoc buoc nhay
// du lon hay nho (5 do, 10 do, 25 do...) mien la lon hon han nhieu tai cho.
//
// Voi < kMinSamples mau, chua du du lieu de uoc luong std dang tin cay ->
// khong bao "nhay" (tranh false-positive ngay sau khi vua reset/khoi dong).
bool PalletApplication::RawJumpDetector::computeStats(cv::Vec3d& mean_pos,
                                                        double& mean_yaw,
                                                        double& pos_std,
                                                        double& yaw_std) const
{
    if (_positions.size() < kMinSamples) return false;

    mean_pos = cv::Vec3d{0.0, 0.0, 0.0};
    mean_yaw = 0.0;
    for (std::size_t i = 0; i < _positions.size(); ++i)
    {
        mean_pos += _positions[i];
        mean_yaw += _yaws[i];
    }
    const double n = static_cast<double>(_positions.size());
    mean_pos *= (1.0 / n);
    mean_yaw /= n;

    double pos_var = 0.0;
    double yaw_var = 0.0;
    for (std::size_t i = 0; i < _positions.size(); ++i)
    {
        const double pd = cv::norm(_positions[i] - mean_pos);
        pos_var += pd * pd;
        const double yd = _yaws[i] - mean_yaw;
        yaw_var += yd * yd;
    }
    pos_std = std::sqrt(pos_var / n);
    yaw_std = std::sqrt(yaw_var / n);
    return true;
}

bool PalletApplication::RawJumpDetector::checkAndUpdate(const cv::Vec3d& position,
                                                          double yaw_deg)
{
    bool jump = false;
    cv::Vec3d mean_pos{0.0, 0.0, 0.0};
    double mean_yaw = 0.0;
    double pos_std = 0.0;
    double yaw_std = 0.0;

    if (computeStats(mean_pos, mean_yaw, pos_std, yaw_std))
    {
        const double pos_threshold =
            std::max(kPositionJumpFloorM, kPositionJumpStdMultiplier * pos_std);
        const double yaw_threshold =
            std::max(kYawJumpFloorDeg, kYawJumpStdMultiplier * yaw_std);

        const double pos_dev = cv::norm(position - mean_pos);
        const double yaw_dev = std::abs(std::remainder(yaw_deg - mean_yaw, 360.0));

        if (pos_dev > pos_threshold || yaw_dev > yaw_threshold)
        {
            jump = true;
        }
    }

    if (jump)
    {
        clear();
    }

    _positions.push_back(position);
    _yaws.push_back(yaw_deg);
    if (_positions.size() > kWindowSize)
    {
        _positions.pop_front();
        _yaws.pop_front();
    }

    return jump;
}

bool PalletApplication::RawJumpDetector::isSettled() const
{
    if (_positions.size() < kSettleMinSamples) return false;
    cv::Vec3d mean_pos{0.0, 0.0, 0.0};
    double mean_yaw = 0.0;
    double pos_std = 0.0;
    double yaw_std = 0.0;
    if (!computeStats(mean_pos, mean_yaw, pos_std, yaw_std)) return false;
    return pos_std <= kPositionSettleStdM && yaw_std <= kYawSettleStdDeg;
}

cv::Vec3d PalletApplication::RawJumpDetector::meanPosition() const
{
    cv::Vec3d mean_pos{0.0, 0.0, 0.0};
    double mean_yaw = 0.0;
    double pos_std = 0.0;
    double yaw_std = 0.0;
    computeStats(mean_pos, mean_yaw, pos_std, yaw_std);
    return mean_pos;
}

double PalletApplication::RawJumpDetector::meanYaw() const
{
    cv::Vec3d mean_pos{0.0, 0.0, 0.0};
    double mean_yaw = 0.0;
    double pos_std = 0.0;
    double yaw_std = 0.0;
    computeStats(mean_pos, mean_yaw, pos_std, yaw_std);
    return mean_yaw;
}

void PalletApplication::RawJumpDetector::clear()
{
    _positions.clear();
    _yaws.clear();
}

void PalletApplication::NormalWindowAverager::update(const cv::Vec3d& normal)
{
    _samples.push_back(normal);
    if (_samples.size() > kWindowSize)
    {
        _samples.pop_front();
    }
}

std::optional<cv::Vec3d> PalletApplication::NormalWindowAverager::mean() const
{
    if (_samples.size() < kMinSamples) return std::nullopt;
    cv::Vec3d sum{0.0, 0.0, 0.0};
    for (const auto& sample : _samples) sum += sample;
    sum *= (1.0 / static_cast<double>(_samples.size()));
    return PalletApplication::normalized(sum);
}