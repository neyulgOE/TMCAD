#pragma once

#include <optional>
#include <random>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "IApplication.h"

// [STATELESS] Service native xu ly theo mo hinh Request-Response dong bo:
// moi request tu Client (vd requestObjectDetection1203) khoi tao/huy pipeline
// rieng, khong co object PalletApplication nao song xuyen suot qua nhieu
// request de tich luy lich su. Vi vay class nay KHONG con giu bat ky state
// EMA/jump-detector/settle nao giua cac lan goi process() -- moi lan goi la
// mot uoc luong doc lap tu 1 frame duy nhat. Viec gom nhieu mau (vd 100
// mau/vi tri) de tinh trung binh/khu nhieu la trach nhiem cua phia Client
// (SampleBuffer). Muc tieu o day la toi da hoa do chinh xac cua TUNG lan
// goi don le (RANSAC + MAD-filter + weighted refit + ray-plane
// intersection), thay vi lam muot theo thoi gian (vo nghia trong kien truc
// nay).
class PalletApplication final : public IApplication
{
public:
    std::string name() const override { return "pallet"; }

    bool initialize(const DetectorConfig& config,
                    std::string& error) override;

    bool process(const RgbdFrame& frame,
                 DetectorOutput& output,
                 std::string& error) override;

private:
    struct Box
    {
        int x1 = 0;
        int y1 = 0;
        int x2 = 0;
        int y2 = 0;
    };

    struct PlaneModel
    {
        cv::Vec3d normal{0.0, 0.0, 0.0};
        double d = 0.0;
        cv::Vec3d centroid{0.0, 0.0, 0.0};
        double rmse_m = 0.0;
        int inlier_count = 0;
        // [MARKER-STYLE] He so mat phang Z = a*X + b*Y + c (X,Y,Z la toa do
        // camera-frame da deproject, don vi met), tu weighted least-squares
        // refit tren cac inlier RANSAC -- dung de tinh depth chinh xac tai
        // BAT KY pixel 2D nao (vd tam bbox) bang giao diem tia-mat phang,
        // thay vi chi lay trung binh cac diem da sample (centroid). Port tu
        // ArucoDetector::fitPlaneRansacZ trong marker_detection.cpp.
        double plane_a = 0.0;
        double plane_b = 0.0;
        double plane_c = 0.0;
        bool has_plane_coeffs = false;
    };

    struct BlockFitInfo
    {
        int points = 0;
        int inliers = 0;
        double rmse_m = 0.0;
        bool ok = false;
        // Fallback neu giao diem tia-mat phang khong tinh duoc.
        cv::Vec3d centroid{0.0, 0.0, 0.0};
        double plane_a = 0.0;
        double plane_b = 0.0;
        double plane_c = 0.0;
        bool has_plane_coeffs = false;
    };

    struct PlaneSummary
    {
        bool ok = false;
        cv::Vec3d normal{0.0, 0.0, 0.0};
        int total_points = 0;
        int total_inliers = 0;
        std::vector<BlockFitInfo> blocks;
    };

    struct PoseAxes
    {
        cv::Vec3d x{1.0, 0.0, 0.0};
        cv::Vec3d y{0.0, 1.0, 0.0};
        cv::Vec3d z{0.0, 0.0, 1.0};
        cv::Matx33d rotation = cv::Matx33d::eye();
    };

    static std::string lower(std::string value);
    static std::optional<cv::Vec3d> normalized(const cv::Vec3d& value);
    static cv::Vec3d deproject(const CameraIntrinsics& intr,
                               double px,
                               double py,
                               double depth_m);
    // [MARKER-STYLE] Giao diem tia-mat phang: tra ve diem 3D tai pixel
    // (px,py) cho mat phang Z = a*X + b*Y + c, bang cach giai he phuong
    // trinh deproject nguoc (X=lx*Z, Y=ly*Z). Chinh xac hon centroid vi
    // khong phu thuoc phan bo/mat can xung cua cac diem da sample (vd block
    // bi khuat 1 phan khi pallet xoay). Tra ve std::nullopt neu suy bien
    // (denom ~ 0) hoac ket qua ngoai [min,max]. Port tu
    // ArucoDetector::fitPlaneRansacZ trong marker_detection.cpp.
    static std::optional<cv::Vec3d> planeDepthAtPixel(const CameraIntrinsics& intr,
                                                       double px,
                                                       double py,
                                                       double plane_a,
                                                       double plane_b,
                                                       double plane_c,
                                                       double min_depth_m,
                                                       double max_depth_m);
    static Box toBox(const DetectionObject& object,
                     int width,
                     int height);
    static Box insetBox(const Box& box, double ratio);

    std::optional<double> patchDepth(const RgbdFrame& frame,
                                     int px,
                                     int py) const;
    bool locallyConsistent(const RgbdFrame& frame,
                           int px,
                           int py,
                           double depth_m) const;
    std::vector<cv::Vec3d> collectBlockPoints(const RgbdFrame& frame,
                                               const Box& box) const;
    std::optional<PlaneModel> ransacPlane(
        const std::vector<cv::Vec3d>& points);
    PlaneSummary fitBlockPlanes(const RgbdFrame& frame,
                                const std::vector<Box>& boxes);

    static std::optional<PoseAxes> buildPoseFromNormal(
        const cv::Vec3d& left,
        const cv::Vec3d& right,
        const cv::Vec3d& normal_towards_camera);
    static std::optional<PoseAxes> buildYawOnlyPose(
        const cv::Vec3d& left,
        const cv::Vec3d& right);
    static cv::Vec3d rotationToEulerDeg(const cv::Matx33d& rotation);
    static double geometricYawDeg(const cv::Vec3d& left,
                                  const cv::Vec3d& right);

private:
    DetectorConfig _config;
    std::mt19937 _rng{std::random_device{}()};
};