#pragma once

#include <optional>
#include <random>
#include <string>
#include <vector>
#include <deque>

#include <opencv2/opencv.hpp>

#include "IApplication.h"

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
        std::vector<uint8_t> inliers;
    };

    struct BlockFitInfo
    {
        int points = 0;
        int inliers = 0;
        double rmse_m = 0.0;
        bool ok = false;
        // [CENTROID-FIX] Trung binh cua tat ca inlier point (mat phang block
        // da RANSAC-loc), dung thay the cho depth tai 1 pixel tam bbox --
        // giam bias hinh hoc do tam bbox 2D khong trung tam vat the 3D khi
        // pallet bi xoay (foreshortening).
        cv::Vec3d centroid{0.0, 0.0, 0.0};
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

    cv::Vec3d smoothVector(std::optional<cv::Vec3d>& state,
                           const cv::Vec3d& value,
                           double alpha,
                           double weight = 1.0) const;
    double smoothScalar(std::optional<double>& state,
                        double value,
                        double alpha) const;
    cv::Vec3d limitNormalJump(const cv::Vec3d& value,
                              double max_deg) const;
    double limitYawJump(double value, double max_deg) const;
    void resetTemporalState();

private:
    DetectorConfig _config;
    std::mt19937 _rng{std::random_device{}()};

    class VectorMedianFilter {
    public:
        explicit VectorMedianFilter(std::size_t window) : _window(std::max<std::size_t>(1, window)) {}
        cv::Vec3d update(const cv::Vec3d& vec);
    private:
        std::size_t _window;
        std::deque<cv::Vec3d> _buffer;
    };

    std::optional<cv::Vec3d> _hole1;
    std::optional<cv::Vec3d> _hole2;
    std::optional<cv::Vec3d> _normal;
    std::optional<double> _yaw;
    VectorMedianFilter _normalMedianFilter{3};

    // [AUTO-RESET] Theo doi lich su GIA TRI THO (truoc EMA) cua vi tri va
    // yaw trong mot cua so truot ngan, de tu phat hien khi frame hien tai
    // la mot vi tri/goc khac han (Techman vua doi vi tri), thay vi chi la
    // nhieu cua cung 1 vi tri dang dung yen. Nguong tu thich nghi theo do
    // lech chuan do duoc trong cua so, khong dung so co dinh -> hoat dong
    // dung voi moi buoc nhay (5 do, 10 do, hay lon hon), khong can biet
    // truoc dai goc quet.
    class RawJumpDetector
    {
    public:
        // Tra ve true neu 'position'/'yaw_deg' la mot buoc nhay bat thuong
        // so voi lich su gan day (tuc: vi tri moi). Luon cap nhat lich su
        // noi bo; neu phat hien nhay thi xoa lich su cu truoc khi nap mau
        // hien tai vao (bat dau theo doi lai tu dau).
        bool checkAndUpdate(const cv::Vec3d& position, double yaw_deg);
        void clear();

        // [SETTLE] Camera/pallet duoc coi la dung yen khi cua so gan day co
        // std (do lech chuan) nho hon nguong rieng (chat hon nguong jump).
        // Dung de biet khi nao co the bo EMA/jump-limit va tra thang gia
        // tri trung binh cua so (khong lag) cho cac lan lay mau tinh.
        bool isSettled() const;
        cv::Vec3d meanPosition() const;
        double meanYaw() const;

    private:
        static constexpr std::size_t kWindowSize = 8;
        static constexpr std::size_t kMinSamples = 4;
        bool computeStats(cv::Vec3d& mean_pos, double& mean_yaw,
                          double& pos_std, double& yaw_std) const;
        std::deque<cv::Vec3d> _positions;
        std::deque<double> _yaws;
    };

    // [SETTLE] Trung binh truot (khong EMA, khong lag) cho plane normal, chi
    // dung khi da xac dinh camera dung yen (qua RawJumpDetector::isSettled()).
    class NormalWindowAverager
    {
    public:
        void update(const cv::Vec3d& normal);
        std::optional<cv::Vec3d> mean() const;
        void clear() { _samples.clear(); }

    private:
        static constexpr std::size_t kWindowSize = 8;
        static constexpr std::size_t kMinSamples = 5;
        std::deque<cv::Vec3d> _samples;
    };

    RawJumpDetector _rawJumpDetector;
    NormalWindowAverager _normalWindowAverager;
};