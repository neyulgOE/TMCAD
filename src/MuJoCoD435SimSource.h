#pragma once

#include "RGBDSource.h"
#include "CameraHardwareProfile.h"
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Dense>

class MuJoCoD435SimSource : public IRGBDSource {
public:
    MuJoCoD435SimSource();
    explicit MuJoCoD435SimSource(const CameraHardwareProfile& profile, int profileId = 1);
    ~MuJoCoD435SimSource() override;

    // IRGBDSource interface implementation
    bool open() override;
    void close() override;
    bool isOpen() const override;
    CameraIntrinsics intrinsics() const override;
    bool grabFrame(RGBDFrame& outFrame) override;
    std::string deviceName() const override;

    // Nạp hồ sơ camera động (Hot-Reload)
    bool loadCameraProfileFromFile(const std::string& filePath, int preferredProfileId = -1);
    bool applyHardwareProfile(const CameraHardwareProfile& profile, int profileId = -1);
    bool setCustomResolution(int targetW, int targetH);

    // Điều khiển tư thế đối tượng mục tiêu & độ cao camera so với sàn
    void setTargetPose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
    void setCameraHeightAboveFloor(double heightMeters);
    double cameraHeightAboveFloor() const;
    void getGroundTruthPose(Eigen::Matrix3d& outR, Eigen::Vector3d& outt) const;
    void getGroundTruthBBox3D(std::vector<Eigen::Vector3d>& corners) const;

    // Điều khiển vật cản (Occluder - lon nấm đè lên)
    void setOccluderEnabled(bool enabled);
    void setOccluderPose(const Eigen::Vector3d& t, const Eigen::Matrix3d& R = Eigen::Matrix3d::Identity());
    bool isOccluderEnabled() const;

    // Thông tin trạng thái
    const CameraHardwareProfile& currentHardwareProfile() const;
    int currentProfileId() const;
    double currentBaselineMeters() const;
    std::string lastErrorMessage() const;

    // Nạp CAD Mesh mới
    bool loadTargetMesh(const std::string& objPath,
                        const std::string& texturePath = "",
                        const Eigen::Vector3d& scale = Eigen::Vector3d(0.001, 0.001, 0.001));

    // ── Chế độ Cửa Sổ Mô Phỏng Tương Tác 3D (MuJoCo Interactive 3D Window) ──
    bool openInteractiveWindow(int winW = 800, int winH = 700, int posX = -1, int posY = -1);
    void closeInteractiveWindow();
    bool isInteractiveWindowOpen() const;
    bool pollInteractiveEvents(); // Xử lý chuột/phím, cập nhật 3D view, trả về false nếu cửa sổ bị đóng
    void setAutoSpin(bool enabled, double speedDegPerFrame = 0.8);
    bool isAutoSpin() const;
    void resetPose();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

