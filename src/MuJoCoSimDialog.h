#pragma once

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QComboBox>
#include <QCheckBox>
#include <QSlider>
#include <Eigen/Dense>
#include "CameraHardwareProfile.h"

struct MuJoCoSimConfig {
    std::string cadPath;
    std::string texPath;
    Eigen::Vector3d cadScale = Eigen::Vector3d(0.001, 0.001, 0.001); // Mặc định mm sang mét
    double distanceMeters = 0.68;
    double camHeightMeters = 0.42; // Độ cao camera so với mặt sàn (m)
    double yawDeg = 20.0;
    double pitchDeg = 15.0;
    double rollDeg = 0.0;
    bool occluderEnabled = false;
    double occlusionPercent = 35.0;
    int profileId = 1; // 720p hoặc 1024x576
    bool isInteractiveMode = true; // Chế độ mở 2 bên (TMCAD + MuJoCo 3D)
};

class MuJoCoSimDialog : public QDialog {
    Q_OBJECT

public:
    explicit MuJoCoSimDialog(const CameraHardwareProfile& profile,
                             const QString& activeOnnxPath = "",
                             const QString& defaultCadPath = "",
                             QWidget* parent = nullptr);
    ~MuJoCoSimDialog() override;

    MuJoCoSimConfig getSimulationConfig() const;

signals:
    void livePoseChanged(double yaw, double pitch, double roll, double dist, bool occluder, double occPct);
    void autoSpinToggled(bool enabled);
    void randomizePoseRequested();
    void stopInteractiveRequested();

private slots:
    void onBrowseCad();
    void onBrowseTex();
    void onAutoDetectFromOnnx();
    void onDistanceChanged(int val);
    void onOcclusionChanged(int val);
    void onSliderLiveChanged();
    void onAutoSpinClicked();
    void onRandomizeClicked();
    void onStopClicked();
    void onStartClicked();

private:
    void setupUi();
    void autoDetectTexture(const QString& cadPath);
    bool autoDetectCadForOnnx(bool showWarningIfNotFound);
    QString resolveProjectRoot() const;

    QLineEdit* cadPathEdit_ = nullptr;
    QLineEdit* texPathEdit_ = nullptr;
    QComboBox* profileCombo_ = nullptr;
    QSlider* distSlider_ = nullptr;
    QLabel* distLabel_ = nullptr;
    QSlider* camHeightSlider_ = nullptr;
    QLabel* camHeightLabel_ = nullptr;
    QSlider* yawSlider_ = nullptr;
    QLabel* yawLabel_ = nullptr;
    QSlider* pitchSlider_ = nullptr;
    QLabel* pitchLabel_ = nullptr;
    QCheckBox* occluderCheck_ = nullptr;
    QSlider* occlusionSlider_ = nullptr;
    QLabel* occlusionLabel_ = nullptr;

    CameraHardwareProfile profile_;
    QString activeOnnxPath_;
    QString defaultCadPath_;
    QComboBox* scaleCombo_ = nullptr;
    MuJoCoSimConfig resultConfig_;
};
