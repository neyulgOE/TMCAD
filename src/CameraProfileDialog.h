#pragma once

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QComboBox>
#include <QCheckBox>
#include <memory>
#include "CameraHardwareProfile.h"

class CameraProfileDialog : public QDialog {
    Q_OBJECT

public:
    explicit CameraProfileDialog(QWidget* parent = nullptr);
    ~CameraProfileDialog() override;

    // Lấy hồ sơ và profile được chọn
    CameraHardwareProfile currentProfile() const;
    int selectedProfileId() const;
    CameraIntrinsics activeIntrinsics() const;

signals:
    void profileApplied(const CameraHardwareProfile& profile, int profileId);

private slots:
    void onBrowseFile();
    void onProfileSelected(int index);
    void onApplyClicked();
    void onSavePresetClicked();

private:
    void setupUi();
    void loadAndDisplayFile(const QString& path);
    void updateIntrinsicsPreview();

    QLineEdit* filePathEdit_ = nullptr;
    QLabel* cameraTypeLabel_ = nullptr;
    QLabel* baselineLabel_ = nullptr;
    QComboBox* profileCombo_ = nullptr;

    QLabel* fxLabel_ = nullptr;
    QLabel* fyLabel_ = nullptr;
    QLabel* cxLabel_ = nullptr;
    QLabel* cyLabel_ = nullptr;
    QLabel* fovLabel_ = nullptr;
    QLabel* resLabel_ = nullptr;

    QCheckBox* irFilterCheck_ = nullptr;
    QCheckBox* stereoShadowCheck_ = nullptr;

    QPushButton* applyBtn_ = nullptr;
    QPushButton* savePresetBtn_ = nullptr;

    CameraHardwareProfile profile_;
};

