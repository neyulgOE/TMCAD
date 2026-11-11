#include "CameraProfileDialog.h"
#include "CameraProfileParser.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>
#include <QDir>
#include <iomanip>
#include <sstream>

CameraProfileDialog::CameraProfileDialog(QWidget* parent)
    : QDialog(parent) {
    setWindowTitle(tr("Quản lý Hồ sơ Camera & Cấu hình Mô phỏng (Camera Profile Manager)"));
    resize(640, 500);
    setupUi();

    // Tự động thử nạp file cấu hình mặc định nếu có
    QString defaultPath = "/home/huyen1905/TMCAD/D435if-dataset";
    if (QFile::exists(defaultPath)) {
        loadAndDisplayFile(defaultPath);
    } else {
        QString presetPath = "/home/huyen1905/TMCAD/configs/camera_profiles/D435if_factory.json";
        if (QFile::exists(presetPath)) {
            loadAndDisplayFile(presetPath);
        }
    }
}

CameraProfileDialog::~CameraProfileDialog() = default;

void CameraProfileDialog::setupUi() {
    auto mainLayout = new QVBoxLayout(this);

    // 1. Nhóm nạp file
    auto fileGroup = new QGroupBox(tr("1. Nạp Tệp Hiệu Chuẩn Phần Cứng (Hardware Calibration Dump)"), this);
    auto fileLayout = new QGridLayout(fileGroup);

    fileLayout->addWidget(new QLabel(tr("Đường dẫn tệp:"), this), 0, 0);
    filePathEdit_ = new QLineEdit(this);
    filePathEdit_->setReadOnly(true);
    fileLayout->addWidget(filePathEdit_, 0, 1);

    auto browseBtn = new QPushButton(tr("Duyệt file..."), this);
    connect(browseBtn, &QPushButton::clicked, this, &CameraProfileDialog::onBrowseFile);
    fileLayout->addWidget(browseBtn, 0, 2);

    cameraTypeLabel_ = new QLabel(tr("Loại Camera: Chưa nạp"), this);
    cameraTypeLabel_->setStyleSheet("font-weight: bold; color: #2b78e4;");
    fileLayout->addWidget(cameraTypeLabel_, 1, 0, 1, 2);

    baselineLabel_ = new QLabel(tr("Khoảng cách Baseline: -- mm"), this);
    fileLayout->addWidget(baselineLabel_, 1, 2);

    mainLayout->addWidget(fileGroup);

    // 2. Nhóm chọn chế độ phân giải
    auto modeGroup = new QGroupBox(tr("2. Chọn Chế độ Phân giải Mô phỏng (Stream Profiles)"), this);
    auto modeLayout = new QGridLayout(modeGroup);

    modeLayout->addWidget(new QLabel(tr("Chế độ Stream:"), this), 0, 0);
    profileCombo_ = new QComboBox(this);
    connect(profileCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &CameraProfileDialog::onProfileSelected);
    modeLayout->addWidget(profileCombo_, 0, 1, 1, 3);

    modeLayout->addWidget(new QLabel(tr("Độ phân giải (W x H):"), this), 1, 0);
    resLabel_ = new QLabel("--", this);
    resLabel_->setStyleSheet("font-weight: bold;");
    modeLayout->addWidget(resLabel_, 1, 1);

    modeLayout->addWidget(new QLabel(tr("Góc mở dọc (FOV Y):"), this), 1, 2);
    fovLabel_ = new QLabel("--", this);
    fovLabel_->setStyleSheet("font-weight: bold;");
    modeLayout->addWidget(fovLabel_, 1, 3);

    modeLayout->addWidget(new QLabel(tr("Tiêu cự fx / fy:"), this), 2, 0);
    fxLabel_ = new QLabel("--", this);
    modeLayout->addWidget(fxLabel_, 2, 1);

    modeLayout->addWidget(new QLabel(tr("Tâm quang học cx / cy:"), this), 2, 2);
    cxLabel_ = new QLabel("--", this);
    modeLayout->addWidget(cxLabel_, 2, 3);

    mainLayout->addWidget(modeGroup);

    // 3. Tùy chọn mô phỏng cảm biến RealSense
    auto simGroup = new QGroupBox(tr("3. Tùy chọn Giả lập Cảm biến (Sensor Simulation Options)"), this);
    auto simLayout = new QVBoxLayout(simGroup);

    irFilterCheck_ = new QCheckBox(tr("Kích hoạt bộ lọc hồng ngoại D435if (IR-Pass Filter 750nm Simulation)"), this);
    irFilterCheck_->setChecked(true);
    simLayout->addWidget(irFilterCheck_);

    stereoShadowCheck_ = new QCheckBox(tr("Mô phỏng vùng mù viền vật thể (Stereo Occlusion Shadow Holes)"), this);
    stereoShadowCheck_->setChecked(true);
    simLayout->addWidget(stereoShadowCheck_);

    mainLayout->addWidget(simGroup);

    // 4. Thanh nút bấm
    auto btnLayout = new QHBoxLayout();
    savePresetBtn_ = new QPushButton(tr("Lưu thành Preset..."), this);
    connect(savePresetBtn_, &QPushButton::clicked, this, &CameraProfileDialog::onSavePresetClicked);
    btnLayout->addWidget(savePresetBtn_);

    btnLayout->addStretch();

    auto cancelBtn = new QPushButton(tr("Đóng"), this);
    connect(cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
    btnLayout->addWidget(cancelBtn);

    applyBtn_ = new QPushButton(tr("Áp dụng vào Mô phỏng"), this);
    applyBtn_->setStyleSheet("background-color: #2b78e4; color: white; font-weight: bold; padding: 6px 16px;");
    connect(applyBtn_, &QPushButton::clicked, this, &CameraProfileDialog::onApplyClicked);
    btnLayout->addWidget(applyBtn_);

    mainLayout->addLayout(btnLayout);
}

void CameraProfileDialog::onBrowseFile() {
    QString filter = tr("Tệp Hiệu chuẩn (*.json *dataset *txt);;Tất cả tệp (*.*)");
    QString path = QFileDialog::getOpenFileName(this, tr("Chọn tệp hiệu chuẩn camera"),
                                                "/home/huyen1905/TMCAD", filter);
    if (!path.isEmpty()) {
        loadAndDisplayFile(path);
    }
}

void CameraProfileDialog::loadAndDisplayFile(const QString& path) {
    CameraHardwareProfile prof;
    if (!CameraProfileParser::loadProfile(path.toStdString(), prof)) {
        QMessageBox::warning(this, tr("Lỗi nạp file"),
                             tr("Không thể đọc thông số camera từ tệp:\n%1").arg(path));
        return;
    }

    profile_ = prof;
    filePathEdit_->setText(path);
    cameraTypeLabel_->setText(tr("Loại Camera: %1").arg(QString::fromStdString(profile_.profileName)));
    baselineLabel_->setText(tr("Khoảng cách Baseline: %1 mm")
                            .arg(profile_.baselineMeters * 1000.0, 0, 'f', 3));

    // Điền ComboBox
    profileCombo_->blockSignals(true);
    profileCombo_->clear();

    auto validList = profile_.listValidProfiles();
    int defaultIdx = 0;
    for (int i = 0; i < static_cast<int>(validList.size()); ++i) {
        const auto& p = validList[i];
        QString text = QString::fromStdString(p.label);
        profileCombo_->addItem(text, p.profileId);
        // Ưu tiên chọn 1280x720 (Profile 1) hoặc 1080p
        if (p.profileId == 1) defaultIdx = i;
    }

    // Thêm tùy chọn tùy biến: 1024x576 (Tỉ lệ 16:9)
    profileCombo_->addItem(tr("1024x576 (Tùy biến - Tự động nội suy từ 16:9)"), -1);

    profileCombo_->setCurrentIndex(defaultIdx);
    profileCombo_->blockSignals(false);

    updateIntrinsicsPreview();
}

void CameraProfileDialog::onProfileSelected(int /*index*/) {
    updateIntrinsicsPreview();
}

void CameraProfileDialog::updateIntrinsicsPreview() {
    CameraIntrinsics intr = activeIntrinsics();
    if (!intr.valid()) return;

    resLabel_->setText(QString("%1 x %2").arg(intr.width).arg(intr.height));
    double fovY = 2.0 * std::atan(static_cast<double>(intr.height) / (2.0 * intr.fy)) * 180.0 / M_PI;
    fovLabel_->setText(QString("%1°").arg(fovY, 0, 'f', 1));

    fxLabel_->setText(QString("fx=%1 / fy=%2").arg(intr.fx, 0, 'f', 2).arg(intr.fy, 0, 'f', 2));
    cxLabel_->setText(QString("cx=%1 / cy=%2").arg(intr.cx, 0, 'f', 2).arg(intr.cy, 0, 'f', 2));
}

CameraHardwareProfile CameraProfileDialog::currentProfile() const {
    return profile_;
}

int CameraProfileDialog::selectedProfileId() const {
    if (!profileCombo_) return -1;
    return profileCombo_->currentData().toInt();
}

CameraIntrinsics CameraProfileDialog::activeIntrinsics() const {
    int pId = selectedProfileId();
    if (pId >= 0) {
        auto itD = profile_.depthProfiles.find(pId);
        if (itD != profile_.depthProfiles.end() && itD->second.isValid()) {
            return itD->second.toCameraIntrinsics();
        }
        auto itC = profile_.colorProfiles.find(pId);
        if (itC != profile_.colorProfiles.end() && itC->second.isValid()) {
            return itC->second.toCameraIntrinsics();
        }
    }
    // Chế độ tùy biến 1024x576
    return profile_.interpolateIntrinsics(1024, 576, true);
}

void CameraProfileDialog::onApplyClicked() {
    if (!profile_.isValid()) {
        QMessageBox::warning(this, tr("Chưa nạp hồ sơ"), tr("Vui lòng nạp một tệp hiệu chuẩn hợp lệ trước khi áp dụng."));
        return;
    }
    emit profileApplied(profile_, selectedProfileId());
    accept();
}

void CameraProfileDialog::onSavePresetClicked() {
    if (!profile_.isValid()) return;

    QString defaultDir = "/home/huyen1905/TMCAD/configs/camera_profiles";
    QDir().mkpath(defaultDir);

    QString savePath = QFileDialog::getSaveFileName(this, tr("Lưu hồ sơ camera thành Preset"),
                                                    defaultDir + "/" + QString::fromStdString(profile_.profileName) + ".json",
                                                    tr("Tệp JSON (*.json)"));
    if (!savePath.isEmpty()) {
        QFile src(filePathEdit_->text());
        if (src.copy(savePath) || QFile::copy(filePathEdit_->text(), savePath)) {
            QMessageBox::information(this, tr("Đã lưu Preset"), tr("Đã lưu hồ sơ camera thành công vào:\n%1").arg(savePath));
        }
    }
}
