#include "MuJoCoSimDialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>
#include <QDir>
#include <QCoreApplication>

MuJoCoSimDialog::MuJoCoSimDialog(const CameraHardwareProfile& profile,
                                 const QString& activeOnnxPath,
                                 const QString& defaultCadPath,
                                 QWidget* parent)
    : QDialog(parent), profile_(profile), activeOnnxPath_(activeOnnxPath), defaultCadPath_(defaultCadPath) {
    setWindowTitle(tr("Thiết lập Mô phỏng Camera Ảo MuJoCo D435if"));
    resize(640, 580);
    setupUi();

    // 1. Ưu tiên tự động tìm CAD khớp với Model ONNX đang được chọn ở Bước 2
    if (!activeOnnxPath_.isEmpty()) {
        autoDetectCadForOnnx(false);
    }
    // 2. Nếu không tìm được từ ONNX, fallback sang CAD đang mở sẵn trong Scene 3D
    if (cadPathEdit_->text().trimmed().isEmpty() && !defaultCadPath_.isEmpty() && QFile::exists(defaultCadPath_)) {
        cadPathEdit_->setText(defaultCadPath_);
        autoDetectTexture(defaultCadPath_);
    }
    // 3. Fallback mặc định cuối cùng (dò từ project root, không hardcode user home)
    if (cadPathEdit_->text().trimmed().isEmpty()) {
        QDir projDir(resolveProjectRoot());
        QString defaultCad = projDir.absoluteFilePath("CAD_DATA/obj_000002/obj_000002.obj");
        if (QFile::exists(defaultCad)) {
            cadPathEdit_->setText(defaultCad);
            autoDetectTexture(defaultCad);
        }
    }
}

MuJoCoSimDialog::~MuJoCoSimDialog() = default;

QString MuJoCoSimDialog::resolveProjectRoot() const {
    auto findRootFrom = [](QDir d) -> QString {
        for (int i = 0; i < 5; ++i) {
            if (d.exists("CAD_DATA") && (d.exists("models") || d.exists("CMakeLists.txt"))) {
                return d.absolutePath();
            }
            if (!d.cdUp()) break;
        }
        return QString();
    };
    if (!activeOnnxPath_.isEmpty()) {
        QString r = findRootFrom(QFileInfo(activeOnnxPath_).absoluteDir());
        if (!r.isEmpty()) return r;
    }
    QString rApp = findRootFrom(QDir(QCoreApplication::applicationDirPath()));
    if (!rApp.isEmpty()) return rApp;
    QString rCwd = findRootFrom(QDir::current());
    if (!rCwd.isEmpty()) return rCwd;
    return QDir::currentPath();
}

void MuJoCoSimDialog::setupUi() {
    auto mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(12);

    // 1. Nhóm chọn file CAD
    auto cadGroup = new QGroupBox(tr("1. Chọn Tệp CAD Mục Tiêu (Target CAD && Texture)"), this);
    auto cadLayout = new QGridLayout(cadGroup);

    cadLayout->addWidget(new QLabel(tr("Tệp CAD (.obj, .stl, .ply):"), this), 0, 0);
    cadPathEdit_ = new QLineEdit(this);
    cadLayout->addWidget(cadPathEdit_, 0, 1);

    auto browseCadBtn = new QPushButton(tr("Duyệt CAD..."), this);
    connect(browseCadBtn, &QPushButton::clicked, this, &MuJoCoSimDialog::onBrowseCad);
    cadLayout->addWidget(browseCadBtn, 0, 2);

    cadLayout->addWidget(new QLabel(tr("Vân bề mặt (Texture .png):"), this), 1, 0);
    texPathEdit_ = new QLineEdit(this);
    texPathEdit_->setPlaceholderText(tr("Để trống để dùng màu Studio CAD kỹ thuật sáng rõ"));
    cadLayout->addWidget(texPathEdit_, 1, 1);

    auto texBtnLayout = new QHBoxLayout();
    auto browseTexBtn = new QPushButton(tr("Duyệt Ảnh..."), this);
    connect(browseTexBtn, &QPushButton::clicked, this, &MuJoCoSimDialog::onBrowseTex);
    auto clearTexBtn = new QPushButton(tr("❌ Màu Studio CAD"), this);
    clearTexBtn->setToolTip(tr("Xóa ảnh vân để chuyển sang màu Studio CAD tương phản cao (Cam kỹ thuật)"));
    connect(clearTexBtn, &QPushButton::clicked, this, [this]() {
        texPathEdit_->clear();
    });
    texBtnLayout->addWidget(browseTexBtn);
    texBtnLayout->addWidget(clearTexBtn);
    cadLayout->addLayout(texBtnLayout, 1, 2);

    cadLayout->addWidget(new QLabel(tr("Đơn vị đo / Tỉ lệ (Scale):"), this), 2, 0);
    scaleCombo_ = new QComboBox(this);
    scaleCombo_->addItem(tr("Milimét (mm → m, ×0.001) [Khuyên dùng cho CAD/BOP]"), 0.001);
    scaleCombo_->addItem(tr("Mét (m → m, ×1.0)"), 1.0);
    scaleCombo_->addItem(tr("Centimét (cm → m, ×0.01)"), 0.01);
    cadLayout->addWidget(scaleCombo_, 2, 1, 1, 2);

    auto autoBtn = new QPushButton(tr("🎯 Tự Động Tìm CAD Khớp với Model ONNX Đang Chọn"), this);
    autoBtn->setStyleSheet("background-color: #00838f; color: white; font-weight: bold; padding: 5px 10px;");
    connect(autoBtn, &QPushButton::clicked, this, &MuJoCoSimDialog::onAutoDetectFromOnnx);
    cadLayout->addWidget(autoBtn, 3, 0, 1, 3);

    mainLayout->addWidget(cadGroup);

    // 2. Nhóm chọn Profile Camera D435if
    auto camGroup = new QGroupBox(tr("2. Độ Phân Giải && Hồ Sơ Camera (Camera Profile)"), this);
    auto camLayout = new QHBoxLayout(camGroup);

    camLayout->addWidget(new QLabel(tr("Chế độ Camera D435if:"), this));
    profileCombo_ = new QComboBox(this);

    auto validList = profile_.listValidProfiles();
    int defaultIdx = 0;
    for (int i = 0; i < static_cast<int>(validList.size()); ++i) {
        const auto& p = validList[i];
        profileCombo_->addItem(QString::fromStdString(p.label), p.profileId);
        if (p.profileId == 1) defaultIdx = i; // 1280x720
    }
    profileCombo_->addItem(tr("1024x576 (Tùy biến - Tự động nội suy từ 16:9)"), -1);
    profileCombo_->setCurrentIndex(defaultIdx);
    camLayout->addWidget(profileCombo_);

    mainLayout->addWidget(camGroup);

    // 3. Nhóm tư thế và che khuất (Occlusion)
    auto poseGroup = new QGroupBox(tr("3. Tư Thế Vật Thể && Mô Phỏng Che Khuất (Occlusion)"), this);
    auto poseLayout = new QGridLayout(poseGroup);

    // Khoảng cách camera
    poseLayout->addWidget(new QLabel(tr("Khoảng cách Camera (Z):"), this), 0, 0);
    distSlider_ = new QSlider(Qt::Horizontal, this);
    distSlider_->setRange(40, 250); // 0.40m - 2.50m
    distSlider_->setValue(58);      // 0.58m chuẩn cự ly RealSense D435
    connect(distSlider_, &QSlider::valueChanged, this, &MuJoCoSimDialog::onDistanceChanged);
    poseLayout->addWidget(distSlider_, 0, 1);
    distLabel_ = new QLabel("0.58 m", this);
    distLabel_->setMinimumWidth(50);
    distLabel_->setStyleSheet("font-weight: bold; color: #4fc3f7;");
    poseLayout->addWidget(distLabel_, 0, 2);

    // Độ cao camera so với mặt sàn
    poseLayout->addWidget(new QLabel(tr("Độ cao Camera (Z_cam):"), this), 1, 0);
    camHeightSlider_ = new QSlider(Qt::Horizontal, this);
    camHeightSlider_->setRange(15, 150); // 0.15m - 1.50m
    camHeightSlider_->setValue(42);
    connect(camHeightSlider_, &QSlider::valueChanged, this, [this](int val) {
        camHeightLabel_->setText(QString::number(val / 100.0, 'f', 2) + " m");
    });
    poseLayout->addWidget(camHeightSlider_, 1, 1);
    camHeightLabel_ = new QLabel("0.42 m", this);
    camHeightLabel_->setMinimumWidth(50);
    camHeightLabel_->setStyleSheet("font-weight: bold; color: #80cbc4;");
    poseLayout->addWidget(camHeightLabel_, 1, 2);

    // Góc xoay Yaw (Xoay quanh trục thẳng đứng trên mặt bàn)
    poseLayout->addWidget(new QLabel(tr("Góc xoay ngang (Yaw):"), this), 2, 0);
    yawSlider_ = new QSlider(Qt::Horizontal, this);
    yawSlider_->setRange(-180, 180);
    yawSlider_->setValue(0);
    connect(yawSlider_, &QSlider::valueChanged, this, [this](int v) {
        yawLabel_->setText(QString("%1°").arg(v));
    });
    poseLayout->addWidget(yawSlider_, 2, 1);
    yawLabel_ = new QLabel("0°", this);
    yawLabel_->setStyleSheet("font-weight: bold;");
    poseLayout->addWidget(yawLabel_, 2, 2);

    // Góc nghiêng Pitch (Nghiêng trên mặt bàn)
    poseLayout->addWidget(new QLabel(tr("Góc nghiêng dọc (Pitch):"), this), 3, 0);
    pitchSlider_ = new QSlider(Qt::Horizontal, this);
    pitchSlider_->setRange(-60, 60);
    pitchSlider_->setValue(0);
    connect(pitchSlider_, &QSlider::valueChanged, this, [this](int v) {
        pitchLabel_->setText(QString("%1°").arg(v));
    });
    poseLayout->addWidget(pitchSlider_, 3, 1);
    pitchLabel_ = new QLabel("0°", this);
    pitchLabel_->setStyleSheet("font-weight: bold;");
    poseLayout->addWidget(pitchLabel_, 3, 2);

    // Vật cản che khuất (Lon nấm đè lên)
    occluderCheck_ = new QCheckBox(tr("Kích hoạt Lon Nấm đè lên để test che khuất (Occlusion Test)"), this);
    occluderCheck_->setChecked(false);
    poseLayout->addWidget(occluderCheck_, 4, 0, 1, 3);

    poseLayout->addWidget(new QLabel(tr("Mức độ che khuất:"), this), 5, 0);
    occlusionSlider_ = new QSlider(Qt::Horizontal, this);
    occlusionSlider_->setRange(0, 75);
    occlusionSlider_->setValue(35);
    connect(occlusionSlider_, &QSlider::valueChanged, this, &MuJoCoSimDialog::onOcclusionChanged);
    poseLayout->addWidget(occlusionSlider_, 5, 1);
    occlusionLabel_ = new QLabel("35%", this);
    occlusionLabel_->setStyleSheet("font-weight: bold; color: #ffb74d;");
    poseLayout->addWidget(occlusionLabel_, 4, 2);

    // Nút điều khiển tương tác trực tiếp
    auto spinBtn = new QPushButton(tr("🔄 Tự Động Xoay 360°"), this);
    spinBtn->setStyleSheet("background-color: #263238; color: #80cbc4; font-weight: bold; padding: 6px 12px; border-radius: 4px;");
    connect(spinBtn, &QPushButton::clicked, this, &MuJoCoSimDialog::onAutoSpinClicked);

    auto randBtn = new QPushButton(tr("🎲 Góc Ngẫu Nhiên"), this);
    randBtn->setStyleSheet("background-color: #263238; color: #ffb74d; font-weight: bold; padding: 6px 12px; border-radius: 4px;");
    connect(randBtn, &QPushButton::clicked, this, &MuJoCoSimDialog::onRandomizeClicked);

    auto liveToolsLayout = new QHBoxLayout();
    liveToolsLayout->addWidget(spinBtn);
    liveToolsLayout->addWidget(randBtn);
    poseLayout->addLayout(liveToolsLayout, 5, 0, 1, 3);

    mainLayout->addWidget(poseGroup);

    // Kết nối live slider signals
    connect(distSlider_, &QSlider::valueChanged, this, &MuJoCoSimDialog::onSliderLiveChanged);
    connect(yawSlider_, &QSlider::valueChanged, this, &MuJoCoSimDialog::onSliderLiveChanged);
    connect(pitchSlider_, &QSlider::valueChanged, this, &MuJoCoSimDialog::onSliderLiveChanged);
    connect(occluderCheck_, &QCheckBox::toggled, this, &MuJoCoSimDialog::onSliderLiveChanged);
    connect(occlusionSlider_, &QSlider::valueChanged, this, &MuJoCoSimDialog::onSliderLiveChanged);

    // 4. Nút bấm
    auto btnLayout = new QHBoxLayout();
    btnLayout->addStretch();

    auto cancelBtn = new QPushButton(tr("Đóng"), this);
    connect(cancelBtn, &QPushButton::clicked, this, [this]() {
        onStopClicked();
    });
    btnLayout->addWidget(cancelBtn);

    auto startBtn = new QPushButton(tr("🚀 BẮT ĐẦU MÔ PHỎNG 2 BÊN (TMCAD + MUJOCO 3D)"), this);
    startBtn->setStyleSheet("background-color: #00897b; color: white; font-weight: bold; font-size: 13px; padding: 10px 22px; border-radius: 4px;");
    connect(startBtn, &QPushButton::clicked, this, &MuJoCoSimDialog::onStartClicked);
    btnLayout->addWidget(startBtn);

    mainLayout->addLayout(btnLayout);
}

void MuJoCoSimDialog::onBrowseCad() {
    QString filter = tr("Tệp CAD (*.obj *.stl *.ply);;Tất cả tệp (*.*)");
    QString initDir = QDir(resolveProjectRoot()).absoluteFilePath("CAD_DATA");
    QString path = QFileDialog::getOpenFileName(this, tr("Chọn tệp CAD"), initDir, filter);
    if (!path.isEmpty()) {
        cadPathEdit_->setText(path);
        autoDetectTexture(path);
    }
}

void MuJoCoSimDialog::onBrowseTex() {
    QString filter = tr("Ảnh Texture (*.png *.jpg *.jpeg);;Tất cả tệp (*.*)");
    QString initDir = QDir(resolveProjectRoot()).absoluteFilePath("CAD_DATA");
    QString path = QFileDialog::getOpenFileName(this, tr("Chọn ảnh vân bề mặt"), initDir, filter);
    if (!path.isEmpty()) {
        texPathEdit_->setText(path);
    }
}

void MuJoCoSimDialog::autoDetectTexture(const QString& cadPath) {
    if (cadPath.isEmpty()) return;
    QFileInfo fi(cadPath);
    QString baseNoExt = fi.dir().absoluteFilePath(fi.completeBaseName());

    // Nếu người dùng chọn file .stl nhưng có file .obj tương ứng cùng tên (có UV texture coordinates):
    // Tự động nâng cấp lên file .obj để hiển thị vân ảnh 3D chân thực!
    if (fi.suffix().toLower() == "stl" && QFile::exists(baseNoExt + ".obj")) {
        cadPathEdit_->setText(baseNoExt + ".obj");
    }

    // Tự động chọn đơn vị Scale nếu là tệp canonical _mesh.ply (mét -> 1.0)
    if (scaleCombo_) {
        if (cadPath.endsWith("_mesh.ply", Qt::CaseInsensitive)) {
            int idx = scaleCombo_->findData(1.0);
            if (idx >= 0) scaleCombo_->setCurrentIndex(idx);
        } else {
            QString unitFile = cadPath + ".unit";
            if (QFile::exists(unitFile)) {
                QFile uf(unitFile);
                if (uf.open(QIODevice::ReadOnly)) {
                    QString u = QString::fromUtf8(uf.readAll()).trimmed().toLower();
                    double factor = (u == "m" || u == "meter" || u == "meters") ? 1.0 :
                                    (u == "cm") ? 0.01 : 0.001;
                    int idx = scaleCombo_->findData(factor);
                    if (idx >= 0) scaleCombo_->setCurrentIndex(idx);
                }
            }
        }
    }

    // Thử tìm file .png hoặc .jpg cùng tên
    if (QFile::exists(baseNoExt + ".png")) {
        texPathEdit_->setText(baseNoExt + ".png");
    } else if (QFile::exists(baseNoExt + ".jpg")) {
        texPathEdit_->setText(baseNoExt + ".jpg");
    } else {
        texPathEdit_->clear();
    }

    const QString upperName = cadPath.toUpper();
    const bool isLargeFloorObj = upperName.contains("TABLE") || upperName.contains("PALLET") || upperName.contains("RACK");
    if (distSlider_) {
        distSlider_->setValue(isLargeFloorObj ? 135 : 58);
    }
    if (camHeightSlider_) {
        camHeightSlider_->setValue(isLargeFloorObj ? 42 : 95);
    }
}

bool MuJoCoSimDialog::autoDetectCadForOnnx(bool showWarningIfNotFound) {
    if (activeOnnxPath_.isEmpty()) {
        if (showWarningIfNotFound) {
            QMessageBox::information(this, tr("Chưa chọn Model"),
                                     tr("Chưa có model ONNX nào được chọn ở Bước 2.\nVui lòng bấm 'Duyệt CAD...' để chọn tệp."));
        }
        return false;
    }

    QFileInfo onnxFi(activeOnnxPath_);
    if (onnxFi.isSymLink()) {
        QFileInfo targetFi(onnxFi.symLinkTarget());
        if (targetFi.exists()) onnxFi = targetFi;
    }
    QString baseName = onnxFi.completeBaseName();
    if (baseName == "model_obs_npu" || baseName == "model_obs_backbone") {
        baseName = onnxFi.dir().dirName();
    }

    QDir projDir(resolveProjectRoot());

    QStringList candidates = {
        projDir.absoluteFilePath("CAD_DATA/" + baseName + "/" + baseName + ".obj"),
        projDir.absoluteFilePath("CAD_DATA/" + baseName + "/" + baseName + ".stl"),
        projDir.absoluteFilePath("CAD_DATA/" + baseName + "/" + baseName + ".STL"),
        projDir.absoluteFilePath("CAD_DATA/" + baseName + "/" + baseName + ".ply"),
        projDir.absoluteFilePath("CAD_DATA/" + baseName + "/" + baseName + "_mesh.ply"),
        projDir.absoluteFilePath("CAD_DATA/" + baseName + ".obj"),
        projDir.absoluteFilePath("CAD_DATA/" + baseName + ".stl"),
        projDir.absoluteFilePath("CAD_DATA/" + baseName + ".STL"),
        projDir.absoluteFilePath("CAD_DATA/" + baseName + "_mesh.ply"),
        projDir.absoluteFilePath("data/hope_sample/cad_models/" + baseName + ".obj"),
        projDir.absoluteFilePath("data/hope_sample/cad_models/" + baseName + ".stl"),
        projDir.absoluteFilePath("data/hope_sample/cad_models/" + baseName + ".ply")
    };

    // Fallback tìm thư mục con trong CAD_DATA khớp tên model (ví dụ CHINSU.onnx -> Chai_Tuong_Ot_CHINSU_...)
    QDir cadRoot(projDir.absoluteFilePath("CAD_DATA"));
    if (cadRoot.exists() && !baseName.isEmpty()) {
        QStringList subDirs = cadRoot.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
        for (const QString& sub : subDirs) {
            if (sub.contains(baseName, Qt::CaseInsensitive) && sub != baseName) {
                candidates << cadRoot.absoluteFilePath(sub + "/" + sub + ".obj")
                           << cadRoot.absoluteFilePath(sub + "/" + sub + ".stl")
                           << cadRoot.absoluteFilePath(sub + "/" + sub + ".STL")
                           << cadRoot.absoluteFilePath(sub + "/" + sub + "_mesh.ply");
            }
        }
    }

    for (const QString& cand : candidates) {
        if (QFile::exists(cand)) {
            cadPathEdit_->setText(cand);
            autoDetectTexture(cand);
            return true;
        }
    }

    QString surfPly = projDir.absoluteFilePath("debug/onnx_cad_surface.ply");
    if (QFile::exists(surfPly)) {
        cadPathEdit_->setText(surfPly);
        texPathEdit_->clear();
        return true;
    }

    if (showWarningIfNotFound) {
        QMessageBox::warning(this, tr("Không tìm thấy CAD tự động"),
                             tr("Không tìm thấy tệp CAD tự động cho %1.\nVui lòng bấm 'Duyệt CAD...' để chọn tệp.")
                             .arg(baseName));
    }
    return false;
}

void MuJoCoSimDialog::onAutoDetectFromOnnx() {
    autoDetectCadForOnnx(true);
}

void MuJoCoSimDialog::onDistanceChanged(int val) {
    double d = static_cast<double>(val) / 100.0;
    distLabel_->setText(QString("%1 m").arg(d, 0, 'f', 2));
}

void MuJoCoSimDialog::onOcclusionChanged(int val) {
    occlusionLabel_->setText(QString("%1%").arg(val));
}

void MuJoCoSimDialog::onSliderLiveChanged() {
    double dist = static_cast<double>(distSlider_->value()) / 100.0;
    double yaw = static_cast<double>(yawSlider_->value());
    double pitch = static_cast<double>(pitchSlider_->value());
    double roll = 0.0;
    bool occ = occluderCheck_->isChecked();
    double occPct = static_cast<double>(occlusionSlider_->value());
    emit livePoseChanged(yaw, pitch, roll, dist, occ, occPct);
}

void MuJoCoSimDialog::onAutoSpinClicked() {
    static bool spin = false;
    spin = !spin;
    emit autoSpinToggled(spin);
}

void MuJoCoSimDialog::onRandomizeClicked() {
    int rYaw = -180 + (rand() % 360);
    int rPitch = -45 + (rand() % 90);
    int rDist = 45 + (rand() % 85); // 0.45m - 1.3m
    yawSlider_->setValue(rYaw);
    pitchSlider_->setValue(rPitch);
    distSlider_->setValue(rDist);
    emit randomizePoseRequested();
}

void MuJoCoSimDialog::onStopClicked() {
    emit stopInteractiveRequested();
    reject();
}

void MuJoCoSimDialog::onStartClicked() {
    QString cadPath = cadPathEdit_->text().trimmed();
    if (cadPath.isEmpty() || !QFile::exists(cadPath)) {
        QMessageBox::warning(this, tr("Thiếu tệp CAD"),
                             tr("Vui lòng chọn một tệp CAD (.obj, .stl, .ply) hợp lệ trước khi bắt đầu mô phỏng."));
        return;
    }

    resultConfig_.cadPath = cadPath.toStdString();
    resultConfig_.texPath = texPathEdit_->text().trimmed().toStdString();
    if (scaleCombo_) {
        double s = scaleCombo_->currentData().toDouble();
        resultConfig_.cadScale = Eigen::Vector3d(s, s, s);
    }
    resultConfig_.distanceMeters = static_cast<double>(distSlider_->value()) / 100.0;
    resultConfig_.camHeightMeters = camHeightSlider_ ? static_cast<double>(camHeightSlider_->value()) / 100.0 : 0.42;
    resultConfig_.yawDeg = yawSlider_->value();
    resultConfig_.pitchDeg = pitchSlider_->value();
    resultConfig_.occluderEnabled = occluderCheck_->isChecked();
    resultConfig_.occlusionPercent = occlusionSlider_->value();
    resultConfig_.profileId = profileCombo_->currentData().toInt();

    accept();
}

MuJoCoSimConfig MuJoCoSimDialog::getSimulationConfig() const {
    return resultConfig_;
}
