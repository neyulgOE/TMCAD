#include "CameraStreamWidget.h"
#include <librealsense2/rs.hpp>
#include <open3d/Open3D.h>

#include <QApplication>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QFileDialog>
#include <QDateTime>
#include <QMessageBox>
#include <QDir>
#include <QPainter>
#include <QFont>
#include <iostream>
#include <fstream>
#include <chrono>

struct CameraStreamWidget::Impl {
    rs2::pipeline pipe;
    rs2::config cfg;
    std::unique_ptr<rs2::align> align;
    rs2::colorizer colorizer;
    bool pipeline_running = false;

    rs2_intrinsics color_intrinsics{};
    double depth_scale = 0.001;
    CameraIntrinsics intrinsics{};
    std::string device_name = "Unknown RealSense";
    std::string serial_number;

    std::mutex frame_mutex;
    rs2::frameset last_frameset;
};

CameraStreamWidget::CameraStreamWidget(QWidget* parent, bool compactMode)
    : QWidget(parent), impl_(std::make_unique<Impl>()) {
    setStyleSheet("background-color: #1a1a20; color: #ffffff;");
    setupUi(compactMode);

    connect(this, &CameraStreamWidget::frameReady,
            this, &CameraStreamWidget::onFrameReady,
            Qt::QueuedConnection);
}

CameraStreamWidget::~CameraStreamWidget() {
    stopStream();
}

void CameraStreamWidget::setupUi(bool compactMode) {
    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(compactMode ? 4 : 8, compactMode ? 4 : 8, compactMode ? 4 : 8, compactMode ? 4 : 8);
    mainLayout->setSpacing(6);

    // 1. Header Bar: Device status, View Mode selector & Center distance
    auto* headerLayout = new QHBoxLayout();
    statusLabel_ = new QLabel("Khởi tạo RealSense D435if...", this);
    statusLabel_->setStyleSheet("font-size: 11px; font-weight: bold; color: #4fc3f7;");
    headerLayout->addWidget(statusLabel_);

    headerLayout->addSpacing(10);
    viewModeCombo_ = new QComboBox(this);
    viewModeCombo_->addItem("🖥️ 3 Khung: Nhận Diện AI | RGB Gốc | Độ Sâu Depth");
    viewModeCombo_->addItem("🎯 Phóng To Khung Nhận Diện AI (BBox HUD)");
    viewModeCombo_->addItem("📷 Chỉ Khung RGB & Độ Sâu Depth");
    viewModeCombo_->setStyleSheet(
        "QComboBox { background: #263238; color: #00e676; font-weight: bold; border: 1px solid #37474f; border-radius: 4px; padding: 2px 8px; font-size: 11px; }"
        "QComboBox::drop-down { border: none; }"
        "QComboBox QAbstractItemView { background: #263238; color: #eceff1; selection-background-color: #00897b; }");
    connect(viewModeCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CameraStreamWidget::onViewModeChanged);
    headerLayout->addWidget(viewModeCombo_);

    headerLayout->addStretch();

    centerDistLabel_ = new QLabel("Tâm ảnh: -- m", this);
    centerDistLabel_->setStyleSheet("font-size: 11px; background: #263238; padding: 2px 6px; border-radius: 4px; color: #00e676;");
    headerLayout->addWidget(centerDistLabel_);
    mainLayout->addLayout(headerLayout);

    // 2. Video Preview Section (3 Panes: AI Detection, Raw RGB, Depth Z16)
    auto* videoContainer = new QWidget(this);
    auto* videoLayout = new QHBoxLayout(videoContainer);
    videoLayout->setContentsMargins(0, 0, 0, 0);
    videoLayout->setSpacing(6);

    int previewW = compactMode ? 280 : 360;
    int previewH = compactMode ? 210 : 270;

    // Khung 1: 🎯 AI Object Detection & Bounding Box HUD
    detectionBox_ = new QGroupBox("🎯 Nhận Diện Đối Tượng (AI Detection & 6DoF HUD)", videoContainer);
    detectionBox_->setStyleSheet(
        "QGroupBox { font-weight: bold; color: #00e676; border: 1px solid #00e676; border-radius: 6px; margin-top: 4px; padding-top: 8px; background: #101518; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }");
    auto* detLayout = new QVBoxLayout(detectionBox_);
    detLayout->setContentsMargins(4, 4, 4, 4);
    detectionViewLabel_ = new QLabel("Chờ nhận diện AI...", detectionBox_);
    detectionViewLabel_->setMinimumSize(previewW, previewH);
    detectionViewLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    detectionViewLabel_->setAlignment(Qt::AlignCenter);
    detectionViewLabel_->setStyleSheet("background: #0a0d0e; border: 1px solid #1b382b; border-radius: 4px;");
    detLayout->addWidget(detectionViewLabel_);
    videoLayout->addWidget(detectionBox_);

    // Khung 2: 📷 Camera RGB Gốc (Raw Stream)
    colorBox_ = new QGroupBox("📷 Camera RGB Gốc (Raw)", videoContainer);
    colorBox_->setStyleSheet(
        "QGroupBox { font-weight: bold; color: #90caf9; border: 1px solid #37474f; border-radius: 6px; margin-top: 4px; padding-top: 8px; background: #121212; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }");
    auto* colLayout = new QVBoxLayout(colorBox_);
    colLayout->setContentsMargins(4, 4, 4, 4);
    colorViewLabel_ = new QLabel("Chờ luồng RGB...", colorBox_);
    colorViewLabel_->setMinimumSize(previewW, previewH);
    colorViewLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    colorViewLabel_->setAlignment(Qt::AlignCenter);
    colorViewLabel_->setStyleSheet("background: #0a0a0a; border: 1px solid #263238; border-radius: 4px;");
    colLayout->addWidget(colorViewLabel_);
    videoLayout->addWidget(colorBox_);

    // Khung 3: 🌐 Bản Đồ Độ Sâu (Depth Z16)
    depthBox_ = new QGroupBox("🌐 Bản Đồ Độ Sâu (Depth Z16)", videoContainer);
    depthBox_->setStyleSheet(
        "QGroupBox { font-weight: bold; color: #ffb74d; border: 1px solid #37474f; border-radius: 6px; margin-top: 4px; padding-top: 8px; background: #121212; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }");
    auto* depLayout = new QVBoxLayout(depthBox_);
    depLayout->setContentsMargins(4, 4, 4, 4);
    depthViewLabel_ = new QLabel("Chờ luồng Depth Z16...", depthBox_);
    depthViewLabel_->setMinimumSize(previewW, previewH);
    depthViewLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    depthViewLabel_->setAlignment(Qt::AlignCenter);
    depthViewLabel_->setStyleSheet("background: #0a0a0a; border: 1px solid #263238; border-radius: 4px;");
    depLayout->addWidget(depthViewLabel_);
    videoLayout->addWidget(depthBox_);

    mainLayout->addWidget(videoContainer);

    // 3. Control & Recording Bar
    auto* ctrlBox = new QGroupBox(compactMode ? "Thu Thập Dữ Liệu Thực Địa" : "Điều Khiển Thu Thập Dữ Liệu", this);
    ctrlBox->setStyleSheet("QGroupBox { font-weight: bold; color: #90caf9; border: 1px solid #37474f; border-radius: 6px; margin-top: 4px; padding-top: 6px; }");
    auto* ctrlLayout = new QVBoxLayout(ctrlBox);
    ctrlLayout->setContentsMargins(4, 4, 4, 4);
    ctrlLayout->setSpacing(4);

    auto* dirLayout = new QHBoxLayout();
    dirLayout->addWidget(new QLabel("Lưu trữ:", ctrlBox));

    QDir projDir(QCoreApplication::applicationDirPath());
    if (projDir.dirName() == "build") projDir.cdUp();
    QString defaultOutDir = projDir.absoluteFilePath("data/realsense_records");

    outputDirEdit_ = new QLineEdit(defaultOutDir, ctrlBox);
    outputDirEdit_->setStyleSheet("background: #263238; color: #ffffff; border: 1px solid #546e7a; padding: 2px 4px; border-radius: 4px; font-size: 11px;");
    dirLayout->addWidget(outputDirEdit_);

    browseBtn_ = new QPushButton("Chọn...", ctrlBox);
    browseBtn_->setStyleSheet("background: #37474f; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px;");
    connect(browseBtn_, &QPushButton::clicked, this, &CameraStreamWidget::onBrowseOutputDir);
    dirLayout->addWidget(browseBtn_);
    ctrlLayout->addLayout(dirLayout);

    // Buttons
    auto* btnLayout = new QHBoxLayout();
    recordBtn_ = new QPushButton("🔴 Bắt Đầu Ghi Video (.bag)", ctrlBox);
    recordBtn_->setFixedHeight(compactMode ? 32 : 36);
    recordBtn_->setStyleSheet("background: #d32f2f; color: white; font-weight: bold; font-size: 12px; border-radius: 4px;");
    connect(recordBtn_, &QPushButton::clicked, this, &CameraStreamWidget::toggleRecord);
    btnLayout->addWidget(recordBtn_);

    snapshotBtn_ = new QPushButton("📸 Chụp Ảnh Mẫu (PNGs)", ctrlBox);
    snapshotBtn_->setFixedHeight(compactMode ? 32 : 36);
    snapshotBtn_->setStyleSheet("background: #1976d2; color: white; font-weight: bold; font-size: 12px; border-radius: 4px;");
    connect(snapshotBtn_, &QPushButton::clicked, this, &CameraStreamWidget::takeSnapshot);
    btnLayout->addWidget(snapshotBtn_);

    ctrlLayout->addLayout(btnLayout);
    mainLayout->addWidget(ctrlBox);
}

bool CameraStreamWidget::startStream(const std::string& recordBagPath) {
    // Nếu truyền vào một tệp .bag đã tồn tại, tự động chuyển sang chế độ Playback
    if (!recordBagPath.empty() && QFile::exists(QString::fromStdString(recordBagPath))) {
        return startPlayback(recordBagPath);
    }

    stopStream();

    try {
        rs2::context ctx;
        auto devices = ctx.query_devices();
        if (devices.size() == 0 && recordBagPath.empty()) {
            statusLabel_->setText("<font color='#f44336'><b>Lỗi: Không tìm thấy camera RealSense D435if cắm qua USB!</b></font>");
            recordBtn_->setEnabled(false);
            snapshotBtn_->setEnabled(false);
            return false;
        }

        if (devices.size() > 0) {
            auto dev = devices[0];
            impl_->device_name = dev.get_info(RS2_CAMERA_INFO_NAME);
            impl_->serial_number = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
        }

        impl_->cfg = rs2::config();
        impl_->cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
        impl_->cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

        if (!recordBagPath.empty()) {
            impl_->cfg.enable_record_to_file(recordBagPath);
        }

        auto profile = impl_->pipe.start(impl_->cfg);

        auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
        impl_->depth_scale = depth_sensor.get_depth_scale();

        auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        impl_->color_intrinsics = color_stream.get_intrinsics();

        impl_->intrinsics.width = impl_->color_intrinsics.width;
        impl_->intrinsics.height = impl_->color_intrinsics.height;
        impl_->intrinsics.fx = impl_->color_intrinsics.fx;
        impl_->intrinsics.fy = impl_->color_intrinsics.fy;
        impl_->intrinsics.cx = impl_->color_intrinsics.ppx;
        impl_->intrinsics.cy = impl_->color_intrinsics.ppy;

        impl_->align = std::make_unique<rs2::align>(RS2_STREAM_COLOR);
        impl_->pipeline_running = true;

        if (recordBagPath.empty()) {
            statusLabel_->setText(QString("● SẴN SÀNG: %1 (SN: %2) - 640x480 @ 30FPS")
                .arg(QString::fromStdString(impl_->device_name))
                .arg(QString::fromStdString(impl_->serial_number)));
        } else {
            statusLabel_->setText(QString("<font color='#ff5252'><b>● ĐANG GHI HÌNH FILE .BAG: %1</b></font>")
                .arg(QFileInfo(QString::fromStdString(recordBagPath)).fileName()));
        }

        recordBtn_->setEnabled(true);
        snapshotBtn_->setEnabled(true);

        captureRunning_.store(true);
        captureThread_ = std::thread(&CameraStreamWidget::captureWorker, this);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "CameraStreamWidget startStream error: " << e.what() << "\n";
        statusLabel_->setText("<font color='#f44336'><b>Lỗi kết nối camera RealSense!</b></font>");
        recordBtn_->setEnabled(false);
        snapshotBtn_->setEnabled(false);
        return false;
    }
}

bool CameraStreamWidget::startPlayback(const std::string& bagPath) {
    stopStream();

    try {
        QFileInfo fi(QString::fromStdString(bagPath));
        if (!fi.exists() || !fi.isFile()) {
            statusLabel_->setText("<font color='#f44336'><b>Lỗi: Không tìm thấy file .bag!</b></font>");
            return false;
        }

        impl_->cfg = rs2::config();
        // Phát lặp lại liên tục (repeat = true), không cần cắm camera USB
        impl_->cfg.enable_device_from_file(bagPath, true);

        auto profile = impl_->pipe.start(impl_->cfg);
        auto dev = profile.get_device();
        impl_->device_name = dev.get_info(RS2_CAMERA_INFO_NAME);
        impl_->serial_number = "Tệp .bag";

        auto playback = dev.as<rs2::playback>();
        if (playback) {
            playback.set_real_time(true); // Phát mượt mà đúng tốc độ thời gian thực ~30 FPS
        }

        try {
            auto depth_sensor = dev.first<rs2::depth_sensor>();
            impl_->depth_scale = depth_sensor.get_depth_scale();
        } catch (...) {
            impl_->depth_scale = 0.001;
        }

        auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        impl_->color_intrinsics = color_stream.get_intrinsics();

        impl_->intrinsics.width = impl_->color_intrinsics.width;
        impl_->intrinsics.height = impl_->color_intrinsics.height;
        impl_->intrinsics.fx = impl_->color_intrinsics.fx;
        impl_->intrinsics.fy = impl_->color_intrinsics.fy;
        impl_->intrinsics.cx = impl_->color_intrinsics.ppx;
        impl_->intrinsics.cy = impl_->color_intrinsics.ppy;

        impl_->align = std::make_unique<rs2::align>(RS2_STREAM_COLOR);
        impl_->pipeline_running = true;
        isPlayback_ = true;
        currentBagPath_ = bagPath;

        statusLabel_->setText(QString("● ĐANG PHÁT LẠI FILE .BAG: %1 (%2x%3 - Lặp lại)")
            .arg(fi.fileName())
            .arg(impl_->intrinsics.width)
            .arg(impl_->intrinsics.height));

        recordBtn_->setText("⏹️ Dừng Phát (.bag)");
        recordBtn_->setStyleSheet("background: #f57c00; color: white; font-weight: bold; font-size: 12px; border-radius: 4px;");
        recordBtn_->setEnabled(true);
        snapshotBtn_->setEnabled(true);

        captureRunning_.store(true);
        captureThread_ = std::thread(&CameraStreamWidget::captureWorker, this);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "CameraStreamWidget startPlayback error: " << e.what() << "\n";
        statusLabel_->setText(QString("<font color='#f44336'><b>Lỗi phát lại file .bag: %1</b></font>").arg(e.what()));
        recordBtn_->setEnabled(false);
        snapshotBtn_->setEnabled(false);
        return false;
    }
}

void CameraStreamWidget::stopStream() {
    captureRunning_.store(false);
    if (captureThread_.joinable()) {
        captureThread_.join();
    }
    if (impl_->pipeline_running) {
        try {
            impl_->pipe.stop();
        } catch (...) {}
        impl_->pipeline_running = false;
    }
    isPlayback_ = false;
    isRecording_ = false;
    recordBtn_->setText("🔴 Bắt Đầu Ghi Video (.bag)");
    recordBtn_->setStyleSheet("background: #d32f2f; color: white; font-weight: bold; font-size: 12px; border-radius: 4px;");
}

bool CameraStreamWidget::isStreaming() const {
    return impl_->pipeline_running;
}

bool CameraStreamWidget::isRecording() const {
    return isRecording_;
}

void CameraStreamWidget::captureWorker() {
    while (captureRunning_.load()) {
        try {
            rs2::frameset frames = impl_->pipe.wait_for_frames(200);
            rs2::video_frame color = frames.get_color_frame();
            rs2::depth_frame depth = frames.get_depth_frame();
            if (!color || !depth) continue;

            {
                std::lock_guard<std::mutex> lock(impl_->frame_mutex);
                impl_->last_frameset = frames;
            }

            int cw = color.get_width();
            int ch = color.get_height();
            int dw = depth.get_width();
            int dh = depth.get_height();

            QImage colorImg(cw, ch, QImage::Format_RGB888);
            std::memcpy(colorImg.bits(), color.get_data(), cw * ch * 3);

            rs2::video_frame colorized_depth = impl_->colorizer.colorize(depth);
            QImage depthImg(dw, dh, QImage::Format_RGB888);
            std::memcpy(depthImg.bits(), colorized_depth.get_data(), dw * dh * 3);

            float center_d = depth.get_distance(dw / 2, dh / 2);

            emit frameReady(colorImg, depthImg, center_d);
        } catch (...) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
}

void CameraStreamWidget::setPoseOverlay3D(const Eigen::Matrix3d& R,
                                          const Eigen::Vector3d& t,
                                          const CameraIntrinsics& intr,
                                          const QString& label,
                                          const QString& poseBanner,
                                          double fitness,
                                          const Eigen::Vector3d& cadMin,
                                          const Eigen::Vector3d& cadMax) {
    CadOverlayData data;
    data.valid = true;
    data.label = label;
    data.poseBanner = poseBanner;

    double w_px = (intr.width > 0) ? intr.width : 640.0;
    const double s = std::clamp(w_px / 1024.0, 0.5, 2.0);
    const float th_thick = std::max(1.0f, static_cast<float>(std::round(3.0 * s)));
    const float th_med   = std::max(1.0f, static_cast<float>(std::round(2.0 * s)));
    const float th_thin  = 1.0f;

    auto proj_f = [&](double x, double y, double z) -> QPointF {
        Eigen::Vector3d pt_cad(x, y, z);
        Eigen::Vector3d pt_cam = R * pt_cad + t;
        double z_eff = std::max(0.05, pt_cam.z());
        qreal u = intr.fx * pt_cam.x() / z_eff + intr.cx;
        qreal v = intr.fy * pt_cam.y() / z_eff + intr.cy;
        return QPointF(u, v);
    };

    const QColor colYellow(255, 238, 0);
    const QColor colGreen(0, 230, 118);
    const QColor colCyan(0, 229, 255);

    // Nếu là vật thể CAD thông thường (không phải TABLE-DEMO): Vẽ khung 3D Bounding Box chính xác
    if ((cadMax - cadMin).norm() > 0.005 && !label.contains("TABLE", Qt::CaseInsensitive)) {
        double dx = (cadMax.x() - cadMin.x()) * 1000.0;
        double dy = (cadMax.y() - cadMin.y()) * 1000.0;
        double dz = (cadMax.z() - cadMin.z()) * 1000.0;
        data.headerBanner = QString("6DoF BOUNDING BOX (%1x%2x%3mm)").arg(dx, 0, 'f', 0).arg(dy, 0, 'f', 0).arg(dz, 0, 'f', 0);

        double x0 = cadMin.x(), x1 = cadMax.x();
        double y0 = cadMin.y(), y1 = cadMax.y();
        double z0 = cadMin.z(), z1 = cadMax.z();

        QPointF c0 = proj_f(x0, y0, z0);
        QPointF c1 = proj_f(x1, y0, z0);
        QPointF c2 = proj_f(x1, y1, z0);
        QPointF c3 = proj_f(x0, y1, z0);
        QPointF c4 = proj_f(x0, y0, z1);
        QPointF c5 = proj_f(x1, y0, z1);
        QPointF c6 = proj_f(x1, y1, z1);
        QPointF c7 = proj_f(x0, y1, z1);

        // Đáy (z0)
        data.segments.push_back({c0, c1, colGreen, th_med});
        data.segments.push_back({c1, c2, colGreen, th_med});
        data.segments.push_back({c2, c3, colGreen, th_med});
        data.segments.push_back({c3, c0, colGreen, th_med});

        // Đỉnh (z1)
        data.segments.push_back({c4, c5, colYellow, th_thick});
        data.segments.push_back({c5, c6, colYellow, th_thick});
        data.segments.push_back({c6, c7, colYellow, th_thick});
        data.segments.push_back({c7, c4, colYellow, th_thick});

        // 4 cột đứng
        data.segments.push_back({c0, c4, colCyan, th_med});
        data.segments.push_back({c1, c5, colCyan, th_med});
        data.segments.push_back({c2, c6, colCyan, th_med});
        data.segments.push_back({c3, c7, colCyan, th_med});

        // Vành đai cắt ngang thân ở giữa (Mid-plane slice theo chuẩn Benchmark 3D)
        double zm = 0.5 * (z0 + z1);
        QPointF m0 = proj_f(x0, y0, zm);
        QPointF m1 = proj_f(x1, y0, zm);
        QPointF m2 = proj_f(x1, y1, zm);
        QPointF m3 = proj_f(x0, y1, zm);
        data.segments.push_back({m0, m1, QColor(0, 230, 118, 180), th_thin});
        data.segments.push_back({m1, m2, QColor(0, 230, 118, 180), th_thin});
        data.segments.push_back({m2, m3, QColor(0, 230, 118, 180), th_thin});
        data.segments.push_back({m3, m0, QColor(0, 230, 118, 180), th_thin});

        // Mũi tên vector 3D từ tâm vật thể chỉ dọc theo trục chính (theo chuẩn Benchmark HOPE/BOP)
        Eigen::Vector3d pt_c = 0.5 * (cadMin + cadMax);
        double axisLenZ = std::max(0.025, (cadMax.z() - cadMin.z()) * 0.40);
        double axisLenX = std::max(0.025, (cadMax.x() - cadMin.x()) * 0.40);
        Eigen::Vector3d pt_z = pt_c + Eigen::Vector3d(0.0, 0.0, axisLenZ);
        Eigen::Vector3d pt_x = pt_c + Eigen::Vector3d(axisLenX, 0.0, 0.0);
        QPointF p_center = proj_f(pt_c.x(), pt_c.y(), pt_c.z());
        QPointF p_arrow  = proj_f(pt_z.x(), pt_z.y(), pt_z.z());
        data.centerAxis = p_center;
        data.axisZ = p_arrow;
        data.axisX = proj_f(pt_x.x(), pt_x.y(), pt_x.z());
        data.hasAxes = true;

        // Đảm bảo đa giác tô bóng nằm ở mặt đỉnh thực tế (Y thấp nhất trong hệ tọa độ quang học camera)
        bool z1_is_top = (R * Eigen::Vector3d(0, 0, z1) + t).y() <= (R * Eigen::Vector3d(0, 0, z0) + t).y();
        if (z1_is_top) {
            data.topSurfacePolygon << c4 << c5 << c6 << c7;
        } else {
            data.topSurfacePolygon << c0 << c1 << c2 << c3;
        }
        setCadOverlay(data);
        return;
    }

    data.headerBanner = "FRONT-FACE ANCHORED (830x505mm) -> EXTRUDE 440mm";

    // Exact CAD dimensions for TABLE-DEMO (synchronized with TMVis TablePoseEstimator.cpp)
    const double cad_x0 = -0.220, cad_x1 = 0.220; // 440mm depth
    const double cad_y0 = -0.415, cad_y1 = 0.415; // 830mm length
    const double z_floor = -0.3355;
    const double z_top   =  0.1665; // 502mm total height

    const double leg_w = 0.040;
    const double beam_h = 0.040;
    const double foot_h = 0.046;

    // Determine Front vs Rear face (Front has smaller Z in camera coordinates)
    Eigen::Vector3d p_pos_x = R * Eigen::Vector3d(cad_x1, 0, 0) + t;
    Eigen::Vector3d p_neg_x = R * Eigen::Vector3d(cad_x0, 0, 0) + t;
    double x_front = (p_pos_x.z() < p_neg_x.z()) ? cad_x1 : cad_x0;
    double x_rear  = (p_pos_x.z() < p_neg_x.z()) ? cad_x0 : cad_x1;

    // Determine Left vs Right in camera coordinates (smaller X in camera is Left)
    Eigen::Vector3d p_pos_y = R * Eigen::Vector3d(0, cad_y1, 0) + t;
    Eigen::Vector3d p_neg_y = R * Eigen::Vector3d(0, cad_y0, 0) + t;
    double y_left  = (p_pos_y.x() < p_neg_y.x()) ? cad_y1 : cad_y0;
    double y_right = (p_pos_y.x() < p_neg_y.x()) ? cad_y0 : cad_y1;

    double y_left_inner  = (y_left < y_right) ? (y_left + leg_w) : (y_left - leg_w);
    double y_right_inner = (y_right < y_left) ? (y_right + leg_w) : (y_right - leg_w);
    double x_front_inner = (x_front < x_rear) ? (x_front + leg_w) : (x_front - leg_w);
    double x_rear_inner  = (x_rear < x_front) ? (x_rear + leg_w) : (x_rear - leg_w);

    double z_top_inner = z_top - beam_h;
    double z_leg_bot   = z_floor + foot_h;

    double y_l_c  = 0.5 * (y_left  + y_left_inner);
    double y_r_c  = 0.5 * (y_right + y_right_inner);

    // Outer 4 corners of tabletop frame
    QPointF p_f_tl = proj_f(x_front, y_left,  z_top);
    QPointF p_f_tr = proj_f(x_front, y_right, z_top);
    QPointF p_r_tl = proj_f(x_rear,  y_left,  z_top);
    QPointF p_r_tr = proj_f(x_rear,  y_right, z_top);

    // Foot contact points on floor
    QPointF p_f_bl = proj_f(x_front,      y_l_c, z_floor);
    QPointF p_f_br = proj_f(x_front,      y_r_c, z_floor);
    QPointF p_r_bl = proj_f(x_rear_inner, y_l_c, z_floor);
    QPointF p_r_br = proj_f(x_rear_inner, y_r_c, z_floor);

    // 1a. Shaded Cyan Tabletop
    data.topSurfacePolygon << p_f_tl << p_f_tr << p_r_tr << p_r_tl;

    // 1b. Floor Footprint Polygon
    data.footPoints = {p_f_bl, p_f_br, p_r_br, p_r_bl};

    // Structural Extrusion Beams
    QPointF p_f_tl_b = proj_f(x_front, y_left_inner,  z_top_inner);
    QPointF p_f_tr_b = proj_f(x_front, y_right_inner, z_top_inner);
    data.segments.push_back({p_f_tl, p_f_tr, colYellow, th_thick});

    // Rear Top Beam
    QPointF p_r_tl_b = proj_f(x_rear_inner, y_left_inner, z_top_inner);
    QPointF p_r_tr_b = proj_f(x_rear_inner, y_right_inner, z_top_inner);
    data.segments.push_back({p_r_tl, p_r_tr, colGreen, th_med});
    data.segments.push_back({p_r_tl_b, p_r_tr_b, colGreen, th_thin});

    // Side Top Beams & Inner Portal Arch under Tabletop
    QPointF rl_top_fo = proj_f(x_rear_inner, y_left,  z_top_inner);
    QPointF rr_top_fo = proj_f(x_rear_inner, y_right, z_top_inner);
    data.segments.push_back({p_f_tl_b, rl_top_fo, colYellow, th_thin});
    data.segments.push_back({p_f_tr_b, rr_top_fo, colGreen,  th_thin});
    data.segments.push_back({proj_f(x_front_inner, y_left_inner, z_top_inner),
                             proj_f(x_rear_inner,  y_left_inner, z_top_inner), colGreen, th_thin});
    data.segments.push_back({proj_f(x_front_inner, y_right_inner, z_top_inner),
                             proj_f(x_rear_inner,  y_right_inner, z_top_inner), colGreen, th_thin});

    // Front-Left Column (Yellow) - 4 vertical edges + bottom caps + stem
    QPointF fl_top_fo = p_f_tl;
    QPointF fl_top_ts = proj_f(x_front,       y_l_c,        z_top);
    QPointF fl_top_fi = proj_f(x_front,       y_left_inner, z_top);
    QPointF fl_top_bi = proj_f(x_front_inner, y_left_inner, z_top_inner);
    QPointF fl_bot_fo = proj_f(x_front,       y_left,       z_leg_bot);
    QPointF fl_bot_ts = proj_f(x_front,       y_l_c,        z_leg_bot);
    QPointF fl_bot_fi = proj_f(x_front,       y_left_inner, z_leg_bot);
    QPointF fl_bot_bi = proj_f(x_front_inner, y_left_inner, z_leg_bot);
    data.segments.push_back({fl_top_fo, fl_bot_fo, colYellow, th_thick});
    data.segments.push_back({fl_top_ts, fl_bot_ts, colGreen,  th_thin});
    data.segments.push_back({fl_top_fi, fl_bot_fi, colYellow, th_med});
    data.segments.push_back({fl_top_bi, fl_bot_bi, colGreen,  th_thin});
    data.segments.push_back({fl_bot_fo, fl_bot_fi, colYellow, th_med});
    data.segments.push_back({fl_bot_fi, fl_bot_bi, colYellow, th_thin});
    data.segments.push_back({fl_bot_ts, p_f_bl,    colYellow, th_med});

    // Front-Right Column (Cyan)
    QPointF fr_top_fo = p_f_tr;
    QPointF fr_top_ts = proj_f(x_front,       y_r_c,         z_top);
    QPointF fr_top_fi = proj_f(x_front,       y_right_inner, z_top);
    QPointF fr_top_bi = proj_f(x_front_inner, y_right_inner, z_top_inner);
    QPointF fr_bot_fo = proj_f(x_front,       y_right,       z_leg_bot);
    QPointF fr_bot_ts = proj_f(x_front,       y_r_c,         z_leg_bot);
    QPointF fr_bot_fi = proj_f(x_front,       y_right_inner, z_leg_bot);
    QPointF fr_bot_bi = proj_f(x_front_inner, y_right_inner, z_leg_bot);
    data.segments.push_back({fr_top_fo, fr_bot_fo, colCyan,  th_thick});
    data.segments.push_back({fr_top_ts, fr_bot_ts, colCyan,  th_thin});
    data.segments.push_back({fr_top_fi, fr_bot_fi, colCyan,  th_med});
    data.segments.push_back({fr_top_bi, fr_bot_bi, colGreen, th_thin});
    data.segments.push_back({fr_bot_fo, fr_bot_fi, colCyan,  th_med});
    data.segments.push_back({fr_bot_fi, fr_bot_bi, colCyan,  th_thin});
    data.segments.push_back({fr_bot_ts, p_f_br,    colCyan,  th_med});

    // Rear-Left Column (Green)
    QPointF rl_top_ts = proj_f(x_rear_inner, y_l_c,        z_top_inner);
    QPointF rl_top_fi = proj_f(x_rear_inner, y_left_inner, z_top_inner);
    QPointF rl_top_bi = proj_f(x_rear,       y_left_inner, z_top_inner);
    QPointF rl_bot_fo = proj_f(x_rear_inner, y_left,       z_leg_bot);
    QPointF rl_bot_ts = proj_f(x_rear_inner, y_l_c,        z_leg_bot);
    QPointF rl_bot_fi = proj_f(x_rear_inner, y_left_inner, z_leg_bot);
    QPointF rl_bot_bi = proj_f(x_rear,       y_left_inner, z_leg_bot);
    data.segments.push_back({rl_top_fo, rl_bot_fo, colGreen, th_med});
    data.segments.push_back({rl_top_ts, rl_bot_ts, colGreen, th_thin});
    data.segments.push_back({rl_top_bi, rl_bot_bi, colGreen, th_thin});
    data.segments.push_back({rl_bot_fo, rl_bot_fi, colGreen, th_med});
    data.segments.push_back({rl_bot_fi, rl_bot_bi, colGreen, th_thin});
    data.segments.push_back({rl_bot_ts, p_r_bl,    colGreen, th_thin});

    // Rear-Right Column (Green)
    QPointF rr_top_ts = proj_f(x_rear_inner, y_r_c,         z_top_inner);
    QPointF rr_top_fi = proj_f(x_rear_inner, y_right_inner, z_top_inner);
    QPointF rr_top_bi = proj_f(x_rear,       y_right_inner, z_top_inner);
    QPointF rr_bot_fo = proj_f(x_rear_inner, y_right,       z_leg_bot);
    QPointF rr_bot_ts = proj_f(x_rear_inner, y_r_c,         z_leg_bot);
    QPointF rr_bot_fi = proj_f(x_rear_inner, y_right_inner, z_leg_bot);
    QPointF rr_bot_bi = proj_f(x_rear,       y_right_inner, z_leg_bot);
    data.segments.push_back({rr_top_fo, rr_bot_fo, colGreen, th_med});
    data.segments.push_back({rr_top_ts, rr_bot_ts, colGreen, th_thin});
    data.segments.push_back({rr_top_bi, rr_bot_bi, colGreen, th_thin});
    data.segments.push_back({rr_bot_fo, rr_bot_fi, colGreen, th_med});
    data.segments.push_back({rr_bot_fi, rr_bot_bi, colGreen, th_thin});
    data.segments.push_back({rr_bot_ts, p_r_br,    colGreen, th_thin});

    // 2. Center 3D Coordinate Axes (X: Red depth, Y: Green length, Z: Blue up)
    QPointF p_center = (p_f_tl + p_f_tr + p_r_tl + p_r_tr + p_f_bl + p_f_br + p_r_bl + p_r_br) * 0.125;
    data.centerAxis = p_center;
    data.axisX = p_center + QPointF(55.0 * s, 0);
    data.axisZ = p_center - QPointF(0, 55.0 * s);
    data.axisY = p_center;
    data.hasAxes = true;

    // Tabletop Center for Header pointer line
    data.tableCenter = (p_f_tl + p_f_tr + p_r_tl + p_r_tr) * 0.25;
    data.hasTableCenter = true;

    // Overall 2D Bounding Box
    qreal min_x = std::min({p_f_tl.x(), p_f_tr.x(), p_r_tl.x(), p_r_tr.x(), p_f_bl.x(), p_f_br.x(), p_r_bl.x(), p_r_br.x()});
    qreal max_x = std::max({p_f_tl.x(), p_f_tr.x(), p_r_tl.x(), p_r_tr.x(), p_f_bl.x(), p_f_br.x(), p_r_bl.x(), p_r_br.x()});
    qreal min_y = std::min({p_f_tl.y(), p_f_tr.y(), p_r_tl.y(), p_r_tr.y(), p_f_bl.y(), p_f_br.y(), p_r_bl.y(), p_r_br.y()});
    qreal max_y = std::max({p_f_tl.y(), p_f_tr.y(), p_r_tl.y(), p_r_tr.y(), p_f_bl.y(), p_f_br.y(), p_r_bl.y(), p_r_br.y()});
    data.bbox = QRectF(min_x, min_y, std::max(10.0, max_x - min_x), std::max(10.0, max_y - min_y));

    setCadOverlay(data);
}

void CameraStreamWidget::setCadOverlay(const CadOverlayData& data) {
    std::lock_guard<std::mutex> lock(overlayMutex_);
    hasOverlay_ = true;
    overlayCadData_ = data;
    overlayCadData_.valid = true;
    overlayBbox_ = data.bbox;
    overlayLabel_ = data.label;
    overlayPoseText_ = data.poseBanner;
    overlayTimestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void CameraStreamWidget::setOverlayDetection(const QRectF& bbox,
                                             const std::vector<QPointF>& corners3DProj,
                                             const QString& label,
                                             const QString& poseText) {
    std::lock_guard<std::mutex> lock(overlayMutex_);
    hasOverlay_ = true;
    overlayBbox_ = bbox;
    overlayCorners3DProj_ = corners3DProj;
    overlayLabel_ = label;
    overlayPoseText_ = poseText;
    overlayTimestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // Construct canonical Target CadOverlayData from 8 corners
    overlayCadData_ = CadOverlayData();
    if (corners3DProj.size() == 8) {
        overlayCadData_.bbox = bbox;
        overlayCadData_.label = label;
        overlayCadData_.poseBanner = poseText;
        overlayCadData_.valid = true;

        const auto& c = corners3DProj;
        auto dist2 = [](const QPointF& a, const QPointF& b) {
            qreal dx = a.x() - b.x(), dy = a.y() - b.y();
            return dx * dx + dy * dy;
        };

        bool face1_is_front = dist2(c[5], c[6]) >= dist2(c[4], c[7]);
        QPointF p_f_1_t = face1_is_front ? c[5] : c[4];
        QPointF p_f_1_b = face1_is_front ? c[1] : c[0];
        QPointF p_f_2_t = face1_is_front ? c[6] : c[7];
        QPointF p_f_2_b = face1_is_front ? c[2] : c[3];

        QPointF p_r_1_t = face1_is_front ? c[4] : c[5];
        QPointF p_r_1_b = face1_is_front ? c[0] : c[1];
        QPointF p_r_2_t = face1_is_front ? c[7] : c[6];
        QPointF p_r_2_b = face1_is_front ? c[3] : c[2];

        QPointF p_f_tl = (p_f_1_t.x() < p_f_2_t.x()) ? p_f_1_t : p_f_2_t;
        QPointF p_f_bl = (p_f_1_t.x() < p_f_2_t.x()) ? p_f_1_b : p_f_2_b;
        QPointF p_f_tr = (p_f_1_t.x() < p_f_2_t.x()) ? p_f_2_t : p_f_1_t;
        QPointF p_f_br = (p_f_1_t.x() < p_f_2_t.x()) ? p_f_2_b : p_f_1_b;

        QPointF p_r_tl = (p_r_1_t.x() < p_r_2_t.x()) ? p_r_1_t : p_r_2_t;
        QPointF p_r_bl = (p_r_1_t.x() < p_r_2_t.x()) ? p_r_1_b : p_r_2_b;
        QPointF p_r_tr = (p_r_1_t.x() < p_r_2_t.x()) ? p_r_2_t : p_r_1_t;
        QPointF p_r_br = (p_r_1_t.x() < p_r_2_t.x()) ? p_r_2_b : p_r_1_b;

        // 1a. Shaded Tabletop (Top Face)
        overlayCadData_.topSurfacePolygon << p_f_tl << p_f_tr << p_r_tr << p_r_tl;

        // 1b. Floor Footprint Polygon
        overlayCadData_.footPoints = {p_f_bl, p_f_br, p_r_br, p_r_bl};

        QColor colYellow(255, 238, 0);
        QColor colGreen(0, 230, 118);
        QColor colCyan(0, 229, 255);

        QPointF v_down_f_l = p_f_bl - p_f_tl;
        QPointF v_down_f_r = p_f_br - p_f_tr;
        QPointF v_down_r_l = p_r_bl - p_r_tl;
        QPointF v_down_r_r = p_r_br - p_r_tr;

        qreal h_beam = 0.08;
        qreal h_leg_bot = 0.89;

        // 1c. Top Beams
        overlayCadData_.segments.push_back({p_f_tl, p_f_tr, colYellow, 3.0f});
        overlayCadData_.segments.push_back({p_f_tl + h_beam * v_down_f_l, p_f_tr + h_beam * v_down_f_r, colYellow, 2.0f});

        overlayCadData_.segments.push_back({p_r_tl, p_r_tr, colGreen, 2.0f});
        overlayCadData_.segments.push_back({p_r_tl + h_beam * v_down_r_l, p_r_tr + h_beam * v_down_r_r, colGreen, 1.5f});

        overlayCadData_.segments.push_back({p_f_tl, p_r_tl, colGreen, 2.0f});
        overlayCadData_.segments.push_back({p_f_tr, p_r_tr, colCyan, 2.0f});

        // Front center vertical support rib
        QPointF p_center_t = (p_f_tl + p_f_tr) / 2.0 + h_beam * (v_down_f_l + v_down_f_r) / 2.0;
        QPointF p_center_b = p_center_t + 0.10 * (v_down_f_l + v_down_f_r) / 2.0;
        overlayCadData_.segments.push_back({p_center_t - QPointF(5, 0), p_center_b - QPointF(5, 0), colYellow, 2.0f});
        overlayCadData_.segments.push_back({p_center_t + QPointF(5, 0), p_center_b + QPointF(5, 0), colYellow, 2.0f});
        overlayCadData_.segments.push_back({p_center_b - QPointF(5, 0), p_center_b + QPointF(5, 0), colYellow, 2.0f});

        // 1d. Front-Left Column (Yellow) - 3 vertical edges + bottom cap + leveling stem
        QPointF p_f_tl_inner_y = p_f_tl + 0.07 * (p_f_tr - p_f_tl);
        QPointF p_f_bl_inner_y = p_f_bl + 0.07 * (p_f_br - p_f_bl);
        QPointF p_f_tl_inner_x = p_f_tl + 0.14 * (p_r_tl - p_f_tl);
        QPointF p_f_bl_inner_x = p_f_bl + 0.14 * (p_r_bl - p_f_bl);

        QPointF leg_bot_l = p_f_tl + h_leg_bot * v_down_f_l;
        QPointF leg_bot_l_iy = p_f_tl_inner_y + h_leg_bot * v_down_f_l;
        QPointF leg_bot_l_ix = p_f_tl_inner_x + h_leg_bot * v_down_f_l;

        overlayCadData_.segments.push_back({p_f_tl, leg_bot_l, colYellow, 3.0f});
        overlayCadData_.segments.push_back({p_f_tl_inner_y + h_beam * v_down_f_l, leg_bot_l_iy, colYellow, 2.0f});
        overlayCadData_.segments.push_back({p_f_tl_inner_x + h_beam * v_down_f_l, leg_bot_l_ix, colYellow, 2.0f});
        overlayCadData_.segments.push_back({leg_bot_l, leg_bot_l_iy, colYellow, 2.0f});
        overlayCadData_.segments.push_back({leg_bot_l, leg_bot_l_ix, colYellow, 2.0f});

        QPointF stem_top_l = (leg_bot_l + leg_bot_l_iy + leg_bot_l_ix) / 3.0;
        overlayCadData_.segments.push_back({stem_top_l, p_f_bl, colYellow, 2.0f});

        // Front left 45 deg gusset
        overlayCadData_.segments.push_back({p_f_tl_inner_y + (h_beam + 0.12) * v_down_f_l,
                                            p_f_tl_inner_y + 0.08 * (p_f_tr - p_f_tl) + h_beam * v_down_f_l, colYellow, 2.0f});

        // 1e. Front-Right Column (Green) - 3 vertical edges + bottom cap + leveling stem
        QPointF p_f_tr_inner_y = p_f_tr + 0.07 * (p_f_tl - p_f_tr);
        QPointF p_f_br_inner_y = p_f_br + 0.07 * (p_f_bl - p_f_br);
        QPointF p_f_tr_inner_x = p_f_tr + 0.14 * (p_r_tr - p_f_tr);
        QPointF p_f_br_inner_x = p_f_br + 0.14 * (p_r_bl - p_f_br);

        QPointF leg_bot_r = p_f_tr + h_leg_bot * v_down_f_r;
        QPointF leg_bot_r_iy = p_f_tr_inner_y + h_leg_bot * v_down_f_r;
        QPointF leg_bot_r_ix = p_f_tr_inner_x + h_leg_bot * v_down_f_r;

        overlayCadData_.segments.push_back({p_f_tr, leg_bot_r, colGreen, 3.0f});
        overlayCadData_.segments.push_back({p_f_tr_inner_y + h_beam * v_down_f_r, leg_bot_r_iy, colGreen, 2.0f});
        overlayCadData_.segments.push_back({p_f_tr_inner_x + h_beam * v_down_f_r, leg_bot_r_ix, colGreen, 2.0f});
        overlayCadData_.segments.push_back({leg_bot_r, leg_bot_r_iy, colGreen, 2.0f});
        overlayCadData_.segments.push_back({leg_bot_r, leg_bot_r_ix, colGreen, 2.0f});

        QPointF stem_top_r = (leg_bot_r + leg_bot_r_iy + leg_bot_r_ix) / 3.0;
        overlayCadData_.segments.push_back({stem_top_r, p_f_br, colGreen, 2.0f});

        // Front right 45 deg gusset
        overlayCadData_.segments.push_back({p_f_tr_inner_y + (h_beam + 0.12) * v_down_f_r,
                                            p_f_tr_inner_y + 0.08 * (p_f_tl - p_f_tr) + h_beam * v_down_f_r, colGreen, 2.0f});

        // 1f. Rear-Left Column (Green)
        QPointF leg_bot_rl = p_r_tl + h_leg_bot * v_down_r_l;
        overlayCadData_.segments.push_back({p_r_tl, leg_bot_rl, colGreen, 2.0f});
        overlayCadData_.segments.push_back({leg_bot_rl, p_r_bl, colGreen, 2.0f});

        // 1g. Rear-Right Column (Cyan)
        QPointF leg_bot_rr = p_r_tr + h_leg_bot * v_down_r_r;
        overlayCadData_.segments.push_back({p_r_tr, leg_bot_rr, colCyan, 2.0f});
        overlayCadData_.segments.push_back({leg_bot_rr, p_r_br, colCyan, 2.0f});

        // 1h. Left Side Frame Inner Cutout (with 45 deg Gussets) - Green
        QPointF s_l_f_top = p_f_tl_inner_x + (h_beam + 0.12) * v_down_f_l;
        QPointF s_l_f_top_g = p_f_tl_inner_x + 0.14 * (p_r_tl - p_f_tl) + h_beam * v_down_f_l;
        QPointF s_l_f_bot = p_f_tl_inner_x + 0.60 * v_down_f_l;
        QPointF s_l_r_bot = p_r_tl + 0.14 * (p_f_tl - p_r_tl) + 0.60 * v_down_r_l;
        QPointF s_l_r_top = p_r_tl + 0.14 * (p_f_tl - p_r_tl) + (h_beam + 0.12) * v_down_r_l;
        QPointF s_l_r_top_g = p_r_tl + 0.28 * (p_f_tl - p_r_tl) + h_beam * v_down_r_l;

        overlayCadData_.segments.push_back({s_l_f_top_g, s_l_f_top, colGreen, 2.0f});
        overlayCadData_.segments.push_back({s_l_f_top, s_l_f_bot, colGreen, 2.0f});
        overlayCadData_.segments.push_back({s_l_f_bot, s_l_r_bot, colGreen, 2.0f});
        overlayCadData_.segments.push_back({s_l_r_bot, s_l_r_top, colGreen, 2.0f});
        overlayCadData_.segments.push_back({s_l_r_top, s_l_r_top_g, colGreen, 2.0f});
        overlayCadData_.segments.push_back({p_f_tl_inner_x + 0.68 * v_down_f_l, p_r_tl + 0.14 * (p_f_tl - p_r_tl) + 0.68 * v_down_r_l, colGreen, 2.0f});

        // 1i. Right Side Frame Inner Cutout (with 45 deg Gussets) - Cyan
        QPointF s_r_f_top = p_f_tr_inner_x + (h_beam + 0.12) * v_down_f_r;
        QPointF s_r_f_top_g = p_f_tr_inner_x + 0.14 * (p_r_tr - p_f_tr) + h_beam * v_down_f_r;
        QPointF s_r_f_bot = p_f_tr_inner_x + 0.60 * v_down_f_r;
        QPointF s_r_r_bot = p_r_tr + 0.14 * (p_f_tr - p_r_tr) + 0.60 * v_down_r_r;
        QPointF s_r_r_top = p_r_tr + 0.14 * (p_f_tr - p_r_tr) + (h_beam + 0.12) * v_down_r_r;
        QPointF s_r_r_top_g = p_r_tr + 0.28 * (p_f_tr - p_r_tr) + h_beam * v_down_r_r;

        overlayCadData_.segments.push_back({s_r_f_top_g, s_r_f_top, colCyan, 2.0f});
        overlayCadData_.segments.push_back({s_r_f_top, s_r_f_bot, colCyan, 2.0f});
        overlayCadData_.segments.push_back({s_r_f_bot, s_r_r_bot, colCyan, 2.0f});
        overlayCadData_.segments.push_back({s_r_r_bot, s_r_r_top, colCyan, 2.0f});
        overlayCadData_.segments.push_back({s_r_r_top, s_r_r_top_g, colCyan, 2.0f});
        overlayCadData_.segments.push_back({p_f_tr_inner_x + 0.68 * v_down_f_r, p_r_tr + 0.14 * (p_f_tr - p_r_tr) + 0.68 * v_down_r_r, colCyan, 2.0f});

        // 1j. Center 3D Coordinate Axes
        QPointF p_center = (p_f_tl + p_f_tr + p_r_tl + p_r_tr + p_f_bl + p_f_br + p_r_bl + p_r_br) / 8.0;
        overlayCadData_.centerAxis = p_center;
        overlayCadData_.axisX = p_center + QPointF(55, 0);
        overlayCadData_.axisZ = p_center - QPointF(0, 55);
        overlayCadData_.axisY = p_center + QPointF(0, 55);
        overlayCadData_.hasAxes = true;
    }
}

void CameraStreamWidget::setOverlayDetection(const QRectF& bbox, const QString& label, const QString& poseText) {
    setOverlayDetection(bbox, {}, label, poseText);
}

void CameraStreamWidget::clearOverlayDetection() {
    std::lock_guard<std::mutex> lock(overlayMutex_);
    hasOverlay_ = false;
    overlayCorners3DProj_.clear();
    overlayCadData_ = CadOverlayData();
}

void CameraStreamWidget::setStaticImageDetection(const QImage& rawColorImg,
                                                 const QImage& annotatedImg,
                                                 const QRectF& bbox,
                                                 const QString& label,
                                                 const QString& statusText) {
    if (detectionBox_) detectionBox_->show();
    if (colorBox_) colorBox_->show();

    if (detectionViewLabel_ && !annotatedImg.isNull()) {
        detectionViewLabel_->setPixmap(QPixmap::fromImage(annotatedImg).scaled(
            detectionViewLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    if (colorViewLabel_ && !rawColorImg.isNull()) {
        colorViewLabel_->setPixmap(QPixmap::fromImage(rawColorImg).scaled(
            colorViewLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    if (statusLabel_) {
        statusLabel_->setText(QString("🎯 %1 | %2").arg(label, statusText));
    }

    {
        std::lock_guard<std::mutex> lock(overlayMutex_);
        hasOverlay_ = true;
        overlayBbox_ = bbox;
        overlayLabel_ = label;
        overlayPoseText_ = statusText;
        overlayTimestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
}

void CameraStreamWidget::displayStaticRGBDFrame(const QImage& colorImg, const QImage& depthImg, float centerDist) {
    if (detectionBox_) detectionBox_->show();
    if (colorBox_) colorBox_->show();
    if (depthBox_) depthBox_->show();
    onFrameReady(colorImg, depthImg, centerDist);
}

void CameraStreamWidget::onViewModeChanged(int index) {
    if (index == 0) {
        // 3 Khung Đồng Thời
        if (detectionBox_) detectionBox_->show();
        if (colorBox_) colorBox_->show();
        if (depthBox_) depthBox_->show();
    } else if (index == 1) {
        // Phóng To Khung Nhận Diện AI
        if (detectionBox_) detectionBox_->show();
        if (colorBox_) colorBox_->hide();
        if (depthBox_) depthBox_->hide();
    } else if (index == 2) {
        // Chỉ Khung RGB & Độ Sâu Depth
        if (detectionBox_) detectionBox_->hide();
        if (colorBox_) colorBox_->show();
        if (depthBox_) depthBox_->show();
    }
}

void CameraStreamWidget::onFrameReady(const QImage& colorImg, const QImage& depthImg, float centerDist) {
    QImage displayImg = colorImg.copy();
    {
        std::lock_guard<std::mutex> lock(overlayMutex_);
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        bool activeOverlay = hasOverlay_ && (now_ms - overlayTimestamp_ < 30000);

        QPainter p(&displayImg);
        p.setRenderHint(QPainter::Antialiasing);

        if (activeOverlay) {
            auto drawChamferedBoxQt = [&](const QRectF& r, qreal c_sz, const QColor& bg, const QColor& border) {
                QPolygonF poly;
                poly << QPointF(r.left() + c_sz, r.top())
                     << QPointF(r.right(), r.top())
                     << QPointF(r.right(), r.bottom() - c_sz)
                     << QPointF(r.right() - c_sz, r.bottom())
                     << QPointF(r.left(), r.bottom())
                     << QPointF(r.left(), r.top() + c_sz);
                p.setBrush(bg);
                p.setPen(QPen(border, 1.0));
                p.drawPolygon(poly);
            };

            if (overlayCadData_.valid) {
                const qreal s = std::clamp(static_cast<qreal>(displayImg.width()) / 1024.0, 0.5, 2.0);

                // 1. Shaded Tabletop (Top Surface Polygon)
                if (!overlayCadData_.topSurfacePolygon.isEmpty()) {
                    p.setBrush(QColor(0, 229, 255, 65)); // 25% tint Cyan
                    p.setPen(QPen(QColor(0, 229, 255), std::max(1.0, 2.0 * s)));
                    p.drawPolygon(overlayCadData_.topSurfacePolygon);
                }

                // 2. Floor Footprint Polygon & Feet
                if (!overlayCadData_.footPoints.empty()) {
                    QPolygonF footPoly;
                    for (const auto& fp : overlayCadData_.footPoints) footPoly << fp;
                    p.setBrush(Qt::NoBrush);
                    p.setPen(QPen(QColor(0, 230, 118), std::max(1.0, 2.0 * s)));
                    p.drawPolygon(footPoly);

                    qreal rx = std::max(4.0, 11.0 * s);
                    qreal ry = std::max(2.0, 4.0 * s);
                    for (const auto& fp : overlayCadData_.footPoints) {
                        p.drawEllipse(fp, rx, ry);
                    }
                }

                // 3. Projected CAD Segments (Direct Geometry)
                for (const auto& seg : overlayCadData_.segments) {
                    p.setPen(QPen(seg.color, seg.thickness));
                    p.drawLine(seg.p0, seg.p1);
                }

                // 4. Center 3D Coordinate Axes (X: Red depth, Y: Green length, Z: Blue up)
                if (overlayCadData_.hasAxes) {
                    QPointF c = overlayCadData_.centerAxis;
                    qreal r_c = std::max(5.0, 8.0 * s);
                    p.setPen(QPen(QColor(10, 10, 10), 1));
                    p.setBrush(QColor(0, 230, 118));
                    p.drawEllipse(c, r_c, r_c);
                    p.setPen(QColor(0, 0, 0));
                    p.setFont(QFont("Arial", std::max(6, (int)(8 * s)), QFont::Bold));
                    p.drawText(QRectF(c.x() - r_c, c.y() - r_c, 2 * r_c, 2 * r_c), Qt::AlignCenter, "Y");

                    // X: Red arrow pointing right
                    p.setPen(QPen(QColor(255, 0, 0), std::max(1.5, 2.0 * s)));
                    p.drawLine(c, overlayCadData_.axisX);
                    p.setFont(QFont("Arial", std::max(7, (int)(9 * s)), QFont::Bold));
                    p.setPen(QColor(255, 0, 0));
                    p.drawText(QRectF(overlayCadData_.axisX.x() + 4 * s, overlayCadData_.axisX.y() - 8 * s, 18 * s, 18 * s), Qt::AlignCenter, "X");

                    // Z: Blue arrow pointing up
                    p.setPen(QPen(QColor(0, 140, 255), std::max(1.5, 2.0 * s)));
                    p.drawLine(c, overlayCadData_.axisZ);
                    p.setPen(QColor(0, 140, 255));
                    p.drawText(QRectF(overlayCadData_.axisZ.x() - 9 * s, overlayCadData_.axisZ.y() - 18 * s, 18 * s, 18 * s), Qt::AlignCenter, "Z");
                }

                // 5. Top-Left Status Tag: [OK] CAD MODEL DETECTED (6DoF)
                QString statusText = overlayCadData_.label.isEmpty() ? overlayLabel_ : overlayCadData_.label;
                if (statusText.isEmpty()) {
                    statusText = "[OK] CAD MODEL DETECTED (6DoF)";
                }
                qreal tag_x = 14.0 * s, tag_y = 14.0 * s;
                qreal tag_w = 260.0 * s, tag_h = 30.0 * s;
                QRectF statusTag(tag_x, tag_y, tag_w, tag_h);
                drawChamferedBoxQt(statusTag, std::max(4.0, 6.0 * s), QColor(20, 20, 20, 220), QColor(0, 230, 118));
                p.setFont(QFont("Arial", std::max(7, (int)(9 * s)), QFont::Bold));
                p.setPen(QColor(0, 230, 118));
                p.drawText(statusTag, Qt::AlignCenter, statusText);

                // 6. Top-Center Header Tag & Pointer Line
                QString headerText = overlayCadData_.headerBanner;
                if (headerText.isEmpty()) {
                    headerText = "FRONT-FACE ANCHORED (830x505mm) -> EXTRUDE 440mm";
                }
                qreal hdr_w = 480.0 * s, hdr_h = 30.0 * s;
                qreal hdr_x = std::clamp(310.0 * s, tag_x + tag_w + 10.0, displayImg.width() - hdr_w - 10.0);
                qreal hdr_y = 36.0 * s;
                QRectF tagTop(hdr_x, hdr_y, hdr_w, hdr_h);
                drawChamferedBoxQt(tagTop, std::max(4.0, 6.0 * s), QColor(20, 20, 20, 220), QColor(0, 230, 118));
                p.setFont(QFont("Monospace", std::max(7, (int)(8 * s)), QFont::Bold));
                p.setPen(QColor(200, 255, 200));
                p.drawText(tagTop, Qt::AlignCenter, headerText);

                // Pointer line down to tabletop center
                if (overlayCadData_.hasTableCenter) {
                    p.setPen(QPen(QColor(0, 230, 118), 1.0));
                    p.drawLine(QPointF(hdr_x + hdr_w / 2.0, hdr_y + hdr_h), overlayCadData_.tableCenter);
                }

                // 7. Bottom Telemetry Banner
                QString banner = overlayCadData_.poseBanner;
                if (banner.isEmpty()) {
                    banner = overlayPoseText_;
                }
                if (banner.isEmpty()) {
                    banner = "CAD MODEL TABLE-DEMO (830x440x505mm)";
                }
                qreal bot_w = 460.0 * s, bot_h = 30.0 * s;
                qreal bot_x = std::clamp((displayImg.width() - bot_w) / 2.0, 10.0, displayImg.width() - bot_w - 10.0);
                qreal bot_y = displayImg.height() - 44.0 * s;
                QRectF tagBot(bot_x, bot_y, bot_w, bot_h);
                drawChamferedBoxQt(tagBot, std::max(3.0, 4.0 * s), QColor(20, 20, 20, 220), QColor(80, 80, 80));
                p.setPen(QColor(230, 230, 230));
                p.setFont(QFont("Monospace", std::max(7, (int)(9 * s)), QFont::Bold));
                p.drawText(tagBot, Qt::AlignCenter, banner);
            } else {
                // Fallback: 2D Neon Bounding Box
                QPen pen(QColor(0, 230, 118), 3);
                p.setPen(pen);
                p.setBrush(Qt::NoBrush);
                p.drawRoundedRect(overlayBbox_, 4, 4);

                QRectF statusTag(14, 14, 195, 24);
                p.fillRect(statusTag, QColor(20, 20, 20, 220));
                p.setPen(QPen(QColor(0, 230, 118), 1));
                p.drawRect(statusTag);
                p.setFont(QFont("Arial", 8, QFont::Bold));
                p.drawText(statusTag, Qt::AlignCenter, "[OK] CAD MODEL DETECTED (2D)");
            }
        } else {
            // Idle Scanning HUD
            QRectF statusTag(12, 12, 220, 24);
            p.fillRect(statusTag, QColor(0, 0, 0, 190));
            p.setPen(QColor(255, 183, 77)); // Amber
            p.setFont(QFont("Arial", 9, QFont::Bold));
            p.drawText(statusTag.adjusted(8, 0, 0, 0), Qt::AlignVCenter | Qt::AlignLeft, "◌ ĐANG QUÉT TÌM VẬT THỂ...");

            // Subtle center crosshair
            QPointF c(displayImg.width() / 2.0, displayImg.height() / 2.0);
            p.setPen(QPen(QColor(255, 255, 255, 100), 1, Qt::DashLine));
            p.drawEllipse(c, 24, 24);
            p.drawLine(QPointF(c.x() - 36, c.y()), QPointF(c.x() + 36, c.y()));
            p.drawLine(QPointF(c.x(), c.y() - 36), QPointF(c.x(), c.y() + 36));
        }
    }

    if (detectionViewLabel_ && detectionViewLabel_->isVisible()) {
        detectionViewLabel_->setPixmap(QPixmap::fromImage(displayImg).scaled(
            detectionViewLabel_->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
    }
    if (colorViewLabel_ && colorViewLabel_->isVisible()) {
        colorViewLabel_->setPixmap(QPixmap::fromImage(colorImg).scaled(
            colorViewLabel_->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
    }
    if (depthViewLabel_ && depthViewLabel_->isVisible()) {
        depthViewLabel_->setPixmap(QPixmap::fromImage(depthImg).scaled(
            depthViewLabel_->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
    }

    centerDistLabel_->setText(QString("Tâm ảnh: %1 m").arg(centerDist, 0, 'f', 2));

    frameCount_++;
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (now_ms - lastFpsTime_ >= 1000) {
        lastFps_ = frameCount_ * 1000.0 / (now_ms - lastFpsTime_);
        frameCount_ = 0;
        lastFpsTime_ = now_ms;
        if (isRecording_) {
            recordBtn_->setText(QString("⏹️ Dừng Ghi (.bag) - %1 FPS").arg(lastFps_, 0, 'f', 1));
        }
    }
}

bool CameraStreamWidget::getLatestRGBDFrame(RGBDFrame& outFrame, CameraIntrinsics& outIntr) {
    rs2::frameset raw_fs;
    {
        std::lock_guard<std::mutex> lock(impl_->frame_mutex);
        raw_fs = impl_->last_frameset;
    }
    if (!raw_fs) return false;

    try {
        if (!impl_->align) {
            impl_->align = std::make_unique<rs2::align>(RS2_STREAM_COLOR);
        }
        rs2::frameset aligned_fs = impl_->align->process(raw_fs);
        rs2::video_frame color_frame = aligned_fs.get_color_frame();
        rs2::depth_frame depth_frame = aligned_fs.get_depth_frame();
        if (!color_frame || !depth_frame) return false;

        int w = color_frame.get_width();
        int h = color_frame.get_height();

        auto colorO3D = std::make_shared<open3d::geometry::Image>();
        colorO3D->Prepare(w, h, 3, 1);
        std::memcpy(colorO3D->data_.data(), color_frame.get_data(), colorO3D->data_.size());

        auto depthO3D = std::make_shared<open3d::geometry::Image>();
        depthO3D->Prepare(w, h, 1, 2);
        std::memcpy(depthO3D->data_.data(), depth_frame.get_data(), depthO3D->data_.size());

        outFrame.color = colorO3D;
        outFrame.depth = depthO3D;
        outFrame.depthScale = 1.0 / impl_->depth_scale;
        outFrame.depthTruncMeters = 3.0;
        outFrame.timestampUs = static_cast<int64_t>(aligned_fs.get_timestamp() * 1000.0);

        outIntr = impl_->intrinsics;
        return true;
    } catch (...) {
        return false;
    }
}

void CameraStreamWidget::toggleRecord() {
    if (isPlayback_) {
        stopStream();
        statusLabel_->setText("Đã dừng phát lại file .bag.");
        return;
    }
    recordBtn_->setEnabled(false);
    if (!isRecording_) {
        QString outDir = outputDirEdit_->text().trimmed();
        QDir().mkpath(outDir);

        QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
        QString bagFile = QString("%1/d435if_record_%2.bag").arg(outDir).arg(timestamp);
        currentBagPath_ = bagFile.toStdString();

        statusLabel_->setText("⏳ Đang chuẩn bị luồng ghi file .bag...");
        QApplication::processEvents();

        if (startStream(currentBagPath_)) {
            isRecording_ = true;
            recordBtn_->setText("⏹️ Dừng Ghi (.bag)");
            recordBtn_->setStyleSheet("background: #2e7d32; color: white; font-weight: bold; font-size: 12px; border-radius: 4px;");
            snapshotBtn_->setEnabled(false);
            emit recordingStarted(QString::fromStdString(currentBagPath_));
        } else {
            statusLabel_->setText("<font color='#f44336'><b>Lỗi: Không thể bắt đầu ghi .bag!</b></font>");
            QMessageBox::warning(this, "Lỗi Ghi Hình", "Không thể bắt đầu ghi ra file .bag:\n" + bagFile);
        }
    } else {
        statusLabel_->setText("⏳ Đang hoàn tất lưu file .bag...");
        QApplication::processEvents();

        isRecording_ = false;
        startStream(""); // restart in preview mode
        recordBtn_->setText("🔴 Bắt Đầu Ghi Video (.bag)");
        recordBtn_->setStyleSheet("background: #d32f2f; color: white; font-weight: bold; font-size: 12px; border-radius: 4px;");
        snapshotBtn_->setEnabled(true);

        QFileInfo fi(QString::fromStdString(currentBagPath_));
        double sizeMb = fi.size() / (1024.0 * 1024.0);
        QMessageBox::information(this, "Ghi Hình Hoàn Tất",
            QString("✅ Đã lưu video .bag thành công!\n\nTập tin: %1\nDung lượng: %2 MB")
            .arg(QString::fromStdString(currentBagPath_))
            .arg(sizeMb, 0, 'f', 1));

        emit recordingStopped(QString::fromStdString(currentBagPath_));
    }
    recordBtn_->setEnabled(true);
}

void CameraStreamWidget::takeSnapshot() {
    RGBDFrame frame;
    CameraIntrinsics intr;
    if (!getLatestRGBDFrame(frame, intr)) {
        QMessageBox::warning(this, "Chụp ảnh", "Chưa nhận được khung hình hợp lệ từ camera!");
        return;
    }

    QString outDir = outputDirEdit_->text().trimmed();
    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
    QString snapDir = QString("%1/snapshot_%2").arg(outDir).arg(timestamp);
    QDir().mkpath(snapDir);

    int w = intr.width;
    int h = intr.height;

    // 1. Lưu color.png
    QImage colorImg(frame.color->data_.data(), w, h, w * 3, QImage::Format_RGB888);
    QString colorPath = snapDir + "/color.png";
    colorImg.save(colorPath, "PNG");

    // 2. Lưu depth.png 16-bit
    QString depthPath = snapDir + "/depth.png";
    open3d::io::WriteImage(depthPath.toStdString(), *frame.depth);

    // 3. Lưu intrinsics.json
    QString intrPath = snapDir + "/intrinsics.json";
    std::ofstream fIntr(intrPath.toStdString());
    if (fIntr.is_open()) {
        fIntr << "{\n";
        fIntr << "  \"width\": " << w << ",\n";
        fIntr << "  \"height\": " << h << ",\n";
        fIntr << "  \"fx\": " << intr.fx << ",\n";
        fIntr << "  \"fy\": " << intr.fy << ",\n";
        fIntr << "  \"cx\": " << intr.cx << ",\n";
        fIntr << "  \"cy\": " << intr.cy << ",\n";
        fIntr << "  \"depth_scale_mm\": 1.0\n";
        fIntr << "}\n";
        fIntr.close();
    }

    snapshotCount_++;
    statusLabel_->setText(QString("<font color='#00e676'><b>📸 Đã chụp snapshot #%1 vào: %2</b></font>")
        .arg(snapshotCount_)
        .arg(QFileInfo(snapDir).fileName()));

    emit snapshotSaved(snapDir);
}

void CameraStreamWidget::onBrowseOutputDir() {
    QString dir = QFileDialog::getExistingDirectory(this, "Chọn thư mục lưu trữ dữ liệu RealSense", outputDirEdit_->text());
    if (!dir.isEmpty()) {
        outputDirEdit_->setText(dir);
    }
}

void CameraStreamWidget::setOutputDir(const QString& dir) {
    if (outputDirEdit_) outputDirEdit_->setText(dir);
}

QString CameraStreamWidget::outputDir() const {
    return outputDirEdit_ ? outputDirEdit_->text().trimmed() : "";
}
