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
#include <QPainterPath>
#include <QFont>
#include <QEvent>
#include <QMouseEvent>
#include <QResizeEvent>
#include <algorithm>
#include <cmath>
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
    detectionBox_ = new QGroupBox("🎯 Nhận Diện Đối Tượng (AI Detection && 6DoF HUD)", videoContainer);
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
    detectionViewLabel_->setMouseTracking(true);
    detectionViewLabel_->installEventFilter(this);
    detectionViewLabel_->setToolTip("Kéo chuột trái để khoanh vùng 2D ROI • Chuột phải / Nhấp đúp để xóa ROI");
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
    colorViewLabel_->setMouseTracking(true);
    colorViewLabel_->installEventFilter(this);
    colorViewLabel_->setToolTip("Kéo chuột trái để khoanh vùng 2D ROI • Chuột phải / Nhấp đúp để xóa ROI");
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
    data.isPhysicalStlMode = true;
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
        double axisLenY = std::max(0.025, (cadMax.y() - cadMin.y()) * 0.40);
        double axisLenX = std::max(0.025, (cadMax.x() - cadMin.x()) * 0.40);
        Eigen::Vector3d pt_z = pt_c + Eigen::Vector3d(0.0, 0.0, axisLenZ);
        Eigen::Vector3d pt_y = pt_c + Eigen::Vector3d(0.0, axisLenY, 0.0);
        Eigen::Vector3d pt_x = pt_c + Eigen::Vector3d(axisLenX, 0.0, 0.0);
        QPointF p_center = proj_f(pt_c.x(), pt_c.y(), pt_c.z());
        QPointF p_arrow  = proj_f(pt_z.x(), pt_z.y(), pt_z.z());
        data.centerAxis = p_center;
        data.axisZ = p_arrow;
        data.axisY = proj_f(pt_y.x(), pt_y.y(), pt_y.z());
        data.axisX = proj_f(pt_x.x(), pt_x.y(), pt_x.z());
        data.hasAxes = true;
        data.tableCenter = p_center;
        data.hasTableCenter = true;

        qreal min_x = std::min({c0.x(), c1.x(), c2.x(), c3.x(), c4.x(), c5.x(), c6.x(), c7.x()});
        qreal max_x = std::max({c0.x(), c1.x(), c2.x(), c3.x(), c4.x(), c5.x(), c6.x(), c7.x()});
        qreal min_y = std::min({c0.y(), c1.y(), c2.y(), c3.y(), c4.y(), c5.y(), c6.y(), c7.y()});
        qreal max_y = std::max({c0.y(), c1.y(), c2.y(), c3.y(), c4.y(), c5.y(), c6.y(), c7.y()});
        data.bbox = QRectF(min_x, min_y, std::max(10.0, max_x - min_x), std::max(10.0, max_y - min_y));

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

    // 8-corner outer OBB (controlled by Show 3D Box checkbox)
    {
        QPointF b0 = proj_f(cad_x0, cad_y0, z_floor), b1 = proj_f(cad_x1, cad_y0, z_floor);
        QPointF b2 = proj_f(cad_x1, cad_y1, z_floor), b3 = proj_f(cad_x0, cad_y1, z_floor);
        QPointF t0 = proj_f(cad_x0, cad_y0, z_top),   t1 = proj_f(cad_x1, cad_y0, z_top);
        QPointF t2 = proj_f(cad_x1, cad_y1, z_top),   t3 = proj_f(cad_x0, cad_y1, z_top);
        const QColor obbCol(0, 230, 118, 170);
        data.segments.push_back({b0, b1, obbCol, th_thin});
        data.segments.push_back({b1, b2, obbCol, th_thin});
        data.segments.push_back({b2, b3, obbCol, th_thin});
        data.segments.push_back({b3, b0, obbCol, th_thin});
        data.segments.push_back({t0, t1, obbCol, th_thin});
        data.segments.push_back({t1, t2, obbCol, th_thin});
        data.segments.push_back({t2, t3, obbCol, th_thin});
        data.segments.push_back({t3, t0, obbCol, th_thin});
        data.segments.push_back({b0, t0, obbCol, th_thin});
        data.segments.push_back({b1, t1, obbCol, th_thin});
        data.segments.push_back({b2, t2, obbCol, th_thin});
        data.segments.push_back({b3, t3, obbCol, th_thin});
    }

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
    data.wireframeSegments.push_back({p_f_tl, p_f_tr, colYellow, th_thick});

    // Rear Top Beam
    QPointF p_r_tl_b = proj_f(x_rear_inner, y_left_inner, z_top_inner);
    QPointF p_r_tr_b = proj_f(x_rear_inner, y_right_inner, z_top_inner);
    data.wireframeSegments.push_back({p_r_tl, p_r_tr, colGreen, th_med});
    data.wireframeSegments.push_back({p_r_tl_b, p_r_tr_b, colGreen, th_thin});

    // Side Top Beams & Inner Portal Arch under Tabletop
    QPointF rl_top_fo = proj_f(x_rear_inner, y_left,  z_top_inner);
    QPointF rr_top_fo = proj_f(x_rear_inner, y_right, z_top_inner);
    data.wireframeSegments.push_back({p_f_tl_b, rl_top_fo, colYellow, th_thin});
    data.wireframeSegments.push_back({p_f_tr_b, rr_top_fo, colGreen,  th_thin});
    data.wireframeSegments.push_back({proj_f(x_front_inner, y_left_inner, z_top_inner),
                             proj_f(x_rear_inner,  y_left_inner, z_top_inner), colGreen, th_thin});
    data.wireframeSegments.push_back({proj_f(x_front_inner, y_right_inner, z_top_inner),
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
    data.wireframeSegments.push_back({fl_top_fo, fl_bot_fo, colYellow, th_thick});
    data.wireframeSegments.push_back({fl_top_ts, fl_bot_ts, colGreen,  th_thin});
    data.wireframeSegments.push_back({fl_top_fi, fl_bot_fi, colYellow, th_med});
    data.wireframeSegments.push_back({fl_top_bi, fl_bot_bi, colGreen,  th_thin});
    data.wireframeSegments.push_back({fl_bot_fo, fl_bot_fi, colYellow, th_med});
    data.wireframeSegments.push_back({fl_bot_fi, fl_bot_bi, colYellow, th_thin});
    data.wireframeSegments.push_back({fl_bot_ts, p_f_bl,    colYellow, th_med});

    // Front-Right Column (Cyan)
    QPointF fr_top_fo = p_f_tr;
    QPointF fr_top_ts = proj_f(x_front,       y_r_c,         z_top);
    QPointF fr_top_fi = proj_f(x_front,       y_right_inner, z_top);
    QPointF fr_top_bi = proj_f(x_front_inner, y_right_inner, z_top_inner);
    QPointF fr_bot_fo = proj_f(x_front,       y_right,       z_leg_bot);
    QPointF fr_bot_ts = proj_f(x_front,       y_r_c,         z_leg_bot);
    QPointF fr_bot_fi = proj_f(x_front,       y_right_inner, z_leg_bot);
    QPointF fr_bot_bi = proj_f(x_front_inner, y_right_inner, z_leg_bot);
    data.wireframeSegments.push_back({fr_top_fo, fr_bot_fo, colCyan,  th_thick});
    data.wireframeSegments.push_back({fr_top_ts, fr_bot_ts, colCyan,  th_thin});
    data.wireframeSegments.push_back({fr_top_fi, fr_bot_fi, colCyan,  th_med});
    data.wireframeSegments.push_back({fr_top_bi, fr_bot_bi, colGreen, th_thin});
    data.wireframeSegments.push_back({fr_bot_fo, fr_bot_fi, colCyan,  th_med});
    data.wireframeSegments.push_back({fr_bot_fi, fr_bot_bi, colCyan,  th_thin});
    data.wireframeSegments.push_back({fr_bot_ts, p_f_br,    colCyan,  th_med});

    // Rear-Left Column (Green)
    QPointF rl_top_ts = proj_f(x_rear_inner, y_l_c,        z_top_inner);
    QPointF rl_top_fi = proj_f(x_rear_inner, y_left_inner, z_top_inner);
    QPointF rl_top_bi = proj_f(x_rear,       y_left_inner, z_top_inner);
    QPointF rl_bot_fo = proj_f(x_rear_inner, y_left,       z_leg_bot);
    QPointF rl_bot_ts = proj_f(x_rear_inner, y_l_c,        z_leg_bot);
    QPointF rl_bot_fi = proj_f(x_rear_inner, y_left_inner, z_leg_bot);
    QPointF rl_bot_bi = proj_f(x_rear,       y_left_inner, z_leg_bot);
    data.wireframeSegments.push_back({rl_top_fo, rl_bot_fo, colGreen, th_med});
    data.wireframeSegments.push_back({rl_top_ts, rl_bot_ts, colGreen, th_thin});
    data.wireframeSegments.push_back({rl_top_bi, rl_bot_bi, colGreen, th_thin});
    data.wireframeSegments.push_back({rl_bot_fo, rl_bot_fi, colGreen, th_med});
    data.wireframeSegments.push_back({rl_bot_fi, rl_bot_bi, colGreen, th_thin});
    data.wireframeSegments.push_back({rl_bot_ts, p_r_bl,    colGreen, th_thin});

    // Rear-Right Column (Green)
    QPointF rr_top_ts = proj_f(x_rear_inner, y_r_c,         z_top_inner);
    QPointF rr_top_fi = proj_f(x_rear_inner, y_right_inner, z_top_inner);
    QPointF rr_top_bi = proj_f(x_rear,       y_right_inner, z_top_inner);
    QPointF rr_bot_fo = proj_f(x_rear_inner, y_right,       z_leg_bot);
    QPointF rr_bot_ts = proj_f(x_rear_inner, y_r_c,         z_leg_bot);
    QPointF rr_bot_fi = proj_f(x_rear_inner, y_right_inner, z_leg_bot);
    QPointF rr_bot_bi = proj_f(x_rear,       y_right_inner, z_leg_bot);
    data.wireframeSegments.push_back({rr_top_fo, rr_bot_fo, colGreen, th_med});
    data.wireframeSegments.push_back({rr_top_ts, rr_bot_ts, colGreen, th_thin});
    data.wireframeSegments.push_back({rr_top_bi, rr_bot_bi, colGreen, th_thin});
    data.wireframeSegments.push_back({rr_bot_fo, rr_bot_fi, colGreen, th_med});
    data.wireframeSegments.push_back({rr_bot_fi, rr_bot_bi, colGreen, th_thin});
    data.wireframeSegments.push_back({rr_bot_ts, p_r_br,    colGreen, th_thin});

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

void CameraStreamWidget::setUniversalPoseOverlays(const std::vector<tmcad::ObjectPose>& poses,
                                                  const tmcad::UniversalPoseEstimator& estimator,
                                                  const CameraIntrinsics& intr,
                                                  const QString& telemetryBanner,
                                                  const QString& headerBanner) {
    if (poses.empty()) {
        clearOverlayDetection();
        return;
    }

    const double w_px = (intr.width > 0) ? intr.width : 640.0;
    const double h_px = (intr.height > 0) ? intr.height : 480.0;
    const double fx = (intr.fx > 100.0) ? intr.fx : 0.95 * w_px;
    const double fy = (intr.fy > 100.0) ? intr.fy : fx;
    const double cx = (intr.cx > 10.0) ? intr.cx : ((intr.ppx > 10.0) ? intr.ppx : 0.5 * w_px);
    const double cy = (intr.cy > 10.0) ? intr.cy : ((intr.ppy > 10.0) ? intr.ppy : 0.5 * h_px);

    const double s = std::clamp(w_px / 1024.0, 0.5, 2.0);
    const float th_thick = std::max(1.2f, static_cast<float>(std::round(2.8 * s)));
    const float th_med   = std::max(1.0f, static_cast<float>(std::round(2.0 * s)));
    const float th_thin  = std::max(1.0f, static_cast<float>(std::round(1.3 * s)));

    const cv::Vec3d cmin = estimator.cadMin();
    const cv::Vec3d cmax = estimator.cadMax();
    const std::vector<tmcad::FeatureEdge3D>& edges3d = estimator.cadFeatureEdges();
    const bool is_table_fallback = edges3d.empty() &&
        QString::fromStdString(estimator.modelName()).contains("TABLE", Qt::CaseInsensitive);

    const QColor palette[4] = {
        QColor(0, 230, 118),   // Instance #1 Green
        QColor(255, 145, 0),   // Instance #2 Orange
        QColor(224, 64, 251),  // Instance #3 Magenta
        QColor(255, 238, 0)    // Instance #4 Yellow
    };

    std::vector<CadOverlayData> overlayList;
    overlayList.reserve(poses.size());

    for (size_t idx = 0; idx < poses.size(); ++idx) {
        const auto& pose = poses[idx];
        if (!pose.valid) continue;

        CadOverlayData d;
        d.valid = true;
        d.isPhysicalStlMode = true;
        const QColor instColor = palette[idx % 4];
        const QColor topColor(255, 238, 0);
        const QColor sideColor(0, 229, 255);

        auto proj_f = [&](double x, double y, double z) -> QPointF {
            cv::Vec3d p = pose.R * cv::Vec3d(x, y, z) + pose.t;
            double pz = std::max(0.05, p[2]);
            return QPointF(fx * p[0] / pz + cx, fy * p[1] / pz + cy);
        };

        // 1. 3D Oriented Bounding Box (8 corners + mid-plane slice) -> d.segments
        const double x0 = cmin[0], x1 = cmax[0];
        const double y0 = cmin[1], y1 = cmax[1];
        const double z0 = cmin[2], z1 = cmax[2];

        QPointF c0 = proj_f(x0, y0, z0), c1 = proj_f(x1, y0, z0);
        QPointF c2 = proj_f(x1, y1, z0), c3 = proj_f(x0, y1, z0);
        QPointF c4 = proj_f(x0, y0, z1), c5 = proj_f(x1, y0, z1);
        QPointF c6 = proj_f(x1, y1, z1), c7 = proj_f(x0, y1, z1);

        d.segments.push_back({c0, c1, instColor, th_med});
        d.segments.push_back({c1, c2, instColor, th_med});
        d.segments.push_back({c2, c3, instColor, th_med});
        d.segments.push_back({c3, c0, instColor, th_med});

        d.segments.push_back({c4, c5, topColor, th_thick});
        d.segments.push_back({c5, c6, topColor, th_thick});
        d.segments.push_back({c6, c7, topColor, th_thick});
        d.segments.push_back({c7, c4, topColor, th_thick});

        d.segments.push_back({c0, c4, sideColor, th_med});
        d.segments.push_back({c1, c5, sideColor, th_med});
        d.segments.push_back({c2, c6, sideColor, th_med});
        d.segments.push_back({c3, c7, sideColor, th_med});

        const double zm = 0.5 * (z0 + z1);
        QPointF m0 = proj_f(x0, y0, zm), m1 = proj_f(x1, y0, zm);
        QPointF m2 = proj_f(x1, y1, zm), m3 = proj_f(x0, y1, zm);
        const QColor midCol(instColor.red(), instColor.green(), instColor.blue(), 170);
        d.segments.push_back({m0, m1, midCol, th_thin});
        d.segments.push_back({m1, m2, midCol, th_thin});
        d.segments.push_back({m2, m3, midCol, th_thin});
        d.segments.push_back({m3, m0, midCol, th_thin});

        qreal min_x = std::min({c0.x(), c1.x(), c2.x(), c3.x(), c4.x(), c5.x(), c6.x(), c7.x()});
        qreal max_x = std::max({c0.x(), c1.x(), c2.x(), c3.x(), c4.x(), c5.x(), c6.x(), c7.x()});
        qreal min_y = std::min({c0.y(), c1.y(), c2.y(), c3.y(), c4.y(), c5.y(), c6.y(), c7.y()});
        qreal max_y = std::max({c0.y(), c1.y(), c2.y(), c3.y(), c4.y(), c5.y(), c6.y(), c7.y()});
        d.bbox = QRectF(min_x, min_y, std::max(10.0, max_x - min_x), std::max(10.0, max_y - min_y));

        const bool z1_is_top = (pose.R * cv::Vec3d(0, 0, z1) + pose.t)[1] <=
                               (pose.R * cv::Vec3d(0, 0, z0) + pose.t)[1];
        if (z1_is_top) {
            d.topSurfacePolygon << c4 << c5 << c6 << c7;
        } else {
            d.topSurfacePolygon << c0 << c1 << c2 << c3;
        }

        // 2. Embedded ONNX CAD Feature / Crease Wireframe -> d.wireframeSegments
        if (!edges3d.empty()) {
            const size_t max_edges = 900;
            const size_t edge_step = std::max<size_t>(1, edges3d.size() / max_edges);
            d.wireframeSegments.reserve(std::min(edges3d.size(), max_edges));
            const QColor wfCol = (poses.size() == 1) ? QColor(0, 229, 255) : instColor.lighter(120);
            for (size_t ei = 0; ei < edges3d.size(); ei += edge_step) {
                const auto& e = edges3d[ei];
                QPointF a = proj_f(e.p0[0], e.p0[1], e.p0[2]);
                QPointF b = proj_f(e.p1[0], e.p1[1], e.p1[2]);
                d.wireframeSegments.push_back({a, b, wfCol, th_thin});
            }
        } else if (is_table_fallback) {
            // Clean structural aluminum frame wireframe fallback if older TABLE ONNX lacks embedded crease edges
            const double lw = 0.040, bh = 0.040;
            QPointF i0 = proj_f(x0 + lw, y0 + lw, z1 - bh), i1 = proj_f(x1 - lw, y0 + lw, z1 - bh);
            QPointF i2 = proj_f(x1 - lw, y1 - lw, z1 - bh), i3 = proj_f(x0 + lw, y1 - lw, z1 - bh);
            QPointF l0 = proj_f(x0 + lw, y0 + lw, z0),      l1 = proj_f(x1 - lw, y0 + lw, z0);
            QPointF l2 = proj_f(x1 - lw, y1 - lw, z0),      l3 = proj_f(x0 + lw, y1 - lw, z0);
            d.wireframeSegments.push_back({i0, i1, sideColor, th_thin});
            d.wireframeSegments.push_back({i1, i2, sideColor, th_thin});
            d.wireframeSegments.push_back({i2, i3, sideColor, th_thin});
            d.wireframeSegments.push_back({i3, i0, sideColor, th_thin});
            d.wireframeSegments.push_back({i0, l0, sideColor, th_thin});
            d.wireframeSegments.push_back({i1, l1, sideColor, th_thin});
            d.wireframeSegments.push_back({i2, l2, sideColor, th_thin});
            d.wireframeSegments.push_back({i3, l3, sideColor, th_thin});
        } else if (!estimator.cadSurfacePts().empty()) {
            // Cross-section bounding ribs for smooth solids without crease edges
            double xm = 0.5 * (x0 + x1), ym = 0.5 * (y0 + y1);
            d.wireframeSegments.push_back({proj_f(xm, y0, z0), proj_f(xm, y1, z0), sideColor, th_thin});
            d.wireframeSegments.push_back({proj_f(xm, y1, z0), proj_f(xm, y1, z1), sideColor, th_thin});
            d.wireframeSegments.push_back({proj_f(xm, y1, z1), proj_f(xm, y0, z1), sideColor, th_thin});
            d.wireframeSegments.push_back({proj_f(xm, y0, z1), proj_f(xm, y0, z0), sideColor, th_thin});
            d.wireframeSegments.push_back({proj_f(x0, ym, z0), proj_f(x1, ym, z0), sideColor, th_thin});
            d.wireframeSegments.push_back({proj_f(x1, ym, z0), proj_f(x1, ym, z1), sideColor, th_thin});
            d.wireframeSegments.push_back({proj_f(x1, ym, z1), proj_f(x0, ym, z1), sideColor, th_thin});
            d.wireframeSegments.push_back({proj_f(x0, ym, z1), proj_f(x0, ym, z0), sideColor, th_thin});
        }

        // 3. Projected Segmented Foreground Point Cloud -> d.segmentedCloudPoints2D
        const std::vector<cv::Vec3f>& segPts = !pose.segmented_points.empty()
            ? pose.segmented_points
            : (idx == 0 ? estimator.lastSegmentedPoints() : std::vector<cv::Vec3f>{});
        if (!segPts.empty()) {
            const size_t max_pts = 480;
            const size_t p_step = std::max<size_t>(1, segPts.size() / max_pts);
            d.segmentedCloudPoints2D.reserve(std::min(segPts.size(), max_pts));
            for (size_t pi = 0; pi < segPts.size(); pi += p_step) {
                const auto& sp = segPts[pi];
                if (sp[2] <= 0.05f) continue;
                qreal u = fx * sp[0] / sp[2] + cx;
                qreal v = fy * sp[1] / sp[2] + cy;
                d.segmentedCloudPoints2D.emplace_back(u, v);
            }
        }

        // 4. Scale-Adaptive True 3D Axes
        const cv::Vec3d pt_c = estimator.isFloorStanding()
            ? cv::Vec3d(0.0, 0.0, 0.0)
            : (0.5 * (cmin + cmax));
        const double axisLen = std::clamp(estimator.realBoundingDiagM() * 0.16, 0.04, 0.16);
        d.centerAxis = proj_f(pt_c[0], pt_c[1], pt_c[2]);
        d.axisX = proj_f(pt_c[0] + axisLen, pt_c[1], pt_c[2]);
        d.axisY = proj_f(pt_c[0], pt_c[1] + axisLen, pt_c[2]);
        d.axisZ = proj_f(pt_c[0], pt_c[1], pt_c[2] + axisLen);
        d.hasAxes = true;
        d.tableCenter = d.centerAxis;
        d.hasTableCenter = (idx == 0);

        // 5. Labels & Banners
        const int fit_pct = qBound(0, static_cast<int>(pose.confidence * 100.0f), 100);
        d.label = QString("[OK] %1 • %2% | %3mm")
                      .arg(QString::fromStdString(pose.model_name))
                      .arg(fit_pct)
                      .arg(pose.rmse_m * 1000.0, 0, 'f', 1);

        if (idx == 0) {
            if (!headerBanner.isEmpty()) {
                d.headerBanner = headerBanner;
            } else {
                const double dx_mm = (cmax[0] - cmin[0]) * 1000.0;
                const double dy_mm = (cmax[1] - cmin[1]) * 1000.0;
                const double dz_mm = (cmax[2] - cmin[2]) * 1000.0;
                d.headerBanner = QString("%1 %2 (%3x%4x%5mm) • %6")
                    .arg(estimator.isFloorStanding() ? "FLOOR LOCKED" : "FREE 6DoF")
                    .arg(QString::fromStdString(estimator.modelName()))
                    .arg(dx_mm, 0, 'f', 0)
                    .arg(dy_mm, 0, 'f', 0)
                    .arg(dz_mm, 0, 'f', 0)
                    .arg(poses.size() > 1 ? QString("%1 inst").arg(poses.size()) : QString::fromStdString(pose.method));
            }
            if (!telemetryBanner.isEmpty()) {
                d.poseBanner = telemetryBanner;
            } else {
                d.poseBanner = QString("%1 | Z=%2m | R:%3° P:%4° Y:%5° | Fit:%6%")
                    .arg(QString::fromStdString(pose.model_name))
                    .arg(pose.t[2], 0, 'f', 3)
                    .arg(pose.roll_deg, 0, 'f', 1)
                    .arg(pose.pitch_deg, 0, 'f', 1)
                    .arg(pose.yaw_deg, 0, 'f', 1)
                    .arg(fit_pct);
            }
        }
        overlayList.push_back(std::move(d));
    }

    if (overlayList.empty()) {
        clearOverlayDetection();
        return;
    }

    {
        std::lock_guard<std::mutex> lock(overlayMutex_);
        hasOverlay_ = true;
        multiOverlayData_ = std::move(overlayList);
        overlayCadData_ = multiOverlayData_.front();
        overlayBbox_ = overlayCadData_.bbox;
        overlayLabel_ = overlayCadData_.label;
        overlayPoseText_ = overlayCadData_.poseBanner;
        overlayTimestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    redrawCurrentFrame();
}

void CameraStreamWidget::setOverlayRenderFlags(const tmcad::OverlayRenderFlags& flags) {
    {
        std::lock_guard<std::mutex> lock(overlayMutex_);
        renderFlags_ = flags;
    }
    redrawCurrentFrame();
}

bool CameraStreamWidget::hasActiveRoi() const noexcept {
    return hasActiveRoi_ && roiNorm_.width() > 0.03 && roiNorm_.height() > 0.03;
}

void CameraStreamWidget::clearInteractiveRoi() {
    const bool was_active = hasActiveRoi();
    hasActiveRoi_ = false;
    isDraggingRoi_ = false;
    roiNorm_ = QRectF();
    redrawCurrentFrame();
    if (was_active) {
        emit roiSelectionChanged(false);
    }
}

cv::Rect CameraStreamWidget::currentRoiHintImageSpace(int img_w, int img_h) const {
    if (!hasActiveRoi() || img_w <= 0 || img_h <= 0) return cv::Rect();
    const int x = std::clamp(static_cast<int>(std::round(roiNorm_.x() * img_w)), 0, std::max(0, img_w - 1));
    const int y = std::clamp(static_cast<int>(std::round(roiNorm_.y() * img_h)), 0, std::max(0, img_h - 1));
    const int w = std::clamp(static_cast<int>(std::round(roiNorm_.width() * img_w)), 1, img_w - x);
    const int h = std::clamp(static_cast<int>(std::round(roiNorm_.height() * img_h)), 1, img_h - y);
    return cv::Rect(x, y, w, h);
}

void CameraStreamWidget::redrawCurrentFrame() {
    if (!hasLastFrame_ || lastColorImg_.isNull()) return;
    onFrameReady(lastColorImg_, lastDepthImg_, lastCenterDist_);
}

bool CameraStreamWidget::eventFilter(QObject* watched, QEvent* event) {
    QLabel* targetLabel = nullptr;
    if (watched == detectionViewLabel_) targetLabel = detectionViewLabel_;
    else if (watched == colorViewLabel_) targetLabel = colorViewLabel_;

    if (targetLabel) {
        auto computePixmapRect = [&](const QLabel* lbl) -> QRectF {
            const QPixmap pm = lbl->pixmap(Qt::ReturnByValue);
            if (!pm.isNull()) {
                const qreal dpr = std::max<qreal>(1.0, pm.devicePixelRatioF());
                const qreal pw = static_cast<qreal>(pm.width()) / dpr;
                const qreal ph = static_cast<qreal>(pm.height()) / dpr;
                return QRectF((lbl->width() - pw) * 0.5, (lbl->height() - ph) * 0.5, pw, ph);
            }
            const int iw = lastColorImg_.isNull() ? 640 : lastColorImg_.width();
            const int ih = lastColorImg_.isNull() ? 480 : lastColorImg_.height();
            if (lbl->width() <= 0 || lbl->height() <= 0 || iw <= 0 || ih <= 0) return QRectF();
            const double scale = std::min(static_cast<double>(lbl->width()) / iw,
                                          static_cast<double>(lbl->height()) / ih);
            const double dw = iw * scale;
            const double dh = ih * scale;
            return QRectF((lbl->width() - dw) * 0.5, (lbl->height() - dh) * 0.5, dw, dh);
        };

        auto mapToNorm = [&](const QPointF& localPos) -> QPointF {
            const QRectF r = computePixmapRect(targetLabel);
            if (r.width() < 4.0 || r.height() < 4.0) {
                return QPointF(std::clamp(localPos.x() / std::max(1, targetLabel->width()), 0.0, 1.0),
                               std::clamp(localPos.y() / std::max(1, targetLabel->height()), 0.0, 1.0));
            }
            const double nx = std::clamp((localPos.x() - r.left()) / r.width(), 0.0, 1.0);
            const double ny = std::clamp((localPos.y() - r.top()) / r.height(), 0.0, 1.0);
            return QPointF(nx, ny);
        };

        if (event->type() == QEvent::MouseButtonDblClick) {
            clearInteractiveRoi();
            return true;
        }
        if (event->type() == QEvent::MouseButtonPress) {
            auto* me = static_cast<QMouseEvent*>(event);
            if (me->button() == Qt::RightButton) {
                clearInteractiveRoi();
                return true;
            }
            if (me->button() == Qt::LeftButton) {
                isDraggingRoi_ = true;
                roiDragStartNorm_ = mapToNorm(me->localPos());
                roiNorm_ = QRectF(roiDragStartNorm_, roiDragStartNorm_);
                redrawCurrentFrame();
                return true;
            }
        } else if (event->type() == QEvent::MouseMove) {
            auto* me = static_cast<QMouseEvent*>(event);
            if (isDraggingRoi_ && (me->buttons() & Qt::LeftButton)) {
                QPointF curNorm = mapToNorm(me->localPos());
                roiNorm_ = QRectF(roiDragStartNorm_, curNorm).normalized();
                redrawCurrentFrame();
                return true;
            }
        } else if (event->type() == QEvent::MouseButtonRelease) {
            auto* me = static_cast<QMouseEvent*>(event);
            if (isDraggingRoi_ && me->button() == Qt::LeftButton) {
                isDraggingRoi_ = false;
                QPointF endNorm = mapToNorm(me->localPos());
                roiNorm_ = QRectF(roiDragStartNorm_, endNorm).normalized();
                const bool valid_roi = (roiNorm_.width() > 0.03 && roiNorm_.height() > 0.03);
                hasActiveRoi_ = valid_roi;
                if (!valid_roi) roiNorm_ = QRectF();
                redrawCurrentFrame();
                emit roiSelectionChanged(hasActiveRoi_);
                return true;
            }
        }
    }
    return QWidget::eventFilter(watched, event);
}

void CameraStreamWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    if (hasLastFrame_ && !isStreaming()) {
        redrawCurrentFrame();
    }
}

void CameraStreamWidget::setCadOverlay(const CadOverlayData& data) {
    std::lock_guard<std::mutex> lock(overlayMutex_);
    hasOverlay_ = true;
    overlayCadData_ = data;
    overlayCadData_.valid = true;
    multiOverlayData_.clear();
    multiOverlayData_.push_back(overlayCadData_);
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
    multiOverlayData_.clear();
    if (overlayCadData_.valid) multiOverlayData_.push_back(overlayCadData_);
}

void CameraStreamWidget::setOverlayDetection(const QRectF& bbox, const QString& label, const QString& poseText) {
    setOverlayDetection(bbox, {}, label, poseText);
}

void CameraStreamWidget::clearOverlayDetection() {
    std::lock_guard<std::mutex> lock(overlayMutex_);
    hasOverlay_ = false;
    overlayCorners3DProj_.clear();
    overlayCadData_ = CadOverlayData();
    multiOverlayData_.clear();
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
    if (&colorImg != &lastColorImg_) lastColorImg_ = colorImg;
    if (&depthImg != &lastDepthImg_) lastDepthImg_ = depthImg;
    lastCenterDist_ = centerDist;
    hasLastFrame_ = !lastColorImg_.isNull();
    if (lastColorImg_.isNull()) return;

    QImage displayImg = lastColorImg_.copy();
    QImage colorRawImg = lastColorImg_;
    bool drawRoiOnColor = false;
    QRectF roiNormSnap;
    bool roiActiveSnap = false;
    bool roiDraggingSnap = false;

    {
        std::lock_guard<std::mutex> lock(overlayMutex_);
        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        const bool activeOverlay = hasOverlay_ && (!captureRunning_.load() || (now_ms - overlayTimestamp_ < 30000));
        const tmcad::OverlayRenderFlags flags = renderFlags_;
        roiNormSnap = roiNorm_;
        roiActiveSnap = hasActiveRoi_;
        roiDraggingSnap = isDraggingRoi_;

        QPainter p(&displayImg);
        p.setRenderHint(QPainter::Antialiasing);
        const qreal s = std::clamp(static_cast<qreal>(displayImg.width()) / 1024.0, 0.5, 2.0);

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

        if (activeOverlay) {
            std::vector<CadOverlayData> itemsToDraw;
            if (!multiOverlayData_.empty()) {
                itemsToDraw = multiOverlayData_;
            } else if (overlayCadData_.valid) {
                itemsToDraw.push_back(overlayCadData_);
            }

            if (!itemsToDraw.empty()) {
                for (size_t itemIdx = 0; itemIdx < itemsToDraw.size(); ++itemIdx) {
                    const auto& item = itemsToDraw[itemIdx];

                    // A. Segmented Foreground Point Cloud Overlay
                    if (flags.show_segmented_cloud && !item.segmentedCloudPoints2D.empty()) {
                        p.setPen(Qt::NoPen);
                        p.setBrush(QColor(0, 255, 210, 125));
                        const qreal ptR = std::max<qreal>(1.2, 1.5 * s);
                        for (const auto& pt2 : item.segmentedCloudPoints2D) {
                            if (pt2.x() >= 0.0 && pt2.x() < displayImg.width() &&
                                pt2.y() >= 0.0 && pt2.y() < displayImg.height()) {
                                p.drawEllipse(pt2, ptR, ptR);
                            }
                        }
                    }

                    // B. Embedded CAD Feature / Crease Wireframe & Structural Silhouette
                    if (flags.show_cad_wireframe) {
                        if (!item.topSurfacePolygon.isEmpty()) {
                            p.setBrush(QColor(0, 229, 255, 48));
                            p.setPen(QPen(QColor(0, 229, 255), std::max(1.0, 1.8 * s)));
                            p.drawPolygon(item.topSurfacePolygon);
                        }
                        if (!item.footPoints.empty()) {
                            QPolygonF footPoly;
                            for (const auto& fp : item.footPoints) footPoly << fp;
                            p.setBrush(Qt::NoBrush);
                            p.setPen(QPen(QColor(0, 230, 118), std::max(1.0, 2.0 * s)));
                            p.drawPolygon(footPoly);
                            const qreal rx = std::max(4.0, 11.0 * s);
                            const qreal ry = std::max(2.0, 4.0 * s);
                            for (const auto& fp : item.footPoints) p.drawEllipse(fp, rx, ry);
                        }
                        for (const auto& seg : item.wireframeSegments) {
                            p.setPen(QPen(seg.color, seg.thickness));
                            p.drawLine(seg.p0, seg.p1);
                        }
                    }

                    // C. 3D Oriented Bounding Box (OBB)
                    const bool drawLegacySegments = !item.isPhysicalStlMode && item.wireframeSegments.empty() &&
                                                    (flags.show_obb_box || flags.show_cad_wireframe);
                    if ((flags.show_obb_box && item.isPhysicalStlMode) || drawLegacySegments) {
                        for (const auto& seg : item.segments) {
                            p.setPen(QPen(seg.color, seg.thickness));
                            p.drawLine(seg.p0, seg.p1);
                        }
                    }

                    // D. Scale-Adaptive 3D Axes & Per-Instance HUD
                    if (flags.show_axes_and_hud) {
                        if (item.hasAxes) {
                            const QPointF c = item.centerAxis;
                            const qreal r_c = std::max(4.0, 6.5 * s);
                            p.setPen(QPen(QColor(10, 10, 10), 1));
                            p.setBrush(QColor(0, 230, 118));
                            p.drawEllipse(c, r_c, r_c);

                            // X axis (Red)
                            p.setPen(QPen(QColor(255, 60, 60), std::max(1.5, 2.0 * s)));
                            p.drawLine(c, item.axisX);
                            p.setFont(QFont("Arial", std::max(7, static_cast<int>(8.5 * s)), QFont::Bold));
                            p.drawText(QRectF(item.axisX.x() + 3 * s, item.axisX.y() - 9 * s, 16 * s, 16 * s), Qt::AlignCenter, "X");

                            // Y axis (Green) when projected offset is distinct
                            const qreal dy_len = std::hypot(item.axisY.x() - c.x(), item.axisY.y() - c.y());
                            if (dy_len > 2.0) {
                                p.setPen(QPen(QColor(50, 235, 80), std::max(1.5, 2.0 * s)));
                                p.drawLine(c, item.axisY);
                                p.drawText(QRectF(item.axisY.x() + 3 * s, item.axisY.y() - 9 * s, 16 * s, 16 * s), Qt::AlignCenter, "Y");
                            }

                            // Z axis (Blue)
                            p.setPen(QPen(QColor(40, 160, 255), std::max(1.5, 2.0 * s)));
                            p.drawLine(c, item.axisZ);
                            p.drawText(QRectF(item.axisZ.x() - 8 * s, item.axisZ.y() - 18 * s, 16 * s, 16 * s), Qt::AlignCenter, "Z");
                        }

                        // Multi-instance per-object floating tag
                        if (itemsToDraw.size() > 1 && !item.label.isEmpty() && item.bbox.isValid()) {
                            const qreal bx = std::clamp(item.bbox.left(), 6.0, std::max(6.0, displayImg.width() - 210.0 * s));
                            const qreal by = std::clamp(item.bbox.top() - 22.0 * s, 48.0 * s, std::max(48.0 * s, displayImg.height() - 56.0 * s));
                            QRectF instTag(bx, by, 200.0 * s, 20.0 * s);
                            drawChamferedBoxQt(instTag, 3.0 * s, QColor(15, 20, 28, 215), QColor(56, 189, 248));
                            p.setFont(QFont("Monospace", std::max(6, static_cast<int>(7.5 * s)), QFont::Bold));
                            p.setPen(QColor(224, 242, 254));
                            p.drawText(instTag.adjusted(4, 0, -4, 0), Qt::AlignVCenter | Qt::AlignLeft, item.label);
                        }
                    }
                }

                // Primary global HUD banners (drawn once even in multi-instance mode)
                if (flags.show_axes_and_hud) {
                    const auto& primary = itemsToDraw.front();
                    QString statusText = primary.label.isEmpty() ? overlayLabel_ : primary.label;
                    if (statusText.isEmpty()) statusText = "[OK] CAD MODEL DETECTED (6DoF)";

                    const qreal tag_x = 14.0 * s, tag_y = 14.0 * s;
                    const qreal tag_w = std::clamp(290.0 * s, 180.0, displayImg.width() * 0.45);
                    const qreal tag_h = 28.0 * s;
                    QRectF statusTag(tag_x, tag_y, tag_w, tag_h);
                    drawChamferedBoxQt(statusTag, std::max(4.0, 6.0 * s), QColor(20, 20, 20, 220), QColor(0, 230, 118));
                    p.setFont(QFont("Arial", std::max(7, static_cast<int>(8.5 * s)), QFont::Bold));
                    p.setPen(QColor(0, 230, 118));
                    p.drawText(statusTag.adjusted(6, 0, -6, 0), Qt::AlignCenter, statusText);

                    QString headerText = primary.headerBanner;
                    if (headerText.isEmpty()) {
                        headerText = "UNIVERSAL BLACK-BOX 6DoF";
                    }
                    const qreal hdr_w = std::min(460.0 * s, std::max(160.0, displayImg.width() - tag_w - 36.0));
                    const qreal hdr_h = 28.0 * s;
                    const qreal hdr_x = std::clamp(tag_x + tag_w + 10.0, tag_x + tag_w + 6.0, std::max(10.0, displayImg.width() - hdr_w - 10.0));
                    const qreal hdr_y = 14.0 * s;
                    QRectF tagTop(hdr_x, hdr_y, hdr_w, hdr_h);
                    drawChamferedBoxQt(tagTop, std::max(4.0, 6.0 * s), QColor(20, 20, 20, 220), QColor(0, 230, 118));
                    p.setFont(QFont("Monospace", std::max(6, static_cast<int>(8.0 * s)), QFont::Bold));
                    p.setPen(QColor(200, 255, 200));
                    p.drawText(tagTop.adjusted(6, 0, -6, 0), Qt::AlignCenter, headerText);

                    if (primary.hasTableCenter) {
                        p.setPen(QPen(QColor(0, 230, 118, 170), 1.0));
                        p.drawLine(QPointF(hdr_x + hdr_w / 2.0, hdr_y + hdr_h), primary.tableCenter);
                    }

                    QString banner = primary.poseBanner.isEmpty() ? overlayPoseText_ : primary.poseBanner;
                    if (!banner.isEmpty()) {
                        const qreal bot_w = std::min(560.0 * s, displayImg.width() - 24.0);
                        const qreal bot_h = 28.0 * s;
                        const qreal bot_x = std::clamp((displayImg.width() - bot_w) / 2.0, 10.0, std::max(10.0, displayImg.width() - bot_w - 10.0));
                        const qreal bot_y = displayImg.height() - 40.0 * s;
                        QRectF tagBot(bot_x, bot_y, bot_w, bot_h);
                        drawChamferedBoxQt(tagBot, std::max(3.0, 4.0 * s), QColor(20, 20, 20, 220), QColor(80, 80, 80));
                        p.setPen(QColor(230, 230, 230));
                        p.setFont(QFont("Monospace", std::max(6, static_cast<int>(8.5 * s)), QFont::Bold));
                        p.drawText(tagBot.adjusted(6, 0, -6, 0), Qt::AlignCenter, banner);
                    }
                }
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

        // E. Interactive 2D ROI Crop Overlay (also painted on raw RGB pane below)
        if ((roiDraggingSnap || roiActiveSnap) && roiNormSnap.width() > 0.005 && roiNormSnap.height() > 0.005) {
            drawRoiOnColor = true;
            auto drawRoiBox = [&](QPainter& painter, int imgW, int imgH) {
                QRectF rx(roiNormSnap.x() * imgW, roiNormSnap.y() * imgH,
                          roiNormSnap.width() * imgW, roiNormSnap.height() * imgH);
                QPainterPath outer;
                outer.addRect(QRectF(0, 0, imgW, imgH));
                QPainterPath inner;
                inner.addRect(rx);
                painter.fillPath(outer.subtracted(inner), QColor(0, 0, 0, 70));

                QPen roiPen(QColor(245, 158, 11), std::max(1.5, 2.2 * s), Qt::DashLine);
                painter.setPen(roiPen);
                painter.setBrush(Qt::NoBrush);
                painter.drawRect(rx);

                QRectF tagRoi(rx.left(), std::max(4.0, rx.top() - 20.0 * s), 165.0 * s, 18.0 * s);
                painter.fillRect(tagRoi, QColor(146, 64, 14, 220));
                painter.setPen(QColor(254, 243, 199));
                painter.setFont(QFont("Monospace", std::max(6, static_cast<int>(7.5 * s)), QFont::Bold));
                painter.drawText(tagRoi, Qt::AlignCenter,
                                 roiDraggingSnap ? "✂️ DRAGGING 2D ROI..." : "✂️ [ROI CROP ACTIVE]");
            };
            drawRoiBox(p, displayImg.width(), displayImg.height());
        }
    }

    if (drawRoiOnColor && !colorRawImg.isNull()) {
        QImage rawCopy = colorRawImg.copy();
        QPainter pr(&rawCopy);
        pr.setRenderHint(QPainter::Antialiasing);
        QRectF rx(roiNormSnap.x() * rawCopy.width(), roiNormSnap.y() * rawCopy.height(),
                  roiNormSnap.width() * rawCopy.width(), roiNormSnap.height() * rawCopy.height());
        pr.setPen(QPen(QColor(245, 158, 11), 2.0, Qt::DashLine));
        pr.setBrush(QColor(245, 158, 11, 25));
        pr.drawRect(rx);
        colorRawImg = std::move(rawCopy);
    }

    const Qt::TransformationMode scaleMode = isStreaming() ? Qt::FastTransformation : Qt::SmoothTransformation;
    if (detectionViewLabel_ && detectionViewLabel_->isVisible()) {
        detectionViewLabel_->setPixmap(QPixmap::fromImage(displayImg).scaled(
            detectionViewLabel_->size(), Qt::KeepAspectRatio, scaleMode));
    }
    if (colorViewLabel_ && colorViewLabel_->isVisible()) {
        colorViewLabel_->setPixmap(QPixmap::fromImage(colorRawImg).scaled(
            colorViewLabel_->size(), Qt::KeepAspectRatio, scaleMode));
    }
    if (depthViewLabel_ && depthViewLabel_->isVisible() && !lastDepthImg_.isNull()) {
        depthViewLabel_->setPixmap(QPixmap::fromImage(lastDepthImg_).scaled(
            depthViewLabel_->size(), Qt::KeepAspectRatio, scaleMode));
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
