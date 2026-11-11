#include "RealSenseRecorderDialog.h"
#include <librealsense2/rs.hpp>
#include <open3d/Open3D.h>

#include <QApplication>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QDateTime>
#include <QMessageBox>
#include <QPainter>
#include <QDir>
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>

struct RealSenseRecorderDialog::Impl {
    rs2::pipeline pipe;
    rs2::config cfg;
    std::unique_ptr<rs2::align> align;
    rs2::colorizer colorizer;
    bool pipeline_running = false;
    bool has_imu = false;

    rs2_intrinsics color_intrinsics{};
    double pitch_deg = 0.0;
    double roll_deg = 0.0;
    double center_dist_m = 0.0;
    std::string device_name = "Unknown RealSense";
    std::string serial_number;

    std::mutex frame_mutex;
    rs2::frameset last_frameset;
};

RealSenseRecorderDialog::RealSenseRecorderDialog(QWidget* parent)
    : QDialog(parent), impl_(std::make_unique<Impl>()) {
    setWindowTitle("Intel RealSense D435if Native Recorder & Dataset Studio (Phase 8)");
    resize(960, 680);
    setStyleSheet("background-color: #1e1e24; color: #ffffff;");

    setupUi();

    connect(this, &RealSenseRecorderDialog::frameReady,
            this, &RealSenseRecorderDialog::onFrameReady,
            Qt::QueuedConnection);

    // Check device and start live preview
    if (!startPipeline("")) {
        statusLabel_->setText("<font color='#f44336'><b>Lỗi: Không tìm thấy camera RealSense D435if cắm qua USB!</b></font>");
        recordBtn_->setEnabled(false);
        snapshotBtn_->setEnabled(false);
    }
}

RealSenseRecorderDialog::~RealSenseRecorderDialog() {
    stopPipeline();
}

void RealSenseRecorderDialog::setupUi() {
    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(12, 12, 12, 12);
    mainLayout->setSpacing(10);

    // 1. Top Header Bar
    auto* headerLayout = new QHBoxLayout();
    statusLabel_ = new QLabel("Khởi tạo RealSense D435if...", this);
    statusLabel_->setStyleSheet("font-size: 13px; font-weight: bold; color: #4fc3f7;");
    headerLayout->addWidget(statusLabel_);
    headerLayout->addStretch();

    imuLabel_ = new QLabel("IMU: Đang kết nối...", this);
    imuLabel_->setStyleSheet("font-size: 11px; background: #263238; padding: 4px 8px; border-radius: 4px; color: #ffca28;");
    headerLayout->addWidget(imuLabel_);

    centerDistLabel_ = new QLabel("Center Dist: -- m", this);
    centerDistLabel_->setStyleSheet("font-size: 11px; background: #263238; padding: 4px 8px; border-radius: 4px; color: #00e676;");
    headerLayout->addWidget(centerDistLabel_);
    mainLayout->addLayout(headerLayout);

    // 2. Video Preview Section (Color + Depth side by side)
    auto* videoBox = new QGroupBox("Luồng Trực Tiếp Thời Gian Thực (C++ Native 30 FPS - Tối Ưu Hóa)", this);
    videoBox->setStyleSheet("QGroupBox { font-weight: bold; color: #90caf9; border: 1px solid #37474f; border-radius: 6px; margin-top: 6px; padding-top: 10px; }");
    auto* videoLayout = new QHBoxLayout(videoBox);

    colorViewLabel_ = new QLabel("Chờ luồng RGB...", videoBox);
    colorViewLabel_->setFixedSize(440, 330);
    colorViewLabel_->setAlignment(Qt::AlignCenter);
    colorViewLabel_->setStyleSheet("background: #121212; border: 1px solid #455a64; border-radius: 4px;");

    depthViewLabel_ = new QLabel("Chờ luồng Depth Z16...", videoBox);
    depthViewLabel_->setFixedSize(440, 330);
    depthViewLabel_->setAlignment(Qt::AlignCenter);
    depthViewLabel_->setStyleSheet("background: #121212; border: 1px solid #455a64; border-radius: 4px;");

    videoLayout->addWidget(colorViewLabel_);
    videoLayout->addWidget(depthViewLabel_);
    mainLayout->addWidget(videoBox);

    // 3. Output Configuration & Control Bar
    auto* ctrlBox = new QGroupBox("Điều Khiển Thu Thập Dữ Liệu", this);
    ctrlBox->setStyleSheet("QGroupBox { font-weight: bold; color: #90caf9; border: 1px solid #37474f; border-radius: 6px; margin-top: 6px; padding-top: 10px; }");
    auto* ctrlLayout = new QVBoxLayout(ctrlBox);

    auto* dirLayout = new QHBoxLayout();
    dirLayout->addWidget(new QLabel("Thư mục lưu trữ:", ctrlBox));

    QDir projDir(QCoreApplication::applicationDirPath());
    if (projDir.dirName() == "build") projDir.cdUp();
    QString defaultOutDir = projDir.absoluteFilePath("data/realsense_records");

    outputDirEdit_ = new QLineEdit(defaultOutDir, ctrlBox);
    outputDirEdit_->setStyleSheet("background: #263238; color: #ffffff; border: 1px solid #546e7a; padding: 4px 6px; border-radius: 4px;");
    dirLayout->addWidget(outputDirEdit_);

    auto* browseBtn = new QPushButton("Chọn thư mục...", ctrlBox);
    browseBtn->setStyleSheet("background: #37474f; color: white; padding: 4px 10px; border-radius: 4px;");
    connect(browseBtn, &QPushButton::clicked, this, &RealSenseRecorderDialog::onBrowseOutputDir);
    dirLayout->addWidget(browseBtn);
    ctrlLayout->addLayout(dirLayout);

    // Action Buttons
    auto* btnLayout = new QHBoxLayout();
    recordBtn_ = new QPushButton("🔴 Bắt Đầu Ghi Video (.bag)", ctrlBox);
    recordBtn_->setFixedHeight(40);
    recordBtn_->setStyleSheet("background: #d32f2f; color: white; font-weight: bold; font-size: 13px; border-radius: 5px;");
    connect(recordBtn_, &QPushButton::clicked, this, &RealSenseRecorderDialog::onToggleRecord);
    btnLayout->addWidget(recordBtn_);

    snapshotBtn_ = new QPushButton("📸 Chụp Ảnh Mẫu (Color + Depth 16-bit PNG)", ctrlBox);
    snapshotBtn_->setFixedHeight(40);
    snapshotBtn_->setStyleSheet("background: #1976d2; color: white; font-weight: bold; font-size: 13px; border-radius: 5px;");
    connect(snapshotBtn_, &QPushButton::clicked, this, &RealSenseRecorderDialog::onTakeSnapshot);
    btnLayout->addWidget(snapshotBtn_);

    auto* closeBtn = new QPushButton("Đóng", ctrlBox);
    closeBtn->setFixedHeight(40);
    closeBtn->setStyleSheet("background: #455a64; color: white; font-weight: bold; font-size: 13px; border-radius: 5px;");
    connect(closeBtn, &QPushButton::clicked, this, &QDialog::accept);
    btnLayout->addWidget(closeBtn);

    ctrlLayout->addLayout(btnLayout);
    mainLayout->addWidget(ctrlBox);
}

bool RealSenseRecorderDialog::startPipeline(const std::string& recordBagPath) {
    stopPipeline();

    try {
        rs2::context ctx;
        auto devices = ctx.query_devices();
        if (devices.size() == 0 && recordBagPath.empty()) {
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

        auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        impl_->color_intrinsics = color_stream.get_intrinsics();

        impl_->pipeline_running = true;

        if (recordBagPath.empty()) {
            statusLabel_->setText(QString("● SẴN SÀNG: %1 (SN: %2) - 640x480 @ 30FPS")
                .arg(QString::fromStdString(impl_->device_name))
                .arg(QString::fromStdString(impl_->serial_number)));
        } else {
            statusLabel_->setText(QString("<font color='#ff5252'><b>● ĐANG GHI HÌNH FILE .BAG: %1</b></font>")
                .arg(QFileInfo(QString::fromStdString(recordBagPath)).fileName()));
        }
        imuLabel_->setText("RGB-D: 640x480 @ 30 FPS");

        captureRunning_.store(true);
        captureThread_ = std::thread(&RealSenseRecorderDialog::captureWorker, this);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "startPipeline error: " << e.what() << "\n";
        return false;
    }
}

void RealSenseRecorderDialog::stopPipeline() {
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
}

void RealSenseRecorderDialog::captureWorker() {
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

            // Chuyển đổi Color buffer sang QImage
            QImage colorImg(cw, ch, QImage::Format_RGB888);
            std::memcpy(colorImg.bits(), color.get_data(), cw * ch * 3);

            // Colorize Depth ở luồng ngầm để Main UI Thread hoàn toàn nhẹ nhàng
            rs2::video_frame colorized_depth = impl_->colorizer.colorize(depth);
            QImage depthImg(dw, dh, QImage::Format_RGB888);
            std::memcpy(depthImg.bits(), colorized_depth.get_data(), dw * dh * 3);

            float center_d = depth.get_distance(dw / 2, dh / 2);

            emit frameReady(colorImg, depthImg, center_d);
        } catch (const rs2::error& e) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        } catch (...) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
}

void RealSenseRecorderDialog::onFrameReady(const QImage& colorImg, const QImage& depthImg, float centerDist) {
    // Render nhanh với FastTransformation (< 0.5ms)
    colorViewLabel_->setPixmap(QPixmap::fromImage(colorImg).scaled(colorViewLabel_->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
    depthViewLabel_->setPixmap(QPixmap::fromImage(depthImg).scaled(depthViewLabel_->size(), Qt::KeepAspectRatio, Qt::FastTransformation));

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

void RealSenseRecorderDialog::onToggleRecord() {
    recordBtn_->setEnabled(false);
    if (!isRecording_) {
        // Start recording to .bag
        QString outDir = outputDirEdit_->text().trimmed();
        QDir().mkpath(outDir);

        QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
        QString bagFile = QString("%1/d435if_record_%2.bag").arg(outDir).arg(timestamp);
        currentBagPath_ = bagFile.toStdString();

        statusLabel_->setText("⏳ Đang chuẩn bị luồng ghi file .bag...");
        QApplication::processEvents();

        if (startPipeline(currentBagPath_)) {
            isRecording_ = true;
            recordBtn_->setText("⏹️ Dừng Ghi (.bag)");
            recordBtn_->setStyleSheet("background: #2e7d32; color: white; font-weight: bold; font-size: 13px; border-radius: 5px;");
            snapshotBtn_->setEnabled(false);
        } else {
            statusLabel_->setText("<font color='#f44336'><b>Lỗi: Không thể bắt đầu ghi .bag!</b></font>");
            QMessageBox::warning(this, "Lỗi Ghi Hình", "Không thể bắt đầu ghi ra file .bag:\n" + bagFile);
        }
    } else {
        // Stop recording
        statusLabel_->setText("⏳ Đang hoàn tất lưu file .bag...");
        QApplication::processEvents();

        isRecording_ = false;
        startPipeline(""); // restart in preview mode
        recordBtn_->setText("🔴 Bắt Đầu Ghi Video (.bag)");
        recordBtn_->setStyleSheet("background: #d32f2f; color: white; font-weight: bold; font-size: 13px; border-radius: 5px;");
        snapshotBtn_->setEnabled(true);

        QFileInfo fi(QString::fromStdString(currentBagPath_));
        double sizeMb = fi.size() / (1024.0 * 1024.0);
        QMessageBox::information(this, "Ghi Hình Hoàn Tất",
            QString("✅ Đã lưu video .bag thành công!\n\nTập tin: %1\nDung lượng: %2 MB")
            .arg(QString::fromStdString(currentBagPath_))
            .arg(sizeMb, 0, 'f', 1));
    }
    recordBtn_->setEnabled(true);
}

void RealSenseRecorderDialog::onTakeSnapshot() {
    rs2::frameset fs;
    {
        std::lock_guard<std::mutex> lock(impl_->frame_mutex);
        fs = impl_->last_frameset;
    }
    if (!fs) {
        QMessageBox::warning(this, "Chụp ảnh", "Chưa nhận được khung hình từ camera!");
        return;
    }

    // Align depth to color specifically for this snapshot
    if (!impl_->align) {
        impl_->align = std::make_unique<rs2::align>(RS2_STREAM_COLOR);
    }
    rs2::frameset aligned_fs = impl_->align->process(fs);
    rs2::video_frame color = aligned_fs.get_color_frame();
    rs2::depth_frame depth = aligned_fs.get_depth_frame();

    if (!color || !depth) {
        QMessageBox::warning(this, "Chụp ảnh", "Không thể căn chỉnh khung hình màu và độ sâu!");
        return;
    }

    QString outDir = outputDirEdit_->text().trimmed();
    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
    QString snapDir = QString("%1/snapshot_%2").arg(outDir).arg(timestamp);
    QDir().mkpath(snapDir);

    int w = color.get_width();
    int h = color.get_height();

    // 1. Lưu color.png
    QImage colorImg(reinterpret_cast<const uchar*>(color.get_data()), w, h, w * 3, QImage::Format_RGB888);
    QString colorPath = snapDir + "/color.png";
    colorImg.save(colorPath, "PNG");

    // 2. Lưu depth.png 16-bit uint16 mm dùng Open3D
    open3d::geometry::Image depthO3D;
    depthO3D.Prepare(w, h, 1, 2); // 1 channel, 2 bytes = Z16
    std::memcpy(depthO3D.data_.data(), depth.get_data(), w * h * 2);
    QString depthPath = snapDir + "/depth.png";
    open3d::io::WriteImage(depthPath.toStdString(), depthO3D);

    // 3. Lưu intrinsics.json
    QString intrPath = snapDir + "/intrinsics.json";
    std::ofstream fIntr(intrPath.toStdString());
    if (fIntr.is_open()) {
        fIntr << "{\n";
        fIntr << "  \"width\": " << w << ",\n";
        fIntr << "  \"height\": " << h << ",\n";
        fIntr << "  \"fx\": " << impl_->color_intrinsics.fx << ",\n";
        fIntr << "  \"fy\": " << impl_->color_intrinsics.fy << ",\n";
        fIntr << "  \"cx\": " << impl_->color_intrinsics.ppx << ",\n";
        fIntr << "  \"cy\": " << impl_->color_intrinsics.ppy << ",\n";
        fIntr << "  \"depth_scale_mm\": 1.0\n";
        fIntr << "}\n";
        fIntr.close();
    }

    // 4. Lưu imu.json
    QString imuPath = snapDir + "/imu.json";
    std::ofstream fImu(imuPath.toStdString());
    if (fImu.is_open()) {
        fImu << "{\n";
        fImu << "  \"available\": " << (impl_->has_imu ? "true" : "false") << ",\n";
        fImu << "  \"pitch_deg\": " << impl_->pitch_deg << ",\n";
        fImu << "  \"roll_deg\": " << impl_->roll_deg << ",\n";
        fImu << "  \"center_dist_m\": " << impl_->center_dist_m << "\n";
        fImu << "}\n";
        fImu.close();
    }

    snapshotCount_++;
    statusLabel_->setText(QString("<font color='#00e676'><b>📸 Đã chụp snapshot #%1 vào: %2</b></font>")
        .arg(snapshotCount_)
        .arg(QFileInfo(snapDir).fileName()));
}

void RealSenseRecorderDialog::onBrowseOutputDir() {
    QString dir = QFileDialog::getExistingDirectory(this, "Chọn thư mục lưu trữ dữ liệu RealSense", outputDirEdit_->text());
    if (!dir.isEmpty()) {
        outputDirEdit_->setText(dir);
    }
}
