#include "ImageDetectionDialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>
#include <QDateTime>
#include <QSplitter>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

ImageDetectionDialog::ImageDetectionDialog(QWidget* parent)
    : QDialog(parent)
{
    setWindowTitle("🎯 Nhận Diện Đối Tượng 2D Trên Ảnh (CAD 2D Detection)");
    resize(1150, 720);
    setStyleSheet("QDialog { background: #12161a; color: #eceff1; font-family: sans-serif; }");

    setupUi();
}

void ImageDetectionDialog::setDetector(std::shared_ptr<CADTemplateDetector> detector, const QString& cadName)
{
    detector_ = detector;
    cadName_ = cadName.isEmpty() ? "TABLE-DEMO" : cadName;
    if (cadModelLabel_) {
        cadModelLabel_->setText(QString("<b>Mô hình mục tiêu:</b> <font color='#00e5ff'>%1</font>").arg(cadName_));
    }
}

void ImageDetectionDialog::setupUi()
{
    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(12, 12, 12, 12);
    mainLayout->setSpacing(10);

    // ── Header Title Banner ──
    auto* headerCard = new QWidget(this);
    headerCard->setStyleSheet("background: #1a2228; border: 1px solid #2a3740; border-radius: 6px; padding: 6px;");
    auto* headerLayout = new QHBoxLayout(headerCard);
    headerLayout->setContentsMargins(8, 4, 8, 4);

    auto* titleLabel = new QLabel("<h3>🎯 NHẬN DIỆN ĐỐI TƯỢNG TRÊN ẢNH (2D CAD DETECTION ENGINE)</h3>", headerCard);
    titleLabel->setStyleSheet("color: #00e676; margin: 0;");
    headerLayout->addWidget(titleLabel);
    headerLayout->addStretch();

    cadModelLabel_ = new QLabel("<b>Mô hình mục tiêu:</b> <font color='#00e5ff'>TABLE-DEMO</font>", headerCard);
    cadModelLabel_->setStyleSheet("font-size: 12px; color: #cfd8dc;");
    headerLayout->addWidget(cadModelLabel_);

    mainLayout->addWidget(headerCard);

    // ── Main Content Splitter (Left: Image Preview, Right: Telemetry & Controls) ──
    auto* splitter = new QSplitter(Qt::Horizontal, this);

    // Left: Image Viewport inside ScrollArea
    scrollArea_ = new QScrollArea(splitter);
    scrollArea_->setStyleSheet("QScrollArea { background: #0a0d0e; border: 1px solid #1f2930; border-radius: 6px; }");
    scrollArea_->setWidgetResizable(true);
    scrollArea_->setAlignment(Qt::AlignCenter);

    imagePreviewLabel_ = new QLabel(scrollArea_);
    imagePreviewLabel_->setAlignment(Qt::AlignCenter);
    imagePreviewLabel_->setStyleSheet("background: transparent;");
    imagePreviewLabel_->setText("<font color='#546e7a'><i>Chưa có ảnh nào được nạp...</i></font>");
    scrollArea_->setWidget(imagePreviewLabel_);
    splitter->addWidget(scrollArea_);

    // Right: Telemetry Card & Option Controls
    auto* rightPanel = new QWidget(splitter);
    rightPanel->setMaximumWidth(400);
    rightPanel->setMinimumWidth(320);
    auto* rightLayout = new QVBoxLayout(rightPanel);
    rightLayout->setContentsMargins(4, 0, 4, 0);
    rightLayout->setSpacing(8);

    // 1. Status Badge Box
    auto* statusBox = new QGroupBox("Trạng Thái Nhận Diện", rightPanel);
    statusBox->setStyleSheet("QGroupBox { font-weight: bold; color: #90a4ae; border: 1px solid #263238; border-radius: 6px; margin-top: 6px; padding-top: 10px; }");
    auto* statusBoxLayout = new QVBoxLayout(statusBox);

    statusBadgeLabel_ = new QLabel("Chưa kiểm thử", statusBox);
    statusBadgeLabel_->setAlignment(Qt::AlignCenter);
    statusBadgeLabel_->setStyleSheet("background: #263238; color: #b0bec5; font-size: 13px; font-weight: bold; padding: 8px; border-radius: 4px;");
    statusBoxLayout->addWidget(statusBadgeLabel_);

    rightLayout->addWidget(statusBox);

    // 2. Metrics Telemetry Card
    auto* metricsBox = new QGroupBox("Thông Số & Tọa Độ Đối Tượng", rightPanel);
    metricsBox->setStyleSheet("QGroupBox { font-weight: bold; color: #90a4ae; border: 1px solid #263238; border-radius: 6px; margin-top: 6px; padding-top: 10px; }");
    auto* grid = new QGridLayout(metricsBox);
    grid->setSpacing(6);

    grid->addWidget(new QLabel("Độ tin cậy (Score):", metricsBox), 0, 0);
    scoreLabel_ = new QLabel("0.0%", metricsBox);
    scoreLabel_->setStyleSheet("font-weight: bold; color: #00e676;");
    grid->addWidget(scoreLabel_, 0, 1);

    scoreProgressBar_ = new QProgressBar(metricsBox);
    scoreProgressBar_->setRange(0, 100);
    scoreProgressBar_->setValue(0);
    scoreProgressBar_->setTextVisible(false);
    scoreProgressBar_->setFixedHeight(8);
    scoreProgressBar_->setStyleSheet("QProgressBar { background: #263238; border-radius: 4px; } QProgressBar::chunk { background: #00e676; border-radius: 4px; }");
    grid->addWidget(scoreProgressBar_, 1, 0, 1, 2);

    grid->addWidget(new QLabel("Góc xoay ước tính (Yaw):", metricsBox), 2, 0);
    yawLabel_ = new QLabel("-", metricsBox);
    yawLabel_->setStyleSheet("color: #40c4ff; font-weight: bold;");
    grid->addWidget(yawLabel_, 2, 1);

    grid->addWidget(new QLabel("Điểm khớp (Matches):", metricsBox), 3, 0);
    matchesLabel_ = new QLabel("-", metricsBox);
    matchesLabel_->setStyleSheet("color: #ffe082;");
    grid->addWidget(matchesLabel_, 3, 1);

    grid->addWidget(new QLabel("Khung bao 2D (ROI):", metricsBox), 4, 0);
    bboxLabel_ = new QLabel("-", metricsBox);
    bboxLabel_->setStyleSheet("color: #80cbc4; font-family: monospace; font-size: 11px;");
    bboxLabel_->setWordWrap(true);
    grid->addWidget(bboxLabel_, 4, 1);

    grid->addWidget(new QLabel("Độ trễ xử lý:", metricsBox), 5, 0);
    latencyLabel_ = new QLabel("-", metricsBox);
    latencyLabel_->setStyleSheet("color: #b388ff; font-weight: bold;");
    grid->addWidget(latencyLabel_, 5, 1);

    rightLayout->addWidget(metricsBox);

    // 3. File Info Box
    auto* fileBox = new QGroupBox("Thông Tin Tệp Ảnh", rightPanel);
    fileBox->setStyleSheet("QGroupBox { font-weight: bold; color: #90a4ae; border: 1px solid #263238; border-radius: 6px; margin-top: 6px; padding-top: 10px; }");
    auto* fileBoxLayout = new QVBoxLayout(fileBox);

    fileInfoLabel_ = new QLabel("Chưa chọn tệp ảnh.", fileBox);
    fileInfoLabel_->setStyleSheet("color: #b0bec5; font-size: 11px; background: #182026; padding: 6px; border-radius: 4px;");
    fileInfoLabel_->setWordWrap(true);
    fileBoxLayout->addWidget(fileInfoLabel_);

    rightLayout->addWidget(fileBox);

    // 4. Display Overlay Toggles
    auto* optionsBox = new QGroupBox("Tùy Chọn Lớp Phủ (Display Layers)", rightPanel);
    optionsBox->setStyleSheet("QGroupBox { font-weight: bold; color: #90a4ae; border: 1px solid #263238; border-radius: 6px; margin-top: 6px; padding-top: 10px; }");
    auto* optLayout = new QVBoxLayout(optionsBox);
    optLayout->setSpacing(4);

    showBboxCheck_ = new QCheckBox("Hiện khung Bounding Box 2D", optionsBox);
    showBboxCheck_->setChecked(true);
    connect(showBboxCheck_, &QCheckBox::toggled, this, &ImageDetectionDialog::onDisplayOptionsChanged);
    optLayout->addWidget(showBboxCheck_);

    showKeypointsCheck_ = new QCheckBox("Hiện điểm đặc trưng inliers (Keypoints)", optionsBox);
    showKeypointsCheck_->setChecked(true);
    connect(showKeypointsCheck_, &QCheckBox::toggled, this, &ImageDetectionDialog::onDisplayOptionsChanged);
    optLayout->addWidget(showKeypointsCheck_);

    showHudCheck_ = new QCheckBox("Hiện bảng thông số HUD trên ảnh", optionsBox);
    showHudCheck_->setChecked(true);
    connect(showHudCheck_, &QCheckBox::toggled, this, &ImageDetectionDialog::onDisplayOptionsChanged);
    optLayout->addWidget(showHudCheck_);

    rightLayout->addWidget(optionsBox);
    rightLayout->addStretch();

    splitter->addWidget(rightPanel);
    splitter->setStretchFactor(0, 7);
    splitter->setStretchFactor(1, 3);
    mainLayout->addWidget(splitter, 1);

    // ── Bottom Action Buttons ──
    auto* bottomBar = new QHBoxLayout();
    bottomBar->setSpacing(8);

    openAnotherBtn_ = new QPushButton("📂 Mở Tệp Ảnh Khác (Select Image)...", this);
    openAnotherBtn_->setFixedHeight(36);
    openAnotherBtn_->setCursor(Qt::PointingHandCursor);
    openAnotherBtn_->setStyleSheet(
        "QPushButton {"
        "  background: #00838f; color: #ffffff; font-weight: bold; padding: 0 16px; border: none; border-radius: 4px;"
        "}"
        "QPushButton:hover { background: #0097a7; }");
    connect(openAnotherBtn_, &QPushButton::clicked, this, &ImageDetectionDialog::onOpenAnotherImage);
    bottomBar->addWidget(openAnotherBtn_);

    saveImageBtn_ = new QPushButton("💾 Lưu Ảnh Đã Nhận Diện (Save As)...", this);
    saveImageBtn_->setFixedHeight(36);
    saveImageBtn_->setEnabled(false);
    saveImageBtn_->setCursor(Qt::PointingHandCursor);
    saveImageBtn_->setStyleSheet(
        "QPushButton {"
        "  background: #2e7d32; color: #ffffff; font-weight: bold; padding: 0 16px; border: none; border-radius: 4px;"
        "}"
        "QPushButton:hover { background: #388e3c; }"
        "QPushButton:disabled { background: #263238; color: #546e7a; }");
    connect(saveImageBtn_, &QPushButton::clicked, this, &ImageDetectionDialog::onSaveAnnotatedImage);
    bottomBar->addWidget(saveImageBtn_);

    bottomBar->addStretch();

    closeBtn_ = new QPushButton("Đóng (Close)", this);
    closeBtn_->setFixedHeight(36);
    closeBtn_->setCursor(Qt::PointingHandCursor);
    closeBtn_->setStyleSheet(
        "QPushButton {"
        "  background: #37474f; color: #eceff1; padding: 0 20px; border: none; border-radius: 4px;"
        "}"
        "QPushButton:hover { background: #455a64; }");
    connect(closeBtn_, &QPushButton::clicked, this, &QDialog::accept);
    bottomBar->addWidget(closeBtn_);

    mainLayout->addLayout(bottomBar);
}

bool ImageDetectionDialog::processImage(const QString& filePath)
{
    if (filePath.isEmpty() || !QFile::exists(filePath)) {
        return false;
    }

    currentFilePath_ = filePath;

    // Đọc ảnh bằng OpenCV trước (hỗ trợ png, jpg, jpeg, bmp, webp, tiff...)
    rawImage_ = cv::imread(filePath.toStdString(), cv::IMREAD_COLOR);

    // Fallback qua QImage nếu OpenCV thiếu codec
    if (rawImage_.empty()) {
        QImage qimg(filePath);
        if (!qimg.isNull()) {
            qimg = qimg.convertToFormat(QImage::Format_RGB888);
            rawImage_ = cv::Mat(qimg.height(), qimg.width(), CV_8UC3,
                                const_cast<uchar*>(qimg.bits()), qimg.bytesPerLine()).clone();
            cv::cvtColor(rawImage_, rawImage_, cv::COLOR_RGB2BGR);
        }
    }

    if (rawImage_.empty()) {
        QMessageBox::warning(this, "Lỗi Tệp Ảnh",
            QString("Không thể mở tệp ảnh được chỉ định:\n%1").arg(filePath));
        return false;
    }

    if (!detector_ || !detector_->isInitialized()) {
        QMessageBox::warning(this, "Chưa Khởi Tạo Detector",
            "Mô hình CAD template detector chưa được khởi tạo!");
        return false;
    }

    // Thực thi nhận diện 2D thuần túy trên ảnh ("chỉ detect thôi")
    lastMatch_ = detector_->detect(rawImage_);

    // Cập nhật giao diện
    updateTelemetryUi();
    renderPreview();

    if (saveImageBtn_) {
        saveImageBtn_->setEnabled(true);
    }

    emit detectionCompleted(filePath, lastMatch_.valid, lastMatch_.latency_ms);
    return true;
}

void ImageDetectionDialog::updateTelemetryUi()
{
    QFileInfo fi(currentFilePath_);
    QString fileInfo = QString(
        "<b>Tên tệp:</b> %1<br>"
        "<b>Định dạng:</b> %2<br>"
        "<b>Độ phân giải:</b> %3 x %4 px")
        .arg(fi.fileName())
        .arg(fi.suffix().toUpper())
        .arg(rawImage_.cols)
        .arg(rawImage_.rows);
    fileInfoLabel_->setText(fileInfo);

    if (lastMatch_.valid) {
        statusBadgeLabel_->setText("🎯 PHÁT HIỆN THÀNH CÔNG (DETECTED)");
        statusBadgeLabel_->setStyleSheet(
            "background: #1b5e20; color: #a5d6a7; font-size: 12px; font-weight: bold; padding: 8px; border-radius: 4px; border: 1px solid #2e7d32;");

        int pct = static_cast<int>(lastMatch_.match_score * 100.0f);
        scoreLabel_->setText(QString("%1%").arg(pct));
        scoreProgressBar_->setValue(pct);

        yawLabel_->setText(QString("~ %1°").arg(lastMatch_.estimated_yaw_deg, 0, 'f', 1));
        matchesLabel_->setText(QString("%1 đặc trưng (%2 inliers)").arg(lastMatch_.good_matches).arg(lastMatch_.ransac_inliers));

        bboxLabel_->setText(QString("[%1, %2, %3, %4]")
            .arg(static_cast<int>(lastMatch_.roi.x))
            .arg(static_cast<int>(lastMatch_.roi.y))
            .arg(static_cast<int>(lastMatch_.roi.width))
            .arg(static_cast<int>(lastMatch_.roi.height)));

        latencyLabel_->setText(QString("%1 ms").arg(lastMatch_.latency_ms, 0, 'f', 1));
    } else {
        statusBadgeLabel_->setText("❌ KHÔNG TÌM THẤY ĐỐI TƯỢNG (NOT FOUND)");
        statusBadgeLabel_->setStyleSheet(
            "background: #4a1414; color: #ef9a9a; font-size: 12px; font-weight: bold; padding: 8px; border-radius: 4px; border: 1px solid #c62828;");

        scoreLabel_->setText("0.0%");
        scoreProgressBar_->setValue(0);
        yawLabel_->setText("-");
        matchesLabel_->setText(QString("%1 đặc trưng (dưới ngưỡng)").arg(lastMatch_.good_matches));
        bboxLabel_->setText("-");
        latencyLabel_->setText(QString("%1 ms").arg(lastMatch_.latency_ms, 0, 'f', 1));
    }
}

void ImageDetectionDialog::renderPreview()
{
    if (rawImage_.empty()) return;

    bool showBbox = showBboxCheck_ && showBboxCheck_->isChecked();
    bool showKpts = showKeypointsCheck_ && showKeypointsCheck_->isChecked();
    bool showHud  = showHudCheck_ && showHudCheck_->isChecked();

    if (showBbox) {
        annotatedImage_ = CADTemplateDetector::annotate(
            rawImage_, lastMatch_, cadName_.toStdString(), showKpts, showHud);
    } else {
        annotatedImage_ = rawImage_.clone();
    }

    // Convert OpenCV BGR Mat to QImage
    cv::Mat rgb;
    cv::cvtColor(annotatedImage_, rgb, cv::COLOR_BGR2RGB);
    QImage qimg(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);

    // Kích thước Viewport động để ảnh hiển thị căng nét, không bị thu nhỏ đốm bé
    QSize vpSize = scrollArea_->viewport()->size();
    if (vpSize.width() < 120 || vpSize.height() < 120) {
        vpSize = scrollArea_->size();
    }
    if (vpSize.width() < 120 || vpSize.height() < 120) {
        vpSize = QSize(750, 550);
    }
    QSize targetSize = vpSize - QSize(16, 16);
    QPixmap pixmap = QPixmap::fromImage(qimg).scaled(targetSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    imagePreviewLabel_->setPixmap(pixmap);
    imagePreviewLabel_->resize(pixmap.size());
}

void ImageDetectionDialog::showEvent(QShowEvent* event)
{
    QDialog::showEvent(event);
    renderPreview();
}

void ImageDetectionDialog::resizeEvent(QResizeEvent* event)
{
    QDialog::resizeEvent(event);
    renderPreview();
}

void ImageDetectionDialog::onDisplayOptionsChanged()
{
    renderPreview();
}

void ImageDetectionDialog::onOpenAnotherImage()
{
    QString dir = currentFilePath_.isEmpty() ? QDir::homePath() : QFileInfo(currentFilePath_).absolutePath();
    QString filter = "Ảnh (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.pbm *.pgm *.ppm);;Tất cả tệp (*)";
    QString newPath = QFileDialog::getOpenFileName(
        this, "Chọn tệp ảnh nhận diện (PNG, JPG, BMP, WEBP...)", dir, filter);

    if (!newPath.isEmpty()) {
        processImage(newPath);
    }
}

void ImageDetectionDialog::onSaveAnnotatedImage()
{
    if (annotatedImage_.empty()) return;

    QFileInfo fi(currentFilePath_);
    QString defaultName = fi.baseName() + "_detected.png";
    QString saveDir = fi.absolutePath() + "/" + defaultName;

    QString savePath = QFileDialog::getSaveFileName(
        this, "Lưu ảnh kết quả nhận diện", saveDir, "PNG (*.png);;JPEG (*.jpg);;All Files (*)");

    if (!savePath.isEmpty()) {
        if (cv::imwrite(savePath.toStdString(), annotatedImage_)) {
            QMessageBox::information(this, "Lưu Ảnh Thành Công",
                QString("Đã lưu ảnh kết quả nhận diện tại:\n%1").arg(savePath));
        } else {
            QMessageBox::warning(this, "Lỗi Lưu Ảnh", "Không thể ghi tệp ảnh vào đường dẫn đã chọn.");
        }
    }
}
