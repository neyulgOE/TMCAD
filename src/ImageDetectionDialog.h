#pragma once

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QProgressBar>
#include <QScrollArea>
#include <QString>
#include <QImage>
#include <opencv2/core.hpp>

#include "CADTemplateDetector.h"

class ImageDetectionDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ImageDetectionDialog(QWidget* parent = nullptr);
    ~ImageDetectionDialog() override = default;

    /**
     * @brief Set target CAD information and detector instance.
     */
    void setDetector(std::shared_ptr<CADTemplateDetector> detector, const QString& cadName);

    /**
     * @brief Load and run 2D detection on the specified image file.
     */
    bool processImage(const QString& filePath);

signals:
    void detectionCompleted(const QString& filePath, bool success, double latencyMs);

private slots:
    void onOpenAnotherImage();
    void onSaveAnnotatedImage();
    void onDisplayOptionsChanged();

protected:
    void showEvent(QShowEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    void setupUi();
    void updateTelemetryUi();
    void renderPreview();

private:
    std::shared_ptr<CADTemplateDetector> detector_;
    QString cadName_ = "TABLE-DEMO";
    QString currentFilePath_;

    cv::Mat rawImage_;
    cv::Mat annotatedImage_;
    CADTemplateDetector::TemplateMatch lastMatch_;

    // UI Widgets
    QLabel* imagePreviewLabel_ = nullptr;
    QScrollArea* scrollArea_ = nullptr;

    QLabel* statusBadgeLabel_ = nullptr;
    QLabel* fileInfoLabel_ = nullptr;
    QLabel* cadModelLabel_ = nullptr;
    QLabel* scoreLabel_ = nullptr;
    QProgressBar* scoreProgressBar_ = nullptr;
    QLabel* yawLabel_ = nullptr;
    QLabel* matchesLabel_ = nullptr;
    QLabel* bboxLabel_ = nullptr;
    QLabel* latencyLabel_ = nullptr;

    QCheckBox* showBboxCheck_ = nullptr;
    QCheckBox* showKeypointsCheck_ = nullptr;
    QCheckBox* showHudCheck_ = nullptr;

    QPushButton* openAnotherBtn_ = nullptr;
    QPushButton* saveImageBtn_ = nullptr;
    QPushButton* closeBtn_ = nullptr;
};
