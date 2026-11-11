#pragma once

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QImage>
#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>

class RealSenseRecorderDialog : public QDialog {
    Q_OBJECT

public:
    explicit RealSenseRecorderDialog(QWidget* parent = nullptr);
    ~RealSenseRecorderDialog() override;

signals:
    void frameReady(const QImage& colorImg, const QImage& depthImg, float centerDist);

private slots:
    void onFrameReady(const QImage& colorImg, const QImage& depthImg, float centerDist);
    void onToggleRecord();
    void onTakeSnapshot();
    void onBrowseOutputDir();

private:
    void setupUi();
    bool startPipeline(const std::string& recordBagPath = "");
    void stopPipeline();
    void captureWorker();

    struct Impl;
    std::unique_ptr<Impl> impl_;

    QLabel* colorViewLabel_ = nullptr;
    QLabel* depthViewLabel_ = nullptr;
    QLabel* statusLabel_ = nullptr;
    QLabel* imuLabel_ = nullptr;
    QLabel* centerDistLabel_ = nullptr;
    QLineEdit* outputDirEdit_ = nullptr;
    QPushButton* recordBtn_ = nullptr;
    QPushButton* snapshotBtn_ = nullptr;

    std::thread captureThread_;
    std::atomic<bool> captureRunning_{false};

    bool isRecording_ = false;
    std::string currentBagPath_;
    int frameCount_ = 0;
    int snapshotCount_ = 0;
    double lastFps_ = 0.0;
    int64_t lastFpsTime_ = 0;
};

