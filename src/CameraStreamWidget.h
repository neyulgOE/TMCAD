#pragma once

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QImage>
#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <QRectF>
#include <QPointF>
#include <QPolygonF>
#include <QColor>
#include <QString>
#include "RGBDTypes.h"
#include "StudioUxControls.h"
#include "UniversalPoseEstimator.h"

class QComboBox;
class QGroupBox;
class QEvent;

struct CadOverlaySegment {
    QPointF p0;
    QPointF p1;
    QColor color;
    float thickness = 2.0f;
};

struct CadOverlayData {
    QRectF bbox;
    std::vector<CadOverlaySegment> segments;          // 3D Oriented Bounding Box (OBB) & generic segments
    std::vector<CadOverlaySegment> wireframeSegments; // Embedded CAD crease / feature wireframe segments
    std::vector<QPointF> segmentedCloudPoints2D;      // Projected segmented cluster points in image coords
    QPolygonF topSurfacePolygon;
    std::vector<QPointF> footPoints;
    QPointF centerAxis;
    QPointF axisX;
    QPointF axisY;
    QPointF axisZ;
    bool hasAxes = false;
    QPointF tableCenter;
    bool hasTableCenter = false;
    QString label;
    QString poseBanner;
    QString headerBanner;
    bool isPhysicalStlMode = false;
    bool valid = false;
};

class CameraStreamWidget : public QWidget {
    Q_OBJECT

public:
    explicit CameraStreamWidget(QWidget* parent = nullptr, bool compactMode = false);
    ~CameraStreamWidget() override;

    bool startStream(const std::string& recordBagPath = "");
    bool startPlayback(const std::string& bagPath);
    void stopStream();
    bool isStreaming() const;
    bool isRecording() const;
    bool isPlayback() const { return isPlayback_; }

    // Lấy nhanh khung hình RGB-D và thông số nội suy camera trực tiếp từ RAM (< 5ms)
    bool getLatestRGBDFrame(RGBDFrame& outFrame, CameraIntrinsics& outIntr);

    // V2.5 / TMVis Synchronized: 3D Wireframe metric projection using exact CAD geometry and Camera Intrinsics
    void setPoseOverlay3D(const Eigen::Matrix3d& R,
                          const Eigen::Vector3d& t,
                          const CameraIntrinsics& intr,
                          const QString& label,
                          const QString& poseBanner,
                          double fitness = 0.94,
                          const Eigen::Vector3d& cadMin = Eigen::Vector3d::Zero(),
                          const Eigen::Vector3d& cadMax = Eigen::Vector3d::Zero());

    // Universal Black-Box ONNX Multi-Instance Overlay (renders OBB + CAD feature wireframe + axes + segmented cloud)
    void setUniversalPoseOverlays(const std::vector<tmcad::ObjectPose>& poses,
                                  const tmcad::UniversalPoseEstimator& estimator,
                                  const CameraIntrinsics& intr,
                                  const QString& telemetryBanner = QString(),
                                  const QString& headerBanner = QString());

    void setOverlayRenderFlags(const tmcad::OverlayRenderFlags& flags);
    tmcad::OverlayRenderFlags overlayRenderFlags() const noexcept { return renderFlags_; }

    bool hasActiveRoi() const noexcept;
    void clearInteractiveRoi();
    cv::Rect currentRoiHintImageSpace(int img_w, int img_h) const;
    void redrawCurrentFrame();

    void setCadOverlay(const CadOverlayData& data);
    void setOverlayDetection(const QRectF& bbox,
                             const std::vector<QPointF>& corners3DProj,
                             const QString& label,
                             const QString& poseText);
    void setOverlayDetection(const QRectF& bbox, const QString& label, const QString& poseText);
    void clearOverlayDetection();
    void setStaticImageDetection(const QImage& rawColorImg,
                                 const QImage& annotatedImg,
                                 const QRectF& bbox,
                                 const QString& label,
                                 const QString& statusText);

    // Hiển thị khung hình ảnh tĩnh (RGB và Depth) kèm lớp phủ 6DoF Pose
    void displayStaticRGBDFrame(const QImage& colorImg, const QImage& depthImg, float centerDist = 0.0f);

    void setOutputDir(const QString& dir);
    QString outputDir() const;

signals:
    void frameReady(const QImage& colorImg, const QImage& depthImg, float centerDist);
    void recordingStarted(const QString& bagPath);
    void recordingStopped(const QString& bagPath);
    void snapshotSaved(const QString& snapDir);
    void roiSelectionChanged(bool active);

public slots:
    void toggleRecord();
    void takeSnapshot();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private slots:
    void onFrameReady(const QImage& colorImg, const QImage& depthImg, float centerDist);
    void onBrowseOutputDir();
    void onViewModeChanged(int index);

private:
    void setupUi(bool compactMode);
    void captureWorker();

    struct Impl;
    std::unique_ptr<Impl> impl_;

    QLabel* detectionViewLabel_ = nullptr;
    QLabel* colorViewLabel_ = nullptr;
    QLabel* depthViewLabel_ = nullptr;

    QGroupBox* detectionBox_ = nullptr;
    QGroupBox* colorBox_ = nullptr;
    QGroupBox* depthBox_ = nullptr;
    QComboBox* viewModeCombo_ = nullptr;
    QLabel* statusLabel_ = nullptr;
    QLabel* centerDistLabel_ = nullptr;
    QLineEdit* outputDirEdit_ = nullptr;
    QPushButton* recordBtn_ = nullptr;
    QPushButton* snapshotBtn_ = nullptr;
    QPushButton* browseBtn_ = nullptr;

    std::thread captureThread_;
    std::atomic<bool> captureRunning_{false};
    bool isRecording_ = false;
    bool isPlayback_ = false;
    std::string currentBagPath_;

    int frameCount_ = 0;
    int snapshotCount_ = 0;
    double lastFps_ = 0.0;
    int64_t lastFpsTime_ = 0;

    // V2.4 & Universal overlay tracking state
    std::mutex overlayMutex_;
    bool hasOverlay_ = false;
    QRectF overlayBbox_;
    std::vector<QPointF> overlayCorners3DProj_;
    QString overlayLabel_;
    QString overlayPoseText_;
    CadOverlayData overlayCadData_;
    std::vector<CadOverlayData> multiOverlayData_;
    int64_t overlayTimestamp_ = 0;

    // Layer toggles & Interactive 2D ROI state
    tmcad::OverlayRenderFlags renderFlags_;
    bool hasActiveRoi_ = false;
    bool isDraggingRoi_ = false;
    QPointF roiDragStartNorm_;
    QRectF roiNorm_;

    // Cached last frame (GUI thread only) so toggling layers or dragging ROI redraws static frames cleanly
    bool hasLastFrame_ = false;
    QImage lastColorImg_;
    QImage lastDepthImg_;
    float lastCenterDist_ = 0.0f;
};
