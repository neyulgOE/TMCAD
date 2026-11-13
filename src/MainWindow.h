#pragma once

#include <QMainWindow>
#include <QDateTime>
#include <QDialog>
#include <QString>
#include <QPlainTextEdit>
#include <QScrollBar>
#include "SceneManager.h"
#include "Viewport3D.h"
#include "PoseEstimator.h"
#include "StudioUxControls.h"
#include "UniversalPoseEstimator.h"

class QListWidget;
class QListWidgetItem;
class QDoubleSpinBox;
class QSpinBox;
class QCheckBox;
class QPushButton;
class QLabel;
class QLineEdit;
class QComboBox;
class QTabWidget;
class QSplitter;
class RealSenseSource;
class RealSenseBagSource;
class CameraStreamWidget;
class CADTemplateDetector;
class ImageDetectionDialog;
class MuJoCoD435SimSource;
class CameraProfileDialog;
#include "CameraHardwareProfile.h"
#include <QProcess>
#include <QProgressDialog>
#include <future>
#include <atomic>
#include <QTimer>
#include <QSlider>
#include <atomic>

// ---------------------------------------------------------------------------
// PoseConfigDialog
// ---------------------------------------------------------------------------
// UI without recompiling. Shown before every "Estimate Pose" run.
// ---------------------------------------------------------------------------
class PoseConfigDialog : public QDialog {
    Q_OBJECT
public:
    explicit PoseConfigDialog(const PoseEstimatorConfig& initial, QWidget* parent = nullptr);

    // Returns the config as edited by the user (only valid if exec() == Accepted).
    PoseEstimatorConfig config() const;

    // Resets all fields to the defaults in PoseEstimatorConfig{}.
    void resetDefaults();

private:
    // Scene preprocessing
    QDoubleSpinBox* voxelSizeSpin_;
    QDoubleSpinBox* outlierStdSpin_;
    QSpinBox*       outlierNeighSpin_;

    // FPFH radii
    QDoubleSpinBox* normalRadiusSpin_;
    QDoubleSpinBox* fpfhRadiusSpin_;

    // RANSAC
    QDoubleSpinBox* ransacDistSpin_;
    QSpinBox*       ransacIterSpin_;
    QDoubleSpinBox* ransacConfSpin_;

    // ICP
    QCheckBox*      icpEnabledCheck_;
    QDoubleSpinBox* icpDistSpin_;

    // Crop
    QCheckBox*      cropEnabledCheck_;
    QDoubleSpinBox* cropMinX_; QDoubleSpinBox* cropMinY_; QDoubleSpinBox* cropMinZ_;
    QDoubleSpinBox* cropMaxX_; QDoubleSpinBox* cropMaxY_; QDoubleSpinBox* cropMaxZ_;
};

// ---------------------------------------------------------------------------
// SyntheticTestParams / SyntheticSceneDialog — lets the user drive
// SyntheticRGBDSource from the UI: place the CAD object at a chosen
// distance/rotation in front of a virtual camera, optionally add
// background-plane clutter and depth noise/dropout, without needing a
// live RealSense or pre-captured image pair. Purpose-built for sweeping
// PoseConfigDialog's crop settings against a scene whose ground-truth
// pose is known exactly and reproducible run-to-run (see roadmap Phase 2
// notes on why live-camera crop testing can't offer that).
// ---------------------------------------------------------------------------
struct SyntheticTestParams {
    double distanceMeters = 1.5;   // object placed this far along camera +Z
    double rotXDeg = 0.0;
    double rotYDeg = 0.0;
    double rotZDeg = 0.0;          // object rotation applied before placement
    double cameraHeightMeters = 0.3; // AMR camera height

    // Realistic active-stereo (D435-style) depth model -- see
    // SyntheticSceneConfig in SyntheticRGBDSource.h for the physics behind
    // each field. Defaults are approximate starting points, not
    // guaranteed factory numbers for this exact unit -- tune against a
    // measured error curve if you need precision.
    bool useRealisticNoiseModel = true;
    double baselineMeters = 0.05;
    double subpixelDisparityErrorPx = 0.08;
    double maxIncidenceAngleDeg = 85.0;
    double minRangeMeters = 0.2;
    double maxRangeMeters = 10.0;

    // Flat/toy noise, used only when useRealisticNoiseModel is off.
    double depthNoiseStdMeters = 0.0;
    // Extra unstructured per-pixel dropout stacked on top of either model.
    double randomDropoutFraction = 0.0;  // NOT Phase 5's visibility grid -- see SyntheticRGBDSource.h

    bool addBackgroundPlane = false;
    double backgroundPlaneDistanceMeters = 3.0;
    double backgroundPlaneSizeMeters = 6.0;

    // Camera intrinsics for the virtual sensor -- defaults match the
    // actual factory-calibrated RealSense D435f on the bench (see
    // MainWindow's camera panel: fx=386.652 fy=386.652 cx=326.089
    // cy=240.785 @ 640x480), NOT a generic guess -- using the real
    // numbers here matters because fx/fy directly set how many pixels
    // (and therefore how many points) the object occupies at a given
    // distance, which feeds straight into whether FPFH has enough
    // neighbors to compute a descriptor.
    int width = 640;
    int height = 480;
    double fx = 386.652;
    double fy = 386.652;
    double cx = 326.089;
    double cy = 240.785;
};

class SyntheticSceneDialog : public QDialog {
    Q_OBJECT
public:
    explicit SyntheticSceneDialog(const SyntheticTestParams& initial, QWidget* parent = nullptr);
    SyntheticTestParams params() const;

private:
    QDoubleSpinBox* distanceSpin_;
    QDoubleSpinBox* rotXSpin_;
    QDoubleSpinBox* rotYSpin_;
    QDoubleSpinBox* rotZSpin_;
    QDoubleSpinBox* cameraHeightSpin_;

    QCheckBox* realisticNoiseCheck_;
    QDoubleSpinBox* baselineSpin_;
    QDoubleSpinBox* subpixelErrorSpin_;
    QDoubleSpinBox* maxIncidenceSpin_;
    QDoubleSpinBox* minRangeSpin_;
    QDoubleSpinBox* maxRangeSpin_;

    QDoubleSpinBox* noiseStdSpin_;
    QDoubleSpinBox* dropoutSpin_;

    QCheckBox* bgPlaneCheck_;
    QDoubleSpinBox* bgDistanceSpin_;
    QDoubleSpinBox* bgSizeSpin_;

    QSpinBox* widthSpin_;
    QSpinBox* heightSpin_;
    QDoubleSpinBox* fxSpin_;
    QDoubleSpinBox* fySpin_;
    QDoubleSpinBox* cxSpin_;
    QDoubleSpinBox* cySpin_;
    QLabel* fovLabel_;  // live-computed H/V FOV from fx/fy/width/height, read-only

    void updateFovLabel();
};

// ---------------------------------------------------------------------------
// StaticRGBDTestDialog — Kiểm thử nhận diện 6DoF với cặp ảnh tĩnh RGB-D
// Tự động nhận diện ảnh depth đi kèm và thông số camera intrinsics
// ---------------------------------------------------------------------------
class StaticRGBDTestDialog : public QDialog {
    Q_OBJECT
public:
    explicit StaticRGBDTestDialog(QWidget* parent = nullptr, const QString& activeModel = QString());

    QString colorPath() const;
    QString depthPath() const;
    CameraIntrinsics intrinsics() const;
    double depthScale() const;

private slots:
    void onBrowseColor();
    void onBrowseDepth();
    void onLoadDumpPreset();
    void onLoadHopePreset();
    void onLoadDatasetPreset();
    void onColorPathChanged(const QString& path);

private:
    void autoDetectCompanion(const QString& colorPath);

    QLineEdit* colorEdit_ = nullptr;
    QLineEdit* depthEdit_ = nullptr;
    QSpinBox* widthSpin_ = nullptr;
    QSpinBox* heightSpin_ = nullptr;
    QDoubleSpinBox* fxSpin_ = nullptr;
    QDoubleSpinBox* fySpin_ = nullptr;
    QDoubleSpinBox* cxSpin_ = nullptr;
    QDoubleSpinBox* cySpin_ = nullptr;
    QDoubleSpinBox* scaleSpin_ = nullptr;
    QLabel* statusLabel_ = nullptr;
};

// ---------------------------------------------------------------------------
// MainWindow
// ---------------------------------------------------------------------------
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;
    void loadCADFile(const QString& path);
    void selectTab(int index);
    void toggleColorRegions() { onToggleColorRegions(); }

private slots:
    void onAddFile();
    void onRemoveSelected();
    void onItemChanged(QListWidgetItem* item);
    void onObjectListChanged();
    void onPlaceOnGrid();
    void onViewportPicked(int id);
    void onTransformChanged();

    void onNormalize();
    void onComputeNormals();
    void onPoissonSample();
    void onShowNormalsToggled(bool checked);
    void onShowAxesToggled(bool checked);
    void onExportSelected();
    void onNormalLengthChanged(double value);
    void onSaveOrientationAsDefault();  // writes current Rot X/Y/Z to "<file>.orientation"
    void onEstimatePoseRealSense();   // Phase 2 — live camera
    void onEstimatePoseRealSenseBag(); // Phase 8 — from recorded .bag file
    void onRecordRealSense();         // Phase 8 — launch D435if recording tool
    void onEstimatePoseFromFile();    // Phase 2 — file-based (no camera needed)
    void onDetectOnImage();           // 2D Object Detection on Image (PNG, JPG, BMP, WEBP...)
    void onEstimatePoseSynthetic();   // Phase 2 — ray-cast synthetic scene, known ground truth
    void onGenerateSyntheticDataset(); // Phase 3/5 — batch dataset (SyntheticDatasetGenerator)
    void onSelectDatasetFile();        // Explicitly choose dataset file (train.txt / dataset_manifest.json)
    void onSelectDatasetDir();         // Explicitly choose dataset directory
    void onTrainAIModel();             // Phase 6 — train GeoPoseNet on geometric surface keypoints
    void onExportOrangePiRKNN();       // Export AI model to RKNN for Orange Pi 5B (RK3588S)
    void onSimulateRKNN();             // Run RKNN NPU PC simulator benchmark
    void onClearCADCache();            // Reset/clear dataset & trained model cache for active CAD
    void onManageCameraProfiles();     // Dynamic Camera Profile Ingestion (D435if, D455, custom)
    void onStartMuJoCoSimulation();    // Run Closed-loop 6DoF Pose Detection Simulation with MuJoCo C++
    void onTrainingStdout();
    void onTrainingStderr();
    void onTrainingFinished(int exitCode, QProcess::ExitStatus exitStatus);

    void onQuickLoadCad(const QString& cadName);
    void onClearAllObjects();
    void onRemoveSelectedCadObject();
    void onReplaceActiveCadObject();
    void onResetAllTasksAndSession();
    void loadRealSenseBagFile(const QString& bagPath);
    void onOpenBagForStepping();
    void stepNextBagFrame();
    void onToggleBagAutoStep();
    void restartBagStepping();
    void openBagForStepping(const QString& bagPath);
    void updateBagTelemetryUi(int frameIdx, double totalFrameMs, bool success);

    // Phase 0 & 1 Canonical CAD Hierarchy & Symmetry slots
    void onBuildCADHierarchy();
    void onSymmetryConfig();
    void onToggleColorRegions();
    void onRunOracleCheck();
    void onRunBaselineBenchmark();

    // Workspace mode switching & UX Studio toolbar
    void onSwitchToSimulationMode();
    void onSwitchToCameraMode();
    void onEstimatePoseLiveSnapshot();
    void onToggleLiveContinuousTracking(bool checked);
    void onLiveTrackingTimerTick();
    void onBrowseOnnxModel();
    void onActiveOnnxModelSwitched(const QString& onnxPath);
    void onInteractiveRoiChanged(bool active);
    void onAutoBuildPipelineRequested();

protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;

private:
    void buildUi();
    void syncPropertiesUi();
    QString prepareCadModelFolder(const QString& sourcePath);
    void refreshOnnxModelList(const QString& selectPath = QString());
    QString selectedOnnxPath() const;
    QString parseDatasetDirFromFile(const QString& path) const;
    QString resolveDatasetDir(const QString& cadName, const QString& explicitDir = QString()) const;
    QString projectRootDir() const;
    bool buildCadHierarchyForObject(SceneObject* obj, bool showSuccessDialog);
    void runOneClickAutoPipeline(const QString& cadFilePath);
    void cacheStaticFrameForRoi(const RGBDFrame& frame,
                                const CameraIntrinsics& intrinsics,
                                const QString& runLabel,
                                const Eigen::Matrix4d* groundTruthCameraFromObject);

    // Shared pose-estimation logic called by onEstimatePoseRealSense(),
    // onEstimatePoseFromFile(), and onEstimatePoseSynthetic(). Returns
    // false if the user cancels the config dialog or no mesh is selected.
    //
    // groundTruthCameraFromObject: optional, non-null only for the
    // synthetic path (a live/file camera has no ground truth to compare
    // against). When provided and the estimate succeeds, e_t/e_R are
    // computed against it using Phase 5's metric formulas and shown
    // alongside the usual fitness/RMSE numbers.
    //
    // runLabel: free-form description of this run (source path/params),
    // written as-is into the "run_label" column of the CSV log
    // (appendPoseRunLog) so sweeps can be told apart later without
    // re-deriving them from a timestamp. Empty by default -- callers that
    // don't care about labeling a sweep can omit it.
    bool runPoseEstimation(const RGBDFrame& frame,
                            const CameraIntrinsics& intrinsics,
                            const PoseEstimatorConfig& cfg,
                            const Eigen::Matrix4d* groundTruthCameraFromObject = nullptr,
                            const QString& runLabel = QString());

    // Phase 6/7 Deep Learning Pose Estimation (Black-Box ONNX C++ Runtime + Python fallback)
    bool runPoseEstimationAI(const RGBDFrame& frame,
                              const CameraIntrinsics& intrinsics,
                              const Eigen::Matrix4d* groundTruthCameraFromObject = nullptr,
                              const QString& runLabel = QString(),
                              bool interactiveRoiRerun = false);

    // Appends one row to /home/huyen1905/TMCAD/logs/pose_runs.csv,
    // creating the file (and its directory) with a header row on first
    // use. Called from runPoseEstimation() for EVERY run, including
    // rejected ones (result.valid == false) -- a failed sweep point still
    // needs its point counts/diagnostics logged to tell WHY it failed
    // apart from every other failed row.
    void appendPoseRunLog(const QString& runLabel,
                           const CADAsset& cadAsset,
                           const PoseEstimatorConfig& cfg,
                           const Pose6DoFResult& result,
                           double e_t = -1.0,
                           double e_R_deg = -1.0);

    SceneManager* scene_;
    Viewport3D* viewport_;
    QListWidget* objectList_;

    // Model Manager UI controls
    QPushButton* addFileBtn_ = nullptr;
    QPushButton* quickLoadBtn_ = nullptr;
    QPushButton* removeSelectedBtn_ = nullptr;
    QPushButton* clearAllBtn_ = nullptr;
    QLabel*      objectCountLabel_ = nullptr;

    // Tabbed workflow container
    QTabWidget*  workflowTabs_ = nullptr;

    // Tab 1: CAD & Topology (Phase 0 & 1)
    QDoubleSpinBox* posX_ = nullptr;
    QDoubleSpinBox* posY_ = nullptr;
    QDoubleSpinBox* posZ_ = nullptr;
    QDoubleSpinBox* rotX_ = nullptr;
    QDoubleSpinBox* rotY_ = nullptr;
    QDoubleSpinBox* rotZ_ = nullptr;
    QPushButton* saveOrientationBtn_ = nullptr;  // writes Rot X/Y/Z to "<file>.orientation" sidecar
    bool updatingUi_ = false;

    QPushButton* normalizeBtn_ = nullptr;
    QPushButton* computeNormalsBtn_ = nullptr;
    QPushButton* poissonSampleBtn_ = nullptr;
    QPushButton* exportBtn_ = nullptr;
    QPushButton* buildHierarchyBtn_ = nullptr;   // Phase 1: Build Canonical Hierarchy (.npz, .ply, .json)
    QPushButton* symmetryConfigBtn_ = nullptr;   // Phase 0: Symmetry config dialog
    QPushButton* colorRegionsBtn_ = nullptr;     // Phase 1: Color mesh by regions
    QPushButton* clearCacheBtn_ = nullptr;       // Wipe dataset & checkpoint cache for active CAD
    QLabel*      cadStatusLabel_ = nullptr;      // Status: Dataset [OK/None] | Model [Trained/None]
    QLabel*      cadHierarchyLabel_ = nullptr;   // Summary: Tris | Edges | Regions | Symmetry
    bool         colorRegionsActive_ = false;

    // Tab 2: Synthetic & Oracle (Phase 1.5 & 2)
    QPushButton* selectDatasetBtn_ = nullptr;    // Explicitly choose dataset file (train.txt / manifest)
    QPushButton* selectDatasetDirBtn_ = nullptr; // Explicitly choose dataset directory
    QPushButton* generateDatasetBtn_ = nullptr;  // Phase 1.5: batch synthetic dataset generation
    QPushButton* runOracleBtn_ = nullptr;        // Phase 2: Oracle geometry check
    QLabel*      oracleResultLabel_ = nullptr;

    // Tab 3: Training & Baselines (Phase 3 & 4)
    QPushButton* trainAiBtn_ = nullptr;          // Phase 4/6: train AI model for active CAD
    QPushButton* exportRknnBtn_ = nullptr;       // Export AI model to RKNN for Orange Pi 5B
    QPushButton* simRknnBtn_ = nullptr;          // Run RKNN NPU PC simulator benchmark
    QPushButton* runBaselineBtn_ = nullptr;      // Phase 3: run FPFH baseline

    // Tab 4: 6DoF Pose & Diagnostics (Phase 5, 6 & 7)
    QComboBox*   poseMethodCombo_ = nullptr;     // Phase 6/7: AI GeoPoseNet vs Classical FPFH
    QComboBox*   onnxModelCombo_ = nullptr;      // Black-Box .ONNX model selector
    QLabel*      onnxStatusLabel_ = nullptr;     // Black-Box .ONNX model status
    QPushButton* estimatePoseBtn_ = nullptr;
    QPushButton* estimatePoseBagBtn_ = nullptr;     // Phase 8: test path from recorded RealSense .bag file
    QPushButton* recordRealSenseBtn_ = nullptr;     // Phase 8: launch D435if recording tool
    QPushButton* estimatePoseFileBtn_ = nullptr; // test path: file-based RGBD
    QPushButton* detectOnImageBtn_ = nullptr;    // 2D Object Detection on Image (PNG, JPG, BMP, WEBP...)
    QPushButton* estimatePoseSyntheticBtn_ = nullptr; // test path: ray-cast synthetic RGBD, known ground truth
    QLabel*      poseResultSummaryLabel_ = nullptr;

    QString      lastImageDir_;
    std::shared_ptr<CADTemplateDetector> imageDetector_;

    QProcess*    trainProcess_ = nullptr;
    QProgressDialog* trainProgressDlg_ = nullptr;
    QString      currentTrainingCadName_;
    QMap<QString, QString> cadDatasetDirMap_;
    QMap<QString, QString> cadDatasetFileMap_;

    // v2 polish
    QLabel* pointCountLabel_ = nullptr;
    QDoubleSpinBox* normalLengthSpin_ = nullptr;

    // Persisted config so the dialog reopens with the last-used values.
    PoseEstimatorConfig lastPoseConfig_;
    SyntheticTestParams lastSyntheticParams_;

    // In-window log panel — shows real-time pipeline steps + a
    // summary block per run (from appendPoseRunLog).
    QPlainTextEdit* logPanel_ = nullptr;

    // High performance live streaming & persistent AI inference
    std::unique_ptr<RealSenseSource> warmRealSense_;
    std::unique_ptr<MuJoCoD435SimSource> activeMuJoCoSim_;
    QTimer* muJoCoSimTimer_ = nullptr;
    std::atomic<bool> isMuJoCoInferencing_{false};
    int muJoCoFrameSkip_ = 0;
    std::unique_ptr<tmcad::UniversalPoseEstimator> liveCppEstimator_;
    std::string liveLoadedOnnxPath_;
    std::future<tmcad::ObjectPose> asyncMuJoCoPoseFuture_;
    std::atomic<bool> asyncWorkerBusy_{false};
    int muJoCoTickCounter_{0};
    CameraIntrinsics lastAsyncIntr_;
    tmcad::ObjectPose lastMuJoCoPose_;
    QPushButton* replaceCadBtn_ = nullptr;
    QPushButton* resetSessionBtn_ = nullptr;
    void onMuJoCoSimTick();
    void stopMuJoCoSimulation();
    CameraHardwareProfile currentCamProfile_;
    int currentCamProfileId_ = 1;
    std::unique_ptr<QProcess> aiDaemonProcess_;
    QString currentDaemonCadPath_;
    QString currentDaemonCkptPath_;
    QDateTime currentDaemonScriptMTime_;
    void stopAiDaemon();

    struct LastAiTelemetry {
        bool valid = false;
        bool fromCppNative = false;
        double aiMs = 0.0;
        double solveMs = 0.0;
        double totalMs = 0.0;
        double tx = 0.0, ty = 0.0, tz = 0.0;
        double roll = 0.0, pitch = 0.0, yaw = 0.0;
        double fitness = 0.0;
        double rmseMm = 0.0;
        int inliers = 0;
        int clusterPts = 0;
        int validDepthPts = 0;
        int numInstances = 0;
        bool floorLocked = false;
        double floorPitchDeg = 0.0;
    };

    std::unique_ptr<tmcad::UniversalPoseEstimator> universalPoseEstimator_;
    tmcad::StudioUxToolbar* uxToolbar_ = nullptr;
    QAction* toggleCamViewAct_ = nullptr;
    QProcess* autoBuildProcess_ = nullptr;
    QProgressDialog* autoBuildProgressDlg_ = nullptr;
    std::atomic<bool> autoBuildCanceled_{false};
    bool lastMultiInstanceFlag_ = true;
    bool isInteractiveRoiUpdating_ = false;

    RGBDFrame lastStaticFrame_;
    CameraIntrinsics lastStaticIntrinsics_;
    QString lastStaticRunLabel_;
    Eigen::Matrix4d lastStaticGtPose_ = Eigen::Matrix4d::Identity();
    bool hasLastStaticGtPose_ = false;
    LastAiTelemetry lastAiTelemetry_;

    // ── Workspace Modes (Mô Phỏng vs Camera Thực Tế) ──────────────────────
    enum class WorkspaceMode {
        SimulationStudio,
        LiveCameraStudio
    };

    WorkspaceMode currentMode_ = WorkspaceMode::SimulationStudio;
    QPushButton* simModeBtn_ = nullptr;
    QPushButton* cameraModeBtn_ = nullptr;
    QWidget* simControlPanel_ = nullptr;
    QWidget* liveCameraControlPanel_ = nullptr;
    CameraStreamWidget* liveCameraStreamWidget_ = nullptr;
    QSplitter* mainViewSplitter_ = nullptr;

    QPushButton* liveSnapshotPoseBtn_ = nullptr;
    QCheckBox*   liveContinuousTrackingCheck_ = nullptr;
    QTimer*      liveTrackingTimer_ = nullptr;
    std::atomic<bool> isLiveTrackingInferencing_{false};
    QComboBox*   livePoseMethodCombo_ = nullptr;
    QLabel*      liveTargetCadLabel_ = nullptr;
    QLabel*      livePoseResultCard_ = nullptr;

    // ── RealSense .bag Frame-by-Frame Stepper & Benchmarking ─────────────
    std::unique_ptr<RealSenseBagSource> activeBagSource_;
    QString      activeBagPath_;
    int          activeBagFrameIndex_ = 0;
    std::atomic<bool> isBagSteppingInferencing_{false};
    bool         isBagAutoStepping_ = false;
    double       bagLatencySum_ = 0.0;
    int          bagLatencyCount_ = 0;
    double       bagMinLatency_ = 9999.0;
    double       bagMaxLatency_ = 0.0;
    QTimer*      bagAutoStepTimer_ = nullptr;
    QLabel*      bagStatusLabel_ = nullptr;
    QLabel*      bagTelemetryCard_ = nullptr;
    QSlider*     bagProgressSlider_ = nullptr;
    QLabel*      bagProgressLabel_ = nullptr;
    QPushButton* bagNextFrameBtn_ = nullptr;
    QPushButton* bagAutoStepBtn_ = nullptr;
    QPushButton* bagRestartBtn_ = nullptr;

    void switchWorkspaceMode(WorkspaceMode mode);

public slots:
    // Appends a colored line to logPanel_.
    void pipelineLog(const QString& line);
};