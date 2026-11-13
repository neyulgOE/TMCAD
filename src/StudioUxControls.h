#pragma once

#include <QWidget>
#include <QComboBox>
#include <QPushButton>
#include <QCheckBox>
#include <QLabel>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QDir>
#include <QFileInfo>
#include <QCoreApplication>
#include <QMetaType>
#include <QSet>
#include <QString>
#include <QVariant>

namespace tmcad {

struct OverlayRenderFlags {
    bool show_obb_box{true};
    bool show_cad_wireframe{true};
    bool show_axes_and_hud{true};
    bool show_segmented_cloud{false};
    bool multi_instance_mode{false};
};

class StudioUxToolbar : public QWidget {
    Q_OBJECT
public:
    explicit StudioUxToolbar(QWidget* parent = nullptr) : QWidget(parent) {
        setupUi();
        refreshOnnxModelList();
    }

    static QString resolveProjectRoot() {
        QDir dir(QCoreApplication::applicationDirPath());
        for (int i = 0; i < 4; ++i) {
            if (dir.exists("models") || (dir.exists("CAD_DATA") && dir.exists("scripts"))) {
                return dir.absolutePath();
            }
            if (!dir.cdUp()) break;
        }
        QDir cwd = QDir::current();
        if (cwd.exists("models") || (cwd.exists("CAD_DATA") && cwd.exists("scripts"))) {
            return cwd.absolutePath();
        }
        QDir fallback(QCoreApplication::applicationDirPath());
        if (fallback.dirName() == "build") fallback.cdUp();
        return fallback.absolutePath();
    }

    static QString resolveModelsDir(const QString& models_dir_hint = QString()) {
        if (!models_dir_hint.isEmpty() && models_dir_hint != "models") {
            QDir explicitDir(models_dir_hint);
            if (explicitDir.exists()) {
                return explicitDir.absolutePath();
            }
        }
        return QDir(resolveProjectRoot()).absoluteFilePath("models");
    }

    void refreshOnnxModelList(const QString& models_dir_hint = QString(),
                              const QString& preferred_path = QString()) {
        _onnx_combo->blockSignals(true);
        QString current = !preferred_path.isEmpty() ? preferred_path : _onnx_combo->currentData().toString();
        _onnx_combo->clear();

        QSet<QString> seen_canonical;
        QSet<QString> seen_names;

        auto addEntry = [&](const QFileInfo& fi) {
            if (!fi.exists() || !fi.isFile() || fi.suffix().toLower() != "onnx") return;
            const QString fname = fi.fileName();
            if (fname == "model_obs_npu.onnx" || fname == "model_obs_backbone.onnx") return;

            QString canon = fi.canonicalFilePath();
            if (canon.isEmpty()) canon = fi.absoluteFilePath();
            const QString base = fi.completeBaseName();

            if (seen_canonical.contains(canon) || seen_names.contains(base)) return;
            seen_canonical.insert(canon);
            seen_names.insert(base);

            _onnx_combo->addItem("📦 " + base + " (.onnx)", fi.absoluteFilePath());
        };

        // 1. Primary models/ store
        QDir models_dir(resolveModelsDir(models_dir_hint));
        if (models_dir.exists()) {
            const QFileInfoList files = models_dir.entryInfoList(QStringList{"*.onnx"}, QDir::Files, QDir::Name);
            // Process regular files before symlinks so canonical names (e.g. CHINSU.onnx over chinsu.onnx) win
            for (const QFileInfo& fi : files) {
                if (!fi.isSymLink()) addEntry(fi);
            }
            for (const QFileInfo& fi : files) {
                if (fi.isSymLink()) addEntry(fi);
            }
        }

        // 2. Fallback deploy_orange_pi_5b/<cad>/<cad>.onnx store
        QDir deploy_dir(QDir(resolveProjectRoot()).absoluteFilePath("deploy_orange_pi_5b"));
        if (deploy_dir.exists()) {
            for (const QString& sub : deploy_dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name)) {
                QDir sub_dir(deploy_dir.absoluteFilePath(sub));
                for (const QFileInfo& fi : sub_dir.entryInfoList(QStringList{"*.onnx"}, QDir::Files, QDir::Name)) {
                    addEntry(fi);
                }
            }
        }

        // 3. Preserve externally browsed model if applicable
        if (!current.isEmpty() && QFileInfo::exists(current)) {
            addEntry(QFileInfo(current));
        }

        if (_onnx_combo->count() == 0) {
            _onnx_combo->addItem("⚠️ No .onnx in models/ (Click Load .onnx)", "");
        } else if (!current.isEmpty()) {
            const QFileInfo cur_fi(current);
            const QString cur_canon = cur_fi.canonicalFilePath();
            const QString cur_base = cur_fi.completeBaseName();
            int match_idx = -1;
            for (int i = 0; i < _onnx_combo->count(); ++i) {
                const QString item_path = _onnx_combo->itemData(i).toString();
                const QFileInfo item_fi(item_path);
                if (item_path == cur_fi.absoluteFilePath() ||
                    (!cur_canon.isEmpty() && item_fi.canonicalFilePath() == cur_canon) ||
                    (!cur_base.isEmpty() && item_fi.completeBaseName() == cur_base)) {
                    match_idx = i;
                    break;
                }
            }
            if (match_idx >= 0) {
                _onnx_combo->setCurrentIndex(match_idx);
            }
        }
        _onnx_combo->blockSignals(false);
    }

    QString activeOnnxPath() const {
        return _onnx_combo ? _onnx_combo->currentData().toString() : QString();
    }

    void setActiveOnnxPath(const QString& onnx_path) {
        if (!_onnx_combo || onnx_path.isEmpty()) return;
        _onnx_combo->blockSignals(true);

        const QFileInfo req_fi(onnx_path);
        const QString req_canon = req_fi.canonicalFilePath();
        const QString req_base = req_fi.completeBaseName();

        int idx = -1;
        for (int i = 0; i < _onnx_combo->count(); ++i) {
            const QString item_path = _onnx_combo->itemData(i).toString();
            if (item_path.isEmpty()) continue;
            const QFileInfo item_fi(item_path);
            if (item_path == req_fi.absoluteFilePath() ||
                (!req_canon.isEmpty() && item_fi.canonicalFilePath() == req_canon) ||
                (!req_base.isEmpty() && item_fi.completeBaseName() == req_base)) {
                idx = i;
                break;
            }
        }

        if (idx < 0 && req_fi.exists()) {
            if (_onnx_combo->count() == 1 && _onnx_combo->itemData(0).toString().isEmpty()) {
                _onnx_combo->clear();
            }
            _onnx_combo->addItem("📦 " + req_base + " (.onnx)", req_fi.absoluteFilePath());
            idx = _onnx_combo->count() - 1;
        }

        if (idx >= 0) {
            _onnx_combo->setCurrentIndex(idx);
        }
        _onnx_combo->blockSignals(false);
    }

    OverlayRenderFlags renderFlags() const noexcept {
        OverlayRenderFlags f;
        f.show_obb_box         = _chk_obb->isChecked();
        f.show_cad_wireframe   = _chk_wireframe->isChecked();
        f.show_axes_and_hud    = _chk_hud->isChecked();
        f.show_segmented_cloud = _chk_cloud->isChecked();
        f.multi_instance_mode  = _chk_multi->isChecked();
        return f;
    }

    void setMultiInstanceChecked(bool enabled) {
        if (!_chk_multi) return;
        _chk_multi->blockSignals(true);
        _chk_multi->setChecked(enabled);
        _chk_multi->blockSignals(false);
    }

    void setRoiActive(bool active) {
        if (!_btn_clear_roi) return;
        if (active) {
            _btn_clear_roi->setText("✂️ Reset 2D ROI [ACTIVE]");
            _btn_clear_roi->setStyleSheet(
                "QPushButton { background-color: #92400e; color: #fef3c7; font-weight: bold; "
                "padding: 3px 9px; border-radius: 4px; border: 1px solid #f59e0b; font-size: 11px; }"
                "QPushButton:hover { background-color: #b45309; }");
        } else {
            _btn_clear_roi->setText("✂️ Reset 2D ROI");
            _btn_clear_roi->setStyleSheet(
                "QPushButton { background-color: #263238; color: #cfd8dc; padding: 3px 9px; "
                "border-radius: 4px; border: 1px solid #37474f; font-size: 11px; }"
                "QPushButton:hover { background-color: #37474f; color: white; }");
        }
    }

    void setAutoBuildRunning(bool running) {
        if (!_btn_one_click) return;
        _btn_one_click->setEnabled(!running);
        _btn_one_click->setText(running ? "⏳ Auto-Building ONNX..." : "⚡ 1-Click Auto-Build (CAD → ONNX)");
    }

    void resetDiagnostics() {
        setPillStyle(_pill_depth, "#cbd5e1", "#1e293b", "⚪ Depth: Idle");
        setPillStyle(_pill_plane, "#cbd5e1", "#1e293b", "⚪ Mode: Auto");
        setPillStyle(_pill_fit,   "#cbd5e1", "#1e293b", "⚪ Fit: --%");
    }

    void updateDiagnostics(int valid_depth_pts,
                           bool floor_locked,
                           double floor_pitch_deg,
                           float confidence,
                           double rmse_mm,
                           int num_instances,
                           bool sparse_depth_fallback = false,
                           bool floor_standing_mode = false) {
        // 1. Depth Pill
        if (valid_depth_pts >= 250 && !sparse_depth_fallback) {
            setPillStyle(_pill_depth, "#065f46", "#d1fae5",
                         QString("🟢 Depth: %1 pts (Good)").arg(valid_depth_pts));
        } else if (valid_depth_pts >= 20) {
            setPillStyle(_pill_depth, "#92400e", "#fef3c7",
                         QString("🟡 Sparse Depth (%1 pts) • 2D Edge Anchor").arg(valid_depth_pts));
        } else {
            setPillStyle(_pill_depth, "#991b1b", "#fee2e2",
                         QString("🔴 Depth Dropout (%1 pts)").arg(valid_depth_pts));
        }

        // 2. Support Mode Pill
        if (floor_locked) {
            setPillStyle(_pill_plane, "#1e40af", "#dbeafe",
                         QString("🟢 FLOOR LOCKED (Pitch %1°)").arg(floor_pitch_deg, 0, 'f', 1));
        } else if (floor_standing_mode) {
            setPillStyle(_pill_plane, "#92400e", "#fef3c7",
                         QString("🟡 Floor Unlock • PCA (%1 inst)").arg(num_instances));
        } else {
            setPillStyle(_pill_plane, "#3730a3", "#e0e7ff",
                         QString("🔵 FREE 6DoF MODE (%1 inst)").arg(num_instances));
        }

        // 3. Confidence & Fit Pill
        int fit_pct = qBound(0, static_cast<int>(confidence * 100.0f), 100);
        if (num_instances <= 0) {
            setPillStyle(_pill_fit, "#991b1b", "#fee2e2", "🔴 Searching Target...");
        } else if (fit_pct >= 65 && rmse_mm <= 16.0) {
            setPillStyle(_pill_fit, "#065f46", "#d1fae5",
                         QString("🎯 Fit: %1% | RMSE: %2 mm").arg(fit_pct).arg(rmse_mm, 0, 'f', 1));
        } else {
            setPillStyle(_pill_fit, "#92400e", "#fef3c7",
                         QString("⚠️ Fit: %1% | RMSE: %2 mm").arg(fit_pct).arg(rmse_mm, 0, 'f', 1));
        }
    }

signals:
    void activeOnnxModelChanged(const QString& onnx_path);
    void overlayFlagsChanged(const tmcad::OverlayRenderFlags& flags);
    void clearRoiRequested();
    void autoBuildPipelineRequested();

private:
    void setupUi() {
        setObjectName("StudioUxToolbar");
        setAttribute(Qt::WA_StyledBackground, true);
        setStyleSheet(
            "#StudioUxToolbar { background-color: #14181f; border-bottom: 1px solid #263238; }"
            "QLabel { color: #e2e8f0; font-size: 11px; }"
            "QCheckBox { color: #cbd5e1; font-size: 11px; spacing: 5px; }"
            "QCheckBox::indicator { width: 14px; height: 14px; border-radius: 3px; border: 1px solid #475569; background: #1e293b; }"
            "QCheckBox::indicator:checked { background: #00b0ff; border-color: #38bdf8; }");

        auto* main_v = new QVBoxLayout(this);
        main_v->setContentsMargins(8, 6, 8, 6);
        main_v->setSpacing(5);

        // Row 1: Hot-Swap ONNX Selector + Multi-Instance + 1-Click Auto-Build Button
        auto* row1 = new QHBoxLayout();
        row1->setSpacing(6);
        auto* lbl_model = new QLabel("<b>Active ONNX:</b>", this);
        lbl_model->setStyleSheet("color: #38bdf8; font-size: 11px;");

        _onnx_combo = new QComboBox(this);
        _onnx_combo->setMinimumWidth(220);
        _onnx_combo->setStyleSheet(
            "QComboBox { background: #102a45; color: #40c4ff; font-weight: bold; border: 1px solid #1e88e5; "
            "border-radius: 4px; padding: 3px 8px; font-size: 11px; }"
            "QComboBox QAbstractItemView { background: #1b2028; color: #ffffff; selection-background-color: #0277bd; }");

        auto* btn_browse = new QPushButton("📂 Load .onnx...", this);
        btn_browse->setStyleSheet(
            "QPushButton { background: #263238; color: #eceff1; padding: 4px 8px; border-radius: 4px; font-size: 11px; }"
            "QPushButton:hover { background: #37474f; }");

        auto* btn_refresh = new QPushButton("🔄", this);
        btn_refresh->setFixedWidth(30);
        btn_refresh->setToolTip("Làm mới danh sách Black-Box .onnx");
        btn_refresh->setStyleSheet(
            "QPushButton { background: #263238; color: #eceff1; padding: 4px; border-radius: 4px; font-size: 11px; }"
            "QPushButton:hover { background: #37474f; }");

        _chk_multi = new QCheckBox("Multi-Object Scene", this);
        _chk_multi->setChecked(false);
        _chk_multi->setToolTip("Phát hiện nhiều vật thể đồng thời trong 1 khung hình (FREE_6DOF)");

        _btn_one_click = new QPushButton("⚡ 1-Click Auto-Build (CAD → ONNX)", this);
        _btn_one_click->setToolTip("Tự động chạy chuỗi CAD Hierarchy → Dataset → Train → Black-Box .ONNX");
        _btn_one_click->setStyleSheet(
            "QPushButton { background-color: #1d4ed8; color: white; font-weight: bold; padding: 5px 12px; border-radius: 4px; font-size: 11px; }"
            "QPushButton:hover { background-color: #2563eb; }"
            "QPushButton:disabled { background-color: #334155; color: #94a3b8; }");

        row1->addWidget(lbl_model);
        row1->addWidget(_onnx_combo, 1);
        row1->addWidget(btn_browse);
        row1->addWidget(btn_refresh);
        row1->addWidget(_chk_multi);
        row1->addWidget(_btn_one_click);
        main_v->addLayout(row1);

        // Row 2: Overlay Layer Toggles + Clear 2D ROI + Traffic-Light Pills
        auto* row2 = new QHBoxLayout();
        row2->setSpacing(8);
        _chk_obb       = new QCheckBox("3D Box (OBB)", this);
        _chk_wireframe = new QCheckBox("CAD Wireframe", this);
        _chk_hud       = new QCheckBox("3D Axes && HUD", this);
        _chk_cloud     = new QCheckBox("Segmented Cloud", this);
        _chk_obb->setChecked(true);
        _chk_wireframe->setChecked(true);
        _chk_hud->setChecked(true);

        _btn_clear_roi = new QPushButton("✂️ Reset 2D ROI", this);
        setRoiActive(false);

        _pill_depth = new QLabel("⚪ Depth: Idle", this);
        _pill_plane = new QLabel("⚪ Mode: Auto", this);
        _pill_fit   = new QLabel("⚪ Fit: --%", this);
        resetDiagnostics();

        row2->addWidget(_chk_obb);
        row2->addWidget(_chk_wireframe);
        row2->addWidget(_chk_hud);
        row2->addWidget(_chk_cloud);
        row2->addWidget(_btn_clear_roi);
        row2->addStretch(1);
        row2->addWidget(_pill_depth);
        row2->addWidget(_pill_plane);
        row2->addWidget(_pill_fit);
        main_v->addLayout(row2);

        // Connections
        connect(_onnx_combo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
            if (idx >= 0) {
                QString path = _onnx_combo->itemData(idx).toString();
                if (!path.isEmpty()) emit activeOnnxModelChanged(path);
            }
        });

        connect(btn_browse, &QPushButton::clicked, this, [this]() {
            QString init_dir = resolveModelsDir();
            if (!QDir(init_dir).exists()) init_dir = resolveProjectRoot();
            QString path = QFileDialog::getOpenFileName(this, "Select Self-Contained Black-Box ONNX",
                                                        init_dir, "ONNX Models (*.onnx)");
            if (!path.isEmpty()) {
                setActiveOnnxPath(path);
                emit activeOnnxModelChanged(QFileInfo(path).absoluteFilePath());
            }
        });

        connect(btn_refresh, &QPushButton::clicked, this, [this]() { refreshOnnxModelList(); });
        connect(_btn_one_click, &QPushButton::clicked, this, &StudioUxToolbar::autoBuildPipelineRequested);
        connect(_btn_clear_roi, &QPushButton::clicked, this, &StudioUxToolbar::clearRoiRequested);

        auto emitFlags = [this]() { emit overlayFlagsChanged(renderFlags()); };
        connect(_chk_obb, &QCheckBox::toggled, this, emitFlags);
        connect(_chk_wireframe, &QCheckBox::toggled, this, emitFlags);
        connect(_chk_hud, &QCheckBox::toggled, this, emitFlags);
        connect(_chk_cloud, &QCheckBox::toggled, this, emitFlags);
        connect(_chk_multi, &QCheckBox::toggled, this, emitFlags);
    }

    static void setPillStyle(QLabel* lbl, const QString& fg, const QString& bg, const QString& text) {
        if (!lbl) return;
        lbl->setText(text);
        lbl->setStyleSheet(QString(
            "QLabel { color: %1; background-color: %2; font-weight: 600; font-size: 11px; "
            "padding: 3px 8px; border-radius: 4px; border: 1px solid %1; }").arg(fg, bg));
    }

private:
    QComboBox* _onnx_combo{nullptr};
    QCheckBox* _chk_multi{nullptr};
    QCheckBox* _chk_obb{nullptr};
    QCheckBox* _chk_wireframe{nullptr};
    QCheckBox* _chk_hud{nullptr};
    QCheckBox* _chk_cloud{nullptr};
    QPushButton* _btn_clear_roi{nullptr};
    QPushButton* _btn_one_click{nullptr};
    QLabel* _pill_depth{nullptr};
    QLabel* _pill_plane{nullptr};
    QLabel* _pill_fit{nullptr};
};

} // namespace tmcad

Q_DECLARE_METATYPE(tmcad::OverlayRenderFlags)

