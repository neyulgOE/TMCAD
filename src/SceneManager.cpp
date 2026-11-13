#include "SceneManager.h"
#include <QVector4D>
#include <QVector3D>
#include <algorithm>
#include <QDebug>
#include <QFileInfo>
#include <cmath>
#include <fstream>

namespace {
    // Different STL sources use different, unrelated up-axis conventions
    // (or none at all) -- there is no metadata in the STL format to detect
    // this from, and no single fix works across files from different
    // origins. The only reliable source of truth is a human looking at the
    // object once and confirming it's oriented correctly.
    //
    // To avoid repeating that fix by hand every time the SAME file is
    // reloaded, MainWindow's "Save as Default Orientation" button (next to
    // the Rot X/Y/Z fields) writes the currently-displayed rotation out to
    // "<file>.orientation" next to the STL. From then on, that exact file
    // loads pre-corrected. This is per-file by design: with mixed-source
    // files there is no shortcut that covers more than one file safely.
    //
    // Sidecar format: one line, three space-separated degrees "rx ry rz",
    // matching the Properties panel's Rot X/Y/Z fields exactly -- applied
    // in the same X-then-Y-then-Z order used by SceneObject::transform(),
    // so what you saw live in the viewport is exactly what gets baked in.
    //
    // IMPORTANT: this is a DISPLAY-ONLY convenience baked into the
    // MeshObject's vertex data via Rotate() below. It does not touch the
    // file on disk, and it is entirely separate from CADPreprocessor::
    // process(), which re-reads the raw file straight from `path` for the
    // pose-estimation pipeline (see CADPreprocessor.cpp). So this has zero
    // effect on CADAsset/realBoundingDiag/pose estimation -- only on how
    // the object looks when you Add File it into the viewport.
    bool readOrientationFix(const std::string& stlPath, Eigen::Matrix3d& outR) {
        std::ifstream in(stlPath + ".orientation");
        if (!in.is_open()) return false;

        double rx = 0.0, ry = 0.0, rz = 0.0;
        if (!(in >> rx >> ry >> rz)) {
            qWarning() << "SceneManager: malformed .orientation sidecar for"
                       << QString::fromStdString(stlPath)
                       << "-- expected 'rx ry rz' in degrees, ignoring.";
            return false;
        }

        const double d2r = M_PI / 180.0;
        // Matches SceneObject::transform()'s rotate(X) then rotate(Y) then
        // rotate(Z) order: combined matrix R = Rx * Ry * Rz.
        Eigen::Matrix3d Rx = Eigen::AngleAxisd(rx * d2r, Eigen::Vector3d::UnitX()).toRotationMatrix();
        Eigen::Matrix3d Ry = Eigen::AngleAxisd(ry * d2r, Eigen::Vector3d::UnitY()).toRotationMatrix();
        Eigen::Matrix3d Rz = Eigen::AngleAxisd(rz * d2r, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        outR = Rx * Ry * Rz;
        return true;
    }
}

int SceneManager::addFile(const std::string& path) {
    QFileInfo info(QString::fromStdString(path));
    QString ext = info.suffix().toLower();
    QString baseName = info.fileName();

    std::unique_ptr<SceneObject> obj;

    if (ext == "stl" || ext == "obj") {
        auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();
        if (!open3d::io::ReadTriangleMesh(path, *mesh) || mesh->vertices_.empty()) {
            qWarning() << "Failed to load mesh:" << QString::fromStdString(path);
            return -1;
        }
        if (ext == "stl") {
            Eigen::Matrix3d R;
            if (readOrientationFix(path, R)) {
                mesh->Rotate(R, Eigen::Vector3d::Zero());

                // Ground vertically using this mesh's OWN bounds only --
                // unlike X/Z centering (Place on Grid), this doesn't
                // depend on any other object in the scene, so it's safe
                // to bake in automatically every time this file loads.
                // X/Z stay untouched here on purpose: those depend on the
                // whole scene's combined bounds at the moment "Place on
                // Grid" is clicked, so a saved X/Z would only be correct
                // for that one specific scene and wrong (or meaningless)
                // in any other -- that's why only rotation is persisted
                // to the sidecar, never position.
                double minZ = mesh->GetMinBound().z();
                mesh->Translate(Eigen::Vector3d(0.0, 0.0, -minZ));
            }
        }
        obj = std::make_unique<MeshObject>(baseName.toStdString(), path, mesh);
    } else if (ext == "pcd" || ext == "xyz") {
        auto cloud = std::make_shared<open3d::geometry::PointCloud>();
        if (!open3d::io::ReadPointCloud(path, *cloud) || cloud->points_.empty()) {
            qWarning() << "Failed to load point cloud:" << QString::fromStdString(path);
            return -1;
        }
        obj = std::make_unique<PointCloudObject>(baseName.toStdString(), path, cloud);
    } else if (ext == "ply") {
        // .ply can be either -- try mesh first, fall back to point cloud.
        auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();
        if (open3d::io::ReadTriangleMesh(path, *mesh) && !mesh->triangles_.empty()) {
            obj = std::make_unique<MeshObject>(baseName.toStdString(), path, mesh);
        } else {
            auto cloud = std::make_shared<open3d::geometry::PointCloud>();
            if (!open3d::io::ReadPointCloud(path, *cloud) || cloud->points_.empty()) {
                qWarning() << "Failed to load .ply as mesh or point cloud:" << QString::fromStdString(path);
                return -1;
            }
            obj = std::make_unique<PointCloudObject>(baseName.toStdString(), path, cloud);
        }
    } else {
        qWarning() << "Unsupported file extension:" << ext;
        return -1;
    }

    int id = nextId_++;
    obj->setId(id);
    pendingUpload_.push_back(obj.get());
    objects_.push_back(std::move(obj));

    emit objectListChanged();
    emit sceneChanged();  // Viewport3D::paintGL will call uploadPendingToGPU() first
    return id;
}

int SceneManager::addPointCloud(const std::string& name,
                                 std::shared_ptr<open3d::geometry::PointCloud> cloud,
                                 const std::string& sourceObjectName,
                                 const std::string& samplingMethod) {
    if (!cloud || cloud->points_.empty()) {
        qWarning() << "addPointCloud: refusing to add a null/empty cloud";
        return -1;
    }

    auto obj = std::make_unique<PointCloudObject>(name, std::string(), cloud);
    if (!samplingMethod.empty() || !sourceObjectName.empty()) {
        obj->setProvenance(sourceObjectName, samplingMethod);
    }

    int id = nextId_++;
    obj->setId(id);
    pendingUpload_.push_back(obj.get());
    objects_.push_back(std::move(obj));

    emit objectListChanged();
    emit sceneChanged();
    return id;
}

void SceneManager::removeObject(int id) {
    auto it = std::find_if(objects_.begin(), objects_.end(),
                            [id](const auto& o) { return o->id() == id; });
    if (it == objects_.end()) return;

    // NOTE: actual GPU buffer release happens in Viewport3D before this
    // object is erased -- see MainWindow::onRemoveSelected(), which calls
    // Viewport3D::releaseObjectBuffers() while the GL context is current,
    // THEN calls removeObject() to erase from this vector.
    objects_.erase(it);
    emit objectListChanged();
    emit sceneChanged();
}

SceneObject* SceneManager::getObject(int id) {
    auto it = std::find_if(objects_.begin(), objects_.end(),
                            [id](const auto& o) { return o->id() == id; });
    return it == objects_.end() ? nullptr : it->get();
}

bool SceneManager::computeSceneBounds(Eigen::Vector3d& outMin, Eigen::Vector3d& outMax) const {
    bool any = false;
    for (const auto& obj : objects_) {
        if (!obj->visible()) continue;
        Eigen::Vector3d mn = obj->getMin();
        Eigen::Vector3d mx = obj->getMax();
        
        QMatrix4x4 t = obj->transform();
        
        QVector3D corners[8] = {
            QVector3D(mn.x(), mn.y(), mn.z()), QVector3D(mn.x(), mn.y(), mx.z()),
            QVector3D(mn.x(), mx.y(), mn.z()), QVector3D(mn.x(), mx.y(), mx.z()),
            QVector3D(mx.x(), mn.y(), mn.z()), QVector3D(mx.x(), mn.y(), mx.z()),
            QVector3D(mx.x(), mx.y(), mn.z()), QVector3D(mx.x(), mx.y(), mx.z())
        };

        for (int i = 0; i < 8; ++i) {
            QVector3D p = t * corners[i];
            Eigen::Vector3d ep(p.x(), p.y(), p.z());
            if (!any) {
                outMin = ep; outMax = ep; any = true;
            } else {
                outMin = outMin.cwiseMin(ep);
                outMax = outMax.cwiseMax(ep);
            }
        }
    }
    return any;
}

void SceneManager::uploadPendingToGPU(QOpenGLFunctions_3_3_Core* gl) {
    for (auto* obj : pendingUpload_) {
        obj->uploadToGPU(gl);
    }
    pendingUpload_.clear();
}
#include "CADPreprocessor.h"

const CADAsset& SceneManager::getOrProcessCADAsset(const std::string& path) {
    auto it = cadAssetCache_.find(path);
    if (it != cadAssetCache_.end()) return it->second;

    CADAsset asset = CADPreprocessor::process(path);
    auto [insertedIt, ok] = cadAssetCache_.emplace(path, std::move(asset));
    return insertedIt->second;
}

void SceneManager::removeObjectAndCache(int id) {
    SceneObject* obj = getObject(id);
    if (obj) {
        const std::string& path = obj->filePath();
        if (!path.empty()) {
            cadAssetCache_.erase(path);
        }
    }
    removeObject(id);
}

void SceneManager::clearInferenceOverlays() {
    std::vector<int> idsToRemove;
    for (const auto& obj : objects_) {
        if (!obj) continue;
        const std::string& n = obj->name();
        if (obj->type() != SceneObjectType::Mesh ||
            n == "ONNX CAD Surface" ||
            n == "Aligned Scene Points" ||
            n == "Scene Cloud" ||
            n == "Segmented Object" ||
            n == "Camera Frustum" ||
            n.find("Aligned") != std::string::npos ||
            n.find("Overlay") != std::string::npos ||
            n.find("Cloud") != std::string::npos) {
            idsToRemove.push_back(obj->id());
        }
    }
    for (int id : idsToRemove) {
        removeObject(id);
    }
}

void SceneManager::clearAllAndResetCache() {
    cadAssetCache_.clear();
    pendingUpload_.clear();
    objects_.clear();
    emit objectListChanged();
    emit sceneChanged();
}