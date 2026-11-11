#pragma once

#include <QObject>
#include <vector>
#include <memory>
#include <string>
#include "SceneObject.h"
#include "CADAsset.h"
#include <unordered_map>

// Owns every loaded object. Knows nothing about widgets or the OpenGL
// context beyond forwarding GL calls it's handed -- MainWindow / Viewport3D
// just ask "what exists now" (objects()) and react to sceneChanged().
class SceneManager : public QObject {
    Q_OBJECT
public:
    explicit SceneManager(QObject* parent = nullptr) : QObject(parent) {}

    // Loads a file via Open3D based on extension:
    //   .stl / .obj / .ply (mesh)  -> MeshObject
    //   .pcd / .ply (point cloud)  -> PointCloudObject   (TODO: disambiguate
    //     .ply mesh-vs-cloud properly; for now assume mesh extensions load
    //     as TriangleMesh and warn if it comes back empty)
    // Returns the new object's id, or -1 on failure.
    int addFile(const std::string& path);

    // Adds a point cloud that was generated in-app rather than loaded from
    // disk (e.g. MeshObject::poissonSample()'s output). sourceObjectName /
    // samplingMethod are optional provenance recorded on the resulting
    // PointCloudObject purely for the exported .json sidecar -- pass empty
    // strings if there's no meaningful provenance.
    // Returns the new object's id, or -1 if cloud is null/empty.
    int addPointCloud(const std::string& name,
                       std::shared_ptr<open3d::geometry::PointCloud> cloud,
                       const std::string& sourceObjectName = "",
                       const std::string& samplingMethod = "");

    void removeObject(int id);
    SceneObject* getObject(int id);
    const std::vector<std::unique_ptr<SceneObject>>& objects() const { return objects_; }

    // Union of all visible objects' bounds -- used by Viewport3D::frameAll().
    bool computeSceneBounds(Eigen::Vector3d& outMin, Eigen::Vector3d& outMax) const;

    // Must be called once, right after each addFile()/addPointCloud(), with
    // a current GL context (Viewport3D calls this from within its own
    // context).
    void uploadPendingToGPU(QOpenGLFunctions_3_3_Core* gl);

    // Returns the processed CADAsset for a path; computes once and caches.
    // Used for Phase 2 (descriptor matching) and Phase 3 (encoder input) --
    // NOT used for display (that remains MeshObject/uploadToGPU).
    const CADAsset& getOrProcessCADAsset(const std::string& path);

signals:
    void sceneChanged();          // repaint trigger
    void objectListChanged();     // list-widget trigger (add/remove, not visibility toggle)

private:
    std::vector<std::unique_ptr<SceneObject>> objects_;
    std::vector<SceneObject*> pendingUpload_;
    std::unordered_map<std::string, CADAsset> cadAssetCache_;
    int nextId_ = 0;
};