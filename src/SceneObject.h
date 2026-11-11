#pragma once

#include <memory>
#include <string>
#include <open3d/Open3D.h>
#include <QOpenGLFunctions_3_3_Core>
#include <QMatrix4x4>
#include <QVector3D>

enum class SceneObjectType { Mesh, PointCloud };

// Base class for anything drawable in the viewport.
//
// Important split of responsibility, keep this as you extend the project:
//   - Geometry processing (normalize, compute normals, sampling, ICP, etc.)
//     lives in Open3D calls, on the CPU-side geometry owned here.
//   - GPU buffers (VAO/VBO) are uploaded ONCE per geometry change, not
//     every frame. paintGL() only ever calls draw(), never uploadToGPU().
class SceneObject {
public:
    SceneObject(std::string name, std::string filePath)
        : name_(std::move(name)), filePath_(std::move(filePath)) {}
    virtual ~SceneObject();

    virtual SceneObjectType type() const = 0;
    virtual Eigen::Vector3d getMin() const = 0;
    virtual Eigen::Vector3d getMax() const = 0;

    // Pushes current CPU-side geometry to GPU buffers. Call this once after
    // construction, and again ONLY if the geometry changed (e.g. after
    // Normalize / Compute Normals / Poisson Sample in the Properties panel).
    // NOTE: only call with a current GL context -- from outside Viewport3D,
    // go through Viewport3D::reuploadObject() instead of calling this
    // directly, since QOpenGLFunctions_3_3_Core is a protected base there.
    virtual void uploadToGPU(QOpenGLFunctions_3_3_Core* gl) = 0;

    // Called every frame from Viewport3D::paintGL(). Must not touch Open3D
    // or CPU-side data -- GPU buffers only.
    virtual void draw(QOpenGLFunctions_3_3_Core* gl) const = 0;

    // Optional per-frame overlay of a small line per vertex along its
    // normal, used for QA'ing geometry by eye. Default is a no-op; only
    // MeshObject implements it (point clouds have no reliable normals yet).
    virtual void drawNormalLines(QOpenGLFunctions_3_3_Core* /*gl*/) const {}

    const std::string& name() const { return name_; }
    const std::string& filePath() const { return filePath_; }
    bool visible() const { return visible_; }
    void setVisible(bool v) { visible_ = v; }
    int id() const { return id_; }
    void setId(int newId) { id_ = newId; }

    QMatrix4x4 transform() const {
        QMatrix4x4 m;
        m.translate(position_);
        m.rotate(rotationEuler_.x(), 1, 0, 0);
        m.rotate(rotationEuler_.y(), 0, 1, 0);
        m.rotate(rotationEuler_.z(), 0, 0, 1);
        return m;
    }

    QVector3D position() const { return position_; }
    void setPosition(const QVector3D& pos) { position_ = pos; }

    QVector3D rotation() const { return rotationEuler_; }
    void setRotation(const QVector3D& rot) { rotationEuler_ = rot; }

    // Số lượng vertex (mesh) hoặc điểm (point cloud) hiện có -- dùng để
    // hiển thị trong Properties panel. Trả về vertexCount_ đã lưu khi
    // uploadToGPU() chạy lần gần nhất.
    virtual size_t vertexCount() const { return vertexCount_; }
    virtual bool hasVertexColors() const { return false; }

    // Releases VAO/VBO/EBO (and, for meshes, the normal-line buffers).
    // Virtual so MeshObject can also free its extra normal-line buffers.
    // Caller must have a current GL context (see uploadToGPU note above).
    virtual void releaseGPUBuffers(QOpenGLFunctions_3_3_Core* gl);

protected:
    std::string name_;
    std::string filePath_;
    bool visible_ = true;
    int id_ = -1;
    QVector3D position_{0, 0, 0};
    QVector3D rotationEuler_{0, 0, 0};

    // GPU handles, created lazily in uploadToGPU().
    unsigned int vao_ = 0;
    unsigned int vboPositions_ = 0;
    unsigned int vboNormals_ = 0;
    unsigned int ebo_ = 0;
    size_t vertexCount_ = 0;
    size_t indexCount_ = 0;
}; // <-- end of class SceneObject

class MeshObject : public SceneObject {
public:
    MeshObject(std::string name, std::string filePath,
               std::shared_ptr<open3d::geometry::TriangleMesh> mesh);

    SceneObjectType type() const override { return SceneObjectType::Mesh; }
    Eigen::Vector3d getMin() const override;
    Eigen::Vector3d getMax() const override;
    void uploadToGPU(QOpenGLFunctions_3_3_Core* gl) override;
    void draw(QOpenGLFunctions_3_3_Core* gl) const override;
    void drawNormalLines(QOpenGLFunctions_3_3_Core* gl) const override;
    void releaseGPUBuffers(QOpenGLFunctions_3_3_Core* gl) override;
    bool hasVertexColors() const override { return hasVertexColors_; }

    open3d::geometry::TriangleMesh& mesh() { return *mesh_; }
    size_t vertexCount() const override { return mesh_ ? mesh_->vertices_.size() : vertexCount_; }

    // --- Phase-1 processing helpers -------------------------------------
    // Each of these mutates mesh_ on the CPU. Caller (MainWindow) is
    // responsible for re-uploading afterwards (Viewport3D::reuploadObject)
    // to refresh what's on screen -- these do NOT touch the GPU themselves.
    void normalize();          // recenters + rescales to a unit-ish bounding box
    void computeNormals();     // fills per-vertex normals via Open3D
    std::shared_ptr<open3d::geometry::PointCloud> poissonSample(size_t numPoints);

    // Hệ số nhân chiều dài normal-line, tính theo % bounding-box diagonal.
    // Mặc định 0.02 (2%). Giảm xuống nếu các line chồng lên nhau ở góc/mối
    // nối có nhiều vertex trùng. Gọi xong nhớ re-upload
    // (Viewport3D::reuploadObject) để áp dụng.
    void setNormalLineScale(float scale) { normalLineScale_ = scale; }
    float normalLineScale() const { return normalLineScale_; }

private:
    void releaseNormalLineBuffers(QOpenGLFunctions_3_3_Core* gl);

    std::shared_ptr<open3d::geometry::TriangleMesh> mesh_;

    // Small line per vertex, from the vertex out along its normal --
    // rebuilt every time uploadToGPU() runs so it never drifts from
    // mesh_. Drawn only when Viewport3D's "Show Normals" toggle is on.
    unsigned int normalVao_ = 0;
    unsigned int normalVbo_ = 0;
    size_t normalLineVertexCount_ = 0;
    float normalLineScale_ = 0.02f;
    unsigned int vboColors_ = 0;
    bool hasVertexColors_ = false;
}; // <-- end of class MeshObject

class PointCloudObject : public SceneObject {
public:
    PointCloudObject(std::string name, std::string filePath,
                      std::shared_ptr<open3d::geometry::PointCloud> cloud);

    SceneObjectType type() const override { return SceneObjectType::PointCloud; }
    Eigen::Vector3d getMin() const override;
    Eigen::Vector3d getMax() const override;
    void uploadToGPU(QOpenGLFunctions_3_3_Core* gl) override;
    void draw(QOpenGLFunctions_3_3_Core* gl) const override;

    open3d::geometry::PointCloud& cloud() { return *cloud_; }
    size_t vertexCount() const override { return cloud_ ? cloud_->points_.size() : vertexCount_; }

    // Provenance, set by SceneManager::addPointCloud() when this cloud came
    // from an in-app operation (e.g. Poisson sampling a mesh) rather than a
    // file load. Used purely for the exported .json sidecar metadata.
    void setProvenance(const std::string& sourceObjectName, const std::string& samplingMethod) {
        sourceObjectName_ = sourceObjectName;
        samplingMethod_ = samplingMethod;
    }
    const std::string& sourceObjectName() const { return sourceObjectName_; }
    const std::string& samplingMethod() const { return samplingMethod_; }

private:
    std::shared_ptr<open3d::geometry::PointCloud> cloud_;
    std::string sourceObjectName_;
    std::string samplingMethod_;
}; // <-- end of class PointCloudObject