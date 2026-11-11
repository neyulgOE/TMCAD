#include "SceneObject.h"

// ---------------------------------------------------------------------------
// SceneObject
// ---------------------------------------------------------------------------

SceneObject::~SceneObject() {
    // NOTE: GPU buffers should already have been released via
    // releaseGPUBuffers() by whoever removes this object from the scene
    // while a valid GL context is current. We can't safely call GL
    // functions from a destructor if the context isn't current, so
    // Viewport3D::releaseObjectBuffers() is responsible for calling that
    // first (see MainWindow::onRemoveSelected()).
}

void SceneObject::releaseGPUBuffers(QOpenGLFunctions_3_3_Core* gl) {
    if (vao_) { gl->glDeleteVertexArrays(1, &vao_); vao_ = 0; }
    if (vboPositions_) { gl->glDeleteBuffers(1, &vboPositions_); vboPositions_ = 0; }
    if (vboNormals_) { gl->glDeleteBuffers(1, &vboNormals_); vboNormals_ = 0; }
    if (ebo_) { gl->glDeleteBuffers(1, &ebo_); ebo_ = 0; }
}

// ---------------------------------------------------------------------------
// MeshObject
// ---------------------------------------------------------------------------

MeshObject::MeshObject(std::string name, std::string filePath,
                        std::shared_ptr<open3d::geometry::TriangleMesh> mesh)
    : SceneObject(std::move(name), std::move(filePath)), mesh_(std::move(mesh)) {
    if (mesh_->vertex_normals_.empty()) {
        mesh_->ComputeVertexNormals();
    }
}

Eigen::Vector3d MeshObject::getMin() const { return mesh_->GetMinBound(); }
Eigen::Vector3d MeshObject::getMax() const { return mesh_->GetMaxBound(); }

void MeshObject::normalize() {
    Eigen::Vector3d center = mesh_->GetCenter();
    mesh_->Translate(-center);
    double scale = mesh_->GetMaxBound().norm();
    if (scale > 1e-9) {
        mesh_->Scale(1.0 / scale, Eigen::Vector3d(0, 0, 0));
    }
}

void MeshObject::computeNormals() {
    mesh_->ComputeVertexNormals();
    mesh_->ComputeTriangleNormals();
}

std::shared_ptr<open3d::geometry::PointCloud> MeshObject::poissonSample(size_t numPoints) {
    return mesh_->SamplePointsPoissonDisk(numPoints);
}

void MeshObject::releaseGPUBuffers(QOpenGLFunctions_3_3_Core* gl) {
    SceneObject::releaseGPUBuffers(gl);
    releaseNormalLineBuffers(gl);
    if (vboColors_) { gl->glDeleteBuffers(1, &vboColors_); vboColors_ = 0; }
    hasVertexColors_ = false;
}

void MeshObject::releaseNormalLineBuffers(QOpenGLFunctions_3_3_Core* gl) {
    if (normalVao_) { gl->glDeleteVertexArrays(1, &normalVao_); normalVao_ = 0; }
    if (normalVbo_) { gl->glDeleteBuffers(1, &normalVbo_); normalVbo_ = 0; }
    normalLineVertexCount_ = 0;
}

void MeshObject::uploadToGPU(QOpenGLFunctions_3_3_Core* gl) {
    releaseGPUBuffers(gl);  // also clears the normal-line buffers (virtual dispatch)

    const auto& verts = mesh_->vertices_;
    const auto& normals = mesh_->vertex_normals_;
    const auto& tris = mesh_->triangles_;

    std::vector<float> posData(verts.size() * 3);
    std::vector<float> normData(normals.size() * 3);
    for (size_t i = 0; i < verts.size(); ++i) {
        posData[i * 3 + 0] = static_cast<float>(verts[i].x());
        posData[i * 3 + 1] = static_cast<float>(verts[i].y());
        posData[i * 3 + 2] = static_cast<float>(verts[i].z());
    }
    for (size_t i = 0; i < normals.size(); ++i) {
        normData[i * 3 + 0] = static_cast<float>(normals[i].x());
        normData[i * 3 + 1] = static_cast<float>(normals[i].y());
        normData[i * 3 + 2] = static_cast<float>(normals[i].z());
    }
    std::vector<unsigned int> indices(tris.size() * 3);
    for (size_t i = 0; i < tris.size(); ++i) {
        indices[i * 3 + 0] = static_cast<unsigned int>(tris[i](0));
        indices[i * 3 + 1] = static_cast<unsigned int>(tris[i](1));
        indices[i * 3 + 2] = static_cast<unsigned int>(tris[i](2));
    }
    vertexCount_ = verts.size();
    indexCount_ = indices.size();

    gl->glGenVertexArrays(1, &vao_);
    gl->glBindVertexArray(vao_);

    gl->glGenBuffers(1, &vboPositions_);
    gl->glBindBuffer(GL_ARRAY_BUFFER, vboPositions_);
    gl->glBufferData(GL_ARRAY_BUFFER, posData.size() * sizeof(float), posData.data(), GL_STATIC_DRAW);
    gl->glEnableVertexAttribArray(0);
    gl->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    gl->glGenBuffers(1, &vboNormals_);
    gl->glBindBuffer(GL_ARRAY_BUFFER, vboNormals_);
    gl->glBufferData(GL_ARRAY_BUFFER, normData.size() * sizeof(float), normData.data(), GL_STATIC_DRAW);
    gl->glEnableVertexAttribArray(1);
    gl->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    if (!mesh_->vertex_colors_.empty() && mesh_->vertex_colors_.size() == verts.size()) {
        std::vector<float> colorData(mesh_->vertex_colors_.size() * 3);
        for (size_t i = 0; i < mesh_->vertex_colors_.size(); ++i) {
            colorData[i * 3 + 0] = static_cast<float>(mesh_->vertex_colors_[i].x());
            colorData[i * 3 + 1] = static_cast<float>(mesh_->vertex_colors_[i].y());
            colorData[i * 3 + 2] = static_cast<float>(mesh_->vertex_colors_[i].z());
        }
        gl->glGenBuffers(1, &vboColors_);
        gl->glBindBuffer(GL_ARRAY_BUFFER, vboColors_);
        gl->glBufferData(GL_ARRAY_BUFFER, colorData.size() * sizeof(float), colorData.data(), GL_STATIC_DRAW);
        gl->glEnableVertexAttribArray(2);
        gl->glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        hasVertexColors_ = true;
    } else {
        hasVertexColors_ = false;
    }

    gl->glGenBuffers(1, &ebo_);
    gl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    gl->glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    gl->glBindVertexArray(0);
    // Data now lives on the GPU. draw() below never touches mesh_ again.

    // --- Normal-visualization lines: 2 verts per mesh vertex, from the
    // vertex out along its normal. Length is scaled to the mesh's own
    // bounding-box diagonal so it stays visible on both tiny and huge
    // parts without a manual knob. Rebuilt every upload so it can never
    // go stale relative to mesh_ (e.g. after Normalize / Compute Normals).
        float lineLen = static_cast<float>((mesh_->GetMaxBound() - mesh_->GetMinBound()).norm()) * normalLineScale_;
    if (lineLen < 1e-6f) lineLen = normalLineScale_;
    
    std::vector<float> lineVerts;
    lineVerts.reserve(verts.size() * 6);
    for (size_t i = 0; i < verts.size(); ++i) {
        const Eigen::Vector3d& p = verts[i];
        Eigen::Vector3d n = i < normals.size() ? normals[i] : Eigen::Vector3d(0.0, 0.0, 0.0);
        Eigen::Vector3d p2 = p + n * lineLen;
        lineVerts.push_back(static_cast<float>(p.x()));
        lineVerts.push_back(static_cast<float>(p.y()));
        lineVerts.push_back(static_cast<float>(p.z()));
        lineVerts.push_back(static_cast<float>(p2.x()));
        lineVerts.push_back(static_cast<float>(p2.y()));
        lineVerts.push_back(static_cast<float>(p2.z()));
    }
    normalLineVertexCount_ = lineVerts.size() / 3;

    gl->glGenVertexArrays(1, &normalVao_);
    gl->glBindVertexArray(normalVao_);
    gl->glGenBuffers(1, &normalVbo_);
    gl->glBindBuffer(GL_ARRAY_BUFFER, normalVbo_);
    gl->glBufferData(GL_ARRAY_BUFFER, lineVerts.size() * sizeof(float), lineVerts.data(), GL_STATIC_DRAW);
    gl->glEnableVertexAttribArray(0);
    gl->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    gl->glBindVertexArray(0);
}

void MeshObject::draw(QOpenGLFunctions_3_3_Core* gl) const {
    if (!visible_ || vao_ == 0) return;
    gl->glBindVertexArray(vao_);
    gl->glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indexCount_), GL_UNSIGNED_INT, nullptr);
    gl->glBindVertexArray(0);
}

void MeshObject::drawNormalLines(QOpenGLFunctions_3_3_Core* gl) const {
    if (!visible_ || normalVao_ == 0 || normalLineVertexCount_ == 0) return;
    gl->glBindVertexArray(normalVao_);
    gl->glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(normalLineVertexCount_));
    gl->glBindVertexArray(0);
}

// ---------------------------------------------------------------------------
// PointCloudObject
// ---------------------------------------------------------------------------

PointCloudObject::PointCloudObject(std::string name, std::string filePath,
                                    std::shared_ptr<open3d::geometry::PointCloud> cloud)
    : SceneObject(std::move(name), std::move(filePath)), cloud_(std::move(cloud)) {}

Eigen::Vector3d PointCloudObject::getMin() const { return cloud_->GetMinBound(); }
Eigen::Vector3d PointCloudObject::getMax() const { return cloud_->GetMaxBound(); }

void PointCloudObject::uploadToGPU(QOpenGLFunctions_3_3_Core* gl) {
    releaseGPUBuffers(gl);

    const auto& pts = cloud_->points_;
    std::vector<float> posData(pts.size() * 3);
    for (size_t i = 0; i < pts.size(); ++i) {
        posData[i * 3 + 0] = static_cast<float>(pts[i].x());
        posData[i * 3 + 1] = static_cast<float>(pts[i].y());
        posData[i * 3 + 2] = static_cast<float>(pts[i].z());
    }
    vertexCount_ = pts.size();

    gl->glGenVertexArrays(1, &vao_);
    gl->glBindVertexArray(vao_);

    gl->glGenBuffers(1, &vboPositions_);
    gl->glBindBuffer(GL_ARRAY_BUFFER, vboPositions_);
    gl->glBufferData(GL_ARRAY_BUFFER, posData.size() * sizeof(float), posData.data(), GL_STATIC_DRAW);
    gl->glEnableVertexAttribArray(0);
    gl->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    gl->glBindVertexArray(0);
}

void PointCloudObject::draw(QOpenGLFunctions_3_3_Core* gl) const {
    if (!visible_ || vao_ == 0) return;
    gl->glBindVertexArray(vao_);
    gl->glPointSize(3.0f);
    gl->glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(vertexCount_));
    gl->glBindVertexArray(0);
}