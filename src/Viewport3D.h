#pragma once

#include "SceneManager.h"
#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QMatrix4x4>
#include <QVector3D>
#include <QPoint>

// Renders whatever SceneManager currently holds. Owns the camera (trackball
// orbit/pan/zoom) and the one shared shader program.
//
// Deliberately does NOT own or process geometry -- that stays in
// SceneManager/SceneObject. paintGL() only ever calls SceneObject::draw(),
// never touches Open3D types directly, so this class stays cheap to call
// every frame regardless of how the mesh/point cloud was produced.
//
// NOTE ON GL ACCESS FROM OUTSIDE: this class privately/protectedly
// inherits QOpenGLFunctions_3_3_Core, so code outside Viewport3D (e.g.
// MainWindow) cannot pass `this` where a QOpenGLFunctions_3_3_Core* is
// expected -- that upcast isn't accessible outside the class. Whenever
// something outside needs a SceneObject's GPU buffers touched (re-upload
// after Normalize/Compute Normals, or release before removal), go through
// reuploadObject() / releaseObjectBuffers() below instead of calling
// SceneObject::uploadToGPU()/releaseGPUBuffers() directly.
class Viewport3D : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
    Q_OBJECT
public:
    explicit Viewport3D(SceneManager* scene, QWidget* parent = nullptr);

    // Re-uploads a single object's CPU-side geometry to its existing GPU
    // buffers. Call this after mutating a MeshObject in place (Normalize,
    // Compute Normals, etc.) so the viewport reflects the change. Makes the
    // GL context current itself, so it's safe to call from MainWindow.
    void reuploadObject(SceneObject* obj);

    // Frees an object's GPU buffers. Call this before erasing the object
    // from SceneManager. Makes the GL context current itself.
    void releaseObjectBuffers(SceneObject* obj);

signals:
    void objectPicked(int id);

public slots:
    void frameAll();               // reset camera to fit the whole scene bounds
    void onSceneChanged();         // triggers a repaint (connected to SceneManager::sceneChanged)
    void setShowNormals(bool on);  // toggles the per-vertex normal-line overlay on meshes
    void setShowAxes(bool on);     // toggles the world coordinate axes overlay

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;

private:
    SceneManager* scene_;
    unsigned int shaderProgram_ = 0;
    int uMvpLoc_ = -1;
    int uModelLoc_ = -1;
    int uColorLoc_ = -1;
    int uUseVertexColorLoc_ = -1;

    unsigned int gridShaderProgram_ = 0;
    int uGridMvpLoc_ = -1;
    unsigned int uGridColorLoc_ = 0;
    
    unsigned int gridVao_ = 0;
    unsigned int gridVbo_ = 0;
    size_t gridVertexCount_ = 0;

    unsigned int axesVao_ = 0;
    unsigned int axesVbo_ = 0;

    bool showNormals_ = false;
    bool showAxes_ = true; // Enabled by default

    // Trackball camera state.
    float yawDeg_ = 30.f;
    float pitchDeg_ = -20.f;
    float distance_ = 3.f;
    QVector3D target_{0, 0, 0};
    QPoint lastMousePos_;
    Qt::MouseButtons lastButtons_ = Qt::NoButton;

    QMatrix4x4 viewMatrix() const;
    QMatrix4x4 projMatrix() const;
    void compileShaders();
};