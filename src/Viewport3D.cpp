#include "Viewport3D.h"
#include <QMouseEvent>
#include <QWheelEvent>
#include <cmath>
#include <QtMath>

// Minimal per-vertex Lambert shader -- enough to tell surfaces apart while
// orbiting. Swap for something fancier later; the SceneObject/GPU-buffer
// layer underneath doesn't need to change when you do.
static const char* kVertexShaderSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;

uniform mat4 uMvp;
uniform mat4 uModel;

out vec3 vNormal;
out vec3 vColor;

void main() {
    gl_Position = uMvp * vec4(aPos, 1.0);
    vNormal = mat3(transpose(inverse(uModel))) * aNormal;
    vColor = aColor;
}
)";

static const char* kFragmentShaderSrc = R"(
#version 330 core
in vec3 vNormal;
in vec3 vColor;
out vec4 FragColor;

uniform vec3 uColor;
uniform int uUseVertexColor;

void main() {
    vec3 baseColor = (uUseVertexColor > 0) ? vColor : uColor;
    vec3 lightDir = normalize(vec3(0.4, 0.6, 0.7));
    float diff = max(dot(normalize(vNormal), lightDir), 0.0);
    vec3 ambient = 0.35 * baseColor;
    vec3 diffuse = 0.65 * diff * baseColor;
    FragColor = vec4(ambient + diffuse, 1.0);
}
)";

static const char* kGridVertexShaderSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMvp;
void main() {
    gl_Position = uMvp * vec4(aPos, 1.0);
}
)";

static const char* kGridFragmentShaderSrc = R"(
#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main() {
    FragColor = vec4(uColor, 1.0);
}
)";

Viewport3D::Viewport3D(SceneManager* scene, QWidget* parent)
    : QOpenGLWidget(parent), scene_(scene) {
    connect(scene_, &SceneManager::sceneChanged, this, &Viewport3D::onSceneChanged);
    setFocusPolicy(Qt::StrongFocus);
}

void Viewport3D::reuploadObject(SceneObject* obj) {
    if (!obj) return;
    makeCurrent();
    obj->uploadToGPU(this);
    doneCurrent();
    update();
}

void Viewport3D::releaseObjectBuffers(SceneObject* obj) {
    if (!obj) return;
    makeCurrent();
    obj->releaseGPUBuffers(this);
    doneCurrent();
}

void Viewport3D::setShowNormals(bool on) {
    showNormals_ = on;
    update();
}

void Viewport3D::setShowAxes(bool on) {
    showAxes_ = on;
    update();
}

void Viewport3D::compileShaders() {
    auto compile = [this](GLenum type, const char* src) {
        unsigned int shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);
        int ok = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[512];
            glGetShaderInfoLog(shader, 512, nullptr, log);
            qWarning("Shader compile error: %s", log);
        }
        return shader;
    };

    unsigned int vs = compile(GL_VERTEX_SHADER, kVertexShaderSrc);
    unsigned int fs = compile(GL_FRAGMENT_SHADER, kFragmentShaderSrc);

    shaderProgram_ = glCreateProgram();
    glAttachShader(shaderProgram_, vs);
    glAttachShader(shaderProgram_, fs);
    glLinkProgram(shaderProgram_);

    glDeleteShader(vs);
    glDeleteShader(fs);

    uMvpLoc_ = glGetUniformLocation(shaderProgram_, "uMvp");
    uModelLoc_ = glGetUniformLocation(shaderProgram_, "uModel");
    uColorLoc_ = glGetUniformLocation(shaderProgram_, "uColor");
    uUseVertexColorLoc_ = glGetUniformLocation(shaderProgram_, "uUseVertexColor");

    unsigned int gridVs = compile(GL_VERTEX_SHADER, kGridVertexShaderSrc);
    unsigned int gridFs = compile(GL_FRAGMENT_SHADER, kGridFragmentShaderSrc);

    gridShaderProgram_ = glCreateProgram();
    glAttachShader(gridShaderProgram_, gridVs);
    glAttachShader(gridShaderProgram_, gridFs);
    glLinkProgram(gridShaderProgram_);

    glDeleteShader(gridVs);
    glDeleteShader(gridFs);

    uGridMvpLoc_ = glGetUniformLocation(gridShaderProgram_, "uMvp");
    uGridColorLoc_ = glGetUniformLocation(gridShaderProgram_, "uColor");
}

void Viewport3D::initializeGL() {
    initializeOpenGLFunctions();
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.12f, 0.12f, 0.14f, 1.0f);
    compileShaders();

    // Generate grid VAO/VBO (unit size -1 to 1)
    std::vector<float> gridVerts;
    int subdivisions = 50; 
    float step = 2.0f / subdivisions;
    for (int i = 0; i <= subdivisions; ++i) {
        float p = -1.0f + i * step;
        // Lines parallel to Y axis (constant X)
        gridVerts.push_back(p); gridVerts.push_back(-1.0f); gridVerts.push_back(0.0f);
        gridVerts.push_back(p); gridVerts.push_back(1.0f); gridVerts.push_back(0.0f);
        // Lines parallel to X axis (constant Y)
        gridVerts.push_back(-1.0f); gridVerts.push_back(p); gridVerts.push_back(0.0f);
        gridVerts.push_back(1.0f); gridVerts.push_back(p); gridVerts.push_back(0.0f);
    }
    gridVertexCount_ = gridVerts.size() / 3;

    glGenVertexArrays(1, &gridVao_);
    glGenBuffers(1, &gridVbo_);
    glBindVertexArray(gridVao_);
    glBindBuffer(GL_ARRAY_BUFFER, gridVbo_);
    glBufferData(GL_ARRAY_BUFFER, gridVerts.size() * sizeof(float), gridVerts.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    // Generate Axes VAO/VBO
    float axisLength = 0.5f; // Initial visual length
    std::vector<float> axesVerts = {
        0.0f, 0.0f, 0.0f,  axisLength, 0.0f, 0.0f, // X
        0.0f, 0.0f, 0.0f,  0.0f, axisLength, 0.0f, // Y
        0.0f, 0.0f, 0.0f,  0.0f, 0.0f, axisLength  // Z
    };
    glGenVertexArrays(1, &axesVao_);
    glGenBuffers(1, &axesVbo_);
    glBindVertexArray(axesVao_);
    glBindBuffer(GL_ARRAY_BUFFER, axesVbo_);
    glBufferData(GL_ARRAY_BUFFER, axesVerts.size() * sizeof(float), axesVerts.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

void Viewport3D::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}

QMatrix4x4 Viewport3D::viewMatrix() const {
    QMatrix4x4 view;
    // Z-UP spherical coordinates
    // yaw rotates around Z axis (XY plane)
    // pitch elevates from XY plane
    QVector3D eye(
        distance_ * std::cos(pitchDeg_ * M_PI / 180.0) * std::cos(yawDeg_ * M_PI / 180.0),
        distance_ * std::cos(pitchDeg_ * M_PI / 180.0) * std::sin(yawDeg_ * M_PI / 180.0),
        distance_ * std::sin(pitchDeg_ * M_PI / 180.0)
    );
    // target_ is added after computing relative eye pos
    eye += target_;
    
    view.lookAt(eye, target_, QVector3D(0, 0, 1));
    return view;
}

QMatrix4x4 Viewport3D::projMatrix() const {
    QMatrix4x4 proj;
    float aspect = width() > 0 && height() > 0
        ? static_cast<float>(width()) / static_cast<float>(height())
        : 1.0f;
    float zNear = std::max(0.01f, distance_ * 0.01f);
    float zFar = std::max(1000.0f, distance_ * 10.0f);
    proj.perspective(45.0f, aspect, zNear, zFar);
    return proj;
}

void Viewport3D::paintGL() {
    // Any object added since the last frame gets its GPU buffers created
    // here, where we know the context is current -- SceneManager itself
    // never touches GL directly outside of this call.
    scene_->uploadPendingToGPU(this);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Draw Grid
    glUseProgram(gridShaderProgram_);
    
    float gridScale = 100.0f;
    Eigen::Vector3d mn, mx;
    if (scene_->computeSceneBounds(mn, mx)) {
        gridScale = std::max(10.0, (mx - mn).norm() * 2.0);
    }
    QMatrix4x4 gridModel;
    // Grid is on X-Y plane, so scale X and Y, not Z
    gridModel.scale(gridScale, gridScale, 1.0f);

    QMatrix4x4 gridMvp = projMatrix() * viewMatrix() * gridModel;
    glUniformMatrix4fv(uGridMvpLoc_, 1, GL_FALSE, gridMvp.constData());
    glUniform3f(uGridColorLoc_, 0.35f, 0.35f, 0.35f);
    
    glBindVertexArray(gridVao_);
    glDrawArrays(GL_LINES, 0, gridVertexCount_);
    glBindVertexArray(0);

    // Draw Axes (if enabled)
    if (showAxes_) {
        // Render without scaling by gridScale so it's always clear, 
        // or scale slightly with scene bounds
        float axesScale = std::max(1.0f, gridScale * 0.1f); 
        QMatrix4x4 axesModel;
        axesModel.scale(axesScale, axesScale, axesScale);
        QMatrix4x4 axesMvp = projMatrix() * viewMatrix() * axesModel;
        
        glUniformMatrix4fv(uGridMvpLoc_, 1, GL_FALSE, axesMvp.constData());
        glBindVertexArray(axesVao_);
        
        // X axis (Red)
        glUniform3f(uGridColorLoc_, 1.0f, 0.2f, 0.2f);
        glDrawArrays(GL_LINES, 0, 2);
        // Y axis (Green)
        glUniform3f(uGridColorLoc_, 0.2f, 1.0f, 0.2f);
        glDrawArrays(GL_LINES, 2, 2);
        // Z axis (Blue)
        glUniform3f(uGridColorLoc_, 0.2f, 0.2f, 1.0f);
        glDrawArrays(GL_LINES, 4, 2);
        
        glBindVertexArray(0);
    }

    // Draw Objects
    glUseProgram(shaderProgram_);

    for (const auto& obj : scene_->objects()) {
        if (!obj->visible()) continue;
        
        QMatrix4x4 model = obj->transform();
        QMatrix4x4 mvp = projMatrix() * viewMatrix() * model;
        glUniformMatrix4fv(uMvpLoc_, 1, GL_FALSE, mvp.constData());
        glUniformMatrix4fv(uModelLoc_, 1, GL_FALSE, model.constData());

        glUniform1i(uUseVertexColorLoc_, obj->hasVertexColors() ? 1 : 0);

        // Simple fixed palette by type for now; swap for per-object colour
        // picking once the list panel needs it.
        if (obj->type() == SceneObjectType::Mesh) {
            glUniform3f(uColorLoc_, 0.75f, 0.75f, 0.8f);
        } else {
            glUniform3f(uColorLoc_, 0.9f, 0.55f, 0.2f);
        }
        obj->draw(this);
    }

    // Normal-line overlay (mesh QA aid) -- reuses the flat-color grid
    // shader since these are just unlit position-only line segments.
    if (showNormals_) {
        glUseProgram(gridShaderProgram_);
        glUniform3f(uGridColorLoc_, 1.0f, 0.9f, 0.2f);

        for (const auto& obj : scene_->objects()) {
            if (!obj->visible()) continue;

            QMatrix4x4 model = obj->transform();
            QMatrix4x4 mvp = projMatrix() * viewMatrix() * model;
            glUniformMatrix4fv(uGridMvpLoc_, 1, GL_FALSE, mvp.constData());
            obj->drawNormalLines(this);  // no-op for non-mesh objects
        }
    }
}

void Viewport3D::onSceneChanged() {
    update();  // schedules paintGL
}

void Viewport3D::frameAll() {
    Eigen::Vector3d mn, mx;
    if (!scene_->computeSceneBounds(mn, mx)) return;

    Eigen::Vector3d center = (mn + mx) * 0.5;
    double radius = (mx - mn).norm() * 0.5;
    target_ = QVector3D(static_cast<float>(center.x()),
                         static_cast<float>(center.y()),
                         static_cast<float>(center.z()));
    distance_ = static_cast<float>(radius) * 2.5f + 0.1f;
    update();
}

// --- Mouse-driven trackball: left-drag orbit, wheel zoom. Middle/pan is
// left as a TODO -- add by translating target_ in the view plane. ---------

#include <limits>

void Viewport3D::mousePressEvent(QMouseEvent* e) {
    lastMousePos_ = e->pos();
    lastButtons_ = e->buttons();

    if (e->button() == Qt::LeftButton) {
        QMatrix4x4 view = viewMatrix();
        QMatrix4x4 proj = projMatrix();
        QMatrix4x4 vpInv = (proj * view).inverted();

        float ndcX = (2.0f * e->pos().x()) / width() - 1.0f;
        float ndcY = 1.0f - (2.0f * e->pos().y()) / height(); 

        QVector4D nearPt(ndcX, ndcY, -1.0f, 1.0f);
        QVector4D farPt(ndcX, ndcY, 1.0f, 1.0f);

        QVector4D worldNear = vpInv * nearPt;
        worldNear /= worldNear.w();
        QVector4D worldFar = vpInv * farPt;
        worldFar /= worldFar.w();

        QVector3D rayOrigin = worldNear.toVector3D();
        QVector3D rayDir = (worldFar.toVector3D() - rayOrigin).normalized();

        int pickedId = -1;
        float minT = std::numeric_limits<float>::max();

        for (const auto& obj : scene_->objects()) {
            if (!obj->visible()) continue;
            
            QMatrix4x4 objInv = obj->transform().inverted();
            QVector3D localOrigin = objInv.map(rayOrigin);
            QVector3D localDir = objInv.mapVector(rayDir).normalized();

            Eigen::Vector3d mn = obj->getMin();
            Eigen::Vector3d mx = obj->getMax();

            float tmin = -std::numeric_limits<float>::max();
            float tmax = std::numeric_limits<float>::max();

            QVector3D boxMin(mn.x(), mn.y(), mn.z());
            QVector3D boxMax(mx.x(), mx.y(), mx.z());

            bool hit = true;
            for (int i = 0; i < 3; ++i) {
                if (std::abs(localDir[i]) < 1e-6f) {
                    if (localOrigin[i] < boxMin[i] || localOrigin[i] > boxMax[i]) hit = false;
                } else {
                    float t1 = (boxMin[i] - localOrigin[i]) / localDir[i];
                    float t2 = (boxMax[i] - localOrigin[i]) / localDir[i];
                    if (t1 > t2) std::swap(t1, t2);
                    tmin = std::max(tmin, t1);
                    tmax = std::min(tmax, t2);
                    if (tmin > tmax) hit = false;
                }
            }

            if (hit && tmax > 0) {
                float t = tmin > 0 ? tmin : tmax;
                if (t < minT) {
                    minT = t;
                    pickedId = obj->id();
                }
            }
        }

        if (pickedId != -1) {
            emit objectPicked(pickedId);
        }
    }
}

void Viewport3D::mouseMoveEvent(QMouseEvent* e) {
    QPoint delta = e->pos() - lastMousePos_;
    lastMousePos_ = e->pos();

    if (lastButtons_ & Qt::LeftButton) {
        yawDeg_ += delta.x() * 0.4f;
        pitchDeg_ = qBound(-89.0f, pitchDeg_ - delta.y() * 0.4f, 89.0f);
        update();
    } else if (lastButtons_ & Qt::RightButton) {
        float factor = 1.0f + (delta.y() * 0.01f);
        distance_ = qBound(0.05f, distance_ * factor, 1000000.0f);
        update();
    }
}

void Viewport3D::wheelEvent(QWheelEvent* e) {
    float factor = e->angleDelta().y() > 0 ? 0.9f : 1.1f;
    distance_ = qBound(0.05f, distance_ * factor, 1000000.0f);
    update();
}