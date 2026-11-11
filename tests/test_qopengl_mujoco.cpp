#include <QApplication>
#include <QOpenGLContext>
#include <QOffscreenSurface>
#include <QOpenGLFunctions>
#include <mujoco/mujoco.h>
#include <iostream>

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    std::cout << "Testing QOpenGLContext + QOffscreenSurface for MuJoCo..." << std::endl;

    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGL);
    format.setProfile(QSurfaceFormat::CompatibilityProfile);
    format.setVersion(3, 3);
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);

    QOpenGLContext ctx;
    ctx.setFormat(format);
    if (!ctx.create()) {
        std::cerr << "Failed to create QOpenGLContext!" << std::endl;
        return 1;
    }

    QOffscreenSurface surface;
    surface.setFormat(format);
    surface.create();
    if (!surface.isValid()) {
        std::cerr << "Failed to create QOffscreenSurface!" << std::endl;
        return 1;
    }

    if (!ctx.makeCurrent(&surface)) {
        std::cerr << "Failed to makeCurrent QOpenGLContext!" << std::endl;
        return 1;
    }
    std::cout << "QOpenGLContext made current successfully on QOffscreenSurface!" << std::endl;

    // Load simple MuJoCo model
    char error[1000] = "";
    mjModel* m = mj_loadXML("/tmp/test_mesh.xml", nullptr, error, 1000);
    if (!m) {
        std::cerr << "mj_loadXML failed: " << error << std::endl;
        return 1;
    }
    std::cout << "MuJoCo model loaded: " << m->nbody << " bodies" << std::endl;

    mjData* d = mj_makeData(m);
    mjvScene scn;
    mjrContext con;
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    mjv_makeScene(m, &scn, 1000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);
    std::cout << "mjr_makeContext SUCCESS with QOpenGLContext!" << std::endl;

    // Render test
    mjrRect viewport = {0, 0, 1024, 576};
    mjvCamera cam;
    mjvOption opt;
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);

    mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);
    std::cout << "mjr_render SUCCESS!" << std::endl;

    mjr_freeContext(&con);
    mjv_freeScene(&scn);
    mj_deleteData(d);
    mj_deleteModel(m);

    ctx.doneCurrent();
    std::cout << "ALL CHECKS PASSED WITH QOPENGLCONTEXT!" << std::endl;
    return 0;
}

