#include <iostream>
#include <EGL/egl.h>
#include <mujoco/mujoco.h>

int main() {
    std::cout << "Testing EGL headless initialization..." << std::endl;
    EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (eglDpy == EGL_NO_DISPLAY) {
        std::cerr << "Could not get EGL display" << std::endl;
        return 1;
    }

    EGLint major, minor;
    if (eglInitialize(eglDpy, &major, &minor) != EGL_TRUE) {
        std::cerr << "Could not initialize EGL" << std::endl;
        return 1;
    }
    std::cout << "EGL Initialized: " << major << "." << minor << std::endl;

    const EGLint configAttribs[] = {
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_STENCIL_SIZE, 8,
        EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };

    EGLint numConfigs;
    EGLConfig eglCfg;
    if (eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs) != EGL_TRUE || numConfigs == 0) {
        std::cerr << "Could not choose EGL config" << std::endl;
        return 1;
    }

    if (eglBindAPI(EGL_OPENGL_API) != EGL_TRUE) {
        std::cerr << "Could not bind EGL OpenGL API" << std::endl;
        return 1;
    }

    EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, nullptr);
    if (eglCtx == EGL_NO_CONTEXT) {
        std::cerr << "Could not create EGL context" << std::endl;
        return 1;
    }

    const EGLint pbufferAttribs[] = {
        EGL_WIDTH, 1024,
        EGL_HEIGHT, 576,
        EGL_NONE
    };
    EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);
    if (eglSurf == EGL_NO_SURFACE) {
        std::cerr << "Could not create EGL Pbuffer Surface" << std::endl;
        return 1;
    }

    if (eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx) != EGL_TRUE) {
        std::cerr << "Could not make EGL context current" << std::endl;
        return 1;
    }
    std::cout << "EGL Context is current!" << std::endl;

    // Test MuJoCo version
    std::cout << "MuJoCo Version: " << mj_versionString() << std::endl;

    eglDestroySurface(eglDpy, eglSurf);
    eglDestroyContext(eglDpy, eglCtx);
    eglTerminate(eglDpy);
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}

