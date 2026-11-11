#include <QApplication>
#include <EGL/egl.h>
#include <iostream>

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    std::cout << "QApplication created." << std::endl;

    EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    std::cout << "eglDpy: " << eglDpy << std::endl;
    if (eglDpy == EGL_NO_DISPLAY) {
        std::cerr << "eglGetDisplay failed, err: 0x" << std::hex << eglGetError() << std::endl;
        return 1;
    }

    EGLint major = 0, minor = 0;
    if (eglInitialize(eglDpy, &major, &minor) != EGL_TRUE) {
        std::cerr << "eglInitialize failed, err: 0x" << std::hex << eglGetError() << std::endl;
        return 1;
    }
    std::cout << "eglInitialize ok: " << major << "." << minor << std::endl;

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

    EGLint numConfigs = 0;
    EGLConfig eglCfg;
    if (eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs) != EGL_TRUE || numConfigs == 0) {
        std::cerr << "eglChooseConfig failed, err: 0x" << std::hex << eglGetError() << " numConfigs=" << numConfigs << std::endl;
        return 1;
    }
    std::cout << "eglChooseConfig ok, numConfigs=" << numConfigs << std::endl;

    if (eglBindAPI(EGL_OPENGL_API) != EGL_TRUE) {
        std::cerr << "eglBindAPI failed, err: 0x" << std::hex << eglGetError() << std::endl;
        return 1;
    }
    std::cout << "eglBindAPI ok" << std::endl;

    EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, nullptr);
    if (eglCtx == EGL_NO_CONTEXT) {
        std::cerr << "eglCreateContext failed, err: 0x" << std::hex << eglGetError() << std::endl;
        return 1;
    }
    std::cout << "eglCreateContext ok" << std::endl;

    const EGLint pbufferAttribs[] = {
        EGL_WIDTH, 1024,
        EGL_HEIGHT, 576,
        EGL_NONE
    };
    EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);
    if (eglSurf == EGL_NO_SURFACE) {
        std::cerr << "eglCreatePbufferSurface failed, err: 0x" << std::hex << eglGetError() << std::endl;
        return 1;
    }
    std::cout << "eglCreatePbufferSurface ok" << std::endl;

    if (eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx) != EGL_TRUE) {
        std::cerr << "eglMakeCurrent failed, err: 0x" << std::hex << eglGetError() << std::endl;
        return 1;
    }
    std::cout << "eglMakeCurrent SUCCESS!" << std::endl;
    return 0;
}

