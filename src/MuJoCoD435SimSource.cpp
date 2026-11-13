#include "MuJoCoD435SimSource.h"
#include "CameraProfileParser.h"
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <EGL/egl.h>
#include <open3d/Open3D.h>
#include <opencv2/core.hpp>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <limits>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>

struct MuJoCoD435SimSource::Impl {
    enum class GLBackend { None, GLFW, EGL };
    GLBackend activeBackend = GLBackend::None;

    // GLFW Context (Desktop GUI)
    GLFWwindow* glfwWin = nullptr;

    // EGL Context (Headless / Server)
    EGLDisplay eglDpy = EGL_NO_DISPLAY;
    EGLConfig eglCfg = nullptr;
    EGLContext eglCtx = EGL_NO_CONTEXT;
    EGLSurface eglSurf = EGL_NO_SURFACE;
    bool eglInitialized = false;

    // MuJoCo Structures
    mjModel* m = nullptr;
    mjData* d = nullptr;
    mjvScene scn;
    mjvCamera cam;
    mjvOption opt;
    mjrContext con;
    bool mujocoInitialized = false;

    // Camera Profile & State
    CameraHardwareProfile hwProfile;
    int activeProfileId = 1;
    CameraIntrinsics activeIntrinsics;
    int currentWidth = 1024;
    int currentHeight = 576;
    bool opened = false;

    // Target Object State (CAD)
    std::string cadObjPath;
    std::string cadTexPath;
    Eigen::Vector3d cadScale = Eigen::Vector3d(0.001, 0.001, 0.001); // mm to m
    Eigen::Vector3d targetSize = Eigen::Vector3d(0.0435, 0.0646, 0.1483); // BBQ bottle bounding box (m)
    Eigen::Vector3d targetMinBound = Eigen::Vector3d(-0.02175, -0.0323, 0.0);
    Eigen::Vector3d targetMaxBound = Eigen::Vector3d(0.02175, 0.0323, 0.1483);
    std::string lastErrorMessage = "";
    Eigen::Matrix3d targetR = Eigen::Matrix3d::Identity();
    Eigen::Vector3d targetT = Eigen::Vector3d(0.0, 0.0, 0.0); // Offset trên mặt bàn (m)
    double currentDistanceMeters = 0.58;

    // Occluder State (Lon nấm hoặc vật cản)
    bool occluderEnabled = false; // Tắt mặc định, chỉ bật khi người dùng muốn test che khuất
    Eigen::Matrix3d occluderR = Eigen::Matrix3d::Identity();
    Eigen::Vector3d occluderT = Eigen::Vector3d(0.08, 0.02, 0.61);

    // Offscreen Buffers
    std::vector<unsigned char> rgbBuffer;
    std::vector<float> depthBuffer;

    // Interactive Window State
    bool isInteractive = false;
    bool button_left = false;
    bool button_right = false;
    bool button_middle = false;
    bool mod_ctrl = false;
    bool mod_shift = false;
    double lastx = 0;
    double lasty = 0;
    bool autoSpin = false;
    double autoSpinSpeed = 0.8;
    int interactiveWinW = 800;
    int interactiveWinH = 700;

    static void mouse_button_cb(GLFWwindow* window, int button, int act, int mods) {
        Impl* impl = static_cast<Impl*>(glfwGetWindowUserPointer(window));
        if (!impl) return;
        impl->button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
        impl->button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
        impl->button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
        impl->mod_ctrl = (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                          glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);
        impl->mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                           glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
        glfwGetCursorPos(window, &impl->lastx, &impl->lasty);
    }

    static void mouse_move_cb(GLFWwindow* window, double xpos, double ypos) {
        Impl* impl = static_cast<Impl*>(glfwGetWindowUserPointer(window));
        if (!impl) return;

        if (!impl->button_left && !impl->button_right && !impl->button_middle) {
            impl->lastx = xpos;
            impl->lasty = ypos;
            return;
        }

        double dx = xpos - impl->lastx;
        double dy = ypos - impl->lasty;
        impl->lastx = xpos;
        impl->lasty = ypos;

        // Giới hạn bước nhảy tránh giật đột ngột
        dx = std::clamp(dx, -40.0, 40.0);
        dy = std::clamp(dy, -40.0, 40.0);

        int width = 800, height = 700;
        glfwGetWindowSize(window, &width, &height);
        if (height <= 0) height = 1;

        if (impl->mod_ctrl) {
            // Giữ Ctrl: Tương tác TRỰC TIẾP lên vật thể CAD trên mặt bàn
            if (impl->button_left) {
                // Chuột trái + Ctrl: Trượt vật thể trên mặt bàn (X / Y)
                impl->targetT.x() += (dx / width) * 0.25;
                impl->targetT.y() -= (dy / height) * 0.25;
                impl->targetT.x() = std::clamp(impl->targetT.x(), -0.35, 0.35);
                impl->targetT.y() = std::clamp(impl->targetT.y(), -0.25, 0.25);
                impl->updateKinematics();
            } else if (impl->button_right) {
                // Chuột phải + Ctrl: Xoay quanh trục thẳng đứng Z trên mặt bàn
                double dYaw = (dx / width) * 120.0;
                Eigen::Matrix3d dR = Eigen::AngleAxisd(dYaw * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
                impl->targetR = dR * impl->targetR;
                impl->updateKinematics();
            }
        } else {
            // Không giữ Ctrl: Xoay góc nhìn Camera tự do của MuJoCo
            mjtMouse action;
            if (impl->button_right) {
                action = impl->mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
                mjv_moveCamera(impl->m, action, dx / height, dy / height, &(impl->scn), &(impl->cam));
            } else if (impl->button_left) {
                action = impl->mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
                mjv_moveCamera(impl->m, action, dx / height, dy / height, &(impl->scn), &(impl->cam));
            }
        }
    }

    static void scroll_cb(GLFWwindow* window, double xoffset, double yoffset) {
        Impl* impl = static_cast<Impl*>(glfwGetWindowUserPointer(window));
        if (!impl) return;

        if (impl->mod_ctrl) {
            // Giữ Ctrl + Cuộn chuột: Điều chỉnh cự ly camera quan sát (gần / xa)
            impl->currentDistanceMeters -= yoffset * 0.03;
            impl->currentDistanceMeters = std::clamp(impl->currentDistanceMeters, 0.35, 3.00);
            impl->updateKinematics();
        } else {
            // Cuộn chuột bình thường: Phóng to/thu nhỏ góc nhìn camera
            mjv_moveCamera(impl->m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &(impl->scn), &(impl->cam));
        }
    }

    static void key_cb(GLFWwindow* window, int key, int scancode, int act, int mods) {
        Impl* impl = static_cast<Impl*>(glfwGetWindowUserPointer(window));
        if (!impl) return;

        if (act == GLFW_PRESS) {
            if (key == GLFW_KEY_SPACE) {
                impl->autoSpin = !impl->autoSpin;
            } else if (key == GLFW_KEY_O) {
                impl->occluderEnabled = !impl->occluderEnabled;
                impl->updateKinematics();
            } else if (key == GLFW_KEY_R) {
                impl->targetR = Eigen::Matrix3d::Identity();
                impl->targetT = Eigen::Vector3d::Zero();
                impl->currentDistanceMeters = impl->isFloorStandingLargeObject() ? 1.35 : 0.58;
                impl->updateKinematics();
            } else if (key == GLFW_KEY_ESCAPE) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
        }
    }

    bool initGLFWInteractive(int width, int height, int posX, int posY) {
        if (!glfwInit()) return false;

        if (glfwWin) {
            glfwDestroyWindow(glfwWin);
            glfwWin = nullptr;
        }

        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
        glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);

        glfwWin = glfwCreateWindow(width, height,
            "🎮 MuJoCo 3D Interactive Simulation [Chuột: Xoay/Kéo/Zoom | Giữ Ctrl: Di Chuyển CAD]",
            nullptr, nullptr);
        if (!glfwWin) return false;

        if (posX >= 0 && posY >= 0) {
            glfwSetWindowPos(glfwWin, posX, posY);
        }

        glfwMakeContextCurrent(glfwWin);
        glfwSwapInterval(1);

        glfwSetWindowUserPointer(glfwWin, this);
        glfwSetMouseButtonCallback(glfwWin, mouse_button_cb);
        glfwSetCursorPosCallback(glfwWin, mouse_move_cb);
        glfwSetScrollCallback(glfwWin, scroll_cb);
        glfwSetKeyCallback(glfwWin, key_cb);

        interactiveWinW = width;
        interactiveWinH = height;
        isInteractive = true;
        activeBackend = GLBackend::GLFW;

        // Khởi tạo camera tự do (free camera) nhìn toàn cảnh bàn làm việc & trạm kiểm tra
        mjv_defaultFreeCamera(m, &cam);
        cam.distance = 1.50;
        cam.elevation = -20;
        cam.azimuth = 135;
        cam.lookat[0] = 0.0;
        cam.lookat[1] = -0.15;
        cam.lookat[2] = 0.78;

        // Recreate scene & context bound to new GLFW window
        mjr_freeContext(&con);
        mjv_freeScene(&scn);
        mjv_makeScene(m, &scn, 2000);
        mjr_makeContext(m, &con, mjFONTSCALE_150);

        rgbBuffer.resize(currentWidth * currentHeight * 3);
        depthBuffer.resize(currentWidth * currentHeight);
        opened = true;
        return true;
    }

    Impl() {
        mjv_defaultCamera(&cam);
        mjv_defaultOption(&opt);
        mjv_defaultScene(&scn);
        mjr_defaultContext(&con);
    }

    ~Impl() {
        cleanup();
    }

    bool initGLContext(int width, int height) {
        // Chiến lược 1: Thử khởi tạo GLFW invisible window trước (chuẩn nhất trên môi trường Desktop GUI có X11/GLX)
        if (initGLFW(width, height)) {
            activeBackend = GLBackend::GLFW;
            return true;
        }

        // Chiến lược 2: Fallback sang EGL (phù hợp khi chạy headless / server không có X11)
        if (initEGL(width, height)) {
            activeBackend = GLBackend::EGL;
            return true;
        }

        std::cerr << "MuJoCoD435SimSource: Tất cả backend OpenGL (GLFW, EGL) đều không thể khởi động!\n";
        lastErrorMessage = "Tất cả backend OpenGL (GLFW, EGL) đều không thể khởi động.";
        return false;
    }

    bool initGLFW(int width, int height) {
        if (glfwWin) {
            glfwMakeContextCurrent(glfwWin);
            currentWidth = width;
            currentHeight = height;
            rgbBuffer.resize(width * height * 3);
            depthBuffer.resize(width * height);
            return true;
        }

        if (!glfwInit()) {
            return false;
        }

        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
        glfwWin = glfwCreateWindow(width, height, "MuJoCo_Virtual_D435if", nullptr, nullptr);
        if (!glfwWin) {
            return false;
        }

        glfwMakeContextCurrent(glfwWin);
        currentWidth = width;
        currentHeight = height;
        rgbBuffer.resize(width * height * 3);
        depthBuffer.resize(width * height);
        return true;
    }

    bool initEGL(int width, int height) {
        if (!eglInitialized) {
            eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
            if (eglDpy == EGL_NO_DISPLAY) {
                return false;
            }

            EGLint major = 0, minor = 0;
            if (eglInitialize(eglDpy, &major, &minor) != EGL_TRUE) {
                return false;
            }

            const EGLint configAttribs[] = {
                EGL_RED_SIZE, 8,
                EGL_GREEN_SIZE, 8,
                EGL_BLUE_SIZE, 8,
                EGL_ALPHA_SIZE, 8,
                EGL_DEPTH_SIZE, 24,
                EGL_STENCIL_SIZE, 8,
                EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,
                EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                EGL_NONE
            };

            EGLint numConfigs = 0;
            if (eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs) != EGL_TRUE || numConfigs == 0) {
                return false;
            }

            eglBindAPI(EGL_OPENGL_API);
            eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, nullptr);
            if (eglCtx == EGL_NO_CONTEXT) {
                return false;
            }
            eglInitialized = true;
        }

        // Thử surfaceless trước (chuẩn của MuJoCo record.cc)
        if (eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, eglCtx) == EGL_TRUE) {
            currentWidth = width;
            currentHeight = height;
            rgbBuffer.resize(width * height * 3);
            depthBuffer.resize(width * height);
            return true;
        }

        // Fallback sang Pbuffer
        const EGLint pbufferAttribs[] = {
            EGL_WIDTH, width,
            EGL_HEIGHT, height,
            EGL_NONE
        };
        eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);
        if (eglSurf != EGL_NO_SURFACE && eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx) == EGL_TRUE) {
            currentWidth = width;
            currentHeight = height;
            rgbBuffer.resize(width * height * 3);
            depthBuffer.resize(width * height);
            return true;
        }

        return false;
    }

    bool isFloorStandingLargeObject() const {
        if (targetSize.norm() > 0.55) return true;
        std::string upper = cadObjPath;
        std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
        return (upper.find("TABLE") != std::string::npos ||
                upper.find("PALLET") != std::string::npos ||
                upper.find("RACK") != std::string::npos);
    }

    std::string generateMJCF(double fovYDeg) {
        std::stringstream ss;
        const bool largeFloorObj = isFloorStandingLargeObject();
        double tableHeight = largeFloorObj ? 0.0 : 0.75;
        // Đặt vật thể sao cho đáy hình học (targetMinBound.z) tiếp xúc chuẩn trên mặt bàn/sàn (tableHeight)
        double baseZ = tableHeight - targetMinBound.z();
        double centerZ = baseZ + 0.5 * (targetMinBound.z() + targetMaxBound.z());

        ss << "<mujoco model='d435if_station'>\n"
           << "  <compiler angle='degree' coordinate='local'/>\n"
           << "  <statistic extent='1.0' center='0 0 0.8'/>\n"
           << "  <visual>\n"
           << "    <global offwidth='" << currentWidth << "' offheight='" << currentHeight << "'/>\n"
           << "    <map znear='0.05' zfar='3.50'/>\n"
           << "    <quality shadowsize='2048'/>\n"
           << "    <headlight ambient='0.75 0.75 0.75' diffuse='0.80 0.80 0.80' specular='0.30 0.30 0.30'/>\n"
           << "    <rgba haze='0.15 0.18 0.22 1'/>\n"
           << "  </visual>\n"
           << "  <asset>\n"
           << "    <!-- Sàn phòng -->\n"
           << "    <texture name='floor_tex' type='2d' builtin='checker' rgb1='0.24 0.26 0.29' rgb2='0.18 0.20 0.23' width='512' height='512'/>\n"
           << "    <material name='floor_mat' texture='floor_tex' texrepeat='12 12' roughness='0.8'/>\n"
           << "    <!-- Mặt bàn kiểm tra kỹ thuật sáng sạch -->\n"
           << "    <material name='table_mat' rgba='0.80 0.82 0.85 1' roughness='0.4' specular='0.2'/>\n"
           << "    <material name='metal_mat' rgba='0.28 0.30 0.33 1' roughness='0.3' specular='0.6'/>\n"
           << "    <material name='rs_case_mat' rgba='0.85 0.86 0.88 1' roughness='0.2' specular='0.7'/>\n"
           << "    <material name='rs_glass_mat' rgba='0.08 0.08 0.08 1' roughness='0.1' specular='0.9'/>\n";

        if (!cadObjPath.empty()) {
            ss << "    <mesh name='target_mesh' file='" << cadObjPath
               << "' scale='" << cadScale.x() << " " << cadScale.y() << " " << cadScale.z() << "'/>\n";
            std::string ext = cadObjPath.size() > 4 ? cadObjPath.substr(cadObjPath.size() - 4) : "";
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            bool isStl = (ext == ".stl");
            if (!cadTexPath.empty() && !isStl) {
                ss << "    <texture name='target_tex' type='2d' file='" << cadTexPath << "'/>\n"
                   << "    <material name='target_mat' texture='target_tex' specular='0.5' shininess='0.4' rgba='1 1 1 1'/>\n";
            } else {
                // Vật liệu đỏ tương ớt Chinsu chuẩn
                ss << "    <material name='target_mat' specular='0.4' shininess='0.3' rgba='0.88 0.18 0.12 1'/>\n";
            }
        }

        ss << "  </asset>\n"
           << "  <worldbody>\n"
           << "    <!-- Sàn phòng thí nghiệm / xưởng tại Z = 0 -->\n"
           << "    <geom name='floor' type='plane' size='5 5 0.1' pos='0 0 0' material='floor_mat'/>\n"
           << "    <!-- Đèn studio 4 điểm chiếu sáng rực rỡ và đèn lấp (fill light) -->\n"
           << "    <light directional='true' pos='0.2 -1.0 2.4' dir='-0.1 0.6 -1' diffuse='0.9 0.9 0.9' specular='0.3 0.3 0.3'/>\n"
           << "    <light directional='true' pos='-1.0 0.2 2.0' dir='0.6 -0.2 -0.8' diffuse='0.7 0.7 0.7'/>\n"
           << "    <light directional='true' pos='0.8 0.6 2.0' dir='-0.4 -0.4 -0.8' diffuse='0.6 0.6 0.6'/>\n"
           << "    <light directional='true' pos='0.0 -0.8 1.1' dir='0.0 0.8 -0.3' diffuse='0.6 0.6 0.6'/>\n";
        if (!largeFloorObj) {
            ss << "    <!-- BÀN LÀM VIỆC CÔNG NGHIỆP: Mặt bàn cao 0.75m, kích thước 1.0m x 0.7m -->\n"
               << "    <body name='worktable' pos='0 0 0'>\n"
               << "      <geom name='table_top' type='box' size='0.50 0.35 0.02' pos='0 0 0.73' material='table_mat'/>\n"
               << "      <geom name='leg1' type='cylinder' size='0.022 0.355' pos=' 0.45  0.30 0.355' material='metal_mat'/>\n"
               << "      <geom name='leg2' type='cylinder' size='0.022 0.355' pos='-0.45  0.30 0.355' material='metal_mat'/>\n"
               << "      <geom name='leg3' type='cylinder' size='0.022 0.355' pos=' 0.45 -0.30 0.355' material='metal_mat'/>\n"
               << "      <geom name='leg4' type='cylinder' size='0.022 0.355' pos='-0.45 -0.30 0.355' material='metal_mat'/>\n"
               << "    </body>\n";
        }
        ss << "    <!-- VẬT THỂ CAD: ĐỨNG THẲNG TỰ NHIÊN TRÊN MẶT BÀN / SÀN -->\n"
           << "    <body name='target_body' pos='0 0 " << baseZ << "' quat='1 0 0 0'>\n";

        if (!cadObjPath.empty()) {
            ss << "      <geom name='target_geom' type='mesh' mesh='target_mesh' material='target_mat'/>\n";
        } else {
            ss << "      <geom name='target_geom' type='box' size='0.022 0.032 0.074' rgba='0.90 0.35 0.25 1'/>\n";
        }
        ss << "    </body>\n"
           << "    <!-- LON ĐỒ HỘP VẬT CẢN (Nếu bật: đứng trên mặt bàn cạnh chai) -->\n"
           << "    <body name='occluder_body' pos='0.08 -0.04 0.795'>\n"
           << "      <geom name='occluder_geom' type='cylinder' size='0.035 0.045' rgba='0.85 0.75 0.5 1'/>\n"
           << "    </body>\n"
           << "    <!-- CAMERA REALSENSE D435 GẮN TRÊN GIÁ ĐỠ -->\n"
           << "    <body name='camera_rig' pos='0 -0.56 0.98'>\n"
           << "      <geom name='rs_case' type='box' size='0.045 0.0125 0.0125' pos='0 0 0' material='rs_case_mat'/>\n"
           << "      <geom name='rs_front_glass' type='box' size='0.043 0.001 0.011' pos='0 0.013 0' material='rs_glass_mat'/>\n"
           << "      <geom name='tripod_pole' type='cylinder' size='0.008 0.10' pos='0 0 -0.11' material='metal_mat'/>\n"
           << "      <geom name='tripod_base' type='cylinder' size='0.05 0.005' pos='0 0 -0.21' material='metal_mat'/>\n"
           << "      <!-- Camera quang học tự động khóa mục tiêu và nhìn thẳng vào vật thể trên bàn -->\n"
           << "      <camera name='d435if_cam' pos='0 0.02 0' mode='targetbody' target='target_body' fovy='" << fovYDeg << "'/>\n"
           << "    </body>\n"
           << "  </worldbody>\n"
           << "</mujoco>\n";

        return ss.str();
    }

    bool initMuJoCo(double fovYDeg) {
        if (mujocoInitialized) {
            mjr_freeContext(&con);
            mjv_freeScene(&scn);
            if (d) mj_deleteData(d);
            if (m) mj_deleteModel(m);
            mujocoInitialized = false;
        }

        std::string xml = generateMJCF(fovYDeg);
        char error[1000] = "Unknown error";

        std::string tmpXml = "/tmp/d435if_sim_scene.xml";
        std::ofstream ofs(tmpXml);
        ofs << xml;
        ofs.close();

        m = mj_loadXML(tmpXml.c_str(), nullptr, error, 1000);
        if (!m) {
            std::cerr << "MuJoCoD435SimSource: mj_loadXML failed: " << error << "\n";
            lastErrorMessage = std::string("MuJoCo XML parser error: ") + error;
            return false;
        }

        d = mj_makeData(m);
        mjv_makeScene(m, &scn, 2000);
        mjr_makeContext(m, &con, mjFONTSCALE_150);

        // Cấu hình camera fixed d435if_cam
        int camId = mj_name2id(m, mjOBJ_CAMERA, "d435if_cam");
        if (camId >= 0) {
            cam.type = mjCAMERA_FIXED;
            cam.fixedcamid = camId;
        } else {
            mjv_defaultFreeCamera(m, &cam);
        }

        mujocoInitialized = true;
        updateKinematics();
        return true;
    }

    void updateKinematics() {
        if (!m || !d) return;

        const bool largeFloorObj = isFloorStandingLargeObject();
        double tableHeight = largeFloorObj ? 0.0 : 0.75;
        double baseZ = tableHeight - targetMinBound.z();
        double centerZ = baseZ + 0.5 * (targetMinBound.z() + targetMaxBound.z());

        // 1. Cập nhật vị trí và tư thế target_body (đứng trên bàn)
        int targetBodyId = mj_name2id(m, mjOBJ_BODY, "target_body");
        if (targetBodyId >= 0) {
            double posX = targetT.x();
            double posY = targetT.y();
            double posZ = baseZ + targetT.z();

            m->body_pos[3 * targetBodyId + 0] = posX;
            m->body_pos[3 * targetBodyId + 1] = posY;
            m->body_pos[3 * targetBodyId + 2] = posZ;

            Eigen::Quaterniond q(targetR);
            m->body_quat[4 * targetBodyId + 0] = q.w();
            m->body_quat[4 * targetBodyId + 1] = q.x();
            m->body_quat[4 * targetBodyId + 2] = q.y();
            m->body_quat[4 * targetBodyId + 3] = q.z();
        }

        // 2. Cập nhật vị trí camera_rig theo cự ly quan sát currentDistanceMeters
        int camRigId = mj_name2id(m, mjOBJ_BODY, "camera_rig");
        if (camRigId >= 0) {
            double dist = currentDistanceMeters > 0.25 ? currentDistanceMeters : 0.58;
            double pitchRad = 22.0 * M_PI / 180.0;
            double camY = -dist * std::cos(pitchRad);
            double camZ = centerZ + dist * std::sin(pitchRad);

            m->body_pos[3 * camRigId + 0] = 0.0;
            m->body_pos[3 * camRigId + 1] = camY;
            m->body_pos[3 * camRigId + 2] = camZ;
        }

        // 3. Cập nhật tư thế occluder_body (vật cản)
        int occBodyId = mj_name2id(m, mjOBJ_BODY, "occluder_body");
        if (occBodyId >= 0) {
            if (occluderEnabled) {
                m->body_pos[3 * occBodyId + 0] = occluderT.x();
                m->body_pos[3 * occBodyId + 1] = occluderT.y();
                m->body_pos[3 * occBodyId + 2] = tableHeight + 0.045;
                Eigen::Quaterniond q(occluderR);
                m->body_quat[4 * occBodyId + 0] = q.w();
                m->body_quat[4 * occBodyId + 1] = q.x();
                m->body_quat[4 * occBodyId + 2] = q.y();
                m->body_quat[4 * occBodyId + 3] = q.z();
            } else {
                m->body_pos[3 * occBodyId + 0] = 100.0;
                m->body_pos[3 * occBodyId + 1] = 100.0;
                m->body_pos[3 * occBodyId + 2] = 100.0;
            }
        }

        mj_forward(m, d);
    }

    void cleanup() {
        isInteractive = false;
        if (mujocoInitialized) {
            mjr_freeContext(&con);
            mjv_freeScene(&scn);
            if (d) mj_deleteData(d);
            if (m) mj_deleteModel(m);
            d = nullptr;
            m = nullptr;
            mujocoInitialized = false;
        }
        if (glfwWin) {
            glfwDestroyWindow(glfwWin);
            glfwWin = nullptr;
            // Không gọi glfwTerminate() để tránh phá hỏng GLFW/GLX context khi đổi ONNX hoặc mở lại MuJoCo
        }
        if (eglInitialized) {
            if (eglSurf != EGL_NO_SURFACE) eglDestroySurface(eglDpy, eglSurf);
            if (eglCtx != EGL_NO_CONTEXT) eglDestroyContext(eglDpy, eglCtx);
            eglTerminate(eglDpy);
            eglSurf = EGL_NO_SURFACE;
            eglCtx = EGL_NO_CONTEXT;
            eglInitialized = false;
        }
        activeBackend = GLBackend::None;
        opened = false;
    }

    static std::string findDefaultCameraProfile() {
        const std::vector<std::string> candidates = {
            "configs/camera_profiles/D435if_factory.json",
            "../configs/camera_profiles/D435if_factory.json"
        };
        for (const auto& p : candidates) {
            if (std::filesystem::exists(p)) return p;
        }
        return "";
    }
};

MuJoCoD435SimSource::MuJoCoD435SimSource()
    : impl_(std::make_unique<Impl>()) {
    const std::string prof = Impl::findDefaultCameraProfile();
    if (!prof.empty()) {
        loadCameraProfileFromFile(prof, 1);
    }
}

MuJoCoD435SimSource::MuJoCoD435SimSource(const CameraHardwareProfile& profile, int profileId)
    : impl_(std::make_unique<Impl>()) {
    applyHardwareProfile(profile, profileId);
}

MuJoCoD435SimSource::~MuJoCoD435SimSource() = default;

bool MuJoCoD435SimSource::open() {
    if (impl_->opened) return true;

    int w = impl_->activeIntrinsics.width > 0 ? impl_->activeIntrinsics.width : 1024;
    int h = impl_->activeIntrinsics.height > 0 ? impl_->activeIntrinsics.height : 576;

    if (!impl_->initGLContext(w, h)) {
        return false;
    }

    double fovY = 58.05;
    if (impl_->activeIntrinsics.fy > 0) {
        fovY = 2.0 * std::atan(static_cast<double>(h) / (2.0 * impl_->activeIntrinsics.fy)) * 180.0 / M_PI;
    }

    if (!impl_->initMuJoCo(fovY)) {
        return false;
    }

    impl_->opened = true;
    return true;
}

void MuJoCoD435SimSource::close() {
    impl_->cleanup();
}

bool MuJoCoD435SimSource::isOpen() const {
    return impl_->opened;
}

CameraIntrinsics MuJoCoD435SimSource::intrinsics() const {
    return impl_->activeIntrinsics;
}

std::string MuJoCoD435SimSource::deviceName() const {
    std::string name = "MuJoCo Virtual RealSense (";
    name += impl_->hwProfile.profileName + ", " +
            std::to_string(impl_->currentWidth) + "x" + std::to_string(impl_->currentHeight) + ")";
    return name;
}

bool MuJoCoD435SimSource::loadCameraProfileFromFile(const std::string& filePath, int preferredProfileId) {
    CameraHardwareProfile prof;
    if (!CameraProfileParser::loadProfile(filePath, prof)) {
        std::cerr << "MuJoCoD435SimSource: Failed to load camera profile from " << filePath << std::endl;
        return false;
    }
    return applyHardwareProfile(prof, preferredProfileId);
}

bool MuJoCoD435SimSource::applyHardwareProfile(const CameraHardwareProfile& profile, int profileId) {
    impl_->hwProfile = profile;
    auto validList = profile.listValidProfiles();
    if (validList.empty()) return false;

    // Tìm profile theo ID hoặc lấy profile hợp lệ đầu tiên
    const StreamProfileEntry* selected = nullptr;
    if (profileId >= 0) {
        auto itD = profile.depthProfiles.find(profileId);
        if (itD != profile.depthProfiles.end() && itD->second.isValid()) {
            selected = &(itD->second);
            impl_->activeProfileId = profileId;
        } else {
            auto itC = profile.colorProfiles.find(profileId);
            if (itC != profile.colorProfiles.end() && itC->second.isValid()) {
                selected = &(itC->second);
                impl_->activeProfileId = profileId;
            }
        }
    }
    if (!selected) {
        selected = &validList[0];
        impl_->activeProfileId = selected->profileId;
    }

    impl_->activeIntrinsics = selected->toCameraIntrinsics();
    impl_->currentWidth = selected->width;
    impl_->currentHeight = selected->height;

    if (impl_->isFloorStandingLargeObject() && impl_->currentDistanceMeters < 1.0) {
        impl_->currentDistanceMeters = 1.35;
    }

    if (impl_->opened || impl_->isInteractive) {
        const bool wasInteractive = impl_->isInteractive;
        int winX = 100, winY = 100;
        int winW = impl_->interactiveWinW, winH = impl_->interactiveWinH;
        if (wasInteractive && impl_->glfwWin) {
            glfwGetWindowPos(impl_->glfwWin, &winX, &winY);
            glfwGetWindowSize(impl_->glfwWin, &winW, &winH);
        }
        close();
        if (!open()) return false;
        if (wasInteractive) {
            openInteractiveWindow(winW, winH, winX, winY);
        }
    }
    return true;
}

bool MuJoCoD435SimSource::setCustomResolution(int targetW, int targetH) {
    if (targetW <= 0 || targetH <= 0) return false;
    impl_->activeIntrinsics = impl_->hwProfile.interpolateIntrinsics(targetW, targetH);
    impl_->currentWidth = targetW;
    impl_->currentHeight = targetH;
    impl_->activeProfileId = -1; // Chế độ tùy biến

    if (impl_->isFloorStandingLargeObject() && impl_->currentDistanceMeters < 1.0) {
        impl_->currentDistanceMeters = 1.35;
    }

    if (impl_->opened || impl_->isInteractive) {
        const bool wasInteractive = impl_->isInteractive;
        int winX = 100, winY = 100;
        int winW = impl_->interactiveWinW, winH = impl_->interactiveWinH;
        if (wasInteractive && impl_->glfwWin) {
            glfwGetWindowPos(impl_->glfwWin, &winX, &winY);
            glfwGetWindowSize(impl_->glfwWin, &winW, &winH);
        }
        close();
        if (!open()) return false;
        if (wasInteractive) {
            openInteractiveWindow(winW, winH, winX, winY);
        }
    }
    return true;
}

void MuJoCoD435SimSource::setTargetPose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    impl_->targetR = R;
    if (t.z() > 0.2) {
        impl_->currentDistanceMeters = (impl_->isFloorStandingLargeObject() && t.z() < 1.0) ? 1.35 : t.z();
        impl_->targetT.z() = 0.0; // Đặt vật thể đứng ngay ngắn trên mặt bàn / sàn
    } else {
        impl_->targetT.z() = t.z();
    }
    impl_->targetT.x() = t.x();
    impl_->targetT.y() = t.y();

    if (impl_->mujocoInitialized) {
        impl_->updateKinematics();
    }
}

void MuJoCoD435SimSource::getGroundTruthPose(Eigen::Matrix3d& outR, Eigen::Vector3d& outt) const {
    if (!impl_->m || !impl_->d) {
        outR = impl_->targetR;
        outt = impl_->targetT;
        return;
    }

    int camId = mj_name2id(impl_->m, mjOBJ_CAMERA, "d435if_cam");
    int targetId = mj_name2id(impl_->m, mjOBJ_BODY, "target_body");
    if (camId < 0 || targetId < 0) {
        outR = impl_->targetR;
        outt = impl_->targetT;
        return;
    }

    // Tọa độ vị trí Camera và Target trong World frame
    Eigen::Vector3d P_cam(impl_->d->cam_xpos[camId*3 + 0],
                          impl_->d->cam_xpos[camId*3 + 1],
                          impl_->d->cam_xpos[camId*3 + 2]);
    Eigen::Vector3d P_body(impl_->d->xpos[targetId*3 + 0],
                           impl_->d->xpos[targetId*3 + 1],
                           impl_->d->xpos[targetId*3 + 2]);

    // MuJoCo cam_xmat lưu dưới dạng 3x3 ma trận thứ tự hàng (row-major):
    // Cột 0: vector Right (+X)
    // Cột 1: vector Up (+Y)
    // Cột 2: vector Backward (+Z) (vì Camera nhìn theo hướng -Z trong quy ước MuJoCo)
    Eigen::Vector3d c_right(impl_->d->cam_xmat[camId*9 + 0],
                            impl_->d->cam_xmat[camId*9 + 3],
                            impl_->d->cam_xmat[camId*9 + 6]);
    Eigen::Vector3d c_up(impl_->d->cam_xmat[camId*9 + 1],
                         impl_->d->cam_xmat[camId*9 + 4],
                         impl_->d->cam_xmat[camId*9 + 7]);
    Eigen::Vector3d c_back(impl_->d->cam_xmat[camId*9 + 2],
                           impl_->d->cam_xmat[camId*9 + 5],
                           impl_->d->cam_xmat[camId*9 + 8]);

    // Quy ước hệ tọa độ quang học chuẩn OpenCV (Optical Frame):
    // X hướng sang phải (+c_right)
    // Y hướng xuống dưới (-c_up)
    // Z hướng thẳng về phía trước theo trục ngắm (-c_back)
    Eigen::Matrix3d R_wc_opt;
    R_wc_opt.col(0) = c_right;
    R_wc_opt.col(1) = -c_up;
    R_wc_opt.col(2) = -c_back;

    // Hướng của vật thể Target trong World frame
    Eigen::Matrix3d R_wo;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_wo(r, c) = impl_->d->xmat[targetId*9 + r*3 + c];
        }
    }

    // Centroid hình học 3D thực tế của vật thể trong World frame:
    Eigen::Vector3d cadCentroidOffset = 0.5 * (impl_->targetMinBound + impl_->targetMaxBound);
    Eigen::Vector3d P_obj = P_body + R_wo * cadCentroidOffset;

    // Chuyển đổi tư thế sang hệ tọa độ quang học của Camera RealSense:
    outR = R_wc_opt.transpose() * R_wo;
    outt = R_wc_opt.transpose() * (P_obj - P_cam);
}

void MuJoCoD435SimSource::getGroundTruthBBox3D(std::vector<Eigen::Vector3d>& corners) const {
    corners.clear();
    double hx = impl_->targetSize.x() * 0.5;
    double hy = impl_->targetSize.y() * 0.5;
    double hz = impl_->targetSize.z() * 0.5;

    Eigen::Vector3d localCorners[8] = {
        {-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, hy, -hz}, {-hx, hy, -hz},
        {-hx, -hy,  hz}, {hx, -hy,  hz}, {hx, hy,  hz}, {-hx, hy,  hz}
    };

    Eigen::Matrix3d gtR;
    Eigen::Vector3d gtt;
    getGroundTruthPose(gtR, gtt);

    for (int i = 0; i < 8; ++i) {
        corners.push_back(gtR * localCorners[i] + gtt);
    }
}

void MuJoCoD435SimSource::setOccluderEnabled(bool enabled) {
    impl_->occluderEnabled = enabled;
    if (impl_->mujocoInitialized) {
        impl_->updateKinematics();
    }
}

void MuJoCoD435SimSource::setOccluderPose(const Eigen::Vector3d& t, const Eigen::Matrix3d& R) {
    impl_->occluderT = t;
    impl_->occluderR = R;
    if (impl_->mujocoInitialized) {
        impl_->updateKinematics();
    }
}

bool MuJoCoD435SimSource::isOccluderEnabled() const {
    return impl_->occluderEnabled;
}

const CameraHardwareProfile& MuJoCoD435SimSource::currentHardwareProfile() const {
    return impl_->hwProfile;
}

int MuJoCoD435SimSource::currentProfileId() const {
    return impl_->activeProfileId;
}

double MuJoCoD435SimSource::currentBaselineMeters() const {
    return impl_->hwProfile.baselineMeters;
}

std::string MuJoCoD435SimSource::lastErrorMessage() const {
    return impl_->lastErrorMessage;
}

bool MuJoCoD435SimSource::loadTargetMesh(const std::string& objPath,
                                         const std::string& texturePath,
                                         const Eigen::Vector3d& scale) {
    impl_->cadObjPath = objPath;
    impl_->cadTexPath = texturePath;
    impl_->cadScale = scale;

    std::string ext = objPath.size() > 4 ? objPath.substr(objPath.size() - 4) : "";
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    bool isBinaryStl = false;
    bool isAsciiStl = false;
    if (ext == ".stl") {
        std::ifstream bifs(objPath, std::ios::binary);
        if (bifs.is_open()) {
            bifs.seekg(0, std::ios::end);
            size_t fileSize = static_cast<size_t>(bifs.tellg());
            bifs.seekg(0, std::ios::beg);
            if (fileSize > 84) {
                bifs.seekg(80, std::ios::beg);
                uint32_t numTriangles = 0;
                bifs.read(reinterpret_cast<char*>(&numTriangles), 4);
                if (fileSize == 84 + static_cast<size_t>(numTriangles) * 50) {
                    isBinaryStl = true;
                }
            }
            if (!isBinaryStl) isAsciiStl = true;
        }
    }

    bool converted = false;
    auto mesh = open3d::io::CreateMeshFromFile(objPath);
    if (mesh && !mesh->vertices_.empty()) {
        Eigen::Vector3d bmin = mesh->GetMinBound();
        Eigen::Vector3d bmax = mesh->GetMaxBound();

        // Tự động phát hiện CAD đã chuẩn hóa sang mét (ví dụ _mesh.ply hoặc extent [0.015m..4.0m])
        Eigen::Vector3d effScale = scale;
        const double rawExtent = (bmax - bmin).norm();
        if (rawExtent > 0.015 && rawExtent < 4.0 && scale.x() <= 0.0015) {
            effScale = Eigen::Vector3d(1.0, 1.0, 1.0);
            impl_->cadScale = effScale;
        }

        impl_->targetMinBound = bmin.cwiseProduct(effScale);
        impl_->targetMaxBound = bmax.cwiseProduct(effScale);
        impl_->targetSize = impl_->targetMaxBound - impl_->targetMinBound;

        const bool tooManyFaces = (mesh->triangles_.size() > 95000);
        if (ext == ".ply" || isAsciiStl || tooManyFaces || !mesh->HasTriangles()) {
            if (!mesh->HasTriangles()) {
                open3d::geometry::PointCloud pcd(mesh->vertices_);
                auto hull = pcd.ComputeConvexHull();
                mesh = std::get<0>(hull);
            }
            if (mesh && mesh->triangles_.size() > 95000) {
                mesh = mesh->SimplifyQuadricDecimation(
                    80000, std::numeric_limits<double>::infinity(), 1.0);
            }
            if (mesh && mesh->HasTriangles()) {
                mesh->ComputeVertexNormals();
                const std::string safeStl = "/tmp/mujoco_safe_target.stl";
                if (open3d::io::WriteTriangleMesh(safeStl, *mesh, false, false)) {
                    impl_->cadObjPath = safeStl;
                    impl_->cadTexPath.clear();
                    converted = true;
                }
            }
        }
    }

    if (!converted && impl_->targetSize.norm() < 1e-6) {
        double minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9, minZ = 1e9, maxZ = -1e9;
        bool hasV = false;
        std::ifstream ifs(objPath);
        std::string line;
        while (ifs.is_open() && std::getline(ifs, line)) {
            if (line.size() > 2 && line[0] == 'v' && line[1] == ' ') {
                std::stringstream ss(line.substr(2));
                double vx, vy, vz;
                if (ss >> vx >> vy >> vz) {
                    minX = std::min(minX, vx); maxX = std::max(maxX, vx);
                    minY = std::min(minY, vy); maxY = std::max(maxY, vy);
                    minZ = std::min(minZ, vz); maxZ = std::max(maxZ, vz);
                    hasV = true;
                }
            }
        }
        if (hasV) {
            impl_->targetMinBound = Eigen::Vector3d(minX, minY, minZ).cwiseProduct(impl_->cadScale);
            impl_->targetMaxBound = Eigen::Vector3d(maxX, maxY, maxZ).cwiseProduct(impl_->cadScale);
            impl_->targetSize = impl_->targetMaxBound - impl_->targetMinBound;
        }
    }

    if (impl_->isFloorStandingLargeObject() && impl_->currentDistanceMeters < 1.0) {
        impl_->currentDistanceMeters = 1.35;
    }

    if (impl_->opened || impl_->isInteractive) {
        const bool wasInteractive = impl_->isInteractive;
        int winX = 100, winY = 100;
        int winW = impl_->interactiveWinW, winH = impl_->interactiveWinH;
        if (wasInteractive && impl_->glfwWin) {
            glfwGetWindowPos(impl_->glfwWin, &winX, &winY);
            glfwGetWindowSize(impl_->glfwWin, &winW, &winH);
        }
        close();
        if (!open()) return false;
        if (wasInteractive) {
            return openInteractiveWindow(winW, winH, winX, winY);
        }
    }
    return true;
}

bool MuJoCoD435SimSource::grabFrame(RGBDFrame& outFrame) {
    if (!impl_->opened && !open()) {
        return false;
    }

    if (impl_->activeBackend == Impl::GLBackend::GLFW && impl_->glfwWin) {
        glfwMakeContextCurrent(impl_->glfwWin);
    } else if (impl_->activeBackend == Impl::GLBackend::EGL && impl_->eglInitialized) {
        eglMakeCurrent(impl_->eglDpy, impl_->eglSurf, impl_->eglSurf, impl_->eglCtx);
    }

    int w = impl_->currentWidth;
    int h = impl_->currentHeight;
    mjrRect viewport = {0, 0, w, h};

    // Render cảnh từ góc nhìn Camera ảo D435if
    mjvCamera simCam;
    int camId = mj_name2id(impl_->m, mjOBJ_CAMERA, "d435if_cam");
    if (camId >= 0) {
        simCam.type = mjCAMERA_FIXED;
        simCam.fixedcamid = camId;
    } else {
        mjv_defaultFreeCamera(impl_->m, &simCam);
    }

    mjv_updateScene(impl_->m, impl_->d, &(impl_->opt), nullptr, &simCam, mjCAT_ALL, &(impl_->scn));
    mjr_render(viewport, &(impl_->scn), &(impl_->con));

    // Đọc RGB và Depth buffers
    mjr_readPixels(impl_->rgbBuffer.data(), impl_->depthBuffer.data(), viewport, &(impl_->con));

    // Tạo đối tượng Open3D Images (dùng cv::flip SIMD thay vòng lặp 2 chiều)
    cv::Mat rawRgb(h, w, CV_8UC3, impl_->rgbBuffer.data());
    cv::Mat flippedRgb;
    cv::flip(rawRgb, flippedRgb, 0);

    auto colorImg = std::make_shared<open3d::geometry::Image>();
    colorImg->Prepare(w, h, 3, 1); // 3 channels, 8-bit unsigned char
    std::memcpy(colorImg->data_.data(), flippedRgb.data, static_cast<size_t>(w * h * 3));

    cv::Mat rawDepth(h, w, CV_32FC1, impl_->depthBuffer.data());
    cv::Mat flippedDepth;
    cv::flip(rawDepth, flippedDepth, 0);

    auto depthImg = std::make_shared<open3d::geometry::Image>();
    depthImg->Prepare(w, h, 1, 2); // 1 channel, 16-bit uint16
    uint16_t* depthData = reinterpret_cast<uint16_t*>(depthImg->data_.data());
    const float* srcDepth = reinterpret_cast<const float*>(flippedDepth.data);

    float extent = impl_->m->stat.extent;
    float zNear = impl_->m->vis.map.znear * extent;
    float zFar = impl_->m->vis.map.zfar * extent;
    static bool printedDepthStats = false;
    if (!printedDepthStats) {
        std::cout << "[MUJOCO DEPTH] extent=" << extent 
                  << " map.znear=" << impl_->m->vis.map.znear 
                  << " map.zfar=" << impl_->m->vis.map.zfar
                  << " -> effective znear=" << zNear 
                  << " zfar=" << zFar << std::endl;
        printedDepthStats = true;
    }

    const int totalPixels = w * h;
    for (int i = 0; i < totalPixels; ++i) {
        float zNorm = srcDepth[i];
        if (zNorm >= 0.9999f || zNorm <= 0.0f) {
            depthData[i] = 0; // Invalid / Sky / Beyond range
        } else {
            float zMeters = (zNear * zFar) / (zFar - zNorm * (zFar - zNear));
            if (zMeters < 0.12f || zMeters > 3.5f) {
                depthData[i] = 0;
            } else {
                depthData[i] = static_cast<uint16_t>(std::clamp(zMeters * 1000.0f + 0.5f, 0.0f, 65535.0f));
            }
        }
    }

    outFrame.color = colorImg;
    outFrame.depth = depthImg;
    outFrame.depthScale = 1000.0;     // mm to meters
    outFrame.depthTruncMeters = 3.5;
    outFrame.timestampUs = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    return true;
}

// ── Chế độ Cửa Sổ Mô Phỏng Tương Tác 3D (MuJoCo Interactive 3D Window) ──

bool MuJoCoD435SimSource::openInteractiveWindow(int winW, int winH, int posX, int posY) {
    if (!open()) return false;
    return impl_->initGLFWInteractive(winW, winH, posX, posY);
}

void MuJoCoD435SimSource::closeInteractiveWindow() {
    impl_->isInteractive = false;
    if (impl_->glfwWin) {
        glfwDestroyWindow(impl_->glfwWin);
        impl_->glfwWin = nullptr;
    }
}

bool MuJoCoD435SimSource::isInteractiveWindowOpen() const {
    return impl_->isInteractive && impl_->glfwWin && !glfwWindowShouldClose(impl_->glfwWin);
}

bool MuJoCoD435SimSource::pollInteractiveEvents() {
    if (!impl_->isInteractive || !impl_->glfwWin) {
        return false;
    }

    if (glfwWindowShouldClose(impl_->glfwWin)) {
        closeInteractiveWindow();
        return false;
    }

    glfwMakeContextCurrent(impl_->glfwWin);
    glfwPollEvents();

    if (impl_->autoSpin) {
        // Tự động quay tròn quanh trục thẳng đứng Z với tốc độ êm ái
        Eigen::Matrix3d dR = Eigen::AngleAxisd(impl_->autoSpinSpeed * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        impl_->targetR = dR * impl_->targetR;
        impl_->updateKinematics();
    }

    // Render góc nhìn tương tác 3D (free camera) ra cửa sổ GLFW
    mjrRect winViewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(impl_->glfwWin, &winViewport.width, &winViewport.height);

    mjv_updateScene(impl_->m, impl_->d, &(impl_->opt), nullptr, &(impl_->cam), mjCAT_ALL, &(impl_->scn));
    mjr_render(winViewport, &(impl_->scn), &(impl_->con));
    glfwSwapBuffers(impl_->glfwWin);

    return true;
}

void MuJoCoD435SimSource::setAutoSpin(bool enabled, double speedDegPerFrame) {
    impl_->autoSpin = enabled;
    impl_->autoSpinSpeed = speedDegPerFrame;
}

bool MuJoCoD435SimSource::isAutoSpin() const {
    return impl_->autoSpin;
}

void MuJoCoD435SimSource::resetPose() {
    impl_->targetR = Eigen::Matrix3d::Identity();
    impl_->targetT = Eigen::Vector3d(0.0, 0.0, 0.0);
    impl_->currentDistanceMeters = 0.58;
    impl_->updateKinematics();
}
