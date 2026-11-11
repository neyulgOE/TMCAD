#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <iostream>

int main() {
    std::cout << "Testing GLFW for MuJoCo invisible window..." << std::endl;
    if (!glfwInit()) {
        std::cerr << "glfwInit failed" << std::endl;
        return 1;
    }
    std::cout << "glfwInit OK" << std::endl;

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(1024, 576, "MuJoCo Offscreen", NULL, NULL);
    if (!window) {
        std::cerr << "glfwCreateWindow failed (probably headless without DISPLAY)" << std::endl;
        glfwTerminate();
        return 2;
    }
    std::cout << "glfwCreateWindow OK!" << std::endl;

    glfwMakeContextCurrent(window);
    std::cout << "glfwMakeContextCurrent OK!" << std::endl;

    char error[1000] = "";
    mjModel* m = mj_loadXML("/tmp/test_mesh.xml", nullptr, error, 1000);
    if (m) {
        std::cout << "Model loaded: " << m->nbody << " bodies" << std::endl;
        mjData* d = mj_makeData(m);
        mjvScene scn;
        mjrContext con;
        mjv_defaultScene(&scn);
        mjr_defaultContext(&con);

        mjv_makeScene(m, &scn, 1000);
        mjr_makeContext(m, &con, mjFONTSCALE_150);
        std::cout << "mjr_makeContext SUCCESS with GLFW!" << std::endl;

        mjr_freeContext(&con);
        mjv_freeScene(&scn);
        mj_deleteData(d);
        mj_deleteModel(m);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}

