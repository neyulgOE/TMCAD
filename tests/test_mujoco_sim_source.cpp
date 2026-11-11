#include "../src/MuJoCoD435SimSource.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "=== TESTING MUJOCO D435IF SIMULATION SOURCE ===" << std::endl;

    MuJoCoD435SimSource sim;
    bool ok = sim.loadCameraProfileFromFile("/home/huyen1905/TMCAD/D435if-dataset", 1);
    if (!ok) {
        std::cerr << "FAILED to load D435if-dataset!" << std::endl;
        return 1;
    }

    std::cout << "Device: " << sim.deviceName() << std::endl;
    auto intr = sim.intrinsics();
    std::cout << "Intrinsics: " << intr.width << "x" << intr.height 
              << " | fx=" << intr.fx << " fy=" << intr.fy 
              << " cx=" << intr.cx << " cy=" << intr.cy << std::endl;

    // Đặt tư thế cho chai BBQ: Đứng thẳng trên bàn, xoay nhẹ góc Yaw = 20 độ, cự ly nhìn = 0.58m
    Eigen::Matrix3d R = Eigen::AngleAxisd(20.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Vector3d t(0.0, 0.0, 0.58);
    sim.setTargetPose(R, t);

    // Mặc định tắt lon nấm để nhìn rõ sản phẩm CAD
    sim.setOccluderEnabled(false);

    sim.loadTargetMesh("/home/huyen1905/TMCAD/CAD_DATA/Chai_Tuong_Ot_CHINSU_500g_Nap_Moi_205mm_CleanSurface-v4/Chai_Tuong_Ot_CHINSU_500g_Nap_Moi_205mm_CleanSurface-v4.stl",
                       "",
                       Eigen::Vector3d(0.001, 0.001, 0.001));

    std::cout << "Opening simulation stream..." << std::endl;
    if (!sim.open()) {
        std::cerr << "FAILED to open simulation stream!" << std::endl;
        return 1;
    }

    std::cout << "Grabbing test frame..." << std::endl;
    RGBDFrame frame;
    if (!sim.grabFrame(frame) || !frame.valid()) {
        std::cerr << "FAILED to grab frame!" << std::endl;
        return 1;
    }

    std::cout << "Frame grabbed successfully!" << std::endl;
    std::cout << "Color image: " << frame.color->width_ << "x" << frame.color->height_ << std::endl;
    std::cout << "Depth image: " << frame.depth->width_ << "x" << frame.depth->height_ << std::endl;

    // Lưu ảnh kiểm tra bằng OpenCV
    int w = frame.color->width_;
    int h = frame.color->height_;
    cv::Mat rgbMat(h, w, CV_8UC3, frame.color->data_.data());
    cv::Mat bgrMat;
    cv::cvtColor(rgbMat, bgrMat, cv::COLOR_RGB2BGR);
    cv::imwrite("/home/huyen1905/.gemini/antigravity/brain/ae2ffe12-fe6e-40fc-891f-e790b255a317/mujoco_sim_test_rgb.png", bgrMat);

    cv::Mat depth16Mat(h, w, CV_16UC1, frame.depth->data_.data());
    cv::imwrite("/home/huyen1905/.gemini/antigravity/brain/ae2ffe12-fe6e-40fc-891f-e790b255a317/mujoco_sim_test_depth.png", depth16Mat);

    // Vẽ thêm 3D Bounding Box từ Ground Truth để kiểm tra
    Eigen::Matrix3d gtR;
    Eigen::Vector3d gtt;
    sim.getGroundTruthPose(gtR, gtt);
    std::cout << "[GROUND TRUTH] Translation: " << gtt.transpose() << std::endl;
    std::cout << "[GROUND TRUTH] Rotation:\n" << gtR << std::endl;

    std::vector<Eigen::Vector3d> bboxCorners;
    sim.getGroundTruthBBox3D(bboxCorners);
    std::cout << "Ground truth 3D bounding box corners: " << bboxCorners.size() << std::endl;

    // Chiếu các điểm 3D lên ảnh 2D
    std::vector<cv::Point2f> pts2d;
    for (const auto& pt3d : bboxCorners) {
        float u = (pt3d.x() * intr.fx / pt3d.z()) + intr.cx;
        float v = (pt3d.y() * intr.fy / pt3d.z()) + intr.cy;
        pts2d.push_back(cv::Point2f(u, v));
    }

    // Vẽ 12 cạnh của 3D bounding box lên ảnh
    cv::Mat overlayMat = bgrMat.clone();
    int edges[12][2] = {
        {0,1}, {1,2}, {2,3}, {3,0},
        {4,5}, {5,6}, {6,7}, {7,4},
        {0,4}, {1,5}, {2,6}, {3,7}
    };
    for (int i = 0; i < 12; ++i) {
        cv::line(overlayMat, pts2d[edges[i][0]], pts2d[edges[i][1]], cv::Scalar(0, 255, 255), 2); // Màu vàng: Ground Truth
    }
    cv::putText(overlayMat, "MuJoCo D435if GT Box", cv::Point(pts2d[0].x, pts2d[0].y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

    cv::imwrite("/home/huyen1905/.gemini/antigravity/brain/ae2ffe12-fe6e-40fc-891f-e790b255a317/mujoco_sim_test_overlay.png", overlayMat);
    std::cout << "SUCCESS: Saved test images to artifacts directory!" << std::endl;

    sim.close();
    return 0;
}

