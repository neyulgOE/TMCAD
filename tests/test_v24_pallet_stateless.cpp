#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>

#include "ObjectDetection_V2.4/ObjectDetection/applications/pallet/PalletApplication.h"
#include "ObjectDetection_V2.4/ObjectDetection/RgbdFrame.h"
#include "ObjectDetection_V2.4/ObjectDetection/DetectionTypes.h"

int main() {
    std::cout << "========================================================\n";
    std::cout << " TEST V2.4 PALLET STATELESS & RAY-PLANE INTERSECTION    \n";
    std::cout << "========================================================\n";

    PalletApplication app;
    DetectorConfig config;
    std::string err;
    if (!app.initialize(config, err)) {
        std::cerr << "[-] Failed to initialize PalletApplication: " << err << "\n";
        return 1;
    }
    std::cout << "[+] PalletApplication initialized successfully.\n";

    // Setup Camera Intrinsics (RealSense D435 640x480)
    RgbdFrame frame;
    frame.width = 640;
    frame.height = 480;
    frame.intrinsics.fx = 385.0;
    frame.intrinsics.fy = 385.0;
    frame.intrinsics.ppx = 320.0;
    frame.intrinsics.ppy = 240.0;
    frame.depth_scale = 0.001; // 1mm per unit
    frame.depth_z16 = cv::Mat::zeros(480, 640, CV_16UC1);

    // Ground truth pallet plane: Z = a*X + b*Y + c
    // At center (0,0), distance = 1.2m
    // Pallet width ~ 0.83m. Suppose blocks are at X = -0.4m, 0.0m, +0.4m
    // Center pixel: px = fx * X / Z + cx, py = fy * Y / Z + cy
    double true_z0 = 1.20; // 1.20m
    double true_yaw_rad = 5.0 * CV_PI / 180.0; // 5 deg
    // Z(X) = Z0 + X * tan(yaw)
    double plane_a = std::tan(true_yaw_rad);
    double plane_b = 0.0;
    double plane_c = true_z0;

    // Block 3D positions
    double x_blocks[3] = {-0.38, 0.0, 0.38};
    double y_block = 0.10; // 10cm below optical axis

    DetectorOutput output;
    output.object_type = "pallet";

    std::vector<cv::Rect> bboxes;
    for (int i = 0; i < 3; ++i) {
        double X = x_blocks[i];
        double Y = y_block;
        double Z = plane_c + plane_a * X + plane_b * Y;

        int px = static_cast<int>(std::round(frame.intrinsics.fx * X / Z + frame.intrinsics.ppx));
        int py = static_cast<int>(std::round(frame.intrinsics.fy * Y / Z + frame.intrinsics.ppy));

        int bw = 40;
        int bh = 30;
        cv::Rect rect(px - bw / 2, py - bh / 2, bw, bh);
        bboxes.push_back(rect);

        // Fill depth inside block box with plane depths + some depth spikes to test MAD filter
        for (int r = rect.y; r < rect.y + rect.height; ++r) {
            for (int c = rect.x; c < rect.x + rect.width; ++c) {
                if (r >= 0 && r < 480 && c >= 0 && c < 640) {
                    double lx = (c - frame.intrinsics.ppx) / frame.intrinsics.fx;
                    double ly = (r - frame.intrinsics.ppy) / frame.intrinsics.fy;
                    double pt_z = plane_c / (1.0 - plane_a * lx - plane_b * ly);
                    // Add 5% depth spike outliers at borders
                    if ((r == rect.y || c == rect.x) && (r % 3 == 0)) {
                        pt_z += 0.25; // 25cm spike (outlier)
                    }
                    frame.depth_z16.at<uint16_t>(r, c) = static_cast<uint16_t>(std::round(pt_z * 1000.0));
                }
            }
        }

        DetectionObject block_obj;
        block_obj.label = "block";
        block_obj.confidence = 0.92f;
        block_obj.x = static_cast<float>(rect.x);
        block_obj.y = static_cast<float>(rect.y);
        block_obj.width = static_cast<float>(rect.width);
        block_obj.height = static_cast<float>(rect.height);
        block_obj.center_x = static_cast<float>(px);
        block_obj.center_y = static_cast<float>(py);
        output.result.objects.push_back(block_obj);
    }

    // Add 1 "front" object encompassing all 3 blocks
    DetectionObject front_obj;
    front_obj.label = "front";
    front_obj.confidence = 0.95f;
    front_obj.x = static_cast<float>(bboxes[0].x - 10);
    front_obj.y = static_cast<float>(bboxes[0].y - 10);
    front_obj.width = static_cast<float>(bboxes[2].x + bboxes[2].width - bboxes[0].x + 20);
    front_obj.height = static_cast<float>(bboxes[0].height + 20);
    front_obj.center_x = front_obj.x + front_obj.width * 0.5f;
    front_obj.center_y = front_obj.y + front_obj.height * 0.5f;
    output.result.objects.push_back(front_obj);

    std::cout << "[+] Running process() on Frame 1...\n";
    bool ok = app.process(frame, output, err);
    if (!ok || output.result.objects.empty()) {
        std::cerr << "[-] process() failed: " << err << " (status: " << output.application_status << ")\n";
        return 1;
    }

    const auto& res = output.result.objects[0];
    std::cout << "[+] Pallet detected successfully!\n";
    std::cout << "    - Status: " << output.application_status << "\n";
    std::cout << "    - 3D Position: X=" << res.x_3d << "m, Y=" << res.y_3d << "m, Z=" << res.z_3d << "m\n";
    std::cout << "    - 6DoF Pose: Roll=" << res.roll_deg << " deg, Pitch=" << res.pitch_deg << " deg, Yaw=" << res.yaw_deg << " deg\n";
    std::cout << "    - Normal: [" << res.plane_normal_x << ", " << res.plane_normal_y << ", " << res.plane_normal_z << "]\n";
    std::cout << "    - Plane RMSE: " << res.plane_rmse_m * 1000.0 << " mm\n";
    std::cout << "    - Inlier Ratio: " << res.plane_inlier_ratio * 100.0 << "%\n";
    std::cout << "    - Pair Distance: " << res.pair_distance_m << " m\n";

    // Validate expectations
    assert(std::abs(res.z_3d - 1.20) < 0.05); // Z within 5cm of 1.2m
    assert(std::abs(res.yaw_deg - 5.0) < 1.5); // Yaw within 1.5 deg of 5.0 deg
    assert(std::abs(res.pair_distance_m - 0.76) < 0.05); // Left-to-right span ~ 0.76m

    std::cout << "\n>>> ALL TESTS PASSED: V2.4 Pallet module verified! <<<\n";
    return 0;
}
