#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>

#include "ObjectDetection_V2.5/ObjectDetection/applications/pallet/PalletApplication.h"
#include "ObjectDetection_V2.5/ObjectDetection/RgbdFrame.h"
#include "ObjectDetection_V2.5/ObjectDetection/DetectionTypes.h"

int main() {
    std::cout << "========================================================\n";
    std::cout << " TEST V2.5 PALLET STATELESS & RAY-PLANE INTERSECTION    \n";
    std::cout << "========================================================\n";

    PalletApplication app;
    DetectorConfig config;
    std::string err;
    if (!app.initialize(config, err)) {
        std::cerr << "[-] Failed to initialize PalletApplication: " << err << "\n";
        return 1;
    }
    std::cout << "[+] PalletApplication initialized successfully.\n";

    RgbdFrame frame;
    frame.width = 640;
    frame.height = 480;
    frame.intrinsics.fx = 385.0;
    frame.intrinsics.fy = 385.0;
    frame.intrinsics.ppx = 320.0;
    frame.intrinsics.ppy = 240.0;
    frame.depth_scale = 0.001;
    frame.depth_z16 = cv::Mat::zeros(480, 640, CV_16UC1);

    double true_z0 = 1.20;
    double true_yaw_rad = 5.0 * CV_PI / 180.0;
    double plane_a = std::tan(true_yaw_rad);
    double plane_b = 0.0;
    double plane_c = true_z0;

    double x_blocks[3] = {-0.38, 0.0, 0.38};
    double y_block = 0.10;

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

        for (int r = rect.y; r < rect.y + rect.height; ++r) {
            for (int c = rect.x; c < rect.x + rect.width; ++c) {
                if (r >= 0 && r < 480 && c >= 0 && c < 640) {
                    double lx = (c - frame.intrinsics.ppx) / frame.intrinsics.fx;
                    double ly = (r - frame.intrinsics.ppy) / frame.intrinsics.fy;
                    double pt_z = plane_c / (1.0 - plane_a * lx - plane_b * ly);
                    if ((r == rect.y || c == rect.x) && (r % 3 == 0)) {
                        pt_z += 0.25;
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

    bool ok = app.process(frame, output, err);
    if (!ok || output.result.objects.empty()) {
        std::cerr << "[-] process() failed: " << err << " (status: " << output.application_status << ")\n";
        return 1;
    }

    const auto& res = output.result.objects[0];
    std::cout << "[+] Pallet detected successfully in V2.5!\n";
    std::cout << "    - Status: " << output.application_status << "\n";
    std::cout << "    - 3D Position: X=" << res.x_3d << "m, Y=" << res.y_3d << "m, Z=" << res.z_3d << "m\n";
    std::cout << "    - 6DoF Pose: Roll=" << res.roll_deg << " deg, Pitch=" << res.pitch_deg << " deg, Yaw=" << res.yaw_deg << " deg\n";

    assert(std::abs(res.z_3d - 1.20) < 0.05);
    assert(std::abs(res.yaw_deg - 5.0) < 1.5);

    std::cout << "\n>>> ALL TESTS PASSED: ObjectDetection_V2.5 verified! <<<\n";
    return 0;
}
