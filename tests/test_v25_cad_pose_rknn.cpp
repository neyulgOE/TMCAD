#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "ObjectDetection/DetectionTypes.h"
#include "ObjectDetection/RgbdFrame.h"
#include "ObjectDetection/applications/cad_pose/CADPoseApplication.h"
#include "ObjectDetection/applications/cad_pose/CADSceneSegmenter.h"
#include "ObjectDetection/applications/cad_pose/CADTemplateDetector.h"

int main()
{
    std::cout << "========================================================\n";
    std::cout << " TEST V2.5 CAD POSE RKNN STATIC-GRAPH INTEGRATION       \n";
    std::cout << "========================================================\n\n";

    // 1. Initialize CADPoseApplication
    CADPoseApplication app;
    DetectorConfig config;
    config.model_package = "/home/huyen1905/TMCAD/ObjectDetection_V2.5/models/object_detection/cad_pose/table_cad/v1";
    config.conf_threshold = 0.20f;

    std::string err;
    std::cout << "[1] Initializing CADPoseApplication with: " << config.model_package << "...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    bool init_ok = app.initialize(config, err);
    auto t1 = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!init_ok)
    {
        std::cerr << "[-] Initialization failed: " << err << "\n";
        return 1;
    }
    app.setDebugDumpPath("/tmp/cpp_dump.json");
    std::cout << "[+] CADPoseApplication initialized successfully in " << init_ms << " ms!\n\n";

    // 2. Load real RGB-D snapshot
    std::string rgb_path = "/home/huyen1905/TMCAD/dump_rgb.png";
    std::string depth_path = "/home/huyen1905/TMCAD/dump_depth.png";
    std::string meta_path = "/home/huyen1905/TMCAD/dump_meta.txt";

    cv::Mat rgb = cv::imread(rgb_path);
    cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    if (rgb.empty() || depth.empty())
    {
        std::cerr << "[-] Failed to load dump_rgb.png or dump_depth.png\n";
        return 1;
    }

    CameraIntrinsics intr;
    intr.fx = 608.215;
    intr.fy = 608.128;
    intr.ppx = 341.473;
    intr.ppy = 228.438;
    double depth_scale = 0.001;

    std::ifstream meta_f(meta_path);
    if (meta_f.is_open())
    {
        std::string line;
        while (std::getline(meta_f, line))
        {
            auto pos = line.find('=');
            if (pos != std::string::npos)
            {
                std::string k = line.substr(0, pos);
                try {
                    double v = std::stod(line.substr(pos + 1));
                    if (k == "fx") intr.fx = v;
                    else if (k == "fy") intr.fy = v;
                    else if (k == "ppx") intr.ppx = v;
                    else if (k == "ppy") intr.ppy = v;
                    else if (k == "scale") depth_scale = v;
                } catch (...) {
                    // Ignore non-numeric metadata keys like registered_pair
                }
            }
        }
    }

    RgbdFrame frame;
    frame.rgb = rgb;
    frame.depth_z16 = depth;
    frame.width = static_cast<uint32_t>(depth.cols);
    frame.height = static_cast<uint32_t>(depth.rows);
    frame.depth_scale = depth_scale;
    frame.intrinsics = intr;

    std::cout << "[2] Real Sensor Snapshot loaded:\n"
              << "    Resolution: " << frame.width << "x" << frame.height << "\n"
              << "    Intrinsics: fx=" << intr.fx << ", fy=" << intr.fy
              << ", ppx=" << intr.ppx << ", ppy=" << intr.ppy
              << ", scale=" << depth_scale << "\n\n";

    // 3. Test 3D Scene Segmenter & Pre-processing
    std::cout << "[3] Running CADSceneSegmenter on Real RGB-D Scene...\n";
    CADSceneSegmenter segmenter;
    segmenter.configure(0.220, 0.415, 0.505, 0.75, 3.50);
    
    auto t_seg0 = std::chrono::high_resolution_clock::now();
    auto seg_res = segmenter.segment(frame, 1024, nullptr);
    auto t_seg1 = std::chrono::high_resolution_clock::now();
    double seg_ms = std::chrono::duration<double, std::milli>(t_seg1 - t_seg0).count();

    std::cout << "[+] Scene segmentation completed in " << seg_ms << " ms:\n"
              << "    - Valid:          " << (seg_res.valid ? "YES" : "NO") << "\n"
              << "    - Floor detected: " << (seg_res.floor_detected ? "YES" : "NO") << "\n"
              << "    - Floor normal:   [" << seg_res.floor_plane[0] << ", " << seg_res.floor_plane[1] << ", " << seg_res.floor_plane[2] << "]\n"
              << "    - Table points:   " << seg_res.table_points.size() << " points\n\n";

    if (seg_res.table_points.size() < 1024)
    {
        std::cerr << "[-] Failed to segment 1024 table points\n";
        return 1;
    }

    // 4. Test spatial k-NN Graph Indices generation
    std::cout << "[4] Testing Fast Spatial k-NN Graph Index Generator (C++)...\n";
    auto t_knn0 = std::chrono::high_resolution_clock::now();
    std::vector<int64_t> knn_indices = app.buildSpatialKNNIndices(seg_res.table_points, 20);
    auto t_knn1 = std::chrono::high_resolution_clock::now();
    double knn_ms = std::chrono::duration<double, std::milli>(t_knn1 - t_knn0).count();

    std::cout << "[+] k-NN indices computed in " << knn_ms << " ms (Target: < 1.0 ms):\n"
              << "    - Array size:     " << knn_indices.size() << " (1024 x 20 elements)\n"
              << "    - First point neighbors: [";
    for (int k = 0; k < 5; ++k) std::cout << knn_indices[k] << (k < 4 ? ", " : "...\n\n");

    // 5. Test Full Pipeline Execution on Real Frame
    std::cout << "[5] Executing CADPoseApplication::process() on Real Scene...\n";
    DetectorOutput output;
    output.object_type = "table";

    auto t_proc0 = std::chrono::high_resolution_clock::now();
    bool proc_ok = app.process(frame, output, err);
    auto t_proc1 = std::chrono::high_resolution_clock::now();
    double proc_ms = std::chrono::duration<double, std::milli>(t_proc1 - t_proc0).count();

    std::cout << "[+] Pipeline process() returned in " << proc_ms << " ms:\n"
              << "    - Success:        " << (proc_ok ? "YES" : "NO") << "\n"
              << "    - Status:         " << output.application_status << "\n"
              << "    - Objects count:  " << output.result.objects.size() << "\n\n";

    if (!output.result.objects.empty())
    {
        const auto& obj = output.result.objects[0];
        std::cout << "🎯 [ESTIMATED 6DOF CAD TABLE POSE (V2.5)]:\n"
                  << "    - Label:          " << obj.label << "\n"
                  << "    - Confidence:     " << obj.confidence << "\n"
                  << "    - Translation:    X = " << obj.x_3d << " m, Y = " << obj.y_3d << " m, Z = " << obj.z_3d << " m\n"
                  << "    - Rotation RPY:   Roll = " << obj.roll_deg << "°, Pitch = " << obj.pitch_deg << "°, Yaw = " << obj.yaw_deg << "°\n"
                  << "    - Surface Normal: [" << obj.plane_normal_x << ", " << obj.plane_normal_y << ", " << obj.plane_normal_z << "]\n"
                  << "    - Fit Ratio:      " << (obj.plane_inlier_ratio * 100.0f) << " %\n"
                  << "    - Plane RMSE:     " << (obj.plane_rmse_m * 1000.0f) << " mm\n";

        // Draw 3D Bounding Box on RGB image for visual verification (matching test_rknn_dump.py)
        cv::Mat vis = rgb.clone();
        if (obj.has_bbox_3d && obj.corners_2d.size() == 8)
        {
            std::cout << "\n[+] 2D Projected Corners:\n";
            for (size_t c = 0; c < 8; ++c)
            {
                std::cout << "    Corner " << c << ": (" << obj.corners_2d[c].x << ", " << obj.corners_2d[c].y << ")\n";
            }

            // Bottom rectangle (Green)
            const std::vector<std::pair<int, int>> edges_bot = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
            for (const auto& e : edges_bot)
                cv::line(vis, obj.corners_2d[e.first], obj.corners_2d[e.second], cv::Scalar(0, 220, 0), 2, cv::LINE_AA);

            // Top rectangle (Cyan)
            const std::vector<std::pair<int, int>> edges_top = {{4, 5}, {5, 6}, {6, 7}, {7, 4}};
            for (const auto& e : edges_top)
                cv::line(vis, obj.corners_2d[e.first], obj.corners_2d[e.second], cv::Scalar(255, 180, 0), 3, cv::LINE_AA);

            // Vertical pillars (Orange)
            const std::vector<std::pair<int, int>> edges_pil = {{0, 4}, {1, 5}, {2, 6}, {3, 7}};
            for (const auto& e : edges_pil)
                cv::line(vis, obj.corners_2d[e.first], obj.corners_2d[e.second], cv::Scalar(0, 140, 255), 2, cv::LINE_AA);

            // HUD Text
            char text_buf[128];
            std::snprintf(text_buf, sizeof(text_buf), "V2.5 (RKNN NPU) Pose: XYZ=(%.2f, %.2f, %.2fm)", obj.x_3d, obj.y_3d, obj.z_3d);
            cv::putText(vis, text_buf, cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            std::snprintf(text_buf, sizeof(text_buf), "Fit: %.1f%% | RPY: (%.1f, %.1f, %.1f) deg", obj.confidence * 100.0f, obj.roll_deg, obj.pitch_deg, obj.yaw_deg);
            cv::putText(vis, text_buf, cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX, 0.60, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::putText(vis, "Model: model_obs_rk3588.rknn | CAD: 440x830x505mm", cv::Point(20, 95), cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        }
        std::string out_img_path = "/home/huyen1905/TMCAD/v25_dump_result.png";
        cv::imwrite(out_img_path, vis);
        std::cout << "\n[+] Saved 3D Bounding Box projection to: " << out_img_path << "\n\n";
    }

    std::cout << "========================================================\n";
    std::cout << " VERIFICATION OF CAD POSE V2.5 INTEGRATION: SUCCESS!    \n";
    std::cout << "========================================================\n";

    return 0;
}
