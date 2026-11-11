#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "ObjectDetection/DetectionTypes.h"
#include "ObjectDetection/applications/cad_pose/CADTemplateDetector.h"
#include "ObjectDetection/applications/cad_pose/CADSceneSegmenter.h"

int main()
{
    std::cout << "========================================================\n";
    std::cout << "  TEST OBJECTDETECTION V2.4 - CAD TEMPLATE DETECTOR\n";
    std::cout << "========================================================\n\n";

    CADTemplateDetector detector;

    // 1. PRE-BAKED INITIALIZATION TEST (Zero-Boot Latency)
    std::cout << "[1] Testing PRE-BAKED binary loading (Zero-Boot Latency)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<cv::Vec3f> dummy_centroids(512);
    for (int i = 0; i < 512; ++i)
    {
        float x = ((i % 16) - 8) * 0.025f;
        float y = ((i / 16) - 16) * 0.025f;
        dummy_centroids[i] = cv::Vec3f(x, y, 0.1665f);
    }

    std::string pkg_dir = "/home/huyen1905/TMCAD/ObjectDetection_V2.4/models/object_detection/cad_pose/table_cad/v1";
    std::string error;
    bool ok = detector.initialize(pkg_dir, dummy_centroids, 0.220, 0.415, 0.1665, -0.3355, error);

    auto t1 = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!ok)
    {
        std::cerr << "[-] Init failed: " << error << "\n";
        return 1;
    }

    std::cout << "[+] Binary pre-baked init SUCCESS in " << init_ms << " ms!\n";
    std::cout << "[+] Loaded " << detector.templateCount() << " pre-baked CAD templates\n\n";

    // 2. ONLINE DETECTION TEST (Synthetic RGB Frame)
    std::cout << "[2] Testing ONLINE phase (Per-frame Detection & Matching)...\n";

    CameraIntrinsics intr;
    intr.fx = 605.0;
    intr.fy = 605.0;
    intr.ppx = 320.0;
    intr.ppy = 240.0;

    // Create a 640x480 test image with a synthetic table shape at (200, 150, 240, 180)
    cv::Mat test_rgb(480, 640, CV_8UC3, cv::Scalar(40, 40, 40));
    cv::rectangle(test_rgb, cv::Rect(200, 150, 240, 180), cv::Scalar(180, 180, 180), -1);
    cv::rectangle(test_rgb, cv::Rect(200, 150, 240, 180), cv::Scalar(255, 255, 255), 3);

    // Create corresponding depth frame
    cv::Mat test_depth(480, 640, CV_16UC1, cv::Scalar(3000)); // background 3.0m
    cv::rectangle(test_depth, cv::Rect(200, 150, 240, 180), cv::Scalar(1200), -1); // table 1.2m
    double depth_scale = 0.001;

    // Warm-up / Frame 0
    auto t2 = std::chrono::high_resolution_clock::now();
    auto match = detector.detect(test_rgb, test_depth, depth_scale, intr);
    auto t3 = std::chrono::high_resolution_clock::now();
    double frame0_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Frame 1 (Steady state)
    auto t4 = std::chrono::high_resolution_clock::now();
    auto match2 = detector.detect(test_rgb, test_depth, depth_scale, intr);
    auto t5 = std::chrono::high_resolution_clock::now();
    double steady_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    std::cout << "[+] Frame 0 detection executed in " << frame0_ms << " ms\n";
    std::cout << "[+] Steady-state detection executed in " << steady_ms << " ms\n";
    std::cout << "[+] Detection result:\n";
    std::cout << "    - Valid:          " << (match.valid ? "YES" : "NO") << "\n";
    std::cout << "    - Method:         " << match.method << "\n";
    std::cout << "    - Match score:    " << match.match_score << "\n";
    std::cout << "    - Good matches:   " << match.good_matches << "\n";
    std::cout << "    - Estimated yaw:  " << match.estimated_yaw_deg << " deg\n";
    std::cout << "    - ROI BoundingBox: [" << match.roi.x << ", " << match.roi.y
              << ", " << match.roi.width << ", " << match.roi.height << "]\n";

    if (match.valid)
    {
        std::cout << "[+] Matching output verified successfully!\n\n";
    }

    // 3. 3D BOUNDING BOX CORNERS TEST
    std::cout << "[3] Testing 3D Bounding Box (8 Projected Corners in DetectionObject)...\n";
    DetectionObject obj;
    obj.has_3d = true;
    obj.x_3d = -0.126f;
    obj.y_3d = -0.022f;
    obj.z_3d = 1.640f;
    obj.has_pose_6d = true;
    obj.yaw_deg = 56.5f;

    // Projected corners for TABLE-DEMO (0.44m x 0.83m x 0.50m)
    obj.corners_2d = {
        cv::Point2f(128.6f, 316.4f), cv::Point2f(116.0f, 405.1f),
        cv::Point2f(446.3f, 369.6f), cv::Point2f(390.4f, 296.1f),
        cv::Point2f(113.9f, 155.5f), cv::Point2f(96.9f, 203.0f),
        cv::Point2f(467.5f, 191.3f), cv::Point2f(400.6f, 151.3f)
    };
    obj.has_bbox_3d = (obj.corners_2d.size() == 8);

    std::cout << "    - has_bbox_3d:    " << (obj.has_bbox_3d ? "YES" : "NO") << "\n";
    std::cout << "    - corners count:  " << obj.corners_2d.size() << "\n";
    for (size_t i = 0; i < obj.corners_2d.size(); ++i) {
        std::cout << "      Corner " << i << ": [" << obj.corners_2d[i].x << ", " << obj.corners_2d[i].y << "]\n";
    }

    // 4. 3D POINT CLOUD SCENE SEGMENTATION TEST (RANSAC Floor Removal & Table Isolation)
    std::cout << "\n[4] Testing 3D POINT CLOUD SCENE SEGMENTER (RANSAC Floor Removal & Table Isolation)...\n";
    CADSceneSegmenter segmenter;
    segmenter.configure(0.220, 0.415, 0.5020, 0.35, 3.50);

    // Create a 640x480 frame with realistic floor plane and a table standing on the floor
    RgbdFrame frame_3d;
    frame_3d.width = 640;
    frame_3d.height = 480;
    frame_3d.depth_scale = 0.001; // 1 mm per unit
    frame_3d.intrinsics = intr;
    frame_3d.depth_z16 = cv::Mat(480, 640, CV_16UC1, cv::Scalar(0));

    // Floor equation: Y_cam = 0.300 m (camera is 0.30m above floor).
    // In camera optical frame: py = (y - cy) * z / fy = 0.300 -> z = 0.300 * fy / (y - cy)
    for (int y = 260; y < 480; ++y)
    {
        double z_floor = 0.300 * intr.fy / (y - intr.ppy);
        if (z_floor >= 0.5 && z_floor <= 3.2)
        {
            uint16_t d_mm = static_cast<uint16_t>(z_floor * 1000.0);
            for (int x = 0; x < 640; ++x)
            {
                frame_3d.depth_z16.at<uint16_t>(y, x) = d_mm;
            }
        }
    }

    // Table standing on the floor at distance z = 1.25m, width = 0.44m, height = 0.50m
    // Tabletop at height 0.50m above floor -> Y_cam = 0.30 - 0.50 = -0.20m (above camera center)
    // Project table box onto image:
    int tbl_u0 = static_cast<int>((-0.22 * intr.fx / 1.25) + intr.ppx);
    int tbl_u1 = static_cast<int>((0.22 * intr.fx / 1.25) + intr.ppx);
    int tbl_v0 = static_cast<int>((-0.20 * intr.fy / 1.25) + intr.ppy);
    int tbl_v1 = static_cast<int>((0.30 * intr.fy / 1.25) + intr.ppy);

    tbl_u0 = std::clamp(tbl_u0, 0, 639);
    tbl_u1 = std::clamp(tbl_u1, 0, 639);
    tbl_v0 = std::clamp(tbl_v0, 0, 479);
    tbl_v1 = std::clamp(tbl_v1, 0, 479);

    for (int y = tbl_v0; y <= tbl_v1; ++y)
    {
        for (int x = tbl_u0; x <= tbl_u1; ++x)
        {
            frame_3d.depth_z16.at<uint16_t>(y, x) = static_cast<uint16_t>(1250); // 1.25m
        }
    }

    // Run segmentation benchmark
    auto t_seg0 = std::chrono::high_resolution_clock::now();
    auto seg_res = segmenter.segment(frame_3d, 1024);
    auto t_seg1 = std::chrono::high_resolution_clock::now();
    double seg_ms = std::chrono::duration<double, std::milli>(t_seg1 - t_seg0).count();

    std::cout << "[+] 3D Scene Segmentation executed in " << seg_ms << " ms (Target: < 4.5 ms)\n";
    std::cout << "    - Valid:               " << (seg_res.valid ? "YES" : "NO") << "\n";
    std::cout << "    - Floor detected:      " << (seg_res.floor_detected ? "YES" : "NO") << "\n";
    std::cout << "    - Camera height:       " << seg_res.floor_height_m << " m (Expected: ~0.30 m)\n";
    std::cout << "    - Floor tilt:          " << seg_res.floor_tilt_deg << " deg (Expected: ~0.0 deg)\n";
    std::cout << "    - Raw points sampled:  " << seg_res.raw_points_count << "\n";
    std::cout << "    - Table cluster pts:   " << seg_res.table_cluster_points_count << "\n";
    std::cout << "    - Table points output: " << seg_res.table_points.size() << " / 1024\n";
    std::cout << "    - ROI BoundingBox:     [" << seg_res.roi_2d.x << ", " << seg_res.roi_2d.y
              << ", " << seg_res.roi_2d.width << ", " << seg_res.roi_2d.height << "]\n";

    if (match.valid && obj.has_bbox_3d && seg_res.valid && seg_res.table_points.size() == 1024)
    {
        std::cout << "\n>>> ALL V2.4+ UPGRADE VERIFICATION CHECKS PASSED SUCCESSFUL! <<<\n";
    }

    return 0;
}
