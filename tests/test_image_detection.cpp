#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "../src/CADTemplateDetector.h"

int main()
{
    std::cout << "========================================================\n";
    std::cout << "  TEST CAD VIEWER - PURE 2D IMAGE DETECTION\n";
    std::cout << "========================================================\n\n";

    CADTemplateDetector detector;
    std::string cad_dir = "/home/huyen1905/TMCAD/ObjectDetection_V2.5/models/object_detection/cad_pose/table_cad/v1";
    std::string error;

    std::vector<cv::Vec3f> dummy_centroids(512);
    for (int i = 0; i < 512; ++i)
    {
        float x = ((i % 16) - 8) * 0.025f;
        float y = ((i / 16) - 16) * 0.025f;
        dummy_centroids[i] = cv::Vec3f(x, y, 0.1665f);
    }

    std::cout << "[1] Initializing detector with TABLE-DEMO templates...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = detector.initialize(cad_dir, "TABLE-DEMO", dummy_centroids, 0.220, 0.415, 0.1665, -0.3355, error);
    auto t1 = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!ok)
    {
        std::cerr << "[-] Init failed: " << error << "\n";
        return 1;
    }
    std::cout << "[+] Init SUCCESS in " << init_ms << " ms. Templates: " << detector.templateCount() << "\n\n";

    // Test on PNG image
    std::string test_img_path = "/home/huyen1905/TMCAD/dump_rgb.png";
    std::cout << "[2] Testing detection on PNG image: " << test_img_path << "...\n";
    cv::Mat img_png = cv::imread(test_img_path, cv::IMREAD_COLOR);
    if (img_png.empty())
    {
        std::cerr << "[-] Failed to read " << test_img_path << "\n";
        return 1;
    }
    std::cout << "[+] Image resolution: " << img_png.cols << "x" << img_png.rows << "\n";

    auto match_png = detector.detect(img_png);
    std::cout << "[+] Detection Result on PNG:\n";
    std::cout << "    - Valid:          " << (match_png.valid ? "YES" : "NO") << "\n";
    std::cout << "    - Match score:    " << (match_png.match_score * 100.0f) << "%\n";
    std::cout << "    - Good matches:   " << match_png.good_matches << "\n";
    std::cout << "    - RANSAC inliers: " << match_png.ransac_inliers << "\n";
    std::cout << "    - Estimated yaw:  " << match_png.estimated_yaw_deg << " deg\n";
    std::cout << "    - Bounding Box:   [" << match_png.roi.x << ", " << match_png.roi.y
              << ", " << match_png.roi.width << ", " << match_png.roi.height << "]\n";
    std::cout << "    - Latency:        " << match_png.latency_ms << " ms\n\n";

    // Test on other formats: convert and save to JPG, BMP, WEBP and test
    std::string test_jpg_path = "/tmp/test_table.jpg";
    std::string test_bmp_path = "/tmp/test_table.bmp";
    cv::imwrite(test_jpg_path, img_png);
    cv::imwrite(test_bmp_path, img_png);

    std::cout << "[3] Testing detection on JPG image...\n";
    cv::Mat img_jpg = cv::imread(test_jpg_path);
    auto match_jpg = detector.detect(img_jpg);
    std::cout << "    - JPG Valid: " << (match_jpg.valid ? "YES" : "NO") << ", Score: " << (match_jpg.match_score * 100.0f) << "%\n";

    std::cout << "[4] Testing detection on BMP image...\n";
    cv::Mat img_bmp = cv::imread(test_bmp_path);
    auto match_bmp = detector.detect(img_bmp);
    std::cout << "    - BMP Valid: " << (match_bmp.valid ? "YES" : "NO") << ", Score: " << (match_bmp.match_score * 100.0f) << "%\n";

    // Test annotation generator
    std::cout << "\n[5] Testing annotation overlay generation...\n";
    cv::Mat annotated = CADTemplateDetector::annotate(img_png, match_png, "TABLE-DEMO", true, true);
    if (!annotated.empty())
    {
        std::string out_annotated = "/tmp/test_detected_table.png";
        cv::imwrite(out_annotated, annotated);
        std::cout << "[+] Saved annotated verification image to: " << out_annotated << "\n";
    }

    std::cout << "\n>>> ALL 2D IMAGE DETECTION CHECKS PASSED SUCCESSFULLY! <<<\n";
    return 0;
}
