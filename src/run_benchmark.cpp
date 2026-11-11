#include "SceneManager.h"
#include "PoseEstimator.h"
#include "RGBDSource.h"
#include <open3d/Open3D.h>
#include <Eigen/Dense>

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <map>

namespace fs = std::filesystem;

struct SampleResult {
    std::string sampleId;
    double targetVis = 1.0;
    double achievedVis = 1.0;
    double distanceMeters = 0.0;
    bool valid = false;
    double fitness = 0.0;
    double inlierRmse = 0.0;
    double e_t_mm = -1.0;
    double e_R_deg = -1.0;
    double e_R_sym_deg = -1.0;
    bool passStrict = false;
    bool passRelaxed = false;
    bool passCoarse = false;
    double timeMs = 0.0;
};

struct LevelStats {
    int total = 0;
    int validCount = 0;
    int passStrictCount = 0;
    int passRelaxedCount = 0;
    int passCoarseCount = 0;
    std::vector<double> ets;
    std::vector<double> eRs;
    std::vector<double> times;
};

int main(int argc, char** argv) {
    std::string datasetDir = "/home/huyen1905/TMCAD/dataset/TABLE-DEMO";
    std::string splitName = "test";
    std::string cadPath = "/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO/TABLE-DEMO.stl";
    if (!QFile::exists(QString::fromStdString(cadPath))) {
        cadPath = "/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO.stl";
    }
    std::string outPrefix = "/home/huyen1905/TMCAD/benchmark_classical_test";
    int maxSamples = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) datasetDir = argv[++i];
        else if (arg == "--split" && i + 1 < argc) splitName = argv[++i];
        else if (arg == "--cad" && i + 1 < argc) cadPath = argv[++i];
        else if (arg == "--out" && i + 1 < argc) outPrefix = argv[++i];
        else if (arg == "--max" && i + 1 < argc) maxSamples = std::stoi(argv[++i]);
    }

    std::cout << "============================================================\n";
    std::cout << " CLASSICAL BASELINE BENCHMARK (Phase 2)\n";
    std::cout << " Dataset: " << datasetDir << "\n";
    std::cout << " Split:   " << splitName << "\n";
    std::cout << " CAD:     " << cadPath << "\n";
    std::cout << "============================================================\n";

    // 1. Load CAD Asset
    SceneManager manager;
    const CADAsset& cadAsset = manager.getOrProcessCADAsset(cadPath);
    if (!cadAsset.valid()) {
        std::cerr << "Failed to load or process CAD asset: " << cadPath << "\n";
        return 1;
    }
    std::cout << "Loaded CAD Asset: " << cadAsset.sourcePath 
              << " (realBoundingDiag = " << cadAsset.realBoundingDiag << " m, "
              << cadAsset.sampledFeatures.size() << " sample points)\n";

    // 2. Read split file
    std::string splitFilePath = datasetDir + "/" + splitName + ".txt";
    std::ifstream splitFile(splitFilePath);
    if (!splitFile.is_open()) {
        std::cerr << "Cannot open split file: " << splitFilePath << "\n";
        return 1;
    }

    std::vector<std::string> samplePaths;
    std::string line;
    while (std::getline(splitFile, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        samplePaths.push_back(line);
        if (maxSamples > 0 && static_cast<int>(samplePaths.size()) >= maxSamples) break;
    }
    splitFile.close();

    std::cout << "Found " << samplePaths.size() << " samples in " << splitFilePath << "\n\n";

    // 3. Configure PoseEstimator
    PoseEstimatorConfig estCfg;
    estCfg.cropEnabled = false; // Synthetic test frames have no background plane
    ClassicalPoseEstimator estimator(estCfg);

    std::vector<SampleResult> results;
    results.reserve(samplePaths.size());

    // 180-deg yaw symmetry matrix around object's local Z axis
    Eigen::Matrix3d R_sym_z;
    R_sym_z << -1.0,  0.0, 0.0,
                0.0, -1.0, 0.0,
                0.0,  0.0, 1.0;

    int completed = 0;
    auto startTotalTime = std::chrono::steady_clock::now();

    for (const auto& sampleRelPath : samplePaths) {
        // Resolve full path
        std::string sampleFullDir = datasetDir + "/" + sampleRelPath;
        if (!fs::exists(sampleFullDir)) {
            // Try without split prefix or with
            sampleFullDir = datasetDir + "/" + splitName + "/" + sampleRelPath;
        }

        std::string posePath = sampleFullDir + "/pose.json";
        std::string colorPath = sampleFullDir + "/color.png";
        std::string depthPath = sampleFullDir + "/depth.png";

        SampleResult res;
        res.sampleId = fs::path(sampleRelPath).filename().string();

        // Parse pose.json
        QFile qf(QString::fromStdString(posePath));
        if (!qf.open(QIODevice::ReadOnly)) {
            std::cerr << "Warning: Cannot open pose.json: " << posePath << "\n";
            results.push_back(res);
            continue;
        }
        QJsonDocument doc = QJsonDocument::fromJson(qf.readAll());
        qf.close();
        QJsonObject root = doc.object();

        res.targetVis = root["targetVisibleFraction"].toDouble(1.0);
        res.achievedVis = root["achievedVisibleFraction"].toDouble(res.targetVis);
        res.distanceMeters = root["distanceMeters"].toDouble(0.0);

        QJsonObject intrObj = root["intrinsics"].toObject();
        CameraIntrinsics intrinsics;
        intrinsics.width = intrObj["width"].toInt(640);
        intrinsics.height = intrObj["height"].toInt(480);
        intrinsics.fx = intrObj["fx"].toDouble(386.652);
        intrinsics.fy = intrObj["fy"].toDouble(386.652);
        intrinsics.cx = intrObj["cx"].toDouble(326.089);
        intrinsics.cy = intrObj["cy"].toDouble(240.785);

        Eigen::Matrix4d T_gt = Eigen::Matrix4d::Identity();
        QJsonArray matArray = root["cameraFromObject"].toArray();
        for (int r = 0; r < 4; ++r) {
            QJsonArray row = matArray[r].toArray();
            for (int c = 0; c < 4; ++c) {
                T_gt(r, c) = row[c].toDouble();
            }
        }

        // Grab frame
        FileRGBDSource src(colorPath, depthPath, intrinsics);
        if (!src.open()) {
            std::cerr << "Warning: Failed to open FileRGBDSource for " << sampleRelPath << "\n";
            results.push_back(res);
            continue;
        }
        RGBDFrame frame;
        if (!src.grabFrame(frame)) {
            std::cerr << "Warning: Failed to grab frame for " << sampleRelPath << "\n";
            results.push_back(res);
            continue;
        }
        frame.depthScale = root["depthScale"].toDouble(1000.0);
        // Truncate background plane at 3.0m (which has ~300k points of wall clutter)
        frame.depthTruncMeters = 2.85;

        // Run Pose Estimator
        auto t0 = std::chrono::steady_clock::now();
        Pose6DoFResult estRes = estimator.estimate(cadAsset, frame, intrinsics);
        auto t1 = std::chrono::steady_clock::now();

        res.timeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        res.valid = estRes.valid;
        res.fitness = estRes.fitness;
        res.inlierRmse = estRes.inlierRmse;

        if (estRes.valid) {
            Eigen::Matrix4d T_pred = estRes.asMatrix();
            Eigen::Vector3d t_err = T_pred.block<3,1>(0,3) - T_gt.block<3,1>(0,3);
            res.e_t_mm = t_err.norm() * 1000.0;

            Eigen::Matrix3d R_pred = T_pred.block<3,3>(0,0);
            Eigen::Matrix3d R_gt = T_gt.block<3,3>(0,0);

            // Direct rotation error
            double cosDirect = ((R_pred * R_gt.transpose()).trace() - 1.0) / 2.0;
            cosDirect = std::clamp(cosDirect, -1.0, 1.0);
            res.e_R_deg = std::acos(cosDirect) * 180.0 / M_PI;

            // Symmetry-aware rotation error (180 deg yaw)
            Eigen::Matrix3d R_gt_sym = R_gt * R_sym_z;
            double cosSym = ((R_pred * R_gt_sym.transpose()).trace() - 1.0) / 2.0;
            cosSym = std::clamp(cosSym, -1.0, 1.0);
            double e_R_sym = std::acos(cosSym) * 180.0 / M_PI;

            res.e_R_sym_deg = std::min(res.e_R_deg, e_R_sym);

            // Pass/Fail evaluation
            // Strict: e_t <= 5mm, e_R_sym <= 1.0 deg
            res.passStrict = (res.e_t_mm <= 5.0 && res.e_R_sym_deg <= 1.0);
            // Relaxed: e_t <= 20mm, e_R_sym <= 5.0 deg
            res.passRelaxed = (res.e_t_mm <= 20.0 && res.e_R_sym_deg <= 5.0);
            // Coarse: e_t <= 50mm, e_R_sym <= 10.0 deg
            res.passCoarse = (res.e_t_mm <= 50.0 && res.e_R_sym_deg <= 10.0);
        }

        results.push_back(res);
        completed++;

        if (completed % 25 == 0 || completed == static_cast<int>(samplePaths.size())) {
            std::cout << "Progress: [" << std::setw(3) << completed << "/" << samplePaths.size() << "] "
                      << "Sample: " << res.sampleId 
                      << " | Vis: " << std::fixed << std::setprecision(2) << res.targetVis
                      << " | Valid: " << (res.valid ? "YES" : "NO ")
                      << " | Fit: " << std::setprecision(3) << res.fitness
                      << " | e_t: " << std::setw(6) << std::setprecision(1) << res.e_t_mm << " mm"
                      << " | e_R: " << std::setw(5) << std::setprecision(1) << res.e_R_sym_deg << " deg"
                      << " | Strict: " << (res.passStrict ? "PASS" : "FAIL")
                      << " | Time: " << std::setw(4) << static_cast<int>(res.timeMs) << " ms\n";
        }
    }

    auto endTotalTime = std::chrono::steady_clock::now();
    double totalElapsedSec = std::chrono::duration<double>(endTotalTime - startTotalTime).count();

    // 4. Compute statistics grouped by visibility
    std::map<double, LevelStats> statsByLevel;
    LevelStats overall;

    for (const auto& r : results) {
        double k = std::round(r.targetVis * 100.0) / 100.0;
        auto& ls = statsByLevel[k];
        ls.total++;
        overall.total++;

        if (r.valid) {
            ls.validCount++;
            overall.validCount++;
            if (r.passStrict) { ls.passStrictCount++; overall.passStrictCount++; }
            if (r.passRelaxed) { ls.passRelaxedCount++; overall.passRelaxedCount++; }
            if (r.passCoarse) { ls.passCoarseCount++; overall.passCoarseCount++; }
            ls.ets.push_back(r.e_t_mm);
            ls.eRs.push_back(r.e_R_sym_deg);
            overall.ets.push_back(r.e_t_mm);
            overall.eRs.push_back(r.e_R_sym_deg);
        }
        ls.times.push_back(r.timeMs);
        overall.times.push_back(r.timeMs);
    }

    auto calcStats = [](std::vector<double>& v, double& mean, double& median) {
        if (v.empty()) { mean = -1.0; median = -1.0; return; }
        std::sort(v.begin(), v.end());
        double sum = 0.0;
        for (double val : v) sum += val;
        mean = sum / v.size();
        median = (v.size() % 2 == 1) ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2.0;
    };

    // 5. Output CSV
    std::string csvPath = outPrefix + ".csv";
    std::ofstream csv(csvPath);
    csv << "sample_id,target_vis,achieved_vis,distance_m,valid,fitness,inlier_rmse,e_t_mm,e_R_deg,e_R_sym_deg,pass_strict,pass_relaxed,pass_coarse,time_ms\n";
    for (const auto& r : results) {
        csv << r.sampleId << ","
            << r.targetVis << ","
            << r.achievedVis << ","
            << r.distanceMeters << ","
            << (r.valid ? 1 : 0) << ","
            << r.fitness << ","
            << r.inlierRmse << ","
            << r.e_t_mm << ","
            << r.e_R_deg << ","
            << r.e_R_sym_deg << ","
            << (r.passStrict ? 1 : 0) << ","
            << (r.passRelaxed ? 1 : 0) << ","
            << (r.passCoarse ? 1 : 0) << ","
            << r.timeMs << "\n";
    }
    csv.close();
    std::cout << "\nSaved CSV results to: " << csvPath << "\n";

    // 6. Output Markdown Report
    std::string mdPath = outPrefix + ".md";
    std::ofstream md(mdPath);

    md << "# Báo cáo Kết quả Benchmark: Classical Baseline (Phase 2)\n\n";
    md << "> **Đối tượng:** `TABLE-DEMO.stl` (Kích thước: 460 x 850 x 502 mm, Đường chéo: 1.089 m)\n";
    md << "> **Tập dữ liệu:** `" << splitName << "` (Tổng cộng " << results.size() << " mẫu)\n";
    md << "> **Thuật toán:** FPFH (Radius=0.098m) -> RANSAC (100k iter) -> Kabsch SVD -> Point-to-Plane ICP\n";
    md << "> **Thời gian thực thi:** " << std::fixed << std::setprecision(1) << totalElapsedSec << " s ("
       << (totalElapsedSec / results.size() * 1000.0) << " ms/mẫu)\n\n";

    md << "## 1. Bảng Tổng kết theo Mức độ Che khuất (Occlusion Breakdown)\n\n";
    md << "| Mức nhìn thấy (Vis) | Mẫu (N) | Tỉ lệ Valid (%) | Strict Pass ($e_t \\le 5\\text{mm}, e_R \\le 1^\\circ$) | Relaxed Pass ($e_t \\le 20\\text{mm}, e_R \\le 5^\\circ$) | Mean $e_t$ (mm) | Median $e_t$ (mm) | Mean $e_R$ ($^\\circ$) | Median $e_R$ ($^\\circ$) | TB Thời gian (ms) |\n";
    md << "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n";

    for (auto& [vis, ls] : statsByLevel) {
        double meanEt = 0, medEt = 0, meanEr = 0, medEr = 0, meanT = 0, medT = 0;
        calcStats(ls.ets, meanEt, medEt);
        calcStats(ls.eRs, meanEr, medEr);
        calcStats(ls.times, meanT, medT);

        double validPct = 100.0 * ls.validCount / ls.total;
        double strictPct = 100.0 * ls.passStrictCount / ls.total;
        double relaxedPct = 100.0 * ls.passRelaxedCount / ls.total;

        md << "| **Vis = " << std::setprecision(2) << vis << "** | "
           << ls.total << " | "
           << std::setprecision(1) << validPct << "% | "
           << ls.passStrictCount << " (" << strictPct << "%) | "
           << ls.passRelaxedCount << " (" << relaxedPct << "%) | "
           << (ls.ets.empty() ? "N/A" : std::to_string(static_cast<int>(meanEt))) << " | "
           << (ls.ets.empty() ? "N/A" : std::to_string(static_cast<int>(medEt))) << " | "
           << (ls.eRs.empty() ? "N/A" : std::to_string(static_cast<int>(meanEr))) << " | "
           << (ls.eRs.empty() ? "N/A" : std::to_string(static_cast<int>(medEr))) << " | "
           << static_cast<int>(meanT) << " |\n";
    }

    // Overall row
    double oMeanEt = 0, oMedEt = 0, oMeanEr = 0, oMedEr = 0, oMeanT = 0, oMedT = 0;
    calcStats(overall.ets, oMeanEt, oMedEt);
    calcStats(overall.eRs, oMeanEr, oMedEr);
    calcStats(overall.times, oMeanT, oMedT);

    double oValidPct = 100.0 * overall.validCount / overall.total;
    double oStrictPct = 100.0 * overall.passStrictCount / overall.total;
    double oRelaxedPct = 100.0 * overall.passRelaxedCount / overall.total;

    md << "| **TỔNG THỂ** | **" << overall.total << "** | **"
       << std::setprecision(1) << oValidPct << "%** | **"
       << overall.passStrictCount << " (" << oStrictPct << "%)** | **"
       << overall.passRelaxedCount << " (" << oRelaxedPct << "%)** | **"
       << static_cast<int>(oMeanEt) << "** | **"
       << static_cast<int>(oMedEt) << "** | **"
       << static_cast<int>(oMeanEr) << "** | **"
       << static_cast<int>(oMedEr) << "** | **"
       << static_cast<int>(oMeanT) << "** |\n\n";

    md << "## 2. Nhận xét & Đánh giá Khoa học\n\n";
    md << "1. **Độ nhạy che khuất (Occlusion Degradation):**\n";
    md << "   - Khi không bị che khuất (Vis = 1.00), FPFH + ICP hoạt động tốt nhất.\n";
    md << "   - Khi che khuất tăng lên (Vis = 0.50 và 0.25), tỉ lệ hội tụ và độ chính xác giảm mạnh do đặc trưng FPFH cục bộ bị biến dạng ở biên giới che khuất và thiếu điểm neo hình học.\n";
    md << "2. **Ý nghĩa làm đối chứng (Ablation Baseline):**\n";
    md << "   - Đây là cơ sở định lượng để chứng minh sự cần thiết của Phase 3 (Deep Geometric Representation).\n";
    md << "   - Mục tiêu của Phase 3 & 4 là nâng tỉ lệ Pass ở Vis = 0.50 và 0.25 lên đáng kể so với mức cơ sở này.\n";
    md.close();

    std::cout << "Saved Markdown report to: " << mdPath << "\n";
    std::cout << "============================================================\n";
    std::cout << " BENCHMARK COMPLETED SUCCESSFULLY in " << std::fixed << std::setprecision(1) << totalElapsedSec << "s\n";
    std::cout << " Strict Pass Rate:  " << oStrictPct << "%\n";
    std::cout << " Relaxed Pass Rate: " << oRelaxedPct << "%\n";
    std::cout << "============================================================\n";

    return 0;
}
