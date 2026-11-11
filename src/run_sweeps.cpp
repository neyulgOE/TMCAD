#include "SceneManager.h"
#include "SyntheticRGBDSource.h"
#include "PoseEstimator.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <algorithm>
#include <QFile>

void runSweepMulti(const CADAsset& cadAsset, std::shared_ptr<open3d::geometry::TriangleMesh> metricMesh,
              double d, double rotX, double rotY, double rotZ, bool bgPlane, bool crop, const std::string& label, int nRuns = 10) {
    
    std::vector<double> fitnesses, e_ts, e_Rs;
    int passCount = 0;
    int correctPoseCount = 0;
    int wrongPoseCount = 0;

    for (int i = 0; i < nRuns; ++i) {
        CameraIntrinsics intrinsics{640, 480, 386.652, 386.652, 326.089, 240.785};
        
        const double floorZ = metricMesh->GetMinBound().z();
        const double camZ = floorZ + 0.23; // real AMR camera is 0.23m above floor
        Eigen::Vector3d camPos(-d, 0.0, camZ);
        Eigen::Vector3d lookTarget(0.0, 0.0, camZ);
        Eigen::Vector3d forward = (lookTarget - camPos).normalized();
        Eigen::Vector3d upWorld(0, 0, 1);
        Eigen::Vector3d right = forward.cross(upWorld).normalized();
        Eigen::Vector3d upCam = right.cross(forward).normalized();

        Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
        Twc.block<3,1>(0,0) = right;
        Twc.block<3,1>(0,1) = -upCam;
        Twc.block<3,1>(0,2) = forward;
        Twc.block<3,1>(0,3) = camPos;

        Eigen::Matrix4d Two = Eigen::Matrix4d::Identity();
        Eigen::Matrix3d Rx = Eigen::AngleAxisd(rotX * M_PI / 180.0, Eigen::Vector3d::UnitX()).toRotationMatrix();
        Eigen::Matrix3d Ry = Eigen::AngleAxisd(rotY * M_PI / 180.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
        Eigen::Matrix3d Rz = Eigen::AngleAxisd(rotZ * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        Two.block<3,3>(0,0) = Rz * Ry * Rx;

        Eigen::Matrix4d gtPose = Twc.inverse() * Two;

        SyntheticSceneConfig sceneCfg;
        sceneCfg.cameraFromObject = gtPose;
        sceneCfg.addBackgroundPlane = bgPlane;
        sceneCfg.backgroundPlaneDistanceMeters = 3.0;
        sceneCfg.backgroundPlaneSizeMeters = 10.0;

        auto source = std::make_shared<SyntheticRGBDSource>(metricMesh, intrinsics, sceneCfg);
        if (!source->open()) continue;

        RGBDFrame syntheticFrame;
        source->grabFrame(syntheticFrame);
        source->close();

        PoseEstimatorConfig cfg;
        cfg.cropEnabled = crop;
        cfg.cropMin = Eigen::Vector3d(-0.5, -0.5, 0.0);
        cfg.cropMax = Eigen::Vector3d(0.5, 0.5, 2.0); // 2.0m max Z to cut out 3.0m bg plane
        cfg.normalRadius = 0.049;
        cfg.fpfhRadius = 0.098;
        cfg.sceneVoxelSize = 0.013; // Reverting to original default 0.013 as requested earlier

        ClassicalPoseEstimator estimator(cfg);
        auto result = estimator.estimate(cadAsset, syntheticFrame, intrinsics);

        double e_t = -1.0;
        double e_R_deg = -1.0;
        if (result.valid) {
            Eigen::Vector3d tGt = gtPose.block<3, 1>(0, 3);
            Eigen::Matrix3d RGt = gtPose.block<3, 3>(0, 0);

            e_t = (result.t - tGt).norm();
            double cosArg = (result.R * RGt.transpose()).trace() - 1.0;
            cosArg = std::clamp(cosArg / 2.0, -1.0, 1.0);
            e_R_deg = std::acos(cosArg) * 180.0 / M_PI;
        }

        fitnesses.push_back(result.fitness);
        if (result.valid) {
            passCount++;
            e_ts.push_back(e_t);
            e_Rs.push_back(e_R_deg);
            
            if (e_R_deg < 5.0) {
                correctPoseCount++;
            } else {
                wrongPoseCount++;
            }
        }
        
        // CSV Logging
        std::string logPath = "/home/huyen1905/TMCAD/logs/pose_runs.csv";
        std::ifstream fIn(logPath);
        bool needHeader = !fIn.good();
        fIn.close();
        
        std::ofstream fOut(logPath, std::ios::app);
        if (needHeader) {
            fOut << "timestamp,run_label,real_bounding_diag,"
                   "crop_enabled,crop_min_x,crop_min_y,crop_min_z,crop_max_x,crop_max_y,crop_max_z,"
                   "voxel_size,outlier_neighbors,outlier_std_ratio,"
                   "normal_radius,fpfh_radius,"
                   "ransac_dist_thresh,ransac_max_iter,ransac_confidence,"
                   "icp_enabled,icp_max_corr_dist,"
                   "raw_scene_points,scene_cloud_points,cad_cloud_points,"
                   "cad_cloud_diag,scene_cloud_diag,"
                   "fitness,inlier_rmse,num_inliers,valid,e_t,e_R_deg\n";
        }
        
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        char timeStr[100];
        std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%dT%H:%M:%S", std::localtime(&now_c));
        
        fOut << timeStr << ",\"" << label << "_run" << i << "\","
             << cadAsset.realBoundingDiag << ","
             << (cfg.cropEnabled ? 1 : 0) << ","
             << cfg.cropMin.x() << "," << cfg.cropMin.y() << "," << cfg.cropMin.z() << ","
             << cfg.cropMax.x() << "," << cfg.cropMax.y() << "," << cfg.cropMax.z() << ","
             << cfg.sceneVoxelSize << "," << cfg.outlierNeighbors << "," << cfg.outlierStdRatio << ","
             << cfg.normalRadius << "," << cfg.fpfhRadius << ","
             << cfg.ransacDistanceThreshold << "," << cfg.ransacMaxIterations << "," << cfg.ransacConfidence << ","
             << (cfg.refineWithICP ? 1 : 0) << "," << cfg.icpMaxCorrespondenceDistance << ","
             << result.numRawScenePoints << "," << result.numSceneCloudPoints << "," << result.numCadCloudPoints << ","
             << result.cadCloudDiag << "," << result.sceneCloudDiag << ","
             << result.fitness << "," << result.inlierRmse << "," << result.numRansacInliers << ","
             << (result.valid ? 1 : 0) << "," << e_t << "," << e_R_deg << "\n";
    }

    auto mean = [](const std::vector<double>& v) {
        return v.empty() ? -1.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    auto stddev = [&mean](const std::vector<double>& v) {
        if (v.size() < 2) return 0.0;
        double m = mean(v);
        double sq = 0;
        for (double x : v) sq += (x-m)*(x-m);
        return std::sqrt(sq / v.size());
    };

    std::cout << std::left << std::setw(35) << label
              << " | pass=" << passCount << "/" << nRuns
              << " | correct=" << correctPoseCount << "/" << nRuns
              << " | wrong=" << wrongPoseCount << "/" << nRuns
              << " | fit=" << std::fixed << std::setprecision(4) << mean(fitnesses) << "±" << stddev(fitnesses)
              << " | e_t=" << std::fixed << std::setprecision(4) << mean(e_ts) << "m"
              << " | e_R=" << std::fixed << std::setprecision(2) << mean(e_Rs) << "°\n";
}

int main() {
    SceneManager manager;
    std::string asymPath = "/home/huyen1905/TMCAD/CAD_DATA/ASYMMETRIC/ASYMMETRIC.stl";
    if (!QFile::exists(QString::fromStdString(asymPath))) {
        asymPath = "/home/huyen1905/TMCAD/CAD_DATA/ASYMMETRIC.stl";
    }
    const CADAsset& cadAssetAsync = manager.getOrProcessCADAsset(asymPath);
    auto metricMeshAsync = SyntheticRGBDSource::meshFromCADAsset(cadAssetAsync);

    std::string tablePath = "/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO/TABLE-DEMO.stl";
    if (!QFile::exists(QString::fromStdString(tablePath))) {
        tablePath = "/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO.stl";
    }
    const CADAsset& cadAssetTable = manager.getOrProcessCADAsset(tablePath);
    auto metricMeshTable = SyntheticRGBDSource::meshFromCADAsset(cadAssetTable);

    std::cout << "========================================================================\n";
    std::cout << "Phase 2 Sweep Evaluation - Multi-run Statistics (N=10)\n";
    std::cout << "========================================================================\n";

    runSweepMulti(cadAssetAsync, metricMeshAsync, 1.5, 0, 0, 0, false, false, "ASYMMETRIC, d=1.5m, Rot(0,0,0)", 10);
    runSweepMulti(cadAssetAsync, metricMeshAsync, 1.5, 45, 0, 0, false, false, "ASYMMETRIC, d=1.5m, Pitch(45,0,0)", 10);
    runSweepMulti(cadAssetAsync, metricMeshAsync, 2.0, 0, 0, 0, false, false, "ASYMMETRIC, d=2.0m, Rot(0,0,0)", 10);
    runSweepMulti(cadAssetTable, metricMeshTable, 1.5, 0, 0, 0, false, true, "TABLE-DEMO, d=1.5m, crop=ON", 10);

    std::cout << "========================================================================\n";
    return 0;
}
