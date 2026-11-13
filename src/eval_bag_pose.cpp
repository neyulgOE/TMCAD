#include "UniversalPoseEstimator.h"
#include "RGBDSource.h"

#include <open3d/Open3D.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static double medianOf(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double percentileOf(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t idx = std::min(v.size() - 1, static_cast<size_t>(std::round(p * (v.size() - 1))));
    return v[idx];
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <bag_path> <onnx_path> <out_dir> [--max-frames N] [--multi] [--temporal]\n";
        return 1;
    }

    const std::string bag_path = argv[1];
    const std::string onnx_path = argv[2];
    const std::string out_dir = argv[3];
    int max_frames = -1;
    bool multi_mode = false;
    bool use_temporal = false;

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--max-frames" && i + 1 < argc) {
            max_frames = std::stoi(argv[++i]);
        } else if (arg == "--multi") {
            multi_mode = true;
        } else if (arg == "--temporal") {
            use_temporal = true;
        }
    }

    fs::create_directories(fs::path(out_dir) / "overlays");

    tmcad::UniversalPoseEstimator estimator;
    if (!estimator.loadModelOnnx(onnx_path)) {
        std::cerr << "[eval_bag_pose] Failed to load ONNX: " << onnx_path << "\n";
        return 2;
    }
    estimator.setMultiInstanceMode(multi_mode, 4);

    RealSenseBagSource bag_src(bag_path);
    if (!bag_src.open()) {
        std::cerr << "[eval_bag_pose] Failed to open .bag: " << bag_path << "\n";
        return 3;
    }

    const CameraIntrinsics base_intr = bag_src.intrinsics();
    const std::string csv_path = (fs::path(out_dir) / "bag_eval.csv").string();
    std::ofstream csv(csv_path);
    csv << "frame_idx,num_accepted,accepted,confidence,rmse_mm,inlier_ratio,"
        << "tx,ty,tz,roll_deg,pitch_deg,yaw_deg,bbox_x,bbox_y,bbox_w,bbox_h,"
        << "valid_depth_pts,ai_ms,solve_ms,total_ms,method\n";

    bool has_last = false;
    tmcad::ObjectPose last_pose;
    int hit_streak = 0;
    int miss_streak = 0;
    cv::Rect auto_roi;

    int frame_idx = 0;
    int accepted_frames = 0;
    std::vector<double> confs, rmses_mm, latencies_ms;

    RGBDFrame frame;
    while ((max_frames < 0 || frame_idx < max_frames) && bag_src.grabFrame(frame)) {
        if (!frame.color || frame.color->IsEmpty()) {
            ++frame_idx;
            continue;
        }
        const int w = frame.color->width_;
        const int h = frame.color->height_;
        cv::Mat rgb(h, w, CV_8UC3, const_cast<uint8_t*>(frame.color->data_.data()));
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

        cv::Mat depth;
        if (frame.depth && !frame.depth->IsEmpty()) {
            cv::Mat raw_d(frame.depth->height_, frame.depth->width_, CV_16UC1,
                          const_cast<uint8_t*>(frame.depth->data_.data()));
            if (raw_d.cols != w || raw_d.rows != h) {
                cv::resize(raw_d, depth, cv::Size(w, h), 0.0, 0.0, cv::INTER_NEAREST);
            } else {
                depth = raw_d;
            }
        }

        CameraIntrinsics intr = base_intr;
        intr.width = w;
        intr.height = h;
        intr.depth_scale = (frame.depthScale > 1e-6) ? (1.0 / frame.depthScale) : 0.001;
        if (intr.ppx <= 1e-6 && intr.cx > 1e-6) intr.ppx = intr.cx;
        if (intr.ppy <= 1e-6 && intr.cy > 1e-6) intr.ppy = intr.cy;
        if (intr.cx <= 1e-6 && intr.ppx > 1e-6) intr.cx = intr.ppx;
        if (intr.cy <= 1e-6 && intr.ppy > 1e-6) intr.cy = intr.ppy;

        const bool used_auto_roi = (use_temporal && !multi_mode && has_last && !auto_roi.empty());
        std::vector<tmcad::ObjectPose> poses =
            estimator.estimateMultiObjects(bgr, depth, intr, used_auto_roi ? auto_roi : cv::Rect());
        if (used_auto_roi && poses.empty()) {
            poses = estimator.estimateMultiObjects(bgr, depth, intr, cv::Rect());
        }

        const double diag_m = estimator.realBoundingDiagM();
        const bool floor_mode = estimator.isFloorStanding();
        poses.erase(std::remove_if(poses.begin(), poses.end(), [&](const tmcad::ObjectPose& p) {
            return !tmcad::DetectionAcceptancePolicy::isAccepted(p, diag_m, floor_mode);
        }), poses.end());

        if (use_temporal && !multi_mode) {
            if (!poses.empty() && has_last && miss_streak < 3) {
                const double jump_m = cv::norm(poses.front().t - last_pose.t);
                if (jump_m > std::max(0.20, 1.5 * diag_m) && poses.front().confidence < 0.85f) {
                    poses.clear();
                }
            }
            if (!poses.empty()) {
                hit_streak = std::min(3, hit_streak + 1);
                miss_streak = 0;
                last_pose = poses.front();
                has_last = true;
                const cv::Rect2f& b = last_pose.bbox_2d;
                if (b.width > 4.0f && b.height > 4.0f) {
                    const int pad = std::max(40, static_cast<int>(std::round(0.45f * std::max(b.width, b.height))));
                    const int rx = std::max(0, static_cast<int>(std::floor(b.x)) - pad);
                    const int ry = std::max(0, static_cast<int>(std::floor(b.y)) - pad);
                    const int rx2 = std::min(w, static_cast<int>(std::ceil(b.x + b.width)) + pad);
                    const int ry2 = std::min(h, static_cast<int>(std::ceil(b.y + b.height)) + pad);
                    if (rx2 - rx >= 64 && ry2 - ry >= 64) {
                        auto_roi = cv::Rect(rx, ry, rx2 - rx, ry2 - ry);
                    }
                }
                if (hit_streak < 2 && poses.front().confidence < 0.85f) {
                    poses.clear();
                }
            } else {
                ++miss_streak;
                if (has_last && miss_streak <= 2) {
                    poses.push_back(last_pose);
                } else {
                    has_last = false;
                    hit_streak = 0;
                    auto_roi = cv::Rect();
                }
            }
        }

        cv::Mat vis = bgr.clone();
        estimator.drawMultiOverlay(vis, poses, intr);
        char fname[64];
        std::snprintf(fname, sizeof(fname), "frame_%04d.jpg", frame_idx);
        cv::imwrite((fs::path(out_dir) / "overlays" / fname).string(), vis);

        const bool acc = !poses.empty();
        if (acc) {
            ++accepted_frames;
            confs.push_back(poses.front().confidence);
            rmses_mm.push_back(poses.front().rmse_m * 1000.0);
        }
        latencies_ms.push_back(estimator.lastTotalLatencyMs());

        if (acc) {
            const auto& p = poses.front();
            csv << frame_idx << "," << poses.size() << ",1,"
                << std::fixed << std::setprecision(4) << p.confidence << ","
                << std::setprecision(2) << (p.rmse_m * 1000.0) << ","
                << std::setprecision(4) << p.inlier_ratio << ","
                << p.t[0] << "," << p.t[1] << "," << p.t[2] << ","
                << std::setprecision(2) << p.roll_deg << "," << p.pitch_deg << "," << p.yaw_deg << ","
                << p.bbox_2d.x << "," << p.bbox_2d.y << "," << p.bbox_2d.width << "," << p.bbox_2d.height << ","
                << estimator.lastValidDepthPts() << ","
                << estimator.lastAiLatencyMs() << ","
                << estimator.lastSolveLatencyMs() << ","
                << estimator.lastTotalLatencyMs() << ","
                << p.method << "\n";
        } else {
            csv << frame_idx << ",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
                << estimator.lastValidDepthPts() << ","
                << std::fixed << std::setprecision(2)
                << estimator.lastAiLatencyMs() << ","
                << estimator.lastSolveLatencyMs() << ","
                << estimator.lastTotalLatencyMs() << ",none\n";
        }
        ++frame_idx;
    }

    bag_src.close();
    csv.close();

    std::cout << "[eval_bag_pose] Frames: " << frame_idx
              << " | Accepted: " << accepted_frames
              << " (" << std::fixed << std::setprecision(1)
              << (frame_idx > 0 ? 100.0 * accepted_frames / frame_idx : 0.0) << "%)"
              << " | Med Conf: " << std::setprecision(3) << medianOf(confs)
              << " | Med RMSE: " << std::setprecision(2) << medianOf(rmses_mm) << " mm"
              << " | Lat p50/p95: " << std::setprecision(1) << medianOf(latencies_ms)
              << " / " << percentileOf(latencies_ms, 0.95) << " ms\n"
              << "[eval_bag_pose] CSV: " << csv_path << "\n";
    return 0;
}

