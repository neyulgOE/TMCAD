#include "RealSenseRecorderDialog.h"
#include <librealsense2/rs.hpp>

#include <QApplication>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <iostream>
#include <string>
#include <csignal>
#include <chrono>
#include <thread>
#include <cmath>
#include <atomic>

static std::atomic<bool> g_stop_requested{false};

static void sigint_handler(int) {
    g_stop_requested.store(true);
}

static void print_usage(const char* prog_name) {
    std::cout << "======================================================================\n"
              << " TMCAD Native C++ Intel RealSense D435if Recorder (Phase 8)\n"
              << "======================================================================\n"
              << "Cách sử dụng:\n"
              << "  " << prog_name << " [tuỳ chọn]\n\n"
              << "Các tuỳ chọn hỗ trợ:\n"
              << "  --help, -h          Hiển thị trợ giúp này\n"
              << "  --output-dir <DIR>  Thư mục lưu trữ dữ liệu (Mặc định: data/realsense_records)\n"
              << "  --record-bag [FILE] Ghi luồng trực tiếp ra file .bag (truyền 'auto' để tự đặt tên)\n"
              << "  --width <W>         Chiều rộng ảnh RGB/Depth (Mặc định: 640)\n"
              << "  --height <H>        Chiều cao ảnh RGB/Depth (Mặc định: 480)\n"
              << "  --fps <FPS>         Tần số khung hình (Mặc định: 30)\n"
              << "  --no-gui            Chế độ dòng lệnh thuần (Headless CLI, không cần X11)\n"
              << "======================================================================\n";
}

int main(int argc, char* argv[]) {
    std::string output_dir = "data/realsense_records";
    std::string record_bag = "";
    int width = 640;
    int height = 480;
    int fps = 30;
    bool no_gui = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--record-bag") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                record_bag = argv[++i];
            } else {
                record_bag = "auto";
            }
        } else if (arg == "--width" && i + 1 < argc) {
            width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            height = std::stoi(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            fps = std::stoi(argv[++i]);
        } else if (arg == "--no-gui") {
            no_gui = true;
        }
    }

    if (!no_gui) {
        // Chế độ GUI Studio tương tác đầy đủ bằng Qt C++
        QApplication app(argc, argv);
        RealSenseRecorderDialog dialog;
        dialog.show();
        return app.exec();
    }

    // Chế độ Headless CLI (không cần màn hình X11, dành cho Robot / nhúng)
    std::signal(SIGINT, sigint_handler);
    std::signal(SIGTERM, sigint_handler);

    std::cout << "[INFO] Khởi động RealSense Native C++ Recorder ở chế độ HEADLESS CLI...\n";
    QDir().mkpath(QString::fromStdString(output_dir));

    std::string bag_path = "";
    if (!record_bag.empty()) {
        if (record_bag == "auto") {
            QString ts = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
            bag_path = output_dir + "/d435if_record_" + ts.toStdString() + ".bag";
        } else {
            bag_path = record_bag;
            if (bag_path.find(".bag") == std::string::npos) bag_path += ".bag";
            if (bag_path.front() != '/' && bag_path.find(output_dir) == std::string::npos) {
                bag_path = output_dir + "/" + bag_path;
            }
        }
    }

    try {
        rs2::context ctx;
        auto devices = ctx.query_devices();
        if (devices.size() == 0) {
            std::cerr << "[ERROR] Không tìm thấy camera Intel RealSense kết nối qua USB!\n";
            return 1;
        }

        auto dev = devices[0];
        std::string dev_name = dev.get_info(RS2_CAMERA_INFO_NAME);
        std::string dev_sn = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
        std::cout << "[FOUND] Thiết bị: " << dev_name << " (S/N: " << dev_sn << ")\n";

        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_RGB8, fps);
        cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);

        if (!bag_path.empty()) {
            std::cout << "[RECORDER] Ghi luồng trực tiếp ra file: " << bag_path << "\n";
            cfg.enable_record_to_file(bag_path);
        }

        pipe.start(cfg);
        std::cout << "[RUNNING] Đang thu nhận luồng dữ liệu " << width << "x" << height << " @ " << fps << " FPS. Nhấn Ctrl+C để dừng...\n";

        int frame_count = 0;
        auto start_time = std::chrono::steady_clock::now();
        auto last_report = start_time;

        while (!g_stop_requested.load()) {
            rs2::frameset frames;
            try {
                frames = pipe.wait_for_frames(100);
            } catch (...) {
                continue;
            }

            frame_count++;
            auto now = std::chrono::steady_clock::now();
            auto diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_report).count();
            if (diff_ms >= 1000) {
                double current_fps = (frame_count * 1000.0) / diff_ms;
                frame_count = 0;
                last_report = now;

                std::cout << "\r[STATUS] Đang thu thập - FPS: " << std::fixed;
                std::cout.precision(1);
                std::cout << current_fps << " (RGB + Depth Z16)" << std::flush;
            }
        }

        std::cout << "\n[STOP] Nhận tín hiệu dừng từ người dùng. Đang đóng luồng...\n";
        pipe.stop();
        std::cout << "[DONE] Đã giải phóng camera an toàn.\n";

        if (!bag_path.empty()) {
            QFileInfo fi(QString::fromStdString(bag_path));
            if (fi.exists()) {
                double sz_mb = fi.size() / (1024.0 * 1024.0);
                std::cout << "[SAVED] File video .bag: " << bag_path << " (" << sz_mb << " MB)\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Ngoại lệ RealSense: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
