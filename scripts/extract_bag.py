#!/usr/bin/env python3
"""
TMCAD Phase 8 -- RealSense .bag Frame Extractor
Trích xuất tự động các khung hình từ file .bag của Intel RealSense D435if:
  - Color (PNG)
  - Depth 16-bit uint16 mm (PNG)
  - Camera Intrinsics (JSON)
  - IMU Gyro/Accel (JSON)
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False


def extract_from_bag(bag_path, output_dir, stride=1, max_frames=None):
    if not os.path.exists(bag_path):
        print(f"[ERROR] Không tìm thấy file .bag: {bag_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    # Nạp file .bag
    config.enable_device_from_file(bag_path, repeat_playback=False)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"[ERROR] Không thể mở file .bag bằng RealSense SDK: {e}")
        return False

    # Lấy thông số nội tại từ luồng Color
    align_to_color = rs.align(rs.stream.color)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    intrinsics_dict = {
        "width": intr.width,
        "height": intr.height,
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "model": str(intr.model),
        "coeffs": [float(c) for c in intr.coeffs],
        "depth_scale_mm": 1.0
    }

    with open(os.path.join(output_dir, "intrinsics.json"), "w") as f:
        json.dump(intrinsics_dict, f, indent=2)

    frame_idx = 0
    saved_count = 0
    imu_records = []

    print(f"[EXTRACTING] Bắt đầu trích xuất từ: {bag_path}")
    print(f"  Độ phân giải: {intr.width}x{intr.height} | Bước nhảy stride={stride}")

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(2000)
            except RuntimeError:
                # Đã duyệt hết file .bag
                break

            # Thu thập dữ liệu IMU nếu có
            for f in frames:
                if f.is_motion_frame():
                    mf = f.as_motion_frame()
                    stream_t = mf.get_profile().stream_type()
                    d = mf.get_motion_data()
                    ts = mf.get_timestamp()
                    if stream_t == rs.stream.accel:
                        imu_records.append({"type": "accel", "ts": ts, "val": [float(d.x), float(d.y), float(d.z)]})
                    elif stream_t == rs.stream.gyro:
                        imu_records.append({"type": "gyro", "ts": ts, "val": [float(d.x), float(d.y), float(d.z)]})

            aligned_frames = align_to_color.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            frame_idx += 1
            if (frame_idx % stride) != 0:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())  # uint16 mm

            color_name = f"color_{saved_count:05d}.png"
            depth_name = f"depth_{saved_count:05d}.png"

            cv2.imwrite(os.path.join(frames_dir, color_name), color_image)
            cv2.imwrite(os.path.join(frames_dir, depth_name), depth_image)

            saved_count += 1
            if saved_count % 10 == 0:
                print(f"  -> Đã trích xuất {saved_count} frames...", end="\r")

            if max_frames and saved_count >= max_frames:
                break

    finally:
        pipeline.stop()

    # Lưu dữ liệu IMU nếu có
    if imu_records:
        with open(os.path.join(output_dir, "imu_stream.json"), "w") as f:
            json.dump(imu_records, f, indent=2)
        print(f"\n[IMU] Đã lưu {len(imu_records)} bản ghi IMU vào imu_stream.json")

    print(f"\n[SUCCESS] Hoàn tất trích xuất {saved_count} khung hình vào thư mục: {frames_dir}")
    print(f"  - Intrinsics: {os.path.join(output_dir, 'intrinsics.json')}")
    return True


def main():
    parser = argparse.ArgumentParser(description="RealSense .bag to RGB-D PNG Extractor")
    parser.add_argument("bag_file", help="Đường dẫn tới file .bag")
    parser.add_argument("--output-dir", default=None, help="Thư mục xuất ảnh (mặc định tạo theo tên file .bag)")
    parser.add_argument("--stride", type=int, default=1, help="Khoảng cách lấy mẫu khung hình (1 = lấy toàn bộ)")
    parser.add_argument("--max-frames", type=int, default=None, help="Số khung hình tối đa cần trích xuất")
    args = parser.parse_args()

    if not HAS_REALSENSE:
        print("[ERROR] pyrealsense2 chưa được cài đặt!")
        sys.exit(1)

    if not args.output_dir:
        base_name = os.path.splitext(os.path.basename(args.bag_file))[0]
        args.output_dir = os.path.join("data/extracted_bags", base_name)

    extract_from_bag(args.bag_file, args.output_dir, args.stride, args.max_frames)


if __name__ == "__main__":
    main()
