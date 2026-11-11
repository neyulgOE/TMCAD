#!/usr/bin/env python3
"""
TMCAD Phase 8 -- Real-World Sensor Preprocessing & Ground Removal Pipeline
Tự động làm sạch dữ liệu cảm biến RealSense D435if:
  1. Lọc nhiễu độ sâu (Bilateral / Statistical Outlier Removal)
  2. Bóc tách mặt phẳng sàn nhà (RANSAC Ground Plane Extraction kết hợp IMU gravity)
  3. Lọc hộp không gian bàn (Workspace Frustum ROI)
  4. Phân cụm mây điểm (DBSCAN) cô lập chiếc bàn khỏi môi trường rác xung quanh
"""

import os
import sys
import json
import argparse
import math
import numpy as np
import open3d as o3d
import cv2


def depth_color_to_point_cloud(color_img, depth_img, intrinsics, min_z=0.35, max_z=2.5):
    """
    Chuyển ảnh RGB và Depth (uint16 mm) thành mây điểm Open3D.
    """
    h, w = depth_img.shape
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_img.astype(np.float32) / 1000.0  # mm to meters

    valid = (z >= min_z) & (z <= max_z)
    z_valid = z[valid]
    u_valid = u_coords[valid]
    v_valid = v_coords[valid]

    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy

    pts = np.stack((x_valid, y_valid, z_valid), axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if color_img is not None:
        if len(color_img.shape) == 3 and color_img.shape[2] == 3:
            # OpenCV BGR -> RGB [0, 1]
            colors = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)[valid].astype(np.float32) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def remove_ground_and_clutter(pcd, imu_accel=None, camera_nominal_height=0.30, debug=False):
    """
    Loại bỏ mặt sàn và các vật thể rác xung quanh mây điểm:
      - pcd: Open3D PointCloud
      - imu_accel: [ax, ay, az] từ gia tốc kế D435if (nếu có)
      - camera_nominal_height: Chiều cao danh định của camera AMR (0.300m)
    """
    if len(pcd.points) < 100:
        return pcd, {"status": "EMPTY_PCD"}

    # 1. Lọc nhiễu thống kê ngoại lai (Statistical Outlier Removal)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    pcd_clean = pcd.select_by_index(ind)

    pts = np.asarray(pcd_clean.points)
    if len(pts) < 100:
        return pcd_clean, {"status": "TOO_FEW_POINTS"}

    # 2. Định hướng vector trọng trường sàn nhà (RealSense convention: Y là hướng xuống sàn)
    # Floor normal hướng ngược chiều Y (lên trên) trong camera frame: n ~ [0, -1, 0]
    expected_floor_normal = np.array([0.0, -1.0, 0.0])
    if imu_accel is not None and len(imu_accel) == 3:
        ax, ay, az = imu_accel
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm > 1.0:
            # Gia tốc kế đo phản lực hướng lên: a ~ [0, -9.8, 0]
            # Normal của sàn hướng thẳng đứng lên: n_floor = - a / norm
            expected_floor_normal = - np.array([ax, ay, az]) / norm
            if debug:
                print(f"[IMU] Hướng trọng trường ước tính n_floor = {expected_floor_normal}")

    # 3. Gravity-constrained floor RANSAC.  Open3D's segment_plane() picks
    # the largest plane without an orientation constraint, which on this
    # scene is frequently the rear wall/tabletop rather than the floor.  The
    # C++ TMVis segmenter already applies this constraint; retain the same
    # rule here so the Qt hierarchy/ICP worker receives the same foreground.
    rng = np.random.default_rng(1337)
    best_count = 0
    best_normal = None
    best_d = 0.0
    best_inliers = None
    floor_threshold = 0.030
    for _ in range(1000):
        ids = rng.choice(len(pts), size=3, replace=False)
        p0, p1, p2 = pts[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-7:
            continue
        normal = normal / norm
        if np.dot(normal, expected_floor_normal) < 0:
            normal = -normal
        # Within ~28 degrees of gravity, and not an implausibly near/far wall.
        if np.dot(normal, expected_floor_normal) < 0.88:
            continue
        d_candidate = -float(np.dot(normal, p0))
        height_candidate = abs(d_candidate)
        if not (0.15 <= height_candidate <= 1.80):
            continue
        h = pts @ normal + d_candidate
        inlier_mask = np.abs(h) < floor_threshold
        count = int(np.sum(inlier_mask))
        n_above = int(np.sum(h > floor_threshold))
        n_below = int(np.sum(h < -floor_threshold))

        # A true floor must have the table/objects standing ABOVE it, not below it!
        # If n_below > n_above * 0.25, this plane is a tabletop or ceiling, NOT the floor.
        if n_above < 200 or n_below > (n_above * 0.25):
            continue

        score = count + n_above * 0.3
        if score > best_count:
            best_count = score
            best_normal = normal
            best_d = d_candidate
            best_inliers = inlier_mask

    if best_normal is None:
        plane_normal = expected_floor_normal
        d = 0.0
        inliers = []
        tilt_angle_deg = 90.0
        est_camera_height = 0.0
        is_floor = False
    else:
        # Least-squares plane refit on the constrained consensus set.
        plane_pts = pts[best_inliers]
        centroid = np.mean(plane_pts, axis=0)
        _, _, vh = np.linalg.svd(plane_pts - centroid, full_matrices=False)
        plane_normal = vh[-1]
        if np.dot(plane_normal, expected_floor_normal) < 0:
            plane_normal = -plane_normal
        d = -float(np.dot(plane_normal, centroid))
        alignment_dot = float(np.dot(plane_normal, expected_floor_normal))
        tilt_angle_deg = math.degrees(math.acos(np.clip(alignment_dot, -1.0, 1.0)))
        est_camera_height = abs(d)
        inliers = np.where(np.abs(pts @ plane_normal + d) < floor_threshold)[0]
        is_floor = (tilt_angle_deg < 28.5) and (0.15 <= est_camera_height <= 1.80)
    floor_removed = False

    if is_floor and len(inliers) > 200:
        floor_removed = True
        if debug:
            print(f"[RANSAC] Đã phát hiện mặt sàn: Cao độ camera = {est_camera_height:.3f} m (Lệch góc: {tilt_angle_deg:.1f}°)")

        # Tính khoảng cách có dấu của từng điểm đến mặt sàn
        dists = np.dot(pts, plane_normal) + d
        # Giữ lại các điểm cao hơn mặt sàn từ 3cm đến 1.4m trong phạm vi làm việc của bàn (z <= 2.10m, |x| <= 0.75m)
        keep_mask = (dists > 0.03) & (dists < 1.40) & (pts[:, 2] <= 2.10) & (np.abs(pts[:, 0]) <= 0.75)
        pcd_above_floor = pcd_clean.select_by_index(np.where(keep_mask)[0])
    else:
        if debug:
            print(f"[RANSAC] Không phát hiện mặt sàn rõ rệt (Lệch={tilt_angle_deg:.1f}°, H={est_camera_height:.2f}m). Dùng bộ lọc độ sâu mặc định.")
        keep_mask = (pts[:, 2] <= 2.10) & (pts[:, 2] >= 0.50) & (np.abs(pts[:, 0]) <= 0.75)
        pcd_above_floor = pcd_clean.select_by_index(np.where(keep_mask)[0])

    if len(pcd_above_floor.points) < 50:
        return pcd_clean, {"status": "NO_POINTS_ABOVE_FLOOR", "camera_height": est_camera_height}

    # 4. Phân cụm mây điểm (DBSCAN Clustering) để cô lập chiếc bàn
    labels = np.array(pcd_above_floor.cluster_dbscan(eps=0.08, min_points=50, print_progress=False))
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    if len(unique_labels) > 0:
        # Chọn cụm bàn nằm ở trung tâm tầm nhìn của camera (ưu tiên cự ly trục quang học)
        best_label = -1
        best_score = -1.0
        pts_above = np.asarray(pcd_above_floor.points)
        for l in unique_labels:
            indices = np.where(labels == l)[0]
            n_pts = len(indices)
            if n_pts < 50:
                continue
            c = np.mean(pts_above[indices], axis=0)
            if c[2] > 2.05 or abs(c[0]) > 0.70:
                continue
            d_center = np.hypot(c[0], c[1]) + c[2] * 0.5
            score = n_pts / (d_center + 0.2)
            if score > best_score:
                best_score = score
                best_label = l

        if best_label >= 0:
            c_best = np.mean(pts_above[np.where(labels == best_label)[0]], axis=0)
            # Hợp nhất các cụm thành phần của bàn (chân bàn + mặt bàn trong bán kính 0.85m của CAD TABLE-DEMO)
            table_indices = []
            for l in unique_labels:
                l_idx = np.where(labels == l)[0]
                c_l = np.mean(pts_above[l_idx], axis=0)
                if np.linalg.norm(c_l - c_best) < 0.85 and c_l[2] <= 2.05:
                    table_indices.extend(l_idx)
            pcd_table = pcd_above_floor.select_by_index(table_indices)
            if debug:
                print(f"[DBSCAN] Đã cô lập và hợp nhất cụm bàn trung tâm: {len(pcd_table.points)} điểm (Loại bỏ {len(pcd_clean.points) - len(pcd_table.points)} điểm rác sàn/tường).")
        else:
            pcd_table = pcd_above_floor
    else:
        pcd_table = pcd_above_floor

    diag_info = {
        "status": "SUCCESS",
        "floor_detected": floor_removed,
        "floor_normal": plane_normal.tolist() if plane_normal is not None else None,
        "floor_d": float(d),
        "camera_height_m": float(est_camera_height),
        "floor_tilt_deg": float(tilt_angle_deg),
        "raw_points": len(pcd.points),
        "clean_points": len(pcd_table.points)
    }

    return pcd_table, diag_info


def process_real_frame(color_path, depth_path, intr_path, imu_path=None, output_ply="debug/clean_scene.ply", debug=True):
    if not os.path.exists(color_path) or not os.path.exists(depth_path) or not os.path.exists(intr_path):
        print(f"[ERROR] Thiếu file đầu vào: {color_path}, {depth_path}, {intr_path}")
        return False

    color = cv2.imread(color_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # 16-bit PNG mm

    with open(intr_path, "r") as f:
        intr = json.load(f)

    imu_accel = None
    if imu_path and os.path.exists(imu_path):
        with open(imu_path, "r") as f:
            imu_dict = json.load(f)
            if imu_dict.get("available") and "accel" in imu_dict:
                imu_accel = imu_dict["accel"]

    pcd = depth_color_to_point_cloud(color, depth, intr)
    clean_pcd, diag = remove_ground_and_clutter(pcd, imu_accel=imu_accel, debug=debug)

    os.makedirs(os.path.dirname(output_ply), exist_ok=True)
    o3d.io.write_point_cloud(output_ply, clean_pcd)
    if debug:
        print(f"[SAVED] Đã lưu mây điểm đã tiền xử lý: {output_ply} ({len(clean_pcd.points)} điểm)")
        print(f"  Chẩn đoán: {diag}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Real-World Sensor Preprocessing & Ground Removal")
    parser.add_argument("--color", help="Đường dẫn file ảnh màu color.png")
    parser.add_argument("--depth", help="Đường dẫn file ảnh độ sâu depth.png (16-bit uint16)")
    parser.add_argument("--intrinsics", help="Đường dẫn file thông số intrinsics.json")
    parser.add_argument("--imu", default=None, help="Đường dẫn file imu.json (tùy chọn)")
    parser.add_argument("--output", default="debug/clean_scene.ply", help="File đầu ra mây điểm .ply đã làm sạch")
    parser.add_argument("--test", action="store_true", help="Chạy kiểm thử với file mây điểm mẫu có sẵn")
    args = parser.parse_args()

    if args.test:
        sample_ply = "debug/test_scene_190.ply"
        if os.path.exists(sample_ply):
            print(f"[TEST] Chạy kiểm thử tiền xử lý trên: {sample_ply}")
            pcd = o3d.io.read_point_cloud(sample_ply)
            clean_pcd, diag = remove_ground_and_clutter(pcd, debug=True)
            o3d.io.write_point_cloud("debug/test_clean_output.ply", clean_pcd)
            print(f"[TEST SUCCESS] Kết quả chẩn đoán: {diag}")
            sys.exit(0)
        else:
            print(f"[TEST ERROR] Không tìm thấy file test: {sample_ply}")
            sys.exit(1)

    if not args.color or not args.depth or not args.intrinsics:
        parser.print_help()
        sys.exit(1)

    process_real_frame(args.color, args.depth, args.intrinsics, args.imu, args.output, debug=True)


if __name__ == "__main__":
    main()
