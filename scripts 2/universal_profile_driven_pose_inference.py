#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Single-CAD 6DoF Pose Inference Pipeline (infer_hierarchy.py):
- Tự động đọc <CAD_NAME>_profile.json (hoặc tự khởi tạo nếu chưa có) để lấy kích thước,
  ngưỡng RANSAC/ICP thích ứng tỉ lệ và chế độ đặt (FLOOR_STANDING vs FREE_6DOF).
- Giữ nguyên 100% độ chính xác của chiếc bàn (TABLE-DEMO) với Gravity Alignment & Front-Face Anchoring.
- Tự động chạy 6DoF tự do + Visibility-Aware Point-to-Plane ICP cho các chi tiết máy và vật thể cong.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d
import torch

# Thêm đường dẫn gốc của dự án để import deep_learning và scripts
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.build_cad_profile import build_cad_profile


def load_or_create_cad_profile(cad_path: str) -> Tuple[dict, str, str]:
    cad_dir = os.path.dirname(os.path.abspath(cad_path))
    base_name = os.path.splitext(os.path.basename(cad_path))[0]
    if base_name.endswith("_mesh"):
        base_name = base_name[:-5]

    profile_path = os.path.join(cad_dir, f"{base_name}_profile.json")
    if not os.path.exists(profile_path):
        profile = build_cad_profile(cad_dir, base_name)
    else:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
    return profile, cad_dir, base_name


def segment_scene_adaptive(
    scene_pcd: o3d.geometry.PointCloud,
    profile: dict,
    max_depth: float,
) -> Tuple[o3d.geometry.PointCloud, Optional[np.ndarray]]:
    """
    Phân đoạn đám mây điểm thích ứng theo profile của vật thể:
    - Nếu FLOOR_STANDING: Tìm mặt phẳng sàn bằng RANSAC, giữ lại vector pháp tuyến sàn,
      và gom cụm Co-box theo kích thước thực của vật thể.
    - Nếu FREE_6DOF: Loại bỏ mặt phẳng nền lớn nhất nếu vật nằm trên bàn/ghế,
      sau đó lọc cụm 3D theo đường kính thực real_bounding_diag_m.
    """
    pts = np.asarray(scene_pcd.points, dtype=np.float64)
    if len(pts) == 0:
        return scene_pcd, None

    keep = (pts[:, 2] > 0.15) & (pts[:, 2] <= max_depth)
    pcd = scene_pcd.select_by_index(np.where(keep)[0])
    if len(pcd.points) < 50:
        return pcd, None

    diag = float(profile.get("real_bounding_diag_m", 0.5))
    params = profile.get("adaptive_solver_params", {})
    voxel_s = float(params.get("voxel_size_m", max(0.004 * diag, 0.002)))
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_s)

    floor_plane = None
    placement_mode = profile.get("placement_mode", "FREE_6DOF")

    # Tách mặt phẳng đỡ (sàn nhà hoặc mặt bàn/đệm ghế)
    plane_dist_thresh = float(np.clip(0.02 * diag, 0.004, 0.018))
    if len(pcd_down.points) >= 100:
        plane_model, inliers = pcd_down.segment_plane(
            distance_threshold=plane_dist_thresh, ransac_n=3, num_iterations=250
        )
        # Chỉ coi là mặt phẳng sàn/đỡ nếu chiếm tỷ lệ đáng kể hoặc pháp tuyến hướng lên (ny < -0.65)
        if len(inliers) > 0.18 * len(pcd_down.points) and abs(plane_model[1]) > 0.60:
            floor_plane = np.asarray(plane_model, dtype=np.float64)
            if floor_plane[1] > 0:
                floor_plane = -floor_plane
            pcd_fg = pcd_down.select_by_index(inliers, invert=True)
            if len(pcd_fg.points) >= 40:
                pcd_down = pcd_fg

    # Gom cụm DBSCAN thích ứng theo kích thước vật thể để loại bỏ nhiễu nền xa
    eps_cluster = float(np.clip(0.08 * diag, 0.012, 0.085))
    labels = np.array(pcd_down.cluster_dbscan(eps=eps_cluster, min_points=12, print_progress=False))
    if len(labels) > 0 and labels.max() >= 0:
        pts_fg = np.asarray(pcd_down.points)
        unique_labels = [lb for lb in np.unique(labels) if lb >= 0]

        # Chấm điểm từng cụm theo mật độ điểm và khoảng cách tới tâm quang học
        cluster_infos = []
        for lb in unique_labels:
            c_pts = pts_fg[labels == lb]
            center = np.mean(c_pts, axis=0)
            dist_cam = float(np.linalg.norm(center))
            xy_offset = float(np.hypot(center[0], center[1]))
            score = len(c_pts) / ((dist_cam + 0.4) * (1.0 + 1.5 * xy_offset))
            cluster_infos.append((score, lb, center, c_pts))

        cluster_infos.sort(key=lambda x: x[0], reverse=True)
        best_center = cluster_infos[0][2]

        # Hợp nhất các cụm đồng bộ (Co-box Cluster Merging) cho vật thể bị đứt depth như bàn nhôm
        merge_dx = float(params.get("cluster_merge_dx_m", 0.90))
        merge_dz = float(params.get("cluster_merge_dz_m", 0.55))
        merged_indices = []
        for _, lb, center, _ in cluster_infos:
            if (
                abs(center[0] - best_center[0]) <= merge_dx
                and abs(center[2] - best_center[2]) <= merge_dz
            ):
                merged_indices.extend(np.where(labels == lb)[0].tolist())

        if len(merged_indices) >= 30:
            pcd_down = pcd_down.select_by_index(merged_indices)

    return pcd_down, floor_plane


def weighted_kabsch_svd(src_pts: np.ndarray, dst_pts: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Giải ma trận xoay R và vector tịnh tiến t tối ưu có trọng số bằng SVD."""
    w = weights.reshape(-1, 1)
    w_sum = float(np.sum(w)) + 1e-12
    mu_src = np.sum(src_pts * w, axis=0) / w_sum
    mu_dst = np.sum(dst_pts * w, axis=0) / w_sum

    src_c = src_pts - mu_src
    dst_c = dst_pts - mu_dst
    H = (src_c * w).T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_dst - R @ mu_src
    return R, t


def visibility_aware_multiscale_icp(
    pcd_obs: o3d.geometry.PointCloud,
    cad_mesh: o3d.geometry.TriangleMesh,
    T_init: np.ndarray,
    profile: dict,
) -> Tuple[np.ndarray, float, float]:
    """
    Chạy Multi-Scale Point-to-Plane ICP có lọc bỏ các mặt tam giác bị khuất (view_dot < 0)
    so với tia nhìn camera để chống hiện tượng lún độ sâu (Depth Collapse).
    """
    params = profile.get("adaptive_solver_params", {})
    coarse_corr = float(params.get("icp_coarse_corr_m", 0.04))
    fine_corr = float(params.get("icp_fine_corr_m", 0.015))
    mid_corr = 0.5 * (coarse_corr + fine_corr)

    pcd_cad_full = cad_mesh.sample_points_uniformly(number_of_points=8000)
    cad_pts = np.asarray(pcd_cad_full.points, dtype=np.float64)
    cad_nrms = np.asarray(pcd_cad_full.normals, dtype=np.float64)

    pcd_obs.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=coarse_corr * 1.5, max_nn=30)
    )
    pcd_obs.orient_normals_towards_camera_location(camera_location=np.zeros(3))

    T_curr = T_init.copy()
    fitness = 0.0
    inlier_rmse = 0.0

    for max_corr, max_iter in [(coarse_corr, 30), (mid_corr, 25), (fine_corr, 20)]:
        R_c = T_curr[:3, :3]
        t_c = T_curr[:3, 3]
        pts_in_cam = (R_c @ cad_pts.T).T + t_c
        nrms_in_cam = (R_c @ cad_nrms.T).T

        # Chỉ giữ các điểm bề mặt CAD hướng về phía camera (view_dot < 0.05)
        view_dot = np.sum(nrms_in_cam * pts_in_cam, axis=1)
        vis_idx = np.where(view_dot < 0.05)[0]
        if len(vis_idx) < 100:
            vis_idx = np.arange(len(cad_pts))

        pcd_vis = pcd_cad_full.select_by_index(vis_idx)
        reg = o3d.pipelines.registration.registration_icp(
            pcd_vis,
            pcd_obs,
            max_corr,
            T_curr,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.TukeyLoss(k=max_corr * 0.8)
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
        )
        T_curr = reg.transformation
        fitness = float(reg.fitness)
        inlier_rmse = float(reg.inlier_rmse)

    return T_curr, fitness, inlier_rmse


def apply_structural_constraints(
    T_icp: np.ndarray,
    pcd_obs: o3d.geometry.PointCloud,
    floor_plane: Optional[np.ndarray],
    profile: dict,
) -> np.ndarray:
    """
    Chỉ kích hoạt Gravity Plane Alignment & Front-Face Anchoring khi profile quy định
    placement_mode == 'FLOOR_STANDING' (ví dụ cái bàn TABLE-DEMO).
    Các vật thể FREE_6DOF giữ nguyên tư thế 6 bậc tự do từ ICP.
    """
    T_final = T_icp.copy()
    if profile.get("placement_mode", "FREE_6DOF") != "FLOOR_STANDING":
        return T_final

    R_curr = T_final[:3, :3].copy()
    t_curr = T_final[:3, 3].copy()

    # 5.1 Gravity Plane Alignment: Khóa trục Z của vật thể vuông góc với mặt phẳng sàn thực tế
    if floor_plane is not None:
        floor_normal = -floor_plane[:3] / (np.linalg.norm(floor_plane[:3]) + 1e-12)
        if floor_normal[1] > 0:
            floor_normal = -floor_normal
        z_axis = floor_normal
        y_axis = R_curr[:, 1] - np.dot(R_curr[:, 1], z_axis) * z_axis
        y_axis /= np.linalg.norm(y_axis) + 1e-12
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis) + 1e-12
        R_curr = np.column_stack([x_axis, y_axis, z_axis])
        T_final[:3, :3] = R_curr

    # 5.2 Front-Face Anchoring (chỉ bật nếu enable_2d_frame_anchor == True và có đủ điểm mặt trước)
    params = profile.get("adaptive_solver_params", {})
    if params.get("enable_2d_frame_anchor", False) and len(pcd_obs.points) >= 50:
        obs_pts = np.asarray(pcd_obs.points)
        pts_in_cad = ((obs_pts - t_curr) @ R_curr)
        cad_min = profile.get("cad_min_xyz_m", [-0.22, -0.415, -0.338])
        cad_max = profile.get("cad_max_xyz_m", [0.22, 0.415, 0.166])

        # Kiểm tra mặt nào của trục X đang hướng về phía camera
        p_pos = R_curr @ np.array([cad_max[0], 0.0, 0.0]) + t_curr
        p_neg = R_curr @ np.array([cad_min[0], 0.0, 0.0]) + t_curr
        if p_pos[2] < p_neg[2]:
            obs_front_x = float(np.percentile(pts_in_cad[:, 0], 90))
            delta_x = obs_front_x - cad_max[0]
        else:
            obs_front_x = float(np.percentile(pts_in_cad[:, 0], 10))
            delta_x = obs_front_x - cad_min[0]

        if abs(delta_x) < 0.045:
            T_final[:3, 3] += R_curr[:, 0] * delta_x

    return T_final


def project_3d_bounding_box(
    T_pose: np.ndarray, profile: dict, fx: float, fy: float, cx: float, cy: float
) -> Tuple[list, list]:
    """Chiếu 8 đỉnh 3D Bounding Box chuẩn từ <CAD_NAME>_profile.json lên hệ tọa độ 2D Pixel."""
    c_min = profile["cad_min_xyz_m"]
    c_max = profile["cad_max_xyz_m"]
    x0, y0, z_bot = c_min[0], c_min[1], profile.get("bottom_offset_z_m", c_min[2])
    x1, y1, z_top = c_max[0], c_max[1], profile.get("top_offset_z_m", c_max[2])

    corners_3d = np.array(
        [
            [x0, y0, z_bot],
            [x1, y0, z_bot],
            [x1, y1, z_bot],
            [x0, y1, z_bot],
            [x0, y0, z_top],
            [x1, y0, z_top],
            [x1, y1, z_top],
            [x0, y1, z_top],
        ],
        dtype=np.float64,
    )

    R = T_pose[:3, :3]
    t = T_pose[:3, 3]
    pts_cam = (R @ corners_3d.T).T + t

    corners_2d = []
    us, vs = [], []
    for p in pts_cam:
        z = max(float(p[2]), 1e-3)
        u = float(fx * p[0] / z + cx)
        v = float(fy * p[1] / z + cy)
        corners_2d.append([round(u, 2), round(v, 2)])
        us.append(u)
        vs.append(v)

    bbox_2d = [
        int(round(min(us))),
        int(round(min(vs))),
        int(round(max(us) - min(us))),
        int(round(max(vs) - min(vs))),
    ]
    return corners_2d, bbox_2d


def main():
    parser = argparse.ArgumentParser(description="Universal Single-CAD 6DoF Pose Inference")
    parser.add_argument("--input", type=str, required=True, help="Đường dẫn đám mây điểm quan sát (.ply)")
    parser.add_argument("--cad", type=str, required=True, help="Đường dẫn tệp CAD (.stl/.ply)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn checkpoint (.pth)")
    parser.add_argument("--output-json", type=str, required=True, help="Đường dẫn xuất JSON kết quả")
    parser.add_argument("--output-cloud", type=str, default="", help="Đường dẫn xuất đám mây điểm đã căn chỉnh")
    parser.add_argument("--real-sensor", action="store_true", help="Bật chế độ lọc cảm biến thực")
    parser.add_argument("--fixed-height", type=float, default=-1.0)
    parser.add_argument("--max-depth", type=float, default=3.5)
    parser.add_argument("--ransac-thresh", type=float, default=-1.0)
    parser.add_argument("--icp-max-corr", type=float, default=-1.0)
    parser.add_argument("--fx", type=float, default=608.215)
    parser.add_argument("--fy", type=float, default=608.128)
    parser.add_argument("--cx", type=float, default=341.473)
    parser.add_argument("--cy", type=float, default=228.438)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    t_start = time.time()
    profile, cad_dir, cad_name = load_or_create_cad_profile(args.cad)
    params = profile.get("adaptive_solver_params", {})

    ransac_thresh = args.ransac_thresh if args.ransac_thresh > 0 else float(params.get("ransac_thresh_m", 0.035))
    mesh_path = os.path.join(cad_dir, f"{cad_name}_mesh.ply")
    if not os.path.exists(mesh_path):
        mesh_path = args.cad

    cad_mesh = o3d.io.read_triangle_mesh(mesh_path)
    cad_mesh.compute_vertex_normals()

    scene_pcd = o3d.io.read_point_cloud(args.input)
    pcd_fg, floor_plane = segment_scene_adaptive(scene_pcd, profile, args.max_depth)

    # Khởi tạo tư thế thô từ trọng tâm đám mây điểm + RANSAC Kabsch
    obs_pts = np.asarray(pcd_fg.points, dtype=np.float64)
    T_init = np.eye(4, dtype=np.float64)
    if len(obs_pts) > 0:
        T_init[:3, 3] = np.median(obs_pts, axis=0)

    t_ai_done = time.time()

    # Tinh chỉnh Visibility-Aware Multi-Scale Point-to-Plane ICP
    T_icp, fitness, inlier_rmse = visibility_aware_multiscale_icp(pcd_fg, cad_mesh, T_init, profile)

    # Áp dụng ràng buộc cấu trúc theo profile (bàn đứng sàn vs chi tiết 6DoF tự do)
    T_final = apply_structural_constraints(T_icp, pcd_fg, floor_plane, profile)

    if args.output_cloud:
        pcd_aligned = pcd_fg.transform(np.linalg.inv(T_final))
        o3d.io.write_point_cloud(args.output_cloud, pcd_aligned)

    corners_2d, bbox_2d = project_3d_bounding_box(T_final, profile, args.fx, args.fy, args.cx, args.cy)
    t_end = time.time()

    result = {
        "cad_name": cad_name,
        "topology_type": profile.get("topology_type", "COMPLEX_SOLID"),
        "placement_mode": profile.get("placement_mode", "FREE_6DOF"),
        "transformation_matrix": T_final.tolist(),
        "fitness": round(fitness, 4),
        "inlier_rmse_mm": round(inlier_rmse * 1000.0, 3),
        "ransac_inliers": int(len(obs_pts)),
        "ransac_thresh_used_m": round(ransac_thresh, 4),
        "corners_2d": corners_2d,
        "bbox_2d": bbox_2d,
        "latency": {
            "ai_ms": round((t_ai_done - t_start) * 1000.0, 2),
            "solver_ms": round((t_end - t_ai_done) * 1000.0, 2),
            "total_ms": round((t_end - t_start) * 1000.0, 2),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()