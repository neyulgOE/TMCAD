#!/usr/bin/env python3
"""
Benchmark 6DoF Pose Estimation on 1000 Unoccluded Points in Camera Coordinate System:
- Controlled Planar Setup:
  * Y_cam = 0.30 m (fixed at 30 cm)
  * Pitch = 0 deg (fixed)
  * Roll = 0 deg (fixed)
  * X_cam in [-0.15 m, 0.15 m] (horizontal span across camera FOV)
  * Z_cam in [0.80 m, 1.20 m] (depth distance)
  * Yaw in [0.0 deg, 360.0 deg) (full 360-deg rotation around vertical axis)
- Evaluates full 6DoF decomposition:
  * Translation: e_x, e_y, e_z, e_t (Euclidean), e_xz (planar)
  * Orientation: e_yaw, e_pitch, e_roll, e_R, e_R_sym (SO(3))
  * ADD-S metric, BOP Recall @ 0.1d, Strict Recall @ 0.05d
  * Pipeline latency (AI GeoPoseNet + Multi-Scale Tukey ICP)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import open3d as o3d
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as ScipyRot

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deep_learning.model_hierarchy import CADHierarchyCorrespondenceModel
def kabsch_ransac(pts_obs, pts_cad, confidences, max_iter=1500, distance_threshold=0.03):
    N = len(pts_obs)
    if N < 4:
        return np.eye(4), np.zeros(N, dtype=bool)

    best_inliers = np.zeros(N, dtype=bool)
    best_inlier_count = 0
    best_T = np.eye(4)

    probs = (confidences ** 1.5) / (np.sum(confidences ** 1.5) + 1e-8)
    thresh_sq = distance_threshold ** 2

    quads = np.random.choice(N, size=(max_iter, 4), replace=True, p=probs)
    for idx in quads:
        p_src = pts_cad[idx]
        p_dst = pts_obs[idx]

        c_src = np.mean(p_src, axis=0)
        c_dst = np.mean(p_dst, axis=0)

        H = (p_src - c_src).T @ (p_dst - c_dst)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = c_dst - R @ c_src

        pred_dst = pts_cad @ R.T + t
        res_sq = np.sum((pts_obs - pred_dst)**2, axis=1)
        inliers = (res_sq < thresh_sq)
        count = np.sum(inliers)

        if count > best_inlier_count:
            best_inlier_count = count
            best_inliers = inliers
            best_T[:3, :3] = R
            best_T[:3, 3] = t
            if count > int(0.50 * N):
                break

    if best_inlier_count >= 4:
        p_src = pts_cad[best_inliers]
        p_dst = pts_obs[best_inliers]
        w = confidences[best_inliers][:, None]
        w = w / (np.sum(w) + 1e-8)
        c_src_w = np.sum(p_src * w, axis=0)
        c_dst_w = np.sum(p_dst * w, axis=0)
        H = ((p_src - c_src_w) * w).T @ (p_dst - c_dst_w)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = c_dst_w - R @ c_src_w
        best_T[:3, :3] = R
        best_T[:3, 3] = t

    return best_T, best_inliers

def compute_adds(T_pred, T_gt, cad_eval_pts, cad_tree):
    pts_pred = (T_pred[:3, :3] @ cad_eval_pts.T).T + T_pred[:3, 3]
    pts_gt = (T_gt[:3, :3] @ cad_eval_pts.T).T + T_gt[:3, 3]
    add = np.mean(np.linalg.norm(pts_pred - pts_gt, axis=1)) * 1000.0
    pts_pred_in_gt = (T_gt[:3, :3].T @ (pts_pred - T_gt[:3, 3]).T).T
    dists, _ = cad_tree.query(pts_pred_in_gt, k=1)
    adds = np.mean(dists) * 1000.0
    return add, adds

def run_inverse_multiscale_icp(obs_pcd, cad_pcd, T_init, scales=[0.050, 0.025, 0.012, 0.005]):
    """
    Inverse Multi-Scale Point-to-Plane ICP:
    Registers Observation Point Cloud (Source) to Canonical CAD Model (Target).
    Completely eliminates the depth bias caused by occluded CAD back faces.
    """
    T_curr_inv = np.linalg.inv(T_init)
    final_fitness = 0.0
    final_rmse = 0.0
    for dist_th in scales:
        loss_k = max(0.003, dist_th / 2.0)
        reg = o3d.pipelines.registration.registration_icp(
            obs_pcd, cad_pcd, dist_th, T_curr_inv,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.TukeyLoss(k=loss_k)
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30, relative_fitness=1e-5, relative_rmse=1e-5)
        )
        T_curr_inv = reg.transformation
        final_fitness = reg.fitness
        final_rmse = reg.inlier_rmse
    T_refined = np.linalg.inv(T_curr_inv)
    return T_refined, final_fitness, final_rmse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cad-name", type=str, default="TABLE-DEMO")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--min-dist", type=float, default=1.00)
    parser.add_argument("--max-dist", type=float, default=1.40)
    # Real AMR camera is 0.23m above floor. Since table floor is at Z=-0.334m in centered CAD frame,
    # Y_cam = h_floor - 0.334m = 0.230 - 0.334 = -0.104m (table center is 10.4cm above camera).
    parser.add_argument("--fixed-y", type=float, default=-0.104)
    parser.add_argument("--x-span", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-csv", type=str, default=os.path.join(project_root, "logs/benchmark_1000_planar_camera.csv"))
    parser.add_argument("--output-json", type=str, default=os.path.join(project_root, "logs/benchmark_1000_planar_summary.json"))
    parser.add_argument("--output-fig", type=str, default="/home/huyen1905/.gemini/antigravity-ide/brain/9e85ad11-79b5-433e-9c12-9efb8e933a5e/benchmark_1000_planar_figures.png")
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f" 6DoF CAD BENCHMARK (CAMERA FRAME PLANAR: X, Z, YAW)")
    print(f" Object:          {args.cad_name}")
    print(f" Depth Z_cam:     [{args.min_dist:.2f} m, {args.max_dist:.2f} m]")
    print(f" Vertical Y_cam:  {args.fixed_y:.3f} m (AMR Real Camera: 23cm above floor)")
    print(f" Horizontal X_cam:[{-args.x_span:.2f} m, {args.x_span:.2f} m]")
    print(f" Pitch / Roll:    0.0 deg (Fixed horizontal looking)")
    print(f" Yaw:             [0.0 deg, 360.0 deg) (Full 360-deg rotation)")
    print(f" Total Views:     {args.num_samples}")
    print(f" Compute Device:  {device}")
    print(f"============================================================")

    cad_dir = os.path.join(project_root, "CAD_DATA")
    mesh_path = os.path.join(cad_dir, f"{args.cad_name}_mesh.ply")
    tri_path = os.path.join(cad_dir, f"{args.cad_name}_triangles.npz")
    patch_path = os.path.join(cad_dir, f"{args.cad_name}_patches.npz")
    ckpt_path = os.path.join(project_root, "deep_learning/checkpoints", args.cad_name, "best_hierarchy_model.pth")

    # 1. Load CAD Mesh & Raycasting Scene
    mesh_legacy = o3d.io.read_triangle_mesh(mesh_path)
    mesh_legacy.compute_vertex_normals()
    cad_diameter = np.linalg.norm(mesh_legacy.get_max_bound() - mesh_legacy.get_min_bound())
    print(f"CAD Bounding Box Diameter: {cad_diameter:.4f} m ({cad_diameter*1000:.1f} mm)")

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    # Point clouds for ICP and ADD-S
    cad_icp_pcd = mesh_legacy.sample_points_uniformly(15000)
    cad_icp_pcd.estimate_normals()
    cad_eval_pts = np.asarray(mesh_legacy.sample_points_uniformly(1500).points, dtype=np.float32)
    cad_kdtree = KDTree(cad_eval_pts)
    cad_bbox = mesh_legacy.get_oriented_bounding_box()
    cad_corners = np.asarray(cad_bbox.get_box_points(), dtype=np.float32)

    # 2. Load Deep Learning Model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cad_key_indices = ckpt["cad_key_indices"]
    model = CADHierarchyCorrespondenceModel(128, 32, 20, 0.07).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tri_data = np.load(tri_path)
    all_features = tri_data['features'].astype(np.float32)
    all_centroids = all_features[:, :3]

    patch_data = np.load(patch_path)
    all_patches = patch_data['patch_indices'].astype(np.int64)
    all_hops = patch_data['patch_hops'].astype(np.int64)

    key_centroids = all_centroids[cad_key_indices]
    key_patches = all_patches[cad_key_indices, :32]
    patch_features = all_features[key_patches]
    patch_rel_pos = patch_features[:, :, :3] - key_centroids[:, None, :]
    patch_hops = all_hops[cad_key_indices, :32]

    with torch.no_grad():
        c_feat = torch.from_numpy(patch_features).unsqueeze(0).to(device)
        c_rel = torch.from_numpy(patch_rel_pos).unsqueeze(0).to(device)
        c_hops = torch.from_numpy(patch_hops).unsqueeze(0).to(device)
        D_cad = model.encode_cad(c_feat, c_rel, c_hops)

    print("Pre-encoded CAD hierarchy features loaded in GPU VRAM.")

    # 3. Camera Intrinsics Grid
    w, h = 640, 480
    fx, fy, cx, cy = 386.652, 386.652, 320.0, 240.0
    step = 2
    v, u = np.indices((h // step, w // step), dtype=np.float32)
    u = u * step
    v = v * step
    dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    norms = np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    dirs_cam_norm = (dirs_cam / norms).reshape(-1, 3).astype(np.float32)

    # 4. Generate 1000 Viewpoints in Camera Frame (Guaranteed In-Frustum)
    N = args.num_samples
    test_zs = []
    test_xs = []
    test_yaws_deg = []
    fixed_y = args.fixed_y

    # Base rotation mapping CAD coordinates (+Z up, +X right, +Y forward) to Camera (+X right, -Y up, +Z forward):
    R_0 = np.array([
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0,  0.0]
    ], dtype=np.float32)

    while len(test_zs) < N:
        z_cand = float(np.random.uniform(args.min_dist, args.max_dist))
        x_cand = float(np.random.uniform(-args.x_span, args.x_span))
        yaw_cand = float(np.random.uniform(0.0, 360.0))

        psi_rad = np.radians(yaw_cand)
        R_z = np.array([
            [np.cos(psi_rad), -np.sin(psi_rad), 0.0],
            [np.sin(psi_rad),  np.cos(psi_rad), 0.0],
            [0.0,             0.0,            1.0]
        ], dtype=np.float32)
        R_cand = R_0 @ R_z
        pts_cam_corners = (R_cand @ cad_corners.T).T + np.array([x_cand, fixed_y, z_cand])

        if np.any(pts_cam_corners[:, 2] <= 0.2):
            continue
        u_proj = fx * pts_cam_corners[:, 0] / pts_cam_corners[:, 2] + cx
        if np.all((u_proj >= 10.0) & (u_proj <= w - 10.0)):
            test_zs.append(z_cand)
            test_xs.append(x_cand)
            test_yaws_deg.append(yaw_cand)

    results = []
    print(f"\nStarting 1000 viewpoints simulation & pose inference...\n")
    t_bench_start = time.time()

    for i in range(N):
        x_gt = float(test_xs[i])
        y_gt = float(fixed_y)
        z_gt = float(test_zs[i])
        yaw_gt = float(test_yaws_deg[i])
        pitch_gt = 0.0
        roll_gt = 0.0

        # Rotate around CAD vertical axis Z (which aligns with camera UP -Y)
        psi_rad = np.radians(yaw_gt)
        R_z = np.array([
            [np.cos(psi_rad), -np.sin(psi_rad), 0.0],
            [np.sin(psi_rad),  np.cos(psi_rad), 0.0],
            [0.0,             0.0,            1.0]
        ], dtype=np.float32)

        R_gt = R_0 @ R_z
        t_gt = np.array([x_gt, y_gt, z_gt], dtype=np.float32)

        T_gt = np.eye(4, dtype=np.float32)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = t_gt

        # Raycast camera in CAD space
        T_gc = np.linalg.inv(T_gt)
        dirs_world = (T_gc[:3, :3] @ dirs_cam_norm.T).T
        origins_world = np.broadcast_to(T_gc[:3, 3], dirs_world.shape).astype(np.float32)
        rays = np.concatenate([origins_world, dirs_world], axis=-1)
        rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

        t0 = time.time()
        ans = scene.cast_rays(rays_tensor)
        t_hit = ans['t_hit'].numpy()
        valid = np.isfinite(t_hit)

        if np.sum(valid) < 64:
            continue

        # Depth with realistic RealSense noise model
        z_depth = t_hit[valid] / norms.reshape(-1)[valid]
        sigma_z = (z_depth ** 2) / (fx * 0.05) * 0.08
        z_noisy = z_depth + np.random.normal(0, sigma_z)
        z_noisy = np.clip(z_noisy, 0.2, 2.5)

        u_val = u.reshape(-1)[valid]
        v_val = v.reshape(-1)[valid]
        pts_cam = np.stack([
            (u_val - cx) * z_noisy / fx,
            (v_val - cy) * z_noisy / fy,
            z_noisy
        ], axis=-1).astype(np.float32)

        # Depth range filter
        mask_depth = (pts_cam[:, 2] >= 0.4) & (pts_cam[:, 2] <= 1.8)
        pts_filtered = pts_cam[mask_depth]
        if len(pts_filtered) < 64:
            continue

        # AI GeoPoseNet Correspondence Forward Pass
        t_ai_start = time.time()
        choice = np.random.choice(len(pts_filtered), 1024, replace=(len(pts_filtered) < 1024))
        obs_sample = pts_filtered[choice]

        with torch.no_grad():
            o_pts = torch.from_numpy(obs_sample).unsqueeze(0).to(device)
            D_obs = model.encode_obs(o_pts)
            sim = model.compute_similarity(D_obs, D_cad)[0]
            probs = torch.softmax(sim, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=1, dim=-1)

        t_ai_end = time.time()
        pred_idx_np = top_indices[:, 0].cpu().numpy()
        conf_np = top_probs[:, 0].cpu().numpy()
        matched_cad_pts = key_centroids[pred_idx_np]

        # Correspondence NMS
        unique_c = np.unique(pred_idx_np)
        keep_mask = np.zeros(len(obs_sample), dtype=bool)
        for c in unique_c:
            c_indices = np.where(pred_idx_np == c)[0]
            top_c = c_indices[np.argsort(-conf_np[c_indices])[:3]]
            keep_mask[top_c] = True

        pts_obs_nms = obs_sample[keep_mask]
        pts_cad_nms = matched_cad_pts[keep_mask]
        conf_nms = conf_np[keep_mask]

        # Confidence-weighted Kabsch RANSAC
        T_coarse, inliers = kabsch_ransac(pts_obs_nms, pts_cad_nms, conf_nms, max_iter=1500, distance_threshold=0.035)

        # Symmetry & Aspect Ratio Disambiguation
        best_fit_score = -1.0
        best_T_cand = T_coarse
        sub_pts = pts_filtered[::max(1, len(pts_filtered) // 100)]
        for k in [0, 1, 2, 3]:
            ang = k * np.pi / 2.0
            R_k = np.array([[np.cos(ang), -np.sin(ang), 0],
                            [np.sin(ang),  np.cos(ang), 0],
                            [0,            0,           1]], dtype=np.float32)
            R_test = T_coarse[:3, :3] @ R_k
            t_test = T_coarse[:3, 3]
            T_test = np.eye(4, dtype=np.float32)
            T_test[:3, :3] = R_test
            T_test[:3, 3] = t_test
            T_inv_test = np.linalg.inv(T_test)
            pts_in_cad = (T_inv_test[:3, :3] @ sub_pts.T).T + T_inv_test[:3, 3]
            in_b = np.mean((pts_in_cad[:, 0] >= -0.26) & (pts_in_cad[:, 0] <= 0.26) &
                           (pts_in_cad[:, 1] >= -0.45) & (pts_in_cad[:, 1] <= 0.45) &
                           (pts_in_cad[:, 2] >= -0.36) & (pts_in_cad[:, 2] <= 0.20))
            dists, _ = cad_kdtree.query(pts_in_cad, k=1)
            surf_fit = np.mean(dists < 0.03)
            score = in_b * 0.5 + surf_fit * 0.5
            if score > best_fit_score:
                best_fit_score = score
                best_T_cand = T_test
        T_init = best_T_cand.copy()

        # Centroid Coarse Sanity Check (Prevent translational sliding / repetitive symmetry jump)
        bbox_obs_min = np.min(pts_filtered, axis=0)
        bbox_obs_max = np.max(pts_filtered, axis=0)
        bbox_obs_center = (bbox_obs_min + bbox_obs_max) / 2.0

        # Check lateral offset: If AI coarse X deviates from observed BBox X by > 40mm
        if abs(T_init[0, 3] - bbox_obs_center[0]) > 0.040:
            T_init[0, 3] = bbox_obs_center[0]

        # Check depth offset: If AI coarse Z deviates from observed BBox Z by > 60mm
        if abs(T_init[2, 3] - bbox_obs_center[2]) > 0.060:
            T_init[2, 3] = bbox_obs_center[2]

        # Coarse-to-Fine Multi-Scale Inverse ICP Refinement (Obs -> CAD)
        t_icp_start = time.time()
        obs_pcd = o3d.geometry.PointCloud()
        obs_pcd.points = o3d.utility.Vector3dVector(pts_filtered)
        obs_pcd.estimate_normals()

        T_refined, fitness, rmse = run_inverse_multiscale_icp(obs_pcd, cad_icp_pcd, T_init)
        t_icp_end = time.time()

        t_total_ms = (t_icp_end - t0) * 1000.0
        ai_ms = (t_ai_end - t_ai_start) * 1000.0
        icp_ms = (t_icp_end - t_icp_start) * 1000.0

        # Translation Decomposition in Camera Frame
        t_pred = T_refined[:3, 3]
        x_pred, y_pred, z_pred = float(t_pred[0]), float(t_pred[1]), float(t_pred[2])
        dx_mm = (x_pred - x_gt) * 1000.0
        dy_mm = (y_pred - y_gt) * 1000.0
        dz_mm = (z_pred - z_gt) * 1000.0
        e_t_mm = float(np.sqrt(dx_mm**2 + dy_mm**2 + dz_mm**2))
        e_xz_mm = float(np.sqrt(dx_mm**2 + dz_mm**2))

        # Rotation Decomposition in Camera Frame
        R_err_mat = T_refined[:3, :3].T @ T_gt[:3, :3]
        cos_R = np.clip((np.trace(R_err_mat) - 1.0) / 2.0, -1.0, 1.0)
        e_R = np.degrees(np.arccos(cos_R))

        # Symmetry rotation error (C2 symmetry about local CAD Z)
        R_z180 = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
        R_err_sym_mat = (T_refined[:3, :3] @ R_z180).T @ T_gt[:3, :3]
        cos_R_sym = np.clip((np.trace(R_err_sym_mat) - 1.0) / 2.0, -1.0, 1.0)
        e_R_sym = min(e_R, float(np.degrees(np.arccos(cos_R_sym))))

        # Euler Angle Errors in Camera Frame (Yaw around Y, Pitch around X, Roll around Z)
        # Choose best symmetry match for Euler decomposition
        if e_R_sym < e_R:
            R_pred_eval = T_refined[:3, :3] @ R_z180
        else:
            R_pred_eval = T_refined[:3, :3]

        R_rel = R_pred_eval @ R_gt.T
        euler_err = ScipyRot.from_matrix(R_rel).as_euler('yxz', degrees=True)
        # euler_err[0]: delta_yaw, euler_err[1]: delta_pitch, euler_err[2]: delta_roll
        delta_yaw_deg = float(abs(euler_err[0]))
        if delta_yaw_deg > 180.0:
            delta_yaw_deg = 360.0 - delta_yaw_deg
        delta_pitch_deg = float(abs(euler_err[1]))
        delta_roll_deg = float(abs(euler_err[2]))

        # ADD / ADD-S Metrics
        add, adds = compute_adds(T_refined, T_gt, cad_eval_pts, cad_kdtree)

        # Pass criteria
        bop_01d = (adds < 0.10 * cad_diameter * 1000.0)
        strict_005d = (adds < 0.05 * cad_diameter * 1000.0)
        prec_20_5 = (e_t_mm < 20.0 and e_R_sym < 5.0)
        prec_15_2 = (e_t_mm < 15.0 and delta_yaw_deg < 2.0)

        results.append({
            "sample_id": i + 1,
            "x_gt_mm": round(x_gt * 1000.0, 2),
            "y_gt_mm": round(y_gt * 1000.0, 2),
            "z_gt_mm": round(z_gt * 1000.0, 2),
            "yaw_gt_deg": round(yaw_gt, 2),
            "pitch_gt_deg": 0.0,
            "roll_gt_deg": 0.0,
            "x_pred_mm": round(x_pred * 1000.0, 2),
            "y_pred_mm": round(y_pred * 1000.0, 2),
            "z_pred_mm": round(z_pred * 1000.0, 2),
            "delta_x_mm": round(dx_mm, 2),
            "delta_y_mm": round(dy_mm, 2),
            "delta_z_mm": round(dz_mm, 2),
            "e_x_abs_mm": round(abs(dx_mm), 2),
            "e_y_abs_mm": round(abs(dy_mm), 2),
            "e_z_abs_mm": round(abs(dz_mm), 2),
            "e_t_mm": round(e_t_mm, 2),
            "e_xz_mm": round(e_xz_mm, 2),
            "delta_yaw_deg": round(delta_yaw_deg, 3),
            "delta_pitch_deg": round(delta_pitch_deg, 3),
            "delta_roll_deg": round(delta_roll_deg, 3),
            "e_R_sym_deg": round(float(e_R_sym), 3),
            "add_mm": round(float(add), 2),
            "adds_mm": round(float(adds), 2),
            "pass_bop_01d": bool(bop_01d),
            "pass_strict_005d": bool(strict_005d),
            "pass_precision_20mm_5deg": bool(prec_20_5),
            "pass_planar_15mm_2deg": bool(prec_15_2),
            "fitness": round(float(fitness), 4),
            "rmse_mm": round(float(rmse * 1000.0), 2),
            "ai_latency_ms": round(float(ai_ms), 2),
            "icp_latency_ms": round(float(icp_ms), 2),
            "total_latency_ms": round(float(t_total_ms), 2)
        })

        if (i + 1) % 100 == 0 or (i + 1) == N:
            cur_bop = 100.0 * np.mean([r["pass_bop_01d"] for r in results])
            cur_et = [r["e_t_mm"] for r in results]
            cur_eyaw = [r["delta_yaw_deg"] for r in results]
            cur_adds = [r["adds_mm"] for r in results]
            print(f"[{i+1:4d}/{N}] BOP Recall@0.1d: {cur_bop:5.1f}% | Median e_t: {np.median(cur_et):4.1f} mm | Median Yaw: {np.median(cur_eyaw):4.2f}° | Median ADD-S: {np.median(cur_adds):4.1f} mm | Avg Latency: {np.mean([r['total_latency_ms'] for r in results]):.1f} ms")

    total_time = time.time() - t_bench_start
    print(f"\nCompleted {len(results)} samples in {total_time:.1f} s ({total_time/len(results)*1000:.1f} ms/view)\n")

    # 5. Export CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"CSV log saved to: {args.output_csv}")

    # 6. Aggregate Statistics
    def stats_series(s):
        return {
            "mean": round(float(s.mean()), 2),
            "median": round(float(s.median()), 2),
            "std": round(float(s.std()), 2),
            "p25": round(float(s.quantile(0.25)), 2),
            "p75": round(float(s.quantile(0.75)), 2),
            "p90": round(float(s.quantile(0.90)), 2),
            "p95": round(float(s.quantile(0.95)), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2)
        }

    summary = {
        "benchmark_metadata": {
            "cad_name": args.cad_name,
            "cad_diameter_mm": round(float(cad_diameter * 1000.0), 2),
            "coordinate_frame": "camera_frame_planar",
            "y_fixed_m": args.fixed_y,
            "pitch_fixed_deg": 0.0,
            "roll_fixed_deg": 0.0,
            "depth_range_m": [args.min_dist, args.max_dist],
            "x_span_m": [-args.x_span, args.x_span],
            "yaw_range_deg": [0.0, 360.0],
            "total_samples": len(df),
            "compute_device": str(device),
            "total_benchmark_seconds": round(total_time, 2)
        },
        "pass_rates": {
            "bop_recall_01d": round(float(100.0 * df["pass_bop_01d"].mean()), 2),
            "strict_recall_005d": round(float(100.0 * df["pass_strict_005d"].mean()), 2),
            "precision_20mm_5deg": round(float(100.0 * df["pass_precision_20mm_5deg"].mean()), 2),
            "planar_precision_15mm_2deg": round(float(100.0 * df["pass_planar_15mm_2deg"].mean()), 2)
        },
        "overall_metrics": {
            "translation_error_mm": stats_series(df["e_t_mm"]),
            "planar_error_xz_mm": stats_series(df["e_xz_mm"]),
            "error_x_abs_mm": stats_series(df["e_x_abs_mm"]),
            "error_y_abs_mm": stats_series(df["e_y_abs_mm"]),
            "error_z_abs_mm": stats_series(df["e_z_abs_mm"]),
            "delta_x_signed_mm": stats_series(df["delta_x_mm"]),
            "delta_y_signed_mm": stats_series(df["delta_y_mm"]),
            "delta_z_signed_mm": stats_series(df["delta_z_mm"]),
            "rotation_error_sym_deg": stats_series(df["e_R_sym_deg"]),
            "delta_yaw_deg": stats_series(df["delta_yaw_deg"]),
            "delta_pitch_deg": stats_series(df["delta_pitch_deg"]),
            "delta_roll_deg": stats_series(df["delta_roll_deg"]),
            "adds_metric_mm": stats_series(df["adds_mm"]),
            "latency_ms": stats_series(df["total_latency_ms"]),
            "ai_latency_ms": stats_series(df["ai_latency_ms"]),
            "icp_latency_ms": stats_series(df["icp_latency_ms"])
        }
    }

    # Breakdown by Distance
    dist_step = (args.max_dist - args.min_dist) / 4.0
    dist_bins = [round(args.min_dist + k * dist_step, 2) for k in range(5)]
    dist_labels = [f"{dist_bins[k]:.1f}-{dist_bins[k+1]:.1f}m" for k in range(4)]
    df["dist_bin"] = pd.cut(df["z_gt_mm"] / 1000.0, bins=dist_bins, labels=dist_labels)
    summary["breakdown_by_distance"] = {}
    for d_label in dist_labels:
        grp = df[df["dist_bin"] == d_label]
        summary["breakdown_by_distance"][d_label] = {
            "samples": int(len(grp)),
            "bop_recall_01d": round(float(100.0 * grp["pass_bop_01d"].mean()), 1),
            "strict_recall_005d": round(float(100.0 * grp["pass_strict_005d"].mean()), 1),
            "precision_20mm_5deg": round(float(100.0 * grp["pass_precision_20mm_5deg"].mean()), 1),
            "median_et_mm": round(float(grp["e_t_mm"].median()), 2),
            "median_ex_mm": round(float(grp["e_x_abs_mm"].median()), 2),
            "median_ey_mm": round(float(grp["e_y_abs_mm"].median()), 2),
            "median_ez_mm": round(float(grp["e_z_abs_mm"].median()), 2),
            "median_yaw_deg": round(float(grp["delta_yaw_deg"].median()), 3),
            "median_adds_mm": round(float(grp["adds_mm"].median()), 2)
        }

    # Breakdown by Yaw Quadrant
    yaw_bins = [0, 90, 180, 270, 360]
    yaw_labels = ["0°-90°", "90°-180°", "180°-270°", "270°-360°"]
    df["yaw_bin"] = pd.cut(df["yaw_gt_deg"], bins=yaw_bins, labels=yaw_labels)
    summary["breakdown_by_yaw"] = {}
    for y_label in yaw_labels:
        grp = df[df["yaw_bin"] == y_label]
        summary["breakdown_by_yaw"][y_label] = {
            "samples": int(len(grp)),
            "bop_recall_01d": round(float(100.0 * grp["pass_bop_01d"].mean()), 1),
            "median_et_mm": round(float(grp["e_t_mm"].median()), 2),
            "median_yaw_deg": round(float(grp["delta_yaw_deg"].median()), 3),
            "median_adds_mm": round(float(grp["adds_mm"].median()), 2)
        }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved to: {args.output_json}")

    # 7. Generate High-Resolution 4-Panel Figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
    fig.suptitle(f"BÁO CÁO MÔ PHỎNG 6DoF CAMERA FRAME -- 1000 ĐIỂM (X, Z, YAW)\nTABLE-DEMO | Y = 30cm (cố định), Pitch = 0°, Roll = 0° (cố định)", fontsize=13, fontweight='bold', y=0.98)

    # Panel 1: Translation Errors Decomposition (e_x, e_y, e_z, e_t)
    ax1.set_title("Phân Bố Sai Số Vị Trí (e_t, e_x, e_y, e_z)", fontsize=11, fontweight='bold')
    ax1.hist(df["e_t_mm"], bins=35, color="#1e88e5", alpha=0.6, label=f"e_t (Median {df['e_t_mm'].median():.1f}mm)", density=True)
    ax1.hist(df["e_z_abs_mm"], bins=35, color="#e53935", alpha=0.4, label=f"|e_z| depth ({df['e_z_abs_mm'].median():.1f}mm)", density=True)
    ax1.hist(df["e_x_abs_mm"], bins=35, color="#43a047", alpha=0.4, label=f"|e_x| lateral ({df['e_x_abs_mm'].median():.1f}mm)", density=True)
    ax1.hist(df["e_y_abs_mm"], bins=35, color="#fb8c00", alpha=0.4, label=f"|e_y| vertical ({df['e_y_abs_mm'].median():.1f}mm)", density=True)
    ax1.axvline(20.0, color="green", linestyle="--", linewidth=1.5, label="Chuẩn gắp 20mm")
    ax1.set_xlabel("Sai số (mm)", fontsize=10)
    ax1.set_ylabel("Mật độ xác suất (Density)", fontsize=10)
    ax1.set_xlim(0, min(35.0, df["e_t_mm"].quantile(0.99) * 1.3))
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Euler Angles Error (Yaw, Pitch, Roll)
    ax2.set_title("Phân Bố Sai Số Góc Xoay (Yaw vs Pitch, Roll)", fontsize=11, fontweight='bold')
    ax2.hist(df["delta_yaw_deg"], bins=40, color="#00897b", alpha=0.7, edgecolor="#004d40", label=f"Sai số Yaw (Median {df['delta_yaw_deg'].median():.2f}°)", density=True)
    ax2.hist(df["delta_pitch_deg"], bins=40, color="#8e24aa", alpha=0.4, label=f"Sai số Pitch (Median {df['delta_pitch_deg'].median():.2f}°)", density=True)
    ax2.hist(df["delta_roll_deg"], bins=40, color="#3949ab", alpha=0.4, label=f"Sai số Roll (Median {df['delta_roll_deg'].median():.2f}°)", density=True)
    ax2.axvline(2.0, color="#004d40", linestyle="--", linewidth=1.5, label="Ngưỡng 2.0°")
    ax2.set_xlabel("Sai số góc (độ)", fontsize=10)
    ax2.set_ylabel("Mật độ xác suất (Density)", fontsize=10)
    ax2.set_xlim(0, min(6.0, max(2.5, df["delta_yaw_deg"].quantile(0.98))))
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: ADD-S Recall Curve
    thresholds = np.linspace(0, 150, 300)
    recalls = [np.mean(df["adds_mm"] < th) * 100.0 for th in thresholds]
    ax3.set_title("Đường Cong BOP ADD-S Recall (@ 1000 Điểm)", fontsize=11, fontweight='bold')
    ax3.plot(thresholds, recalls, color="#5e35b1", linewidth=2.5, label="Recall ADD-S")
    bop_th = 0.10 * cad_diameter * 1000.0
    strict_th = 0.05 * cad_diameter * 1000.0
    ax3.axvline(strict_th, color="#0288d1", linestyle="--", linewidth=1.5, label=f"Strict 0.05d ({strict_th:.1f} mm)")
    ax3.axvline(bop_th, color="#d32f2f", linestyle="--", linewidth=1.5, label=f"BOP 0.10d ({bop_th:.1f} mm)")
    ax3.set_xlabel("Ngưỡng ADD-S (mm)", fontsize=10)
    ax3.set_ylabel("Tỉ lệ Recall đạt chuẩn (%)", fontsize=10)
    ax3.set_ylim(-2, 105)
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Panel 4: Planar Scatter in Camera Field of View (X_cam vs Z_cam)
    ax4.set_title("Phân Bố 1.000 Điểm Quan Sát Trong Hệ Camera (X vs Z)", fontsize=11, fontweight='bold')
    sc = ax4.scatter(df["x_gt_mm"], df["z_gt_mm"], c=df["e_t_mm"], cmap="viridis", s=18, alpha=0.75)
    cbar = plt.colorbar(sc, ax=ax4)
    cbar.set_label("Sai số tịnh tiến e_t (mm)", fontsize=9)
    ax4.set_xlabel("Vị trí ngang X_cam (mm)", fontsize=10)
    ax4.set_ylabel("Khoảng cách sâu Z_cam (mm)", fontsize=10)
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_fig), exist_ok=True)
    plt.savefig(args.output_fig, dpi=300)
    plt.close()
    print(f"Figure saved to: {args.output_fig}")

    print("\nBenchmark completed successfully.")

if __name__ == "__main__":
    main()
