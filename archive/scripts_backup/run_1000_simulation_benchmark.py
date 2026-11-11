#!/usr/bin/env python3
"""
Benchmark 1000 Unoccluded Synthetic Viewpoints for 6DoF CAD Pose Estimation
- CAD Object: TABLE-DEMO.stl (Diameter = 1.089 m)
- Occlusion: 0% (Fully unoccluded CAD)
- Distance: 0.8 m to 1.2 m
- Angles: Upper hemisphere, Azimuth [0, 360 deg), Elevation [15, 65 deg], Roll [-5, 5 deg]
- Sensor: Intel RealSense D435 Realistic Optical & Noise Model
- Total Points: 1000 points
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deep_learning.model_hierarchy import CADHierarchyCorrespondenceModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of simulated viewpoints")
    parser.add_argument("--cad-name", type=str, default="TABLE-DEMO")
    parser.add_argument("--min-dist", type=float, default=0.8)
    parser.add_argument("--max-dist", type=float, default=1.2)
    parser.add_argument("--min-elev", type=float, default=15.0)
    parser.add_argument("--max-elev", type=float, default=65.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv-out", type=str, default="logs/benchmark_1000_unoccluded_simulation.csv")
    parser.add_argument("--json-out", type=str, default="logs/benchmark_1000_unoccluded_summary.json")
    parser.add_argument("--fig-out", type=str, default="/home/huyen1905/.gemini/antigravity-ide/brain/9e85ad11-79b5-433e-9c12-9efb8e933a5e/benchmark_1000_figures.png")
    return parser.parse_args()

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

def run_multiscale_icp(cad_pcd, obs_pcd, T_init, scales=[0.03, 0.015, 0.008, 0.004]):
    T_curr = T_init.copy()
    final_fitness = 0.0
    final_rmse = 0.0
    for dist_th in scales:
        loss_k = max(0.003, dist_th / 2.0)
        reg = o3d.pipelines.registration.registration_icp(
            cad_pcd, obs_pcd, dist_th, T_curr,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.TukeyLoss(k=loss_k)
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25, relative_fitness=1e-5, relative_rmse=1e-5)
        )
        T_curr = reg.transformation
        final_fitness = reg.fitness
        final_rmse = reg.inlier_rmse
    return T_curr, final_fitness, final_rmse

def compute_adds(T_pred, T_gt, cad_eval_pts, cad_tree):
    # Transformed points
    pts_pred = (T_pred[:3, :3] @ cad_eval_pts.T).T + T_pred[:3, 3]
    pts_gt = (T_gt[:3, :3] @ cad_eval_pts.T).T + T_gt[:3, 3]
    
    # Standard ADD
    add = np.mean(np.linalg.norm(pts_pred - pts_gt, axis=1)) * 1000.0 # mm
    
    # ADD-S for symmetric objects: nearest neighbor distance in CAD point cloud
    pts_pred_in_gt = (T_gt[:3, :3].T @ (pts_pred - T_gt[:3, 3]).T).T
    dists, _ = cad_tree.query(pts_pred_in_gt, k=1)
    adds = np.mean(dists) * 1000.0 # mm
    return add, adds

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f" 6DoF CAD SIMULATION BENCHMARK -- 1000 UNOCCLUDED VIEWS")
    print(f" Object:        {args.cad_name}")
    print(f" Distance:      [{args.min_dist:.2f} m, {args.max_dist:.2f} m]")
    print(f" Elevation:     [{args.min_elev:.1f} deg, {args.max_elev:.1f} deg]")
    print(f" Azimuth:       [0.0 deg, 360.0 deg)")
    print(f" Total Views:   {args.num_samples}")
    print(f" Compute Dev:   {device}")
    print(f"============================================================")

    cad_dir = os.path.join(project_root, "CAD_DATA")
    mesh_path = os.path.join(cad_dir, f"{args.cad_name}_mesh.ply")
    tri_path = os.path.join(cad_dir, f"{args.cad_name}_triangles.npz")
    patch_path = os.path.join(cad_dir, f"{args.cad_name}_patches.npz")
    sym_path = os.path.join(cad_dir, f"{args.cad_name}_symmetry.json")
    ckpt_path = os.path.join(project_root, "deep_learning/checkpoints", args.cad_name, "best_hierarchy_model.pth")

    # 1. Load CAD Mesh & Raycasting Scene
    mesh_legacy = o3d.io.read_triangle_mesh(mesh_path)
    mesh_legacy.compute_vertex_normals()
    cad_diameter = np.linalg.norm(mesh_legacy.get_max_bound() - mesh_legacy.get_min_bound())
    print(f"CAD Bounding Box Diameter: {cad_diameter:.4f} m ({cad_diameter*1000:.1f} mm)")

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    # CAD point clouds for ICP & ADD-S
    cad_icp_pcd = mesh_legacy.sample_points_uniformly(10000)
    cad_icp_pcd.estimate_normals()

    cad_eval_pts = np.asarray(mesh_legacy.sample_points_uniformly(1500).points, dtype=np.float32)
    cad_kdtree = KDTree(cad_eval_pts)

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

    # Pre-encode CAD once in VRAM
    with torch.no_grad():
        c_feat = torch.from_numpy(patch_features).unsqueeze(0).to(device)
        c_rel = torch.from_numpy(patch_rel_pos).unsqueeze(0).to(device)
        c_hops = torch.from_numpy(patch_hops).unsqueeze(0).to(device)
        D_cad = model.encode_cad(c_feat, c_rel, c_hops)

    print("Pre-encoded CAD hierarchy features loaded in GPU VRAM.")

    # 3. Camera Pinhole Grid
    w, h = 640, 480
    fx, fy, cx, cy = 386.652, 386.652, 320.0, 240.0
    step = 2 # 320x240 raycast grid (~76.8k rays)
    v, u = np.indices((h // step, w // step), dtype=np.float32)
    u = u * step
    v = v * step
    dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    norms = np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    dirs_cam_norm = (dirs_cam / norms).reshape(-1, 3).astype(np.float32)
    num_rays = len(dirs_cam_norm)

    # 4. Generate 1000 Viewpoints
    N = args.num_samples
    distances = np.random.uniform(args.min_dist, args.max_dist, size=N)
    azimuths_deg = np.random.uniform(0.0, 360.0, size=N)
    elevations_deg = np.random.uniform(args.min_elev, args.max_elev, size=N)
    rolls_deg = np.random.uniform(-5.0, 5.0, size=N)

    # Target centers (table center is around z=0)
    target_center = np.array([0.0, 0.0, -0.003], dtype=np.float32)

    results = []
    print(f"\nStarting 1000 viewpoints simulation & pose inference...\n")
    t_bench_start = time.time()

    for i in range(N):
        d = distances[i]
        az_rad = np.radians(azimuths_deg[i])
        el_rad = np.radians(elevations_deg[i])
        roll_rad = np.radians(rolls_deg[i])

        cam_pos = np.array([
            d * np.cos(el_rad) * np.cos(az_rad),
            d * np.cos(el_rad) * np.sin(az_rad),
            d * np.sin(el_rad)
        ], dtype=np.float32) + target_center

        forward = (target_center - cam_pos)
        forward /= np.linalg.norm(forward)
        up_world = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(forward, up_world)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-4:
            right = np.array([1, 0, 0], dtype=np.float32)
        else:
            right /= right_norm
        down = np.cross(forward, right)

        R_wc = np.stack([right, down, forward], axis=1)
        if abs(rolls_deg[i]) > 1e-3:
            # in-plane camera roll
            R_roll = np.array([
                [np.cos(roll_rad), -np.sin(roll_rad), 0],
                [np.sin(roll_rad),  np.cos(roll_rad), 0],
                [0, 0, 1]
            ], dtype=np.float32)
            R_wc = R_wc @ R_roll

        T_wc = np.eye(4, dtype=np.float32)
        T_wc[:3, :3] = R_wc
        T_wc[:3, 3] = cam_pos
        T_gt = np.linalg.inv(T_wc)

        # Raycast
        t0 = time.time()
        dirs_world = (R_wc @ dirs_cam_norm.T).T
        origins_world = np.broadcast_to(cam_pos, dirs_world.shape).astype(np.float32)
        rays = np.concatenate([origins_world, dirs_world], axis=-1)
        rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

        ans = scene.cast_rays(rays_tensor)
        t_hit = ans['t_hit'].numpy()
        valid = np.isfinite(t_hit)

        if np.sum(valid) < 64:
            continue

        # Depth with realistic RealSense noise
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
        ], axis=-1)

        # 5mm voxel downsampling
        obs_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_cam))
        obs_pcd = obs_pcd.voxel_down_sample(0.005)
        obs_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
        pts_filtered = np.asarray(obs_pcd.points, dtype=np.float32)

        if len(pts_filtered) < 64:
            continue

        # AI Sample
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

        # NMS
        unique_c = np.unique(pred_idx_np)
        keep_mask = np.zeros(len(obs_sample), dtype=bool)
        for c in unique_c:
            c_indices = np.where(pred_idx_np == c)[0]
            top_c = c_indices[np.argsort(-conf_np[c_indices])[:3]]
            keep_mask[top_c] = True

        pts_obs_nms = obs_sample[keep_mask]
        pts_cad_nms = matched_cad_pts[keep_mask]
        conf_nms = conf_np[keep_mask]

        # Kabsch RANSAC
        T_coarse, inliers = kabsch_ransac(pts_obs_nms, pts_cad_nms, conf_nms, max_iter=1500, distance_threshold=0.035)

        # Multi-candidate symmetry and rectangular aspect ratio disambiguation (0, 90, 180, 270 deg)
        # Because the table is 0.85m x 0.46m, candidates rotated by 90/270 deg will overflow the 0.46m width!
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
        T_init = best_T_cand

        # Coarse-to-Fine Multi-Scale ICP Refinement
        t_icp_start = time.time()
        T_refined, fitness, rmse = run_multiscale_icp(cad_icp_pcd, obs_pcd, T_init)
        t_icp_end = time.time()

        t_total_ms = (t_icp_end - t0) * 1000.0
        ai_ms = (t_ai_end - t_ai_start) * 1000.0
        icp_ms = (t_icp_end - t_icp_start) * 1000.0

        # Compute Errors
        e_t = np.linalg.norm(T_refined[:3, 3] - T_gt[:3, 3]) * 1000.0 # mm
        
        R_err_mat = T_refined[:3, :3].T @ T_gt[:3, :3]
        cos_R = np.clip((np.trace(R_err_mat) - 1.0) / 2.0, -1.0, 1.0)
        e_R = np.degrees(np.arccos(cos_R))

        # Symmetry rotation error (C2 symmetry about Z)
        R_z180 = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
        R_err_sym_mat = (T_refined[:3, :3] @ R_z180).T @ T_gt[:3, :3]
        cos_R_sym = np.clip((np.trace(R_err_sym_mat) - 1.0) / 2.0, -1.0, 1.0)
        e_R_sym = min(e_R, float(np.degrees(np.arccos(cos_R_sym))))

        # ADD / ADD-S
        add, adds = compute_adds(T_refined, T_gt, cad_eval_pts, cad_kdtree)

        # Pass criteria
        bop_01d = (adds < 0.10 * cad_diameter * 1000.0)
        strict_005d = (adds < 0.05 * cad_diameter * 1000.0)
        prec_20_5 = (e_t < 20.0 and e_R_sym < 5.0)
        prec_10_2 = (e_t < 10.0 and e_R_sym < 2.0)

        results.append({
            "sample_id": i + 1,
            "distance_m": round(float(d), 3),
            "azimuth_deg": round(float(azimuths_deg[i]), 1),
            "elevation_deg": round(float(elevations_deg[i]), 1),
            "roll_deg": round(float(rolls_deg[i]), 1),
            "valid": True,
            "e_t_mm": round(float(e_t), 2),
            "e_R_deg": round(float(e_R), 2),
            "e_R_sym_deg": round(float(e_R_sym), 2),
            "add_mm": round(float(add), 2),
            "adds_mm": round(float(adds), 2),
            "pass_bop_01d": bool(bop_01d),
            "pass_strict_005d": bool(strict_005d),
            "pass_precision_20mm_5deg": bool(prec_20_5),
            "pass_ultra_10mm_2deg": bool(prec_10_2),
            "fitness": round(float(fitness), 4),
            "rmse_mm": round(float(rmse * 1000.0), 2),
            "num_inliers": int(np.sum(inliers)),
            "ai_latency_ms": round(float(ai_ms), 1),
            "icp_latency_ms": round(float(icp_ms), 1),
            "total_latency_ms": round(float(t_total_ms), 1)
        })

        if (i + 1) % 100 == 0 or (i + 1) == N:
            cur_adds = [r["adds_mm"] for r in results]
            cur_et = [r["e_t_mm"] for r in results]
            cur_er = [r["e_R_sym_deg"] for r in results]
            cur_bop = np.mean([r["pass_bop_01d"] for r in results]) * 100.0
            print(f"[{i+1:4d}/{N}] BOP Recall@0.1d: {cur_bop:5.1f}% | Median e_t: {np.median(cur_et):4.1f} mm | Median e_R: {np.median(cur_er):4.2f}° | Median ADD-S: {np.median(cur_adds):4.1f} mm | Avg Latency: {np.mean([r['total_latency_ms'] for r in results]):.1f} ms")

    total_bench_time = time.time() - t_bench_start
    print(f"\nCompleted {len(results)} viewpoints in {total_bench_time:.1f} s ({total_bench_time/len(results)*1000:.1f} ms/view)!")

    # 5. Save DataFrame to CSV
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.csv_out, index=False)
    print(f"Saved full run data to: {args.csv_out}")

    # 6. Compute Statistical Summary
    summary = {
        "benchmark_metadata": {
            "cad_object": args.cad_name,
            "cad_diameter_mm": round(cad_diameter * 1000.0, 2),
            "bop_threshold_01d_mm": round(0.10 * cad_diameter * 1000.0, 2),
            "strict_threshold_005d_mm": round(0.05 * cad_diameter * 1000.0, 2),
            "total_viewpoints_tested": len(df),
            "distance_range_m": [args.min_dist, args.max_dist],
            "elevation_range_deg": [args.min_elev, args.max_elev],
            "azimuth_range_deg": [0.0, 360.0],
            "occlusion": "0% (Full unoccluded CAD)",
            "sensor_noise": "Intel RealSense D435 realistic disparity noise",
            "total_benchmark_duration_s": round(total_bench_time, 2)
        },
        "overall_metrics": {
            "bop_recall_01d_percent": round(float(df["pass_bop_01d"].mean() * 100.0), 2),
            "strict_recall_005d_percent": round(float(df["pass_strict_005d"].mean() * 100.0), 2),
            "precision_20mm_5deg_percent": round(float(df["pass_precision_20mm_5deg"].mean() * 100.0), 2),
            "ultra_precision_10mm_2deg_percent": round(float(df["pass_ultra_10mm_2deg"].mean() * 100.0), 2),
            "translation_error_mm": {
                "mean": round(float(df["e_t_mm"].mean()), 2),
                "median": round(float(df["e_t_mm"].median()), 2),
                "std": round(float(df["e_t_mm"].std()), 2),
                "p25": round(float(df["e_t_mm"].quantile(0.25)), 2),
                "p75": round(float(df["e_t_mm"].quantile(0.75)), 2),
                "p90": round(float(df["e_t_mm"].quantile(0.90)), 2),
                "p95": round(float(df["e_t_mm"].quantile(0.95)), 2),
                "min": round(float(df["e_t_mm"].min()), 2),
                "max": round(float(df["e_t_mm"].max()), 2)
            },
            "rotation_error_sym_deg": {
                "mean": round(float(df["e_R_sym_deg"].mean()), 2),
                "median": round(float(df["e_R_sym_deg"].median()), 2),
                "std": round(float(df["e_R_sym_deg"].std()), 2),
                "p25": round(float(df["e_R_sym_deg"].quantile(0.25)), 2),
                "p75": round(float(df["e_R_sym_deg"].quantile(0.75)), 2),
                "p90": round(float(df["e_R_sym_deg"].quantile(0.90)), 2),
                "p95": round(float(df["e_R_sym_deg"].quantile(0.95)), 2),
                "min": round(float(df["e_R_sym_deg"].min()), 2),
                "max": round(float(df["e_R_sym_deg"].max()), 2)
            },
            "adds_metric_mm": {
                "mean": round(float(df["adds_mm"].mean()), 2),
                "median": round(float(df["adds_mm"].median()), 2),
                "std": round(float(df["adds_mm"].std()), 2),
                "p25": round(float(df["adds_mm"].quantile(0.25)), 2),
                "p75": round(float(df["adds_mm"].quantile(0.75)), 2),
                "p90": round(float(df["adds_mm"].quantile(0.90)), 2),
                "p95": round(float(df["adds_mm"].quantile(0.95)), 2)
            },
            "quality_telemetry": {
                "mean_fitness_percent": round(float(df["fitness"].mean() * 100.0), 2),
                "mean_inlier_rmse_mm": round(float(df["rmse_mm"].mean()), 2),
                "mean_inliers_count": round(float(df["num_inliers"].mean()), 1)
            },
            "latency_ms": {
                "mean_ai_ms": round(float(df["ai_latency_ms"].mean()), 2),
                "mean_icp_ms": round(float(df["icp_latency_ms"].mean()), 2),
                "mean_total_ms": round(float(df["total_latency_ms"].mean()), 2),
                "median_total_ms": round(float(df["total_latency_ms"].median()), 2),
                "p95_total_ms": round(float(df["total_latency_ms"].quantile(0.95)), 2),
                "equivalent_fps": round(1000.0 / float(df["total_latency_ms"].mean()), 1)
            }
        },
        "breakdown_by_distance": {},
        "breakdown_by_elevation": {}
    }

    # Breakdown by distance bands
    df["dist_bin"] = pd.cut(df["distance_m"], bins=[0.8, 0.9, 1.0, 1.1, 1.2], labels=["0.8-0.9m", "0.9-1.0m", "1.0-1.1m", "1.1-1.2m"])
    for bin_name, grp in df.groupby("dist_bin", observed=True):
        summary["breakdown_by_distance"][bin_name] = {
            "samples": len(grp),
            "bop_recall_01d": round(float(grp["pass_bop_01d"].mean() * 100.0), 1),
            "strict_recall_005d": round(float(grp["pass_strict_005d"].mean() * 100.0), 1),
            "precision_20mm_5deg": round(float(grp["pass_precision_20mm_5deg"].mean() * 100.0), 1),
            "median_et_mm": round(float(grp["e_t_mm"].median()), 2),
            "median_eR_deg": round(float(grp["e_R_sym_deg"].median()), 2),
            "median_adds_mm": round(float(grp["adds_mm"].median()), 2),
            "mean_fitness": round(float(grp["fitness"].mean() * 100.0), 1)
        }

    # Breakdown by elevation bands
    df["elev_bin"] = pd.cut(df["elevation_deg"], bins=[15, 30, 45, 65], labels=["15°-30° (Shallow)", "30°-45° (Mid-Slope)", "45°-65° (Top-Down)"])
    for bin_name, grp in df.groupby("elev_bin", observed=True):
        summary["breakdown_by_elevation"][bin_name] = {
            "samples": len(grp),
            "bop_recall_01d": round(float(grp["pass_bop_01d"].mean() * 100.0), 1),
            "strict_recall_005d": round(float(grp["pass_strict_005d"].mean() * 100.0), 1),
            "precision_20mm_5deg": round(float(grp["pass_precision_20mm_5deg"].mean() * 100.0), 1),
            "median_et_mm": round(float(grp["e_t_mm"].median()), 2),
            "median_eR_deg": round(float(grp["e_R_sym_deg"].median()), 2),
            "median_adds_mm": round(float(grp["adds_mm"].median()), 2),
            "mean_fitness": round(float(grp["fitness"].mean() * 100.0), 1)
        }

    with open(args.json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary metrics to: {args.json_out}")

    # 7. Generate High-Quality Visual Charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Subplot 1: Translation Error Histogram & CDF
    ax1 = axes[0, 0]
    ax1.hist(df["e_t_mm"], bins=40, color="#0288d1", alpha=0.7, edgecolor="#01579b", density=True, label="Histogram")
    sorted_et = np.sort(df["e_t_mm"])
    yvals = np.arange(len(sorted_et)) / float(len(sorted_et))
    ax1_twin = ax1.twinx()
    ax1_twin.plot(sorted_et, yvals, color="#d32f2f", linewidth=2.2, label="CDF (Tích lũy)")
    ax1_twin.axvline(20.0, color="#388e3c", linestyle="--", linewidth=1.5, label="Ngưỡng 20 mm")
    ax1_twin.set_ylabel("Tỉ lệ tích lũy (Cumulative Fraction)", color="#d32f2f", fontsize=10)
    ax1.set_title("Phân Bố Sai Số Tịnh Tiến $e_t$ (mm)", fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel("Sai số tịnh tiến $e_t$ (mm)", fontsize=10)
    ax1.set_ylabel("Mật độ xác suất (Density)", fontsize=10)
    ax1.set_xlim(0, min(80.0, df["e_t_mm"].quantile(0.98)))
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Rotation Error Histogram & CDF
    ax2 = axes[0, 1]
    ax2.hist(df["e_R_sym_deg"], bins=40, color="#00897b", alpha=0.7, edgecolor="#004d40", density=True)
    sorted_er = np.sort(df["e_R_sym_deg"])
    yvals_r = np.arange(len(sorted_er)) / float(len(sorted_er))
    ax2_twin = ax2.twinx()
    ax2_twin.plot(sorted_er, yvals_r, color="#e65100", linewidth=2.2, label="CDF")
    ax2_twin.axvline(5.0, color="#388e3c", linestyle="--", linewidth=1.5, label="Ngưỡng 5°")
    ax2_twin.set_ylabel("Tỉ lệ tích lũy (Cumulative Fraction)", color="#e65100", fontsize=10)
    ax2.set_title("Phân Bố Sai Số Góc Xoay Đối Xứng $e_{R,sym}$ (độ)", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Sai số góc xoay đối xứng $e_{R,sym}$ (°)", fontsize=10)
    ax2.set_ylabel("Mật độ xác suất (Density)", fontsize=10)
    ax2.set_xlim(0, min(15.0, df["e_R_sym_deg"].quantile(0.98)))
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Subplot 3: ADD-S Curve vs BOP Standards
    ax3 = axes[1, 0]
    sorted_adds = np.sort(df["adds_mm"])
    yvals_adds = np.arange(len(sorted_adds)) / float(len(sorted_adds)) * 100.0
    ax3.plot(sorted_adds, yvals_adds, color="#7b1fa2", linewidth=2.5, label="Đường cong Recall ADD-S")
    ax3.axvline(0.05 * cad_diameter * 1000.0, color="#0288d1", linestyle="--", linewidth=1.8, label=f"Strict 0.05d ({0.05*cad_diameter*1000:.1f} mm)")
    ax3.axvline(0.10 * cad_diameter * 1000.0, color="#d32f2f", linestyle="--", linewidth=1.8, label=f"BOP 0.10d ({0.10*cad_diameter*1000:.1f} mm)")
    ax3.set_title("Đường Cong BOP ADD-S Recall (@ 1000 Điểm)", fontsize=12, fontweight="bold", pad=8)
    ax3.set_xlabel("Ngưỡng ADD-S (mm)", fontsize=10)
    ax3.set_ylabel("Tỉ lệ Recall đạt chuẩn (%)", fontsize=10)
    ax3.set_xlim(0, 150)
    ax3.set_ylim(0, 105)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower right", fontsize=9)

    # Subplot 4: Error vs Distance Scatter with Moving Average
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df["distance_m"], df["e_t_mm"], c=df["adds_mm"], cmap="viridis", alpha=0.6, s=25, edgecolor="none")
    cb = plt.colorbar(scatter, ax=ax4)
    cb.set_label("ADD-S Metric (mm)", fontsize=9)
    # Moving median trend
    dist_bins = np.linspace(args.min_dist, args.max_dist, 9)
    bin_centers = 0.5 * (dist_bins[:-1] + dist_bins[1:])
    binned_median = []
    for j in range(len(dist_bins)-1):
        mask = (df["distance_m"] >= dist_bins[j]) & (df["distance_m"] < dist_bins[j+1])
        binned_median.append(df.loc[mask, "e_t_mm"].median() if np.sum(mask) > 0 else 0)
    ax4.plot(bin_centers, binned_median, color="#d32f2f", linewidth=2.5, marker="o", label="Trung vị $e_t$ theo cự ly")
    ax4.set_title("Tương Quan Sai Số Tịnh Tiến Theo Khoảng Cách", fontsize=12, fontweight="bold", pad=8)
    ax4.set_xlabel("Khoảng cách Camera - Vật thể (m)", fontsize=10)
    ax4.set_ylabel("Sai số tịnh tiến $e_t$ (mm)", fontsize=10)
    ax4.set_ylim(0, min(80.0, df["e_t_mm"].quantile(0.98)))
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left", fontsize=9)

    plt.suptitle(f"BÁO CÁO THỰC NGHIỆM MÔ PHỎNG 6DoF POSE -- 1000 ĐIỂM (CỰ LY 0.8m - 1.2m)\nMô hình: {args.cad_name} (Đường kính {cad_diameter*1000:.1f} mm) | Không che khuất (100% CAD)", fontsize=13, fontweight="bold", y=0.98)
    
    os.makedirs(os.path.dirname(args.fig_out), exist_ok=True)
    plt.savefig(args.fig_out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved evaluation chart figure to: {args.fig_out}")

if __name__ == "__main__":
    main()
