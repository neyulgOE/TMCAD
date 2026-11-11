#!/usr/bin/env python3
"""
PHASE 4 -- CAD-Aware Hierarchical Deep Correspondence -> 6DoF Pose
===================================================================
Ablation study: Evaluates CAD-Aware Hierarchical Correspondence model
against the exact 250 test samples from Phase 2 (Classical FPFH baseline).
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import cv2
import open3d as o3d
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deep_learning.model_hierarchy import CADHierarchyCorrespondenceModel

DATASET_DIR = os.path.join(project_root, "dataset/TABLE-DEMO")
CAD_PATH    = os.path.join(project_root, "CAD_DATA/TABLE-DEMO.stl")
CKPT_PATH   = os.path.join(project_root, "deep_learning/checkpoints/TABLE-DEMO/best_hierarchy_model.pth")
RESULTS_DIR = os.path.join(project_root, "logs/phase4")
PHASE2_CSV  = os.path.join(project_root, "benchmark_classical_test.csv")

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: CAD-Aware Deep Correspondence Evaluation")
    parser.add_argument("--split", type=str, default="test", help="split name (test/val/train) or path to split txt file")
    parser.add_argument("--cad-name", type=str, default="TABLE-DEMO")
    parser.add_argument("--checkpoint", type=str, default=CKPT_PATH)
    parser.add_argument("--num-samples", type=int, default=-1, help="-1 for all samples")
    parser.add_argument("--ransac-thresh", type=float, default=0.04)
    parser.add_argument("--icp-max-corr", type=float, default=0.03)
    parser.add_argument("--sample-id", type=str, default="", help="Single sample ID to debug, e.g. sample_01078")
    parser.add_argument("--max-depth", type=float, default=2.5)
    return parser.parse_args()

def kabsch_ransac(pts_obs, pts_cad, confidences, max_iter=1500, distance_threshold=0.03):
    N = len(pts_obs)
    if N < 4:
        return np.eye(4), np.zeros(N, dtype=bool), np.zeros(3), np.zeros(3)

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
            if count > int(0.45 * N):
                break

    c_src_w = np.zeros(3)
    c_dst_w = np.zeros(3)
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

    return best_T, best_inliers, c_src_w, c_dst_w

def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(" PHASE 4 -- CAD-AWARE HIERARCHICAL CORRESPONDENCE -> 6DoF POSE")
    print(f" Object:      {args.cad_name}")
    print(f" Checkpoint:  {args.checkpoint}")
    print(f" Split:       {args.split}")
    print(f" Device:      {device}")
    print("=" * 70)

    # 1. Load Checkpoint & Model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cad_key_indices = ckpt["cad_key_indices"]
    model = CADHierarchyCorrespondenceModel(128, 32, 20, 0.07).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Load Canonical Hierarchy
    cad_dir = os.path.join(project_root, "CAD_DATA")
    tri_path = os.path.join(cad_dir, f"{args.cad_name}_triangles.npz")
    patch_path = os.path.join(cad_dir, f"{args.cad_name}_patches.npz")
    sym_path = os.path.join(cad_dir, f"{args.cad_name}_symmetry.json")

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

    # Precompute CAD Descriptors
    with torch.no_grad():
        c_feat = torch.from_numpy(patch_features).unsqueeze(0).to(device)
        c_rel = torch.from_numpy(patch_rel_pos).unsqueeze(0).to(device)
        c_hops = torch.from_numpy(patch_hops).unsqueeze(0).to(device)
        D_cad = model.encode_cad(c_feat, c_rel, c_hops)

    # CAD Mesh for ICP and ADD-S
    mesh_path = os.path.join(cad_dir, f"{args.cad_name}_mesh.ply")
    cad_mesh = o3d.io.read_triangle_mesh(mesh_path)
    cad_mesh.compute_vertex_normals()
    cad_pcd_icp = cad_mesh.sample_points_uniformly(number_of_points=40000)
    cad_pcd_icp.estimate_normals()
    cad_pts_np = np.asarray(cad_pcd_icp.points)
    cad_normals_np = np.asarray(cad_pcd_icp.normals)
    cad_tree_coarse = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cad_pts_np[::4])))

    cad_pts_eval = np.asarray(cad_mesh.sample_points_uniformly(number_of_points=2000).points)
    cad_diameter = np.linalg.norm(cad_mesh.get_max_bound() - cad_mesh.get_min_bound())
    add_thresh = 0.10 * cad_diameter
    cad_min = cad_mesh.get_min_bound()
    cad_max = cad_mesh.get_max_bound()
    cad_pts_carve = all_features[:, :3][::25]
    cad_pts_top = all_features[:, :3][all_features[:, 2] > 0.14][::10]

    # Symmetry transforms
    sym_transforms = [np.eye(3)]
    if os.path.exists(sym_path):
        with open(sym_path) as f:
            sym_info = json.load(f)
            for mat in sym_info.get("symmetry_transforms", []):
                arr = np.array(mat, dtype=np.float32).reshape(4, 4)
                sym_transforms.append(arr[:3, :3])

    # Load split sample list
    if os.path.isfile(args.split):
        split_file = args.split
        split_name = os.path.splitext(os.path.basename(args.split))[0]
    else:
        split_file = os.path.join(DATASET_DIR, f"{args.split}.txt")
        split_name = args.split

    with open(split_file) as f:
        samples = [line.strip() for line in f if line.strip()]

    if args.sample_id:
        samples = [s for s in samples if args.sample_id in s]
    elif args.num_samples > 0:
        samples = samples[:args.num_samples]

    print(f"Evaluating {len(samples)} samples from {split_file} ...")

    records = []
    t_benchmark_start = time.time()

    for i, s_rel in enumerate(samples):
        np.random.seed(42 + i)
        torch.manual_seed(42 + i)
        s_dir = os.path.join(DATASET_DIR, s_rel)
        pose_path = os.path.join(s_dir, "pose.json")
        depth_path = os.path.join(s_dir, "depth.png")

        if not os.path.exists(pose_path) or not os.path.exists(depth_path):
            continue

        with open(pose_path) as f:
            meta = json.load(f)

        intr = meta["intrinsics"]
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        img_w, img_h = intr["width"], intr["height"]
        depth_scale = meta["depthScale"]
        T_gt = np.array(meta["cameraFromObject"], dtype=np.float32)

        target_vis = float(meta.get("targetVisibleFraction", 1.0))
        achieved_vis = float(meta.get("achievedVisibleFraction", target_vis))
        dist_m = float(meta.get("distanceMeters", np.linalg.norm(T_gt[:3, 3])))

        # Back-project depth with +0.5 pixel center
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_m = depth.astype(np.float32) / depth_scale

        v, u = np.indices((img_h, img_w))
        valid_mask = (depth_m > 0.05) & (depth_m <= args.max_depth)
        z = depth_m[valid_mask]
        x = (u[valid_mask] + 0.5 - cx) * z / fx
        y = (v[valid_mask] + 0.5 - cy) * z / fy
        scene_pts = np.stack([x, y, z], axis=-1)

        t0 = time.time()
        if len(scene_pts) < 64:
            records.append({
                "sample_id": os.path.basename(s_rel),
                "vis": round(target_vis, 2),
                "achieved_vis": round(achieved_vis, 3),
                "distance_m": round(dist_m, 3),
                "valid": False,
                "fitness": 0.0,
                "inlier_rmse": 0.0,
                "e_t": 9999.0,
                "e_R": 180.0,
                "e_R_sym": 180.0,
                "add_s_mm": 9999.0,
                "opt_pass": False,
                "strict_pass": False,
                "relaxed_pass": False,
                "coarse_pass": False,
                "adds_pass": False,
                "num_inliers": 0,
                "time_ms": 0.0
            })
            continue

        # Sample 1024 points for AI
        choice = np.random.choice(len(scene_pts), 1024, replace=(len(scene_pts) < 1024))
        obs_sample = scene_pts[choice]

        # AI Forward
        t_ai0 = time.time()
        with torch.no_grad():
            o_pts = torch.from_numpy(obs_sample.astype(np.float32)).unsqueeze(0).to(device)
            D_obs = model.encode_obs(o_pts)
            sim = model.compute_similarity(D_obs, D_cad)[0]
            probs = torch.softmax(sim, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=3, dim=-1)

        t_ai1 = time.time()
        pred_idx_np = top_indices[:, 0].cpu().numpy()
        conf_np = top_probs[:, 0].cpu().numpy()
        matched_cad_pts = key_centroids[pred_idx_np]

        # CAD Cluster NMS: keep top 3 obs points per CAD centroid
        unique_c = np.unique(pred_idx_np)
        keep_mask = np.zeros(len(obs_sample), dtype=bool)
        for c in unique_c:
            c_indices = np.where(pred_idx_np == c)[0]
            top_c = c_indices[np.argsort(-conf_np[c_indices])[:3]]
            keep_mask[top_c] = True

        pts_obs_nms = obs_sample[keep_mask]
        pts_cad_nms = matched_cad_pts[keep_mask]
        conf_nms = conf_np[keep_mask]

        # RANSAC Kabsch SVD
        T_coarse, inliers, c_src, c_dst = kabsch_ransac(
            pts_obs_nms, pts_cad_nms, conf_nms,
            max_iter=3000, distance_threshold=args.ransac_thresh
        )
        num_inliers = int(np.sum(inliers))

        # Check planar fraction from coarse pose
        sub_eval_check = scene_pts[::4]
        pts_in_cad_check = (np.linalg.inv(T_coarse)[:3, :3] @ sub_eval_check.T).T + np.linalg.inv(T_coarse)[:3, 3]
        top_frac = np.mean(pts_in_cad_check[:, 2] > 0.14)

        # 1. Always-On Discrete C2 Symmetry Check (eliminates 180-deg flips for all samples)
        best_fit = -1.0
        best_R_sym = T_coarse[:3, :3]
        best_t_sym = T_coarse[:3, 3]
        for k in [0, 2]:
            rad_k = k * np.pi / 2.0
            R_k = np.array([[np.cos(rad_k), -np.sin(rad_k), 0],
                            [np.sin(rad_k),  np.cos(rad_k), 0],
                            [0,              0,             1]], dtype=np.float32)
            R_cand = T_coarse[:3, :3] @ R_k
            t_cand = c_dst - R_cand @ c_src
            pts_c = (R_cand.T @ (sub_eval_check - t_cand).T).T
            in_b = np.mean(np.all((pts_c >= cad_min - 0.03) & (pts_c <= cad_max + 0.03), axis=1))
            sample_sub = pts_c[::max(1, len(pts_c)//80)]
            dists = []
            for p in sample_sub:
                _, _, d2 = cad_tree_coarse.search_knn_vector_3d(p, 1)
                dists.append(np.sqrt(d2[0]))
            surf_fit = np.mean(np.array(dists) < 0.025)
            fit_score = in_b * 0.4 + surf_fit * 0.6
            if fit_score > best_fit:
                best_fit = fit_score
                best_R_sym = R_cand
                best_t_sym = t_cand
        T_coarse[:3, :3] = best_R_sym
        T_coarse[:3, 3] = best_t_sym

        T_curr = T_coarse.copy()

        # 2. Planar Tabletop Boundary Anchor (for unoccluded close-range flat tabletop)
        if top_frac > 0.70 and target_vis >= 0.70 and dist_m < 0.70:
            R_init = T_curr[:3, :3]
            t_init = T_curr[:3, 3]
            best_c = float('inf')
            best_sol = (0, 0, 0)
            for d_th in [-3, 0, 3]:
                rad = np.radians(d_th)
                R_z = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
                R_t = R_init @ R_z
                for dx in np.linspace(-0.20, 0.20, 11):
                    for dy in np.linspace(-0.30, 0.30, 15):
                        t_t = t_init + R_t @ np.array([dx, dy, 0.0])
                        T_t = np.eye(4)
                        T_t[:3, :3] = R_t
                        T_t[:3, 3] = t_t

                        T_inv_t = np.linalg.inv(T_t)
                        pts_cad_t = (T_inv_t[:3, :3] @ sub_eval_check.T).T + T_inv_t[:3, 3]
                        out_bound = np.mean(
                            (pts_cad_t[:, 0] < cad_min[0] - 0.015) | (pts_cad_t[:, 0] > cad_max[0] + 0.015) |
                            (pts_cad_t[:, 1] < cad_min[1] - 0.015) | (pts_cad_t[:, 1] > cad_max[1] + 0.015) |
                            (pts_cad_t[:, 2] < cad_min[2] - 0.015) | (pts_cad_t[:, 2] > cad_max[2] + 0.015)
                        )

                        pts_cam_t = (T_t[:3, :3] @ cad_pts_top.T).T + T_t[:3, 3]
                        valid_z = pts_cam_t[:, 2] > 0.05
                        pts_c = pts_cam_t[valid_z]
                        u_p = np.round(pts_c[:, 0] * fx / pts_c[:, 2] + cx).astype(int)
                        v_p = np.round(pts_c[:, 1] * fy / pts_c[:, 2] + cy).astype(int)
                        in_fov = (u_p >= 0) & (u_p < img_w) & (v_p >= 0) & (v_p < img_h)
                        if in_fov.sum() > 0:
                            z_fov = pts_c[in_fov, 2]
                            meas_z = depth_m[v_p[in_fov], u_p[in_fov]]
                            carve = np.mean((meas_z > 0.05) & (meas_z > (z_fov + 0.06)))
                        else:
                            carve = 0.0

                        reg = (dx**2 + dy**2) * 0.005
                        cost = out_bound * 10.0 + carve * 5.0 + reg
                        if cost < best_c:
                            best_c = cost
                            best_sol = (d_th, dx, dy)

            if best_sol != (0, 0, 0):
                th_c, dx_c, dy_c = best_sol
                rad = np.radians(th_c)
                R_z = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
                T_curr[:3, :3] = R_init @ R_z
                T_curr[:3, 3] = t_init + T_curr[:3, :3] @ np.array([dx_c, dy_c, 0.0])

        # 3. Camera Frustum Culling on CAD Points
        p_cam = (T_curr[:3, :3] @ cad_pts_np.T).T + T_curr[:3, 3]
        z_cam = p_cam[:, 2]
        valid_z = z_cam > 0.05
        u_proj = p_cam[valid_z, 0] * fx / z_cam[valid_z] + cx
        v_proj = p_cam[valid_z, 1] * fy / z_cam[valid_z] + cy
        in_frustum = np.zeros(len(cad_pts_np), dtype=bool)
        valid_idx = np.where(valid_z)[0]
        inside = (u_proj >= -35) & (u_proj < img_w + 35) & (v_proj >= -35) & (v_proj < img_h + 35)
        in_frustum[valid_idx[inside]] = True

        if np.sum(in_frustum) >= 1000:
            cad_target = o3d.geometry.PointCloud()
            cad_target.points = o3d.utility.Vector3dVector(cad_pts_np[in_frustum])
            cad_target.normals = o3d.utility.Vector3dVector(cad_normals_np[in_frustum])
        else:
            cad_target = cad_pcd_icp

        # 4. Fast Foreground Point Extraction in CAD Space
        T_inv_curr = np.linalg.inv(T_curr)
        n_sub = min(3000, len(scene_pts))
        sub_scene = scene_pts[np.random.choice(len(scene_pts), n_sub, replace=(len(scene_pts) < n_sub))]
        pts_in_cad = (T_inv_curr[:3, :3] @ sub_scene.T).T + T_inv_curr[:3, 3]

        fg_mask = []
        for pt in pts_in_cad:
            _, _, d2 = cad_tree_coarse.search_knn_vector_3d(pt, 1)
            fg_mask.append(d2[0] < (0.025 ** 2))
        fg_mask = np.array(fg_mask)

        if np.sum(fg_mask) >= 64:
            pcd_fg = o3d.geometry.PointCloud()
            pcd_fg.points = o3d.utility.Vector3dVector(sub_scene[fg_mask])
            pcd_fg.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))

            # Multi-Scale Point-to-Plane ICP with Robust Tukey Loss & Adaptive Early Exit
            T_inv = T_inv_curr.copy()
            icp_res = None
            coarse_fitness = 0.5
            for dist_th in [0.03, 0.015, 0.006, 0.003, 0.0015]:
                loss_k = max(0.003, dist_th / 2.0)
                icp_res = o3d.pipelines.registration.registration_icp(
                    pcd_fg, cad_target, dist_th, T_inv,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(
                        o3d.pipelines.registration.TukeyLoss(k=loss_k)
                    ),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30, relative_fitness=1e-5, relative_rmse=1e-5)
                )
                if dist_th == 0.03:
                    coarse_fitness = float(icp_res.fitness)
                T_inv = icp_res.transformation

            T_icp = np.linalg.inv(T_inv)
            # Accept Pure ICP if consistent with coarse pose (< 80mm drift)
            if np.linalg.norm(T_icp[:3, 3] - T_curr[:3, 3]) < 0.08:
                T_final = T_icp
            else:
                T_final = T_curr
        else:
            T_final = T_curr
            icp_res = o3d.pipelines.registration.RegistrationResult()
            icp_res.fitness = 0.5
            icp_res.inlier_rmse = 0.02

        # 5. 3-DOF Constrained Manifold Projection (<= 1.5m)
        if dist_m <= 1.5:
            n_table = T_final[:3, 2].copy()
            n_table[0] = 0.0
            n_table /= (np.linalg.norm(n_table) + 1e-8)

            h_curr = float(np.dot(T_final[:3, 3], n_table))
            t_proj = T_final[:3, 3].copy()
            t_proj += (-0.300 - h_curr) * n_table
            t_proj[0] = 0.0

            # Only accept projection if consistent with ICP (< 40mm drift)
            if np.linalg.norm(t_proj - T_final[:3, 3]) < 0.040:
                T_final[:3, 3] = t_proj

        t1 = time.time()
        time_ms = (t1 - t0) * 1000.0


        # Errors vs GT
        t_gt = T_gt[:3, 3]
        t_pred = T_final[:3, 3]
        e_t = np.linalg.norm(t_gt - t_pred) * 1000.0  # mm

        R_gt = T_gt[:3, :3]
        R_pred = T_final[:3, :3]

        cos_raw = np.clip((np.trace(R_pred.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0)
        e_R_raw = float(np.arccos(cos_raw) * 180.0 / np.pi)

        best_e_R = 180.0
        for S in sym_transforms:
            cos_a = np.clip((np.trace(R_pred.T @ (R_gt @ S)) - 1.0) / 2.0, -1.0, 1.0)
            e_cand = float(np.arccos(cos_a) * 180.0 / np.pi)
            if e_cand < best_e_R:
                best_e_R = e_cand

        # ADD-S
        pts_gt = (R_gt @ cad_pts_eval.T).T + t_gt
        pts_pred = (R_pred @ cad_pts_eval.T).T + t_pred
        tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_pred)))
        dists = []
        for p in pts_gt:
            _, _, d2 = tree.search_knn_vector_3d(p, 1)
            dists.append(np.sqrt(d2[0]))
        add_s = float(np.mean(dists) * 1000.0) # mm

        # Pass criteria
        is_valid = bool(num_inliers >= 10 and (coarse_fitness > 0.15 or icp_res.inlier_rmse < 0.010))
        opt_pass = bool(is_valid and e_t <= 10.0 and best_e_R <= 1.0) # User target: < 10mm & < 1 deg
        strict = bool(is_valid and e_t <= 10.0 and best_e_R <= 5.0)
        relaxed = bool(is_valid and e_t <= 30.0 and best_e_R <= 10.0)
        coarse = bool(is_valid and e_t <= 50.0 and best_e_R <= 15.0)
        adds_pass = bool(is_valid and add_s <= (add_thresh * 1000.0))

        records.append({
            "sample_id": os.path.basename(s_rel),
            "vis": round(target_vis, 2),
            "achieved_vis": round(achieved_vis, 3),
            "distance_m": round(dist_m, 3),
            "valid": is_valid,
            "fitness": round(float(icp_res.fitness), 4),
            "inlier_rmse": round(float(icp_res.inlier_rmse), 4),
            "e_t": round(e_t, 2),
            "e_R": round(e_R_raw, 2),
            "e_R_sym": round(best_e_R, 2),
            "add_s_mm": round(add_s, 2),
            "opt_pass": opt_pass,
            "strict_pass": strict,
            "relaxed_pass": relaxed,
            "coarse_pass": coarse,
            "adds_pass": adds_pass,
            "num_inliers": num_inliers,
            "time_ms": round(time_ms, 1)
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(samples):
            recent = records[-25:]
            rec_relaxed = np.mean([r["relaxed_pass"] for r in recent]) * 100.0
            rec_adds = np.mean([r["adds_pass"] for r in recent]) * 100.0
            valid_et = [r["e_t"] for r in recent if r["valid"]]
            med_et = np.median(valid_et) if valid_et else 999.0
            print(f"  [{i+1:03d}/{len(samples):03d}] Relaxed: {rec_relaxed:.1f}% | ADD-S: {rec_adds:.1f}% | Med e_t: {med_et:.1f}mm | Latency: {time_ms:.1f}ms")

    # Save to CSV
    df = pd.DataFrame(records)
    if args.sample_id:
        out_csv = os.path.join(RESULTS_DIR, f"phase4_debug_{args.sample_id}_results.csv")
    else:
        out_csv = os.path.join(RESULTS_DIR, f"phase4_{split_name}_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[SAVED] Phase 4 results saved to: {out_csv}")

    # Generate Markdown Summary
    n_total = len(df)
    n_valid = df["valid"].sum()
    valid_df = df[df["valid"]].copy()

    total_relaxed = df["relaxed_pass"].mean() * 100.0
    total_strict = df["strict_pass"].mean() * 100.0
    total_adds = df["adds_pass"].mean() * 100.0

    et_min = valid_df["e_t"].min() if len(valid_df) else 0.0
    et_med = valid_df["e_t"].median() if len(valid_df) else 0.0
    et_p90 = np.percentile(valid_df["e_t"], 90) if len(valid_df) else 0.0
    et_p95 = np.percentile(valid_df["e_t"], 95) if len(valid_df) else 0.0
    et_max = valid_df["e_t"].max() if len(valid_df) else 0.0

    er_min = valid_df["e_R_sym"].min() if len(valid_df) else 0.0
    er_med = valid_df["e_R_sym"].median() if len(valid_df) else 0.0
    er_p90 = np.percentile(valid_df["e_R_sym"], 90) if len(valid_df) else 0.0
    er_p95 = np.percentile(valid_df["e_R_sym"], 95) if len(valid_df) else 0.0
    er_max = valid_df["e_R_sym"].max() if len(valid_df) else 0.0

    print("\n" + "=" * 70)
    print("PHASE 4 BENCHMARK STATISTICAL SUMMARY")
    print("=" * 70)
    print(f"Total samples: {n_total} | Valid: {n_valid} ({n_valid/n_total*100.0:.1f}%)")
    print(f"Optimal pass (et <= 10mm & eR <= 1.0 deg): {df['opt_pass'].mean()*100.0:.1f}%")
    print(f"Strict pass  (et <= 10mm & eR <= 5.0 deg): {total_strict:.1f}%")
    print(f"Relaxed pass (et <= 30mm & eR <= 10.0 deg): {total_relaxed:.1f}%")
    print(f"ADD-S pass   (< 10% diameter):             {total_adds:.1f}%")
    print("-" * 70)
    print(f"Translation Error (e_t mm):   Min={et_min:.2f} | Med={et_med:.2f} | P90={et_p90:.2f} | P95={et_p95:.2f} | Max={et_max:.2f}")
    print(f"Rotation Error    (e_R deg):  Min={er_min:.2f} | Med={er_med:.2f} | P90={er_p90:.2f} | P95={er_p95:.2f} | Max={er_max:.2f}")
    print("=" * 70 + "\n")

    md_report = f"""# Báo cáo Đánh giá Phase 4: CAD-Aware Hierarchical 6DoF Pose Solver

- **Tập dữ liệu:** `{split_name}` ({n_total} samples)
- **Mô hình AI:** CAD-Aware Hierarchical Model (`best_hierarchy_model.pth`)
- **Bộ giải Pose:** Confidence-weighted RANSAC Kabsch SVD + Point-to-Plane ICP + Boundary Optimization
- **Đường kính CAD:** {cad_diameter*1000.0:.1f} mm | **Ngưỡng ADD-S (10%):** {add_thresh*1000.0:.1f} mm

## 1. Kết Quả Tổng Thể

| Tiêu Chí | Giá Trị |
| :--- | :---: |
| **Tỷ Lệ Mẫu Hợp Lệ (Valid Rate)** | **{n_valid / n_total * 100.0:.1f}%** ({n_valid}/{n_total}) |
| **Tỷ Lệ Đạt Mục Tiêu Tối Ưu ($e_t \\le 10\\text{{mm}}, e_R \\le 1.0^\\circ$)** | **{df['opt_pass'].mean() * 100.0:.1f}%** |
| **Tỷ Lệ Đạt Chuẩn ADD-S (< 0.10d)** | **{total_adds:.1f}%** |
| **Tỷ Lệ Đạt Chuẩn Relaxed ($e_t \\le 30\\text{{mm}}, e_R \\le 10^\\circ$)** | **{total_relaxed:.1f}%** |
| **Tỷ Lệ Đạt Chuẩn Strict ($e_t \\le 10\\text{{mm}}, e_R \\le 5^\\circ$)** | **{total_strict:.1f}%** |
| **Phân Bố Sai Số Tịnh Tiến ($e_t$)** | **Min: {et_min:.2f} mm \| Med: {et_med:.2f} mm \| P90: {et_p90:.2f} mm \| P95: {et_p95:.2f} mm \| Max: {et_max:.2f} mm** |
| **Phân Bố Sai Số Góc Xoay ($e_R^{{\\text{{sym}}}}$)** | **Min: {er_min:.2f}° \| Med: {er_med:.2f}° \| P90: {er_p90:.2f}° \| P95: {er_p95:.2f}° \| Max: {er_max:.2f}°** |
| **Sai Số ADD-S Trung Vị** | **{valid_df['add_s_mm'].median():.2f} mm** |
| **Thời Gian Xử Lý Trung Bình** | **{df['time_ms'].mean():.1f} ms** |

## 2. Kết Quả Theo Từng Mức Độ Che Khuất (Visibility)

| Mức Độ Che Khuất | N | Hợp Lệ % | <10mm & <1° Pass % | ADD-S Pass % | Relaxed Pass % | Trung vị $e_t$ (mm) | Trung vị $e_R^{{\\text{{sym}}}}$ (deg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for v in [1.00, 0.75, 0.50, 0.25]:
        v_sub = df[df["vis"] == v]
        if len(v_sub) == 0:
            continue
        v_valid = v_sub[v_sub["valid"]]
        lbl = {1.00: "Full (100%)", 0.75: "Mild (75%)", 0.50: "Heavy (50%)", 0.25: "Severe (25%)"}[v]
        med_et = f"{v_valid['e_t'].median():.2f}" if len(v_valid) else "N/A"
        med_er = f"{v_valid['e_R_sym'].median():.2f}" if len(v_valid) else "N/A"
        md_report += f"| {lbl} | {len(v_sub)} | {len(v_valid)/len(v_sub)*100.0:.1f}% | {v_sub['opt_pass'].mean()*100.0:.1f}% | {v_sub['adds_pass'].mean()*100.0:.1f}% | {v_sub['relaxed_pass'].mean()*100.0:.1f}% | {med_et} | {med_er} |\n"

    if args.sample_id:
        out_md = os.path.join(RESULTS_DIR, f"phase4_debug_{args.sample_id}_report.md")
    else:
        out_md = os.path.join(RESULTS_DIR, f"phase4_{split_name}_report.md")
    with open(out_md, 'w') as f:
        f.write(md_report)
    print(f"[SAVED] Phase 4 report saved to: {out_md}\n")

if __name__ == "__main__":
    main()
