#!/usr/bin/env python3
"""
PHASE 7 -- Coarse-to-Fine ICP Refinement Benchmark (250 Test Samples)
Pipelines GeoPoseNet (Coarse Pose, ~3.7 ms) -> Point-to-Plane ICP (Fine Refinement, ~10 ms)
Demonstrating sub-millimeter / precision refinement on the full test split.
"""

import os
import sys
import time
import json
import argparse
import cv2
import numpy as np
import pandas as pd
import open3d as o3d
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from deep_learning.dataset_e2e import TableE2EDataset, get_cad_keypoints, load_cad_points
from deep_learning.model_e2e import GeoPoseNet

CAD_PATH    = "/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO.stl"
DATASET_DIR = "/home/huyen1905/TMCAD/dataset/TABLE-DEMO"
CKPT_PATH   = "/home/huyen1905/TMCAD/deep_learning/checkpoints/best_e2e_pose_model.pth"
RESULTS_DIR = "/home/huyen1905/TMCAD/logs/phase7"


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 7 Coarse-to-Fine Benchmark")
    parser.add_argument("--num-samples", type=int, default=250)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_cad_pcd_with_normals(cad_path=CAD_PATH, num_points=2500):
    mesh = o3d.io.read_triangle_mesh(cad_path)
    center = mesh.get_center()
    mesh.translate(-center)
    mesh.scale(0.001, np.zeros(3))
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(num_points)
    pcd.estimate_normals()
    return pcd


def run_point_to_plane_icp(cad_pcd, obs_pcd, T_init, max_corr_dist=0.04, max_iter=40):
    """
    Refines T_init using Point-to-Plane ICP.
    """
    reg = o3d.pipelines.registration.registration_icp(
        cad_pcd, obs_pcd, max_corr_dist, T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter, relative_fitness=1e-6, relative_rmse=1e-6)
    )
    return reg.transformation, reg.fitness, reg.inlier_rmse


def compute_pose_errors(T_pred, T_gt):
    e_t = np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3]) * 1000.0  # mm
    R_diff = T_pred[:3, :3].T @ T_gt[:3, :3]
    cos_theta = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    e_R = np.degrees(np.arccos(cos_theta))
    return e_t, e_R


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 75)
    print(" PHASE 7 -- COARSE-TO-FINE REFINEMENT (GeoPoseNet + Point-to-Plane ICP)")
    print(f" Device:         {device}")
    print(f" Target Samples: {args.num_samples if not args.dry_run else 15}")
    print("=" * 75)

    # 1. Load CAD
    cad_pcd = load_cad_pcd_with_normals(CAD_PATH, 2000)
    cad_eval_pts = np.asarray(cad_pcd.points, dtype=np.float32)

    # 2. Load GeoPoseNet model
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cad_kp = ckpt["cad_keypoints"]
    k_neighbors = ckpt.get("args", {}).get("k_neighbors", 20)
    feat_dim = ckpt.get("args", {}).get("feat_dim", 128)

    model = GeoPoseNet(cad_keypoints=cad_kp, k_neighbors=k_neighbors, feat_dim=feat_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Load dataset
    test_file = os.path.join(DATASET_DIR, "test.txt")
    with open(test_file) as f:
        samples = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    N = min(len(samples), args.num_samples) if not args.dry_run else 15
    rows = []
    by_vis = defaultdict(list)

    print(f"\n{'#':<5} {'Sample':<18} {'Vis':<6} {'e_t Coarse':<11} {'e_t ICP':<11} {'e_R Coarse':<11} {'e_R ICP':<11} {'ICP Fit':<8} {'Time'}")
    print("-" * 92)

    t_geo_total, t_icp_total = 0.0, 0.0

    for i in range(N):
        sample_id = samples[i]
        sample_dir = os.path.join(DATASET_DIR, sample_id)
        if not os.path.isdir(sample_dir):
            sample_dir = os.path.join(DATASET_DIR, "test", sample_id)

        with open(os.path.join(sample_dir, "pose.json")) as f:
            pdata = json.load(f)

        intr = pdata["intrinsics"]
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        T_gt = np.array(pdata["cameraFromObject"], dtype=np.float64)
        vis = round(pdata.get("targetVisibleFraction", 1.0), 2)

        depth = cv2.imread(os.path.join(sample_dir, "depth.png"), cv2.IMREAD_UNCHANGED)
        v_idx, u_idx = np.where((depth > 200) & (depth < 2850))

        if len(v_idx) >= 1024:
            sel = np.random.choice(len(v_idx), 1024, replace=False)
        else:
            sel = np.random.choice(len(v_idx), 1024, replace=True)

        z_m = depth[v_idx[sel], u_idx[sel]].astype(np.float32) / 1000.0
        x_m = (u_idx[sel].astype(np.float32) - cx) * z_m / fx
        y_m = (v_idx[sel].astype(np.float32) - cy) * z_m / fy
        obs_pts = np.stack([x_m, y_m, z_m], axis=1)

        # Estimate normals for GeoPoseNet
        obs_pcd_full = o3d.geometry.PointCloud()
        obs_pcd_full.points = o3d.utility.Vector3dVector(obs_pts.astype(np.float64))
        obs_pcd_full.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=20))
        obs_pcd_full.orient_normals_towards_camera_location(np.zeros(3))
        obs_normals = np.asarray(obs_pcd_full.normals, dtype=np.float32)

        obs_6d = np.concatenate([obs_pts, obs_normals], axis=1)
        obs_tensor = torch.from_numpy(obs_6d).unsqueeze(0).to(device)

        # ── Stage 1: GeoPoseNet Coarse Pose ──────────────────────────────
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            out = model(obs_tensor)
            R_coarse = out["R"][0].cpu().numpy().astype(np.float64)
            t_coarse = out["t"][0].cpu().numpy().astype(np.float64)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_geo_ms = (time.time() - t0) * 1000.0
        t_geo_total += t_geo_ms

        T_coarse = np.eye(4, dtype=np.float64)
        T_coarse[:3, :3] = R_coarse
        T_coarse[:3, 3] = t_coarse

        e_t_coarse, e_R_coarse = compute_pose_errors(T_coarse, T_gt)

        # ── Stage 2: Point-to-Plane ICP Fine Refinement ─────────────────
        t1 = time.time()
        # Voxel downsampled observation for fast ICP
        obs_pcd_icp = obs_pcd_full.voxel_down_sample(0.01)
        T_icp_raw, fitness, rmse = run_point_to_plane_icp(cad_pcd, obs_pcd_icp, T_coarse, max_corr_dist=0.05, max_iter=35)
        t_icp_ms = (time.time() - t1) * 1000.0
        t_icp_total += t_icp_ms

        # Industry standard: verification gate. If ICP fitness is good, accept ICP; otherwise keep coarse pose
        if fitness >= 0.20 and rmse <= 0.035:
            T_final = T_icp_raw
            icp_accepted = True
        else:
            T_final = T_coarse
            icp_accepted = False

        e_t_final, e_R_final = compute_pose_errors(T_final, T_gt)

        # Check passes
        strict_coarse  = (e_t_coarse <= 5.0) and (e_R_coarse <= 1.0)
        relaxed_coarse = (e_t_coarse <= 20.0) and (e_R_coarse <= 5.0)

        strict_final  = (e_t_final <= 5.0) and (e_R_final <= 1.0)
        relaxed_final = (e_t_final <= 20.0) and (e_R_final <= 5.0)

        row = {
            "sample_id": sample_id,
            "vis": vis,
            "et_coarse": e_t_coarse,
            "eR_coarse": e_R_coarse,
            "et_final": e_t_final,
            "eR_final": e_R_final,
            "icp_accepted": icp_accepted,
            "relaxed_coarse": relaxed_coarse,
            "relaxed_final": relaxed_final,
            "strict_coarse": strict_coarse,
            "strict_final": strict_final,
            "fitness": fitness,
            "rmse": rmse,
            "time_geo_ms": t_geo_ms,
            "time_icp_ms": t_icp_ms,
            "time_total_ms": t_geo_ms + t_icp_ms
        }
        rows.append(row)
        by_vis[vis].append(row)

        if i < 20 or i % 25 == 0 or i == N - 1:
            print(f"{i+1:<5} {sample_id:<18} {vis:<6.2f} {e_t_coarse:<11.1f} {e_t_final:<11.1f} {e_R_coarse:<11.1f} {e_R_final:<11.1f} {fitness:<8.3f} {'Y' if icp_accepted else 'N':<4} {t_geo_ms+t_icp_ms:.1f}ms")

    df_res = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "phase7_test_results.csv")
    df_res.to_csv(csv_path, index=False)

    print("\n" + "=" * 95)
    print(f" PHASE 7 SUMMARY (N={N}) | Avg Latency: GeoPoseNet={t_geo_total/N:.1f}ms + ICP={t_icp_total/N:.1f}ms -> Total={(t_geo_total+t_icp_total)/N:.1f}ms")
    print("=" * 95)

    for vis_val in [1.00, 0.75, 0.50, 0.25]:
        v_rows = by_vis.get(vis_val, [])
        if not v_rows:
            continue
        v_df = pd.DataFrame(v_rows)
        n = len(v_df)
        print(f" Vis={vis_val:.2f} (N={n:<3}) | "
              f"Coarse: MedEt={v_df['et_coarse'].median():5.1f}mm MedEr={v_df['eR_coarse'].median():5.1f}° | "
              f"After ICP: MedEt={v_df['et_final'].median():5.1f}mm MedEr={v_df['eR_final'].median():5.1f}° Relax%={(v_df['relaxed_final'].mean()*100.0):.1f}%")

    print("-" * 95)
    print(f" TOTAL (N={N:<3})      | "
          f"Coarse: MedEt={df_res['et_coarse'].median():5.1f}mm MedEr={df_res['eR_coarse'].median():5.1f}° | "
          f"After ICP: MedEt={df_res['et_final'].median():5.1f}mm MedEr={df_res['eR_final'].median():5.1f}° Relax%={(df_res['relaxed_final'].mean()*100.0):.1f}% Strict%={(df_res['strict_final'].mean()*100.0):.1f}%")
    print("=" * 95)
    print(f"\n[DONE] Phase 7 results saved to: {csv_path}")


if __name__ == "__main__":
    main()
