#!/usr/bin/env python3
"""
PHASE 6 -- End-to-End GeoPoseNet Benchmark on Test Split (250 samples)
Evaluates GeoPoseNet (6D Normals + Keypoint Voting + Differentiable Kabsch SVD)
and compares directly with Phase 2 (FPFH) and Phase 4 (DGCNN 3D + RANSAC).
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from deep_learning.dataset_e2e import TableE2EDataset, get_cad_keypoints, load_cad_points
from deep_learning.model_e2e import GeoPoseNet
from deep_learning.procrustes import geodesic_rotation_loss

CAD_PATH_DEFAULT    = "/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO.stl"
DATASET_DIR_DEFAULT = "/home/huyen1905/TMCAD/dataset/TABLE-DEMO"
CKPT_PATH_DEFAULT   = "/home/huyen1905/TMCAD/deep_learning/checkpoints/best_e2e_pose_model.pth"
RESULTS_DIR         = "/home/huyen1905/TMCAD/logs/phase6"
PHASE2_CSV          = "/home/huyen1905/TMCAD/benchmark_classical_test.csv"
PHASE4_CSV          = "/home/huyen1905/TMCAD/logs/phase4/phase4_test_results.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GeoPoseNet on Test Split")
    parser.add_argument("--checkpoint",  type=str, default=CKPT_PATH_DEFAULT)
    parser.add_argument("--dataset-dir",  type=str, default=DATASET_DIR_DEFAULT)
    parser.add_argument("--cad-path",     type=str, default=CAD_PATH_DEFAULT)
    parser.add_argument("--split",        type=str, default="test")
    parser.add_argument("--num-points",   type=int, default=1024)
    parser.add_argument("--dry-run",      action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 75)
    print(" PHASE 6 -- END-TO-END GEO-POSE-NET BENCHMARK")
    print(f" Checkpoint:  {args.checkpoint}")
    print(f" Device:      {device}")
    print(f" Split:       {args.split}")
    print("=" * 75)

    assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    k_neighbors = saved_args.get("k_neighbors", 20)
    feat_dim = saved_args.get("feat_dim", 128)
    cad_kp = ckpt["cad_keypoints"]

    model = GeoPoseNet(cad_keypoints=cad_kp, k_neighbors=k_neighbors, feat_dim=feat_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cad_eval_pts = torch.from_numpy(load_cad_points(args.cad_path, 500)).to(device)

    dataset = TableE2EDataset(
        dataset_dir=args.dataset_dir, split=args.split, cad_path=args.cad_path,
        num_points=args.num_points, augment=False
    )

    N = len(dataset) if not args.dry_run else 20
    rows = []
    by_vis = defaultdict(list)

    print(f"\n{'#':<5} {'Sample':<18} {'Vis':<6} {'e_t(mm)':<10} {'e_R(deg)':<10} {'ADD(mm)':<10} {'Str':<4} {'Rlx':<4} {'ms'}")
    print("-" * 80)

    total_inference_ms = 0.0

    for i in range(N):
        sample = dataset[i]
        sample_id = sample["sample_id"]

        # Read visibility
        sample_dir = os.path.join(args.dataset_dir, sample_id)
        if not os.path.isdir(sample_dir):
            sample_dir = os.path.join(args.dataset_dir, args.split, sample_id)
        with open(os.path.join(sample_dir, "pose.json")) as f:
            pdata = json.load(f)
        vis = round(pdata.get("targetVisibleFraction", 1.0), 2)

        obs_6d = sample["obs_6d"].unsqueeze(0).to(device)  # (1, N, 6)
        R_gt   = sample["R_gt"].unsqueeze(0).to(device)    # (1, 3, 3)
        t_gt   = sample["t_gt"].unsqueeze(0).to(device)    # (1, 3)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            out = model(obs_6d)
            R_pred = out["R"]
            t_pred = out["t"]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - t0) * 1000.0
        total_inference_ms += elapsed_ms

        e_t = torch.norm(t_pred - t_gt, dim=-1).item() * 1000.0  # mm

        R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
        e_R = torch.rad2deg(torch.acos(cos_theta)).item()  # deg

        # ADD
        p_pred = torch.bmm(cad_eval_pts.unsqueeze(0), R_pred.transpose(1, 2)) + t_pred.unsqueeze(1)
        p_gt   = torch.bmm(cad_eval_pts.unsqueeze(0), R_gt.transpose(1, 2))   + t_gt.unsqueeze(1)
        e_add  = torch.norm(p_pred - p_gt, dim=-1).mean().item() * 1000.0

        strict = (e_t <= 5.0) and (e_R <= 1.0)
        relaxed = (e_t <= 20.0) and (e_R <= 5.0)

        row = {
            "sample_id": sample_id,
            "vis": vis,
            "valid": True,
            "e_t": e_t,
            "e_R": e_R,
            "e_add": e_add,
            "strict_pass": strict,
            "relaxed_pass": relaxed,
            "ms": elapsed_ms
        }
        rows.append(row)
        by_vis[vis].append(row)

        if i < 25 or i % 25 == 0 or i == N - 1:
            print(f"{i+1:<5} {sample_id:<18} {vis:<6.2f} {e_t:<10.1f} {e_R:<10.1f} {e_add:<10.1f} "
                  f"{'Y' if strict else 'N':<4} {'Y' if relaxed else 'N':<4} {elapsed_ms:.1f}")

    df_results = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, f"phase6_{args.split}_results.csv")
    df_results.to_csv(csv_path, index=False)

    print("\n" + "=" * 85)
    print(f" PHASE 6 SUMMARY -- {args.split} (N={N}) | Avg Inference: {total_inference_ms/N:.2f} ms")
    print("=" * 85)

    summary_rows = []
    for vis_val in [1.00, 0.75, 0.50, 0.25]:
        v_rows = by_vis.get(vis_val, [])
        if not v_rows:
            continue
        v_df = pd.DataFrame(v_rows)
        n = len(v_df)
        strict_n = v_df["strict_pass"].sum()
        relaxed_n = v_df["relaxed_pass"].sum()
        summary_rows.append({
            "vis": vis_val,
            "N": n,
            "strict_pct": (strict_n / n) * 100.0,
            "relaxed_pct": (relaxed_n / n) * 100.0,
            "mean_et": v_df["e_t"].mean(),
            "median_et": v_df["e_t"].median(),
            "mean_er": v_df["e_R"].mean(),
            "median_er": v_df["e_R"].median(),
            "mean_add": v_df["e_add"].mean()
        })
        print(f" Vis={vis_val:.2f}  N={n:<3} Strict={strict_n}/{n} ({(strict_n/n)*100.0:4.1f}%) "
              f"Relaxed={relaxed_n}/{n} ({(relaxed_n/n)*100.0:4.1f}%) "
              f"MedEt={v_df['e_t'].median():5.1f}mm  MedEr={v_df['e_R'].median():5.1f}deg  MeanADD={v_df['e_add'].mean():5.1f}mm")

    print("-" * 85)
    tot_strict = df_results["strict_pass"].sum()
    tot_relaxed = df_results["relaxed_pass"].sum()
    print(f" TOTAL     N={N:<3} Strict={tot_strict}/{N} ({(tot_strict/N)*100.0:4.1f}%) "
          f"Relaxed={tot_relaxed}/{N} ({(tot_relaxed/N)*100.0:4.1f}%) "
          f"MedEt={df_results['e_t'].median():5.1f}mm  MedEr={df_results['e_R'].median():5.1f}deg  MeanADD={df_results['e_add'].mean():5.1f}mm")
    print("=" * 85)

    # Comparison with Phase 2 and Phase 4 if available
    p2_df = pd.read_csv(PHASE2_CSV) if os.path.exists(PHASE2_CSV) else None
    p4_df = pd.read_csv(PHASE4_CSV) if os.path.exists(PHASE4_CSV) else None

    print("\n" + "=" * 95)
    print(" 3-GENERATION COMPARISON: Phase 2 (FPFH) vs Phase 4 (DGCNN 3D) vs Phase 6 (GeoPoseNet 6D)")
    print("=" * 95)
    print(f" {'Vis':<6} | {'P2 Relax%':<10} {'P2 MedEt':<10} | {'P4 Relax%':<10} {'P4 MedEt':<10} | {'P6 Relax%':<10} {'P6 MedEt':<10} {'P6 MeanADD':<10}")
    print("-" * 95)

    for vis_val in [1.00, 0.75, 0.50, 0.25]:
        p2_r = p2_df[p2_df["target_vis"] == vis_val]["pass_relaxed"].mean() * 100.0 if p2_df is not None else 0.0
        p2_et = p2_df[p2_df["target_vis"] == vis_val]["e_t_mm"].median() if p2_df is not None else 0.0

        p4_r = p4_df[p4_df["vis"] == vis_val]["relaxed_pass"].mean() * 100.0 if p4_df is not None else 0.0
        p4_et = p4_df[p4_df["vis"] == vis_val]["e_t"].median() if p4_df is not None else 0.0

        p6_sub = df_results[df_results["vis"] == vis_val]
        p6_r = p6_sub["relaxed_pass"].mean() * 100.0
        p6_et = p6_sub["e_t"].median()
        p6_add = p6_sub["e_add"].mean()

        print(f" {vis_val:<6.2f} | {p2_r:<10.1f} {p2_et:<10.1f} | {p4_r:<10.1f} {p4_et:<10.1f} | {p6_r:<10.1f} {p6_et:<10.1f} {p6_add:<10.1f}")

    print("=" * 95)
    print(f"\n[DONE] Phase 6 evaluation completed! Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
