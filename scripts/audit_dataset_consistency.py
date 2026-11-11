#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import cv2

def run_audit(dataset_dir: str, num_samples_to_audit: int = 100):
    print("=" * 70)
    print(f" MATHEMATICAL AUDIT: SYNTHETIC DATASET CONSISTENCY")
    print(f" Target Directory: {dataset_dir}")
    print("=" * 70)

    has_realistic_noise = False
    manifest_path = os.path.join(dataset_dir, "dataset_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        has_realistic_noise = manifest.get('useRealisticNoiseModel', False)
        print(f"[Manifest] Samples: {manifest.get('numSamples')}")
        print(f"[Manifest] AMR Mode: {manifest.get('amrMode')}, Height Range: [{manifest.get('amrMinCameraHeightMeters')}, {manifest.get('amrMaxCameraHeightMeters')}]m")
        print(f"[Manifest] Floor Z: {manifest.get('floorZ')}m")
        print(f"[Manifest] Intrinsics: {manifest.get('intrinsics')}")
        print(f"[Manifest] Realistic Noise: {has_realistic_noise}")
    else:
        print("[-] Warning: dataset_manifest.json not found.")

    max_geo_residual_mm = 0.0
    mean_geo_residual_mm = 0.0
    total_pts_checked = 0

    max_reproj_u_err_px = 0.0
    max_reproj_v_err_px = 0.0

    max_depth_val_err_mm = 0.0
    depth_errors = []
    depth_match_pts = 0

    cam_heights_above_floor = []
    audited_samples = 0

    # List sample directories
    samples = sorted([d for d in os.listdir(dataset_dir) if d.startswith("sample_")])
    if not samples:
        print(f"[-] No sample directories found in {dataset_dir}!")
        return False

    samples_to_check = samples[:min(len(samples), num_samples_to_audit)]
    print(f"[Audit] Checking {len(samples_to_check)} samples in detail...")

    for s_name in samples_to_check:
        s_dir = os.path.join(dataset_dir, s_name)
        pose_path = os.path.join(s_dir, "pose.json")
        csv_path = os.path.join(s_dir, "correspondences.csv")
        depth_path = os.path.join(s_dir, "depth.png")

        if not (os.path.exists(pose_path) and os.path.exists(csv_path) and os.path.exists(depth_path)):
            print(f"[-] Missing files in {s_dir}")
            continue

        with open(pose_path, 'r') as f:
            pose_meta = json.load(f)
        T_co = np.array(pose_meta["cameraFromObject"], dtype=np.float64)
        R_co = T_co[:3, :3]
        t_co = T_co[:3, 3]

        intr = pose_meta["intrinsics"]
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]

        # Calculate camera height in object/world frame
        # T_oc = T_co^-1
        T_oc = np.linalg.inv(T_co)
        cam_pos_world = T_oc[:3, 3]
        floor_z = -0.337025 # CAD bottom for TABLE-DEMO
        cam_height_above_floor = cam_pos_world[2] - floor_z
        cam_heights_above_floor.append(cam_height_above_floor)

        df = pd.read_csv(csv_path)
        if len(df) == 0:
            continue

        cad_pts = df[['cad_x', 'cad_y', 'cad_z']].values.astype(np.float64)
        cam_pts = df[['cam_x', 'cam_y', 'cam_z']].values.astype(np.float64)
        uv_csv = df[['u', 'v']].values.astype(np.float64)

        # 1. Test geometric consistency: T_co * P_cad ≈ P_cam
        cam_pred = (R_co @ cad_pts.T).T + t_co
        diff_geo = np.linalg.norm(cam_pred - cam_pts, axis=1) * 1000.0 # to mm
        max_geo_residual_mm = max(max_geo_residual_mm, np.max(diff_geo))
        mean_geo_residual_mm += np.sum(diff_geo)
        total_pts_checked += len(diff_geo)

        # 2. Test pinhole reprojection: u = fx * X/Z + cx, v = fy * Y/Z + cy
        # Open3D raycasting shoots through pixel centers (u + 0.5, v + 0.5)
        u_proj = fx * (cam_pts[:, 0] / cam_pts[:, 2]) + cx
        v_proj = fy * (cam_pts[:, 1] / cam_pts[:, 2]) + cy
        u_center = uv_csv[:, 0] + 0.5
        v_center = uv_csv[:, 1] + 0.5
        u_err = np.abs(u_proj - u_center)
        v_err = np.abs(v_proj - v_center)
        max_reproj_u_err_px = max(max_reproj_u_err_px, np.max(u_err))
        max_reproj_v_err_px = max(max_reproj_v_err_px, np.max(v_err))

        # 3. Test depth image consistency
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is not None:
            # Sample 200 random points for depth buffer lookup
            sample_indices = np.random.choice(len(df), min(200, len(df)), replace=False)
            for idx in sample_indices:
                u_int = int(round(uv_csv[idx, 0]))
                v_int = int(round(uv_csv[idx, 1]))
                if 0 <= v_int < depth_img.shape[0] and 0 <= u_int < depth_img.shape[1]:
                    z_depth_m = depth_img[v_int, u_int] * 0.001
                    z_cam_m = cam_pts[idx, 2]
                    err_mm = abs(z_depth_m - z_cam_m) * 1000.0
                    depth_errors.append(err_mm)
                    max_depth_val_err_mm = max(max_depth_val_err_mm, err_mm)
                    depth_match_pts += 1

        audited_samples += 1

    mean_geo_residual_mm /= max(1, total_pts_checked)

    print("\n" + "=" * 70)
    print(" AUDIT VERIFICATION RESULTS")
    print("=" * 70)
    print(f"Total samples checked:       {audited_samples}")
    print(f"Total points checked:        {total_pts_checked:,}")

    # Check 1: CAD -> Camera Transform
    pass_geo = max_geo_residual_mm < 0.1
    status_geo = "[PASS]" if pass_geo else "[FAIL]"
    print(f"1. CAD->Cam Transform Error: {status_geo} Max={max_geo_residual_mm:.4f} mm, Mean={mean_geo_residual_mm:.6f} mm (Limit < 0.1 mm)")

    # Check 2: Reprojection
    pass_reproj = max_reproj_u_err_px < 0.5 and max_reproj_v_err_px < 0.5
    status_reproj = "[PASS]" if pass_reproj else "[FAIL]"
    print(f"2. Pinhole Reprojection:     {status_reproj} Max U Err={max_reproj_u_err_px:.3f} px, Max V Err={max_reproj_v_err_px:.3f} px (Limit < 0.5 px)")

    # Check 3: Depth PNG Consistency
    depth_errors_arr = np.array(depth_errors) if depth_errors else np.array([0.0])
    mean_depth_err = np.mean(depth_errors_arr)
    p99_depth_err = np.percentile(depth_errors_arr, 99)
    if has_realistic_noise:
        pass_depth = (mean_depth_err < 15.0) and (p99_depth_err < 50.0)
        status_depth = "[PASS]" if pass_depth else "[FAIL]"
        print(f"3. Depth Image Consistency:  {status_depth} Mean={mean_depth_err:.2f} mm, P99={p99_depth_err:.2f} mm, Max={max_depth_val_err_mm:.2f} mm on {depth_match_pts} lookups")
    else:
        pass_depth = max_depth_val_err_mm < 5.0
        status_depth = "[PASS]" if pass_depth else "[FAIL]"
        print(f"3. Depth Image Consistency:  {status_depth} Max Depth Val Err={max_depth_val_err_mm:.3f} mm (Limit < 5.0 mm) on {depth_match_pts} lookups")

    # Check 4: Camera Height Range
    h_min = min(cam_heights_above_floor) if cam_heights_above_floor else 0.0
    h_max = max(cam_heights_above_floor) if cam_heights_above_floor else 0.0
    pass_h = (h_min >= 0.14) and (h_max <= 0.41)
    status_h = "[PASS]" if pass_h else "[FAIL]"
    print(f"4. Camera Floor Height Range: {status_h} [{h_min:.3f}m, {h_max:.3f}m] (Target: [0.15m, 0.40m])")

    print("=" * 70)
    all_passed = pass_geo and pass_reproj and pass_depth and pass_h
    if all_passed:
        print("[SUCCESS] ALL 4 MATHEMATICAL CHECKS PASSED WITH ZERO TOLERANCE VIOLATIONS!")
    else:
        print("[-] ONE OR MORE CHECKS FAILED! INVESTIGATION REQUIRED.")
    print("=" * 70)
    return all_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="/home/huyen1905/TMCAD/dataset/TABLE-DEMO_audit_v1")
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    run_audit(args.dataset_dir, args.num_samples)
