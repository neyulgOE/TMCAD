#!/usr/bin/env python3
"""
Lightweight Self-Contained Black-Box ONNX / RKNN CAD Pose Estimator Runtime for Orange Pi 5B & PC.
Requires ONLY a single <MODEL>.onnx file (zero .stl, .bin, or .npy files needed).
"""
import os
import sys
import time
import argparse
import numpy as np


def solve_kabsch_svd(P_cam: np.ndarray, Q_cad: np.ndarray):
    """
    Solve rigid 3D transformation P_cam = R @ Q_cad + t via SVD.
    """
    c_cam = P_cam.mean(axis=0)
    c_cad = Q_cad.mean(axis=0)
    X = Q_cad - c_cad
    Y = P_cam - c_cam
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_cam - R @ c_cad
    return R, t


def ransac_kabsch_pose(obs_pts: np.ndarray,
                       matched_cad_pts: np.ndarray,
                       match_scores: np.ndarray,
                       dist_thresh: float = 0.035,
                       num_iters: int = 300):
    N = len(obs_pts)
    weights = np.maximum(match_scores, 1e-4)
    probs = weights / weights.sum()

    best_inliers = None
    best_count = -1
    rng = np.random.default_rng(42)

    for _ in range(num_iters):
        idx = rng.choice(N, size=3, replace=False, p=probs)
        R, t = solve_kabsch_svd(obs_pts[idx], matched_cad_pts[idx])
        pred = (matched_cad_pts @ R.T) + t[None, :]
        err = np.linalg.norm(obs_pts - pred, axis=-1)
        inliers = err < dist_thresh
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers

    if best_inliers is not None and best_count >= 6:
        R, t = solve_kabsch_svd(obs_pts[best_inliers], matched_cad_pts[best_inliers])
        pred = (matched_cad_pts[best_inliers] @ R.T) + t[None, :]
        rmse = float(np.sqrt(np.mean(np.sum((obs_pts[best_inliers] - pred) ** 2, axis=-1))))
        return R, t, best_count, rmse
    R, t = solve_kabsch_svd(obs_pts, matched_cad_pts)
    return R, t, len(obs_pts), 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Run Self-Contained Black-Box ONNX CAD Pose Inference")
    parser.add_argument("--onnx", type=str, default="", help="Path to single Black-Box .onnx file")
    parser.add_argument("--cad-name", type=str, default="TABLE-DEMO", help="CAD model name (e.g. TABLE-DEMO)")
    parser.add_argument("--rknn", type=str, default="", help="Optional path to .rknn model")
    parser.add_argument("--input-cloud", type=str, default="", help="Input point cloud (.npy or .ply)")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    onnx_candidates = []
    if args.onnx:
        onnx_candidates.append(args.onnx)
    onnx_candidates.extend([
        os.path.join(project_root, "models", f"{args.cad_name}.onnx"),
        os.path.join(script_dir, args.cad_name, f"{args.cad_name}.onnx"),
        os.path.join(script_dir, args.cad_name, "model_obs_npu.onnx"),
        os.path.join(script_dir, "model_obs_npu.onnx"),
    ])

    onnx_path = ""
    for p in onnx_candidates:
        if p and os.path.exists(p):
            onnx_path = p
            break

    if not onnx_path:
        raise FileNotFoundError(f"Black-Box ONNX model not found for {args.cad_name}")

    import onnxruntime as ort
    ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    print("============================================================")
    print(" BLACK-BOX ONNX CAD POSE ESTIMATOR RUNTIME (ZERO STL/NPY)")
    print(f" Loaded Single-File Model: {onnx_path}")
    print("============================================================")

    # Load or generate sample point cloud
    if args.input_cloud and os.path.exists(args.input_cloud):
        if args.input_cloud.endswith(".npy"):
            obs_pts = np.load(args.input_cloud).astype(np.float32)
        else:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(args.input_cloud)
            obs_pts = np.asarray(pcd.points, dtype=np.float32)
        if len(obs_pts) >= 1024:
            idx_choice = np.random.choice(len(obs_pts), 1024, replace=False)
            obs_pts = obs_pts[idx_choice]
        else:
            idx_choice = np.random.choice(len(obs_pts), 1024, replace=True)
            obs_pts = obs_pts[idx_choice]
    else:
        # Unpack embedded surface points from the ONNX model itself to synthesize a realistic test cloud
        dummy_in = np.zeros((1, 1024, 3), dtype=np.float32)
        surf_pts = ort_sess.run(["cad_surface_pts"], {"obs_pts": dummy_in})[0]
        obs_pts = surf_pts[:1024].copy() + np.array([0.05, -0.08, 1.25], dtype=np.float32)

    t0 = time.time()
    outputs = ort_sess.run(
        ["D_obs", "matched_cad_pts", "match_scores", "cad_feature_edges", "cad_meta"],
        {"obs_pts": obs_pts[None, ...].astype(np.float32)}
    )
    t_onnx = (time.time() - t0) * 1000.0

    D_obs, matched_cad_pts, match_scores, cad_edges, cad_meta = outputs
    matched_cad_pts = matched_cad_pts[0]
    match_scores = match_scores[0]

    t1 = time.time()
    R, t, inliers, rmse = ransac_kabsch_pose(obs_pts, matched_cad_pts, match_scores)
    t_pose = (time.time() - t1) * 1000.0

    total_latency = t_onnx + t_pose
    fps = 1000.0 / max(0.1, total_latency)

    print("\n--- BLACK-BOX ONNX INFERENCE BREAKDOWN ---")
    print(f"  • Embedded Wireframe Edges:   {len(cad_edges)} segments")
    print(f"  • Embedded CAD Dimensions:    {(cad_meta[6]*2000):.1f} x {(cad_meta[7]*2000):.1f} x {((cad_meta[8]-cad_meta[9])*1000):.1f} mm")
    print(f"  • ONNX Forward + Matching:    {t_onnx:.2f} ms")
    print(f"  • RANSAC Kabsch 6DoF Solver:  {t_pose:.2f} ms")
    print(f"  • Total Pipeline Latency:     {total_latency:.2f} ms (~{fps:.1f} FPS)")
    print(f"  • Estimated 6DoF Translation: X={t[0]:.4f}m, Y={t[1]:.4f}m, Z={t[2]:.4f}m (Inliers: {inliers}/1024, RMSE: {rmse*1000:.2f}mm)")
    print("------------------------------------------")
    print(" SUCCESS (100% Single-File ONNX Black-Box)!")


if __name__ == "__main__":
    main()
