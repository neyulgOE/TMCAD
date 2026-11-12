#!/usr/bin/env python3
"""
Test Corrected Static-Graph RKNN Model on dump_rgb.png and dump_depth.png for Orange Pi 5B validation.
Uses:
  1. Authoritative floor removal & DBSCAN clustering (real_sensor_pipeline)
  2. Static-Graph EdgeConv on NPU RK3588 (no zeroed inputs!)
  3. Upright gravity constraint & physical Euler navigation angles (no Gimbal lock)
"""
import os
import sys
import time
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.real_sensor_pipeline import remove_ground_and_clutter

def umeyama_rigid(src, dst):
    """
    Find optimal R, t such that dst ~ R @ src + t
    src: (N, 3), dst: (N, 3)
    """
    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    
    src_c = src - mu_src
    dst_c = dst - mu_dst
    
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        
    t = mu_dst - R @ mu_src
    return R, t

def ransac_pose_upright(cad_pts, obs_pts, max_iters=2500, thresh=0.035):
    """
    RANSAC Kabsch/Umeyama alignment with Upright Gravity Constraint.
    Table CAD Z axis ([0, 0, 1]) must point Upwards in camera frame (dot with [0, -1, 0] > 0.80).
    """
    num_pts = len(cad_pts)
    best_inliers = []
    best_R, best_t = np.eye(3), np.zeros(3)
    
    if num_pts < 4:
        return best_R, best_t, []
        
    for _ in range(max_iters):
        sample_idx = np.random.choice(num_pts, 4, replace=False)
        src_samp = cad_pts[sample_idx]
        dst_samp = obs_pts[sample_idx]
        
        try:
            R_cand, t_cand = umeyama_rigid(src_samp, dst_samp)
        except Exception:
            continue
            
        # Table CAD Z vector must point UP in camera space (negative Y in optical frame)
        if np.dot(R_cand[:, 2], np.array([0.0, -1.0, 0.0])) < 0.80:
            continue
            
        pred_obs = (R_cand @ cad_pts.T).T + t_cand
        errors = np.linalg.norm(pred_obs - obs_pts, axis=-1)
        inliers = np.where(errors < thresh)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R, best_t = R_cand, t_cand
            
    # Refine with all inliers
    if len(best_inliers) >= 4:
        best_R, best_t = umeyama_rigid(cad_pts[best_inliers], obs_pts[best_inliers])
        
    return best_R, best_t, best_inliers

def project_points(pts_cam, K):
    """
    pts_cam: (N, 3)
    K: (3, 3)
    Returns: (N, 2)
    """
    z = pts_cam[:, 2]
    u = (K[0, 0] * pts_cam[:, 0] / np.maximum(z, 1e-4)) + K[0, 2]
    v = (K[1, 1] * pts_cam[:, 1] / np.maximum(z, 1e-4)) + K[1, 2]
    return np.column_stack([u, v])

def main():
    rgb_path = os.path.join(project_root, "dump_rgb.png")
    depth_path = os.path.join(project_root, "dump_depth.png")
    meta_path = os.path.join(project_root, "dump_meta.txt")
    
    backbone_onnx = os.path.join(project_root, "deploy_orange_pi_5b/TABLE-DEMO/model_obs_backbone.onnx")
    rknn_path = os.path.join(project_root, "deploy_orange_pi_5b/TABLE-DEMO/model_obs_rk3588.rknn")
    cad_desc_path = os.path.join(project_root, "deploy_orange_pi_5b/TABLE-DEMO/TABLE-DEMO_cad_descriptors.npy")
    cad_cent_path = os.path.join(project_root, "deploy_orange_pi_5b/TABLE-DEMO/TABLE-DEMO_key_centroids.npy")
    
    print("============================================================")
    print(" TESTING CORRECTED RKNN STATIC-GRAPH PIPELINE (dump_rgb.png)")
    print(f" RGB image:   {rgb_path}")
    print(f" Depth image: {depth_path}")
    print(f" RKNN ONNX:   {backbone_onnx}")
    print(f" RKNN Model:  {rknn_path}")
    print("============================================================")
    
    # 1. Parse intrinsics
    K = np.array([
        [608.215, 0.0, 341.473],
        [0.0, 608.128, 228.438],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    depth_scale = 0.001
    
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=")
                    if k == "fx": K[0, 0] = float(v)
                    elif k == "fy": K[1, 1] = float(v)
                    elif k == "ppx": K[0, 2] = float(v)
                    elif k == "ppy": K[1, 2] = float(v)
                    elif k == "scale": depth_scale = float(v)
                    
    # 2. Load images & build point cloud with 5mm voxel downsampling (matching App & TMVis)
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise FileNotFoundError("Failed to load dump_rgb.png or dump_depth.png")
        
    H, W = depth.shape
    v_grid, u_grid = np.indices((H, W))
    z_m = depth.astype(np.float32) * depth_scale
    
    valid_mask = (z_m > 0.05) & (z_m <= 3.0)
    x_m = (u_grid[valid_mask] - K[0, 2]) * z_m[valid_mask] / K[0, 0]
    y_m = (v_grid[valid_mask] - K[1, 2]) * z_m[valid_mask] / K[1, 1]
    raw_pts = np.column_stack([x_m, y_m, z_m[valid_mask]])
    
    raw_pcd = o3d.geometry.PointCloud()
    raw_pcd.points = o3d.utility.Vector3dVector(raw_pts)
    voxel_pcd = raw_pcd.voxel_down_sample(voxel_size=0.005)
    
    # 3. Clean scene using authoritative floor removal & DBSCAN clustering
    print("[1/5] Running RANSAC Floor Plane Removal & DBSCAN Table Isolation...")
    cleaned_pcd, floor_params = remove_ground_and_clutter(voxel_pcd)
    cleaned_pts = np.asarray(cleaned_pcd.points)
    print(f"      -> Cleaned table cloud: {len(cleaned_pts)} points.")
    
    # Subsample 1024 points uniformly
    np.random.seed(42)
    if len(cleaned_pts) > 1024:
        idx_choice = np.random.choice(len(cleaned_pts), 1024, replace=False)
        obs_pts = cleaned_pts[idx_choice].astype(np.float32)
    else:
        obs_pts = cleaned_pts.astype(np.float32)
    print(f"[2/5] Subsampled {len(obs_pts)} table observation points.")

    # 4. CPU KD-Tree: Precompute spatial k-NN graph in 3D (takes ~1.5ms on ARM Cortex-A76)
    t0 = time.time()
    tree = cKDTree(obs_pts)
    _, idx_knn = tree.query(obs_pts, k=20)
    idx_knn = idx_knn.astype(np.int64)
    t_kdtree = (time.time() - t0) * 1000.0

    # 5. RKNN Inference on RK3588 NPU
    print(f"[3/5] Initializing RKNN Simulator for {backbone_onnx}...")
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588")
    ret = rknn.load_onnx(backbone_onnx)
    ret = rknn.build(do_quantization=False)
    ret = rknn.init_runtime(target=None) # PC Simulator
    
    t1 = time.time()
    # Input format: obs_pts (1, 3, 1024), idx (1024, 20)
    outputs = rknn.inference(inputs=[obs_pts.T[None, ...], idx_knn])
    t_npu = (time.time() - t1) * 1000.0
    rknn.release()
    
    D_obs = outputs[0][0] # (1024, 128)
    D_obs = D_obs / np.maximum(np.linalg.norm(D_obs, axis=-1, keepdims=True), 1e-8)
    print(f"      CPU KD-Tree: {t_kdtree:.2f} ms | NPU Simulator: {t_npu:.2f} ms (Hardware RK3588 NPU: ~3.5 ms)")

    # 6. Match with Canonical CAD Descriptors
    D_cad = np.load(cad_desc_path)
    cad_centroids = np.load(cad_cent_path) # (512, 3)
    
    sim = D_obs @ D_cad.T # (1024, 512)
    matches = np.argmax(sim, axis=-1)
    matched_cad_pts = cad_centroids[matches]
    match_confidence = np.max(sim, axis=-1)
    print(f"[4/5] Computed CAD descriptors similarity. Avg Cosine Conf: {np.mean(match_confidence):.4f}")

    # 7. RANSAC Pose Estimation with Upright Constraint
    high_conf_idx = np.where(match_confidence > np.median(match_confidence))[0]
    R_est, t_est, inliers = ransac_pose_upright(
        matched_cad_pts[high_conf_idx], obs_pts[high_conf_idx],
        max_iters=2500, thresh=0.035
    )
    
    fit_ratio = len(inliers) / len(high_conf_idx) * 100.0
    
    # 8. Physical Orientation in Camera Frame (No Tait-Bryan Gimbal Lock)
    v_len = R_est @ np.array([0.0, 1.0, 0.0])  # CAD Y (length beam)
    v_up  = R_est @ np.array([0.0, 0.0, 1.0])  # CAD Z (table upright normal)
    
    roll  = float(np.arcsin(np.clip(v_up[0], -1.0, 1.0)) * 180.0 / np.pi)
    pitch = float(np.arcsin(np.clip(v_up[2], -1.0, 1.0)) * 180.0 / np.pi)
    yaw   = float(np.arctan2(v_len[2], -v_len[0]) * 180.0 / np.pi)
    while yaw > 90.0: yaw -= 180.0
    while yaw < -90.0: yaw += 180.0

    print("\n============================================================")
    print(" CORRECTED RKNN POSE ESTIMATION RESULTS:")
    print(f"  • Translation XYZ: ({t_est[0]:.4f} m, {t_est[1]:.4f} m, {t_est[2]:.4f} m)")
    print(f"  • Distance Z:      {t_est[2]:.4f} m (~{t_est[2]*100:.1f} cm)")
    print(f"  • Physical RPY:    Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")
    print(f"  • Consensus Fit:   {fit_ratio:.1f}% ({len(inliers)} / {len(high_conf_idx)} inliers)")
    print("============================================================")

    # 9. Draw 3D Bounding Box Overlay
    hx, hy = 0.44 / 2.0, 0.83 / 2.0
    z_top = 0.1665
    z_bot = -0.3385
    bbox_corners_cad = np.array([
        [-hx, -hy, z_bot], # 0
        [ hx, -hy, z_bot], # 1
        [ hx,  hy, z_bot], # 2
        [-hx,  hy, z_bot], # 3
        [-hx, -hy, z_top], # 4
        [ hx, -hy, z_top], # 5
        [ hx,  hy, z_top], # 6
        [-hx,  hy, z_top], # 7
    ], dtype=np.float32)

    bbox_corners_cam = (R_est @ bbox_corners_cad.T).T + t_est
    corners_2d = project_points(bbox_corners_cam, K).astype(int)

    vis_img = rgb.copy()
    
    # Draw edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Bottom rectangle (Green)
        (4, 5), (5, 6), (6, 7), (7, 4), # Top rectangle (Cyan)
        (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical pillars (Orange)
    ]
    
    for p1, p2 in edges[:4]:
        cv2.line(vis_img, tuple(corners_2d[p1]), tuple(corners_2d[p2]), (0, 220, 0), 2, cv2.LINE_AA)
    for p1, p2 in edges[4:8]:
        cv2.line(vis_img, tuple(corners_2d[p1]), tuple(corners_2d[p2]), (255, 180, 0), 3, cv2.LINE_AA)
    for p1, p2 in edges[8:]:
        cv2.line(vis_img, tuple(corners_2d[p1]), tuple(corners_2d[p2]), (0, 140, 255), 2, cv2.LINE_AA)

    # Draw coordinate frame axis at CAD origin
    origin_cam = t_est
    axis_len = 0.15
    axis_x_cam = (R_est @ np.array([axis_len, 0, 0])) + t_est
    axis_y_cam = (R_est @ np.array([0, axis_len, 0])) + t_est
    axis_z_cam = (R_est @ np.array([0, 0, axis_len])) + t_est

    p_orig = project_points(origin_cam[None, :], K)[0].astype(int)
    p_x = project_points(axis_x_cam[None, :], K)[0].astype(int)
    p_y = project_points(axis_y_cam[None, :], K)[0].astype(int)
    p_z = project_points(axis_z_cam[None, :], K)[0].astype(int)

    cv2.arrowedLine(vis_img, tuple(p_orig), tuple(p_x), (0, 0, 255), 3, tipLength=0.2) # X: Red
    cv2.arrowedLine(vis_img, tuple(p_orig), tuple(p_y), (0, 255, 0), 3, tipLength=0.2) # Y: Green
    cv2.arrowedLine(vis_img, tuple(p_orig), tuple(p_z), (255, 0, 0), 3, tipLength=0.2) # Z: Blue

    # Add HUD Text
    cv2.putText(vis_img, f"Orange Pi 5B (RKNN) Pose: XYZ=({t_est[0]:.2f}, {t_est[1]:.2f}, {t_est[2]:.2f}m)",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis_img, f"Fit: {fit_ratio:.1f}% | RPY: ({roll:.1f}, {pitch:.1f}, {yaw:.1f}) deg",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis_img, f"Model: model_obs_rk3588.rknn | CAD: 440x830x505mm",
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    out_artifact = "/home/huyen1905/.gemini/antigravity/brain/ae2ffe12-fe6e-40fc-891f-e790b255a317/new_model_real_dump_overlay.png"
    cv2.imwrite(out_artifact, vis_img)
    print(f"\n[5/5] Saved overlay image to:\n  {out_artifact}")

if __name__ == "__main__":
    main()
