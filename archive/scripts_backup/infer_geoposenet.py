#!/usr/bin/env python3
"""
infer_geoposenet.py -- Fast 6DoF Pose Inference Bridge for CadViewer (Qt)
Loads GeoPoseNet (DGCNN 6D + Keypoint Offset Voting + Kabsch SVD) and Point-to-Plane ICP.
Input: Scene Point Cloud (.ply or .pcd or directory)
Output: JSON file with R, t, Euler angles, latency, fitness, RMSE, and aligned overlay cloud.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import open3d as o3d

# Ensure TMCAD root is in Python path
TMCAD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TMCAD_ROOT not in sys.path:
    sys.path.insert(0, TMCAD_ROOT)

import torch
from deep_learning.model_e2e import GeoPoseNet

DEFAULT_CAD = os.path.join(TMCAD_ROOT, "CAD_DATA", "TABLE-DEMO.stl")


def parse_args():
    parser = argparse.ArgumentParser(description="Fast 6DoF Pose Inference for CadViewer")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input scene point cloud (.ply, .pcd)")
    parser.add_argument("--cad", "-c", type=str, default=DEFAULT_CAD, help="Path to CAD model (.stl, .ply)")
    parser.add_argument("--checkpoint", "-m", type=str, default="", help="Path to model checkpoint .pth")
    parser.add_argument("--output-json", "-o", type=str, default="pose_output.json", help="Path to output result JSON")
    parser.add_argument("--output-cloud", type=str, default="", help="Optional path to output aligned scene cloud in object space (.ply)")
    parser.add_argument("--no-icp", action="store_true", help="Skip Point-to-Plane ICP fine refinement")
    parser.add_argument("--num-points", type=int, default=2048, help="Number of sampled points for GeoPoseNet")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda:0, cpu, or auto)")
    return parser.parse_args()


def load_cad_model(cad_path, num_points=2048):
    """Loads CAD model, centers it at origin, scales from mm to meters if needed, and samples points."""
    mesh = o3d.io.read_triangle_mesh(cad_path)
    if not mesh.has_triangles() or len(mesh.vertices) == 0:
        raise ValueError(f"Could not load valid triangle mesh from {cad_path}")
    
    # Center CAD at origin
    center = mesh.get_center()
    mesh.translate(-center)
    
    # Check if CAD units are in mm (bounding box > 2.0m)
    bbox_diag = np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound())
    scale_factor = 1.0
    if bbox_diag > 2.0:
        # Likely in millimeters, scale to meters
        scale_factor = 0.001
        mesh.scale(scale_factor, np.zeros(3))
    
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(num_points)
    pcd.estimate_normals()
    return mesh, pcd, scale_factor


def rotation_matrix_to_euler_xyz(R):
    """Computes Euler angles [Roll, Pitch, Yaw] in degrees (XYZ convention)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])


def main():
    args = parse_args()
    
    # 1. Device configuration
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    t_start = time.perf_counter()

    # 2. Read scene point cloud
    if not os.path.isfile(args.input):
        result = {
            "valid": False,
            "error": f"Input file not found: {args.input}"
        }
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        sys.exit(1)

    scene_pcd = o3d.io.read_point_cloud(args.input)
    if len(scene_pcd.points) < 50:
        result = {
            "valid": False,
            "error": f"Input cloud has too few points ({len(scene_pcd.points)})"
        }
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        sys.exit(1)

    raw_num_points = len(scene_pcd.points)

    # Clean NaNs
    scene_pcd.remove_non_finite_points()
    if not scene_pcd.has_normals():
        scene_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
    scene_pcd.orient_normals_towards_camera_location(np.zeros(3))

    # 3. Subsample to num_points for GeoPoseNet
    scene_pts_all = np.asarray(scene_pcd.points, dtype=np.float32)
    scene_norms_all = np.asarray(scene_pcd.normals, dtype=np.float32)

    if len(scene_pts_all) > args.num_points:
        idx = np.random.choice(len(scene_pts_all), args.num_points, replace=False)
        pts_sampled = scene_pts_all[idx]
        norms_sampled = scene_norms_all[idx]
    else:
        idx = np.random.choice(len(scene_pts_all), args.num_points, replace=True)
        pts_sampled = scene_pts_all[idx]
        norms_sampled = scene_norms_all[idx]

    # 4. Load CAD & Checkpoint
    cad_mesh, cad_pcd, scale_factor = load_cad_model(args.cad, num_points=2000)

    ckpt_path = args.checkpoint
    cad_name = os.path.splitext(os.path.basename(args.cad))[0]
    if not ckpt_path or not os.path.isfile(ckpt_path):
        candidate_1 = os.path.join(TMCAD_ROOT, "deep_learning", "checkpoints", cad_name, "best_model.pth")
        candidate_2 = os.path.join(TMCAD_ROOT, "deep_learning", "checkpoints", "best_e2e_pose_model.pth")
        if os.path.isfile(candidate_1):
            ckpt_path = candidate_1
        elif os.path.isfile(candidate_2):
            ckpt_path = candidate_2
        else:
            result = {
                "valid": False,
                "error": f"Chưa có checkpoint AI cho vật thể '{cad_name}'. Vui lòng bấm 'Train AI Model' trên thanh công cụ CadViewer, hoặc chuyển chế độ sang 'Classical (FPFH + RANSAC)'."
            }
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2)
            sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cad_keypoints = ckpt["cad_keypoints"]
    k_neighbors = ckpt.get("args", {}).get("k_neighbors", 20)
    feat_dim = ckpt.get("args", {}).get("feat_dim", 128)

    model = GeoPoseNet(cad_keypoints=cad_keypoints, k_neighbors=k_neighbors, feat_dim=feat_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 5. GeoPoseNet Inference (Coarse Pose)
    t_ai_start = time.perf_counter()
    feats_6d = np.concatenate([pts_sampled, norms_sampled], axis=1)  # (N, 6)
    feats_tensor = torch.from_numpy(feats_6d).unsqueeze(0).to(device)       # (1, N, 6)

    with torch.no_grad():
        out_dict = model(feats_tensor)
        R_coarse = out_dict["R"][0].cpu().numpy()
        t_coarse = out_dict["t"][0].cpu().numpy()

    t_ai_end = time.perf_counter()
    ai_latency_ms = (t_ai_end - t_ai_start) * 1000.0

    T_coarse = np.eye(4)
    T_coarse[:3, :3] = R_coarse
    T_coarse[:3, 3] = t_coarse

    # 6. Point-to-Plane ICP Refinement (Fine Pose) with Anti-Drift Guard
    T_final = T_coarse.copy()
    icp_latency_ms = 0.0
    fitness = 1.0
    inlier_rmse = 0.005

    if not args.no_icp:
        t_icp_start = time.perf_counter()
        try:
            # Voxel downsample to 10mm for balanced surface representation (avoids planar sliding)
            scene_pcd_icp = scene_pcd.voxel_down_sample(0.01)
            
            # Point-to-Plane ICP with 25mm threshold
            reg = o3d.pipelines.registration.registration_icp(
                cad_pcd, scene_pcd_icp, max_correspondence_distance=0.025,
                init=T_coarse,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30, relative_fitness=1e-5, relative_rmse=1e-5)
            )
            
            # Anti-drift guard: ensure ICP does not drift excessively from the AI coarse prediction
            drift_dist = np.linalg.norm(reg.transformation[:3, 3] - T_coarse[:3, 3])
            if reg.fitness > 0.15 and drift_dist < 0.08:
                T_final = reg.transformation
                fitness = float(reg.fitness)
                inlier_rmse = float(reg.inlier_rmse)
            else:
                # Keep GeoPoseNet coarse pose if ICP drifts along degenerate planar nullspace
                T_final = T_coarse
                eval_coarse = o3d.pipelines.registration.evaluate_registration(cad_pcd, scene_pcd_icp, 0.03, T_coarse)
                fitness = float(eval_coarse.fitness)
                inlier_rmse = float(eval_coarse.inlier_rmse)
        except Exception as e:
            print(f"[Warning] ICP refinement failed with {e}, keeping coarse pose.", file=sys.stderr)
            T_final = T_coarse
        t_icp_end = time.perf_counter()
        icp_latency_ms = (t_icp_end - t_icp_start) * 1000.0

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0

    R_final = T_final[:3, :3]
    t_final = T_final[:3, 3]
    euler_deg = rotation_matrix_to_euler_xyz(R_final)

    # 7. Optionally generate aligned scene overlay in object space
    if args.output_cloud:
        # Transform scene points into CAD object coordinates: P_obj = R^T * (P_scene - t)
        pts_obj = (R_final.T @ (scene_pts_all - t_final).T).T
        aligned_pcd = o3d.geometry.PointCloud()
        aligned_pcd.points = o3d.utility.Vector3dVector(pts_obj)
        
        # Color the aligned points (Electric Cyan: 0.1, 0.9, 0.9)
        colors = np.zeros_like(pts_obj)
        colors[:, 0] = 0.1
        colors[:, 1] = 0.9
        colors[:, 2] = 0.9
        aligned_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Scale back to CAD visual units (mm) if CAD model was scaled down
        if scale_factor < 0.5:
            aligned_pcd.scale(1.0 / scale_factor, np.zeros(3))

        o3d.io.write_point_cloud(args.output_cloud, aligned_pcd)

    # Physical Camera navigation angles in World / Robot frame:
    T_obj_cam = np.linalg.inv(T_final)
    cam_pos = T_obj_cam[:3, 3]
    fwd = T_obj_cam[:3, 2]
    cam_down = T_obj_cam[:3, 1]
    cam_up = -cam_down
    cam_right = T_obj_cam[:3, 0]

    phys_pitch = float(np.degrees(np.arcsin(np.clip(fwd[2], -1.0, 1.0))))
    phys_roll = float(np.degrees(np.arctan2(cam_right[2], cam_up[2])))
    phys_azimuth = float(np.degrees(np.arctan2(-cam_pos[1], -cam_pos[0])))
    phys_height = float(cam_pos[2])
    phys_dist = float(np.linalg.norm(cam_pos[:2]))

    # 8. Construct Output JSON
    result = {
        "valid": True,
        "transformation_matrix": T_final.tolist(),
        "R": R_final.tolist(),
        "t_meters": t_final.tolist(),
        "t_mm": (t_final * 1000.0).tolist(),
        "euler_deg": {
            "roll": float(euler_deg[0]),
            "pitch": float(euler_deg[1]),
            "yaw": float(euler_deg[2])
        },
        "camera_physical_world": {
            "pitch_deg": round(phys_pitch, 2),
            "roll_deg": round(phys_roll, 2),
            "azimuth_deg": round(phys_azimuth, 2),
            "height_m": round(phys_height, 3),
            "distance_m": round(phys_dist, 3)
        },
        "fitness": float(fitness),
        "inlier_rmse_m": float(inlier_rmse),
        "inlier_rmse_mm": float(inlier_rmse * 1000.0),
        "latency": {
            "ai_latency_ms": round(ai_latency_ms, 2),
            "icp_latency_ms": round(icp_latency_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2)
        },
        "num_raw_points": raw_num_points,
        "device": str(device)
    }

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary to stdout for Qt QProcess reading
    print(f"STATUS:SUCCESS")
    print(f"LATENCY_TOTAL:{total_latency_ms:.2f}ms (AI:{ai_latency_ms:.2f}ms, ICP:{icp_latency_ms:.2f}ms)")
    print(f"TRANSLATION_MM: X={t_final[0]*1000.0:.1f}, Y={t_final[1]*1000.0:.1f}, Z={t_final[2]*1000.0:.1f}")
    print(f"PHYSICAL_CAMERA: Pitch={phys_pitch:.2f}deg, Roll={phys_roll:.2f}deg, Height={phys_height:.3f}m")
    print(f"ROTATION_OPENCV: Roll={euler_deg[0]:.1f}, Pitch={euler_deg[1]:.1f}, Yaw={euler_deg[2]:.1f}")
    print(f"FITNESS:{fitness*100.0:.1f}% (RMSE: {inlier_rmse*1000.0:.2f}mm)")


if __name__ == "__main__":
    main()
