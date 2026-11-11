#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import open3d as o3d
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deep_learning.model_hierarchy import CADHierarchyCorrespondenceModel

def test_pilot():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cad_name = "TABLE-DEMO"
    cad_dir = os.path.join(project_root, "CAD_DATA")
    mesh_path = os.path.join(cad_dir, f"{cad_name}_mesh.ply")
    tri_path = os.path.join(cad_dir, f"{cad_name}_triangles.npz")
    patch_path = os.path.join(cad_dir, f"{cad_name}_patches.npz")
    sym_path = os.path.join(cad_dir, f"{cad_name}_symmetry.json")
    ckpt_path = os.path.join(project_root, "deep_learning/checkpoints", cad_name, "best_hierarchy_model.pth")

    # 1. Load CAD Mesh & Raycasting Scene
    mesh_legacy = o3d.io.read_triangle_mesh(mesh_path)
    mesh_legacy.compute_vertex_normals()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    # 2. Load Model
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

    print("Pre-encoded CAD successfully!")

    # 3. Raycast 1 view
    d = 1.0
    azimuth = 45.0 * np.pi / 180.0
    elevation = 35.0 * np.pi / 180.0

    cam_pos = np.array([
        d * np.cos(elevation) * np.cos(azimuth),
        d * np.cos(elevation) * np.sin(azimuth),
        d * np.sin(elevation)
    ], dtype=np.float32)

    forward = -cam_pos / np.linalg.norm(cam_pos)
    up_world = np.array([0, 0, 1], dtype=np.float32)
    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    R_wc = np.stack([right, down, forward], axis=1)
    T_wc = np.eye(4, dtype=np.float32)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = cam_pos
    T_gt = np.linalg.inv(T_wc)

    # Raycast
    w, h = 640, 480
    fx, fy, cx, cy = 386.652, 386.652, 320.0, 240.0
    v, u = np.indices((h, w), dtype=np.float32)
    dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    norms = np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    dirs_cam_norm = dirs_cam / norms

    # Transform dirs to world
    dirs_world = (R_wc @ dirs_cam_norm.reshape(-1, 3).T).T.astype(np.float32)
    origins_world = np.broadcast_to(cam_pos, dirs_world.shape).astype(np.float32)

    rays = np.concatenate([origins_world, dirs_world], axis=-1)
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    t0 = time.time()
    ans = scene.cast_rays(rays_tensor)
    t_hit = ans['t_hit'].numpy()
    ray_time = (time.time() - t0) * 1000.0

    valid = np.isfinite(t_hit)
    hit_count = np.sum(valid)
    print(f"Raycast time: {ray_time:.2f} ms, Valid hit points: {hit_count}")

    # Depth map in meters
    z_depth = t_hit[valid] / norms.reshape(-1)[valid]
    
    # Add disparity noise
    sigma_z = (z_depth ** 2) / (fx * 0.05) * 0.08
    z_noisy = z_depth + np.random.normal(0, sigma_z)

    u_val = u.reshape(-1)[valid]
    v_val = v.reshape(-1)[valid]
    pts_cam = np.stack([
        (u_val - cx) * z_noisy / fx,
        (v_val - cy) * z_noisy / fy,
        z_noisy
    ], axis=-1)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_cam))
    pcd = pcd.voxel_down_sample(0.005)
    pts_filtered = np.asarray(pcd.points, dtype=np.float32)
    print(f"Filtered points (5mm): {len(pts_filtered)}")

if __name__ == "__main__":
    test_pilot()
