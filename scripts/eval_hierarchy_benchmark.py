import os
import sys
import json
import time
import argparse
import numpy as np
import cv2
import open3d as o3d
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deep_learning.model_hierarchy import CADHierarchyCorrespondenceModel

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CAD-Aware Hierarchical Correspondence Model")
    parser.add_argument("--cad-name", type=str, default="TABLE-DEMO")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of val samples to evaluate (or -1 for all)")
    parser.add_argument("--ransac-thresh", type=float, default=0.04)
    parser.add_argument("--icp-max-corr", type=float, default=0.03)
    parser.add_argument("--max-depth", type=float, default=2.5)
    return parser.parse_args()

def kabsch_ransac(pts_obs, pts_cad, confidences, max_iter=2000, distance_threshold=0.04):
    N = len(pts_obs)
    if N < 4:
        return np.eye(4), np.zeros(N, dtype=bool)

    best_inliers = np.zeros(N, dtype=bool)
    best_inlier_count = 0
    best_T = np.eye(4)

    probs = confidences / (np.sum(confidences) + 1e-8)

    for _ in range(max_iter):
        idx = np.random.choice(N, 4, replace=False, p=probs)
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
        residuals = np.linalg.norm(pts_obs - pred_dst, axis=1)
        inliers = (residuals < distance_threshold)
        count = np.sum(inliers)

        if count > best_inlier_count:
            best_inlier_count = count
            best_inliers = inliers
            best_T = np.eye(4)
            best_T[:3, :3] = R
            best_T[:3, 3] = t

    if best_inlier_count >= 4:
        p_src = pts_cad[best_inliers]
        p_dst = pts_obs[best_inliers]
        c_src = np.mean(p_src, axis=0)
        c_dst = np.mean(p_dst, axis=0)
        H = (p_src - c_src).T @ (p_dst - c_dst)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = c_dst - R @ c_src
        best_T[:3, :3] = R
        best_T[:3, 3] = t

    return best_T, best_inliers

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(project_root, "deep_learning/checkpoints", args.cad_name, "best_hierarchy_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cad_key_indices = ckpt["cad_key_indices"]
    model = CADHierarchyCorrespondenceModel(128, 32, 20, 0.07).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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

    # Pre-encode CAD
    with torch.no_grad():
        c_feat = torch.from_numpy(patch_features).unsqueeze(0).to(device)
        c_rel = torch.from_numpy(patch_rel_pos).unsqueeze(0).to(device)
        c_hops = torch.from_numpy(patch_hops).unsqueeze(0).to(device)
        D_cad = model.encode_cad(c_feat, c_rel, c_hops)

    # CAD Mesh for ICP and ADD-S
    mesh_path = os.path.join(cad_dir, f"{args.cad_name}_mesh.ply")
    cad_mesh = o3d.io.read_triangle_mesh(mesh_path)
    cad_mesh.compute_vertex_normals()
    cad_pcd_icp = cad_mesh.sample_points_uniformly(number_of_points=10000)
    cad_pcd_icp.estimate_normals()

    # CAD diameter and evaluation points for ADD-S
    cad_pts_eval = np.asarray(cad_mesh.sample_points_uniformly(number_of_points=2000).points)
    cad_diameter = np.linalg.norm(cad_mesh.get_max_bound() - cad_mesh.get_min_bound())
    add_threshold = 0.10 * cad_diameter # 10% diameter threshold

    # Symmetry transforms
    sym_transforms = [np.eye(3)]
    if os.path.exists(sym_path):
        with open(sym_path) as f:
            sym_info = json.load(f)
            for mat in sym_info.get("symmetry_transforms", []):
                arr = np.array(mat, dtype=np.float32).reshape(4, 4)
                sym_transforms.append(arr[:3, :3])

    # Val samples
    val_txt = os.path.join(project_root, "dataset", args.cad_name, "val.txt")
    with open(val_txt) as f:
        val_samples = [line.strip() for line in f if line.strip()]

    if args.num_samples > 0:
        val_samples = val_samples[:args.num_samples]

    print(f"============================================================")
    print(f" BENCHMARK: CAD-Aware Hierarchical Correspondence")
    print(f" Object: {args.cad_name} | CAD Diameter: {cad_diameter*1000:.1f} mm")
    print(f" ADD Threshold (10%): {add_threshold*1000:.1f} mm | Total samples: {len(val_samples)}")
    print(f"============================================================")

    errors_t = []
    errors_R = []
    errors_add = []
    latencies_ai = []
    latencies_icp = []
    latencies_tot = []
    fitnesses = []
    inlier_counts = []

    for idx, s_rel in enumerate(val_samples):
        s_dir = os.path.join(project_root, "dataset", args.cad_name, s_rel)
        with open(os.path.join(s_dir, "pose.json")) as f:
            meta = json.load(f)

        intr = meta["intrinsics"]
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        w, h = intr["width"], intr["height"]
        depth_scale = meta["depthScale"]
        T_gt = np.array(meta["cameraFromObject"], dtype=np.float32)

        depth = cv2.imread(os.path.join(s_dir, "depth.png"), cv2.IMREAD_UNCHANGED)
        depth_m = depth.astype(np.float32) / depth_scale

        v, u = np.indices((h, w))
        valid = (depth_m > 0.05) & (depth_m <= args.max_depth)
        z = depth_m[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy
        scene_pts = np.stack([x, y, z], axis=-1)

        if len(scene_pts) < 64:
            continue

        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_pts)
        scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))

        # Sample observation
        t_start = time.time()
        choice = np.random.choice(len(scene_pts), 1024, replace=(len(scene_pts) < 1024))
        obs_sample = scene_pts[choice]

        # AI Forward
        t_ai0 = time.time()
        with torch.no_grad():
            o_pts = torch.from_numpy(obs_sample.astype(np.float32)).unsqueeze(0).to(device)
            D_obs = model.encode_obs(o_pts)
            sim = model.compute_similarity(D_obs, D_cad)[0]
            probs = torch.softmax(sim, dim=-1)
            confidences, pred_idx = torch.max(probs, dim=-1)

        t_ai1 = time.time()
        ai_lat = (t_ai1 - t_ai0) * 1000.0

        conf_np = confidences.cpu().numpy()
        matched_cad_pts = key_centroids[pred_idx.cpu().numpy()]

        # RANSAC Kabsch
        t_sol0 = time.time()
        T_coarse, inliers = kabsch_ransac(
            obs_sample, matched_cad_pts, conf_np,
            max_iter=2000, distance_threshold=args.ransac_thresh
        )
        num_inliers = int(np.sum(inliers))

        # ICP
        icp_res = o3d.pipelines.registration.registration_icp(
            cad_pcd_icp, scene_pcd, args.icp_max_corr, T_coarse,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        T_final = icp_res.transformation if icp_res.fitness > 0.15 else T_coarse
        t_sol1 = time.time()
        sol_lat = (t_sol1 - t_sol0) * 1000.0
        tot_lat = (t_sol1 - t_start) * 1000.0

        # Evaluation vs GT
        t_gt = T_gt[:3, 3]
        t_pred = T_final[:3, 3]
        e_t = np.linalg.norm(t_gt - t_pred)

        R_gt = T_gt[:3, :3]
        R_pred = T_final[:3, :3]

        best_e_R = 180.0
        for S in sym_transforms:
            cos_a = np.clip((np.trace(R_pred.T @ (R_gt @ S)) - 1.0) / 2.0, -1.0, 1.0)
            e_R_cand = np.arccos(cos_a) * 180.0 / np.pi
            if e_R_cand < best_e_R:
                best_e_R = e_R_cand

        # ADD-S metric (average distance to closest model point under symmetry)
        pts_gt = (R_gt @ cad_pts_eval.T).T + t_gt
        pts_pred = (R_pred @ cad_pts_eval.T).T + t_pred
        
        # Point-to-point symmetric distance
        # Using KD-Tree for nearest neighbors
        pcd_pred_tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_pred)))
        dists = []
        for p in pts_gt:
            _, _, d2 = pcd_pred_tree.search_knn_vector_3d(p, 1)
            dists.append(np.sqrt(d2[0]))
        add_s = np.mean(dists)

        errors_t.append(e_t * 1000.0)
        errors_R.append(best_e_R)
        errors_add.append(add_s * 1000.0)
        latencies_ai.append(ai_lat)
        latencies_icp.append(sol_lat)
        latencies_tot.append(tot_lat)
        fitnesses.append(icp_res.fitness * 100.0)
        inlier_counts.append(num_inliers)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(val_samples):
            print(f"  [{idx+1:02d}/{len(val_samples):02d}] "
                  f"Med e_t: {np.median(errors_t):.1f}mm | Med e_R: {np.median(errors_R):.1f}° | "
                  f"ADD-S: {np.mean(np.array(errors_add) <= (add_threshold*1000.0))*100:.1f}% | "
                  f"Inliers: {np.median(inlier_counts):.0f} | Latency: {np.mean(latencies_tot):.0f}ms")

    # Final summary
    pass_rate = np.mean(np.array(errors_add) <= (add_threshold * 1000.0)) * 100.0
    print(f"\n============================================================")
    print(f" BENCHMARK RESULTS ({len(errors_t)} samples)")
    print(f"============================================================")
    print(f"  • Median Translation Error (e_t):   {np.median(errors_t):.2f} mm  (Mean: {np.mean(errors_t):.2f} mm)")
    print(f"  • Median Rotation Error (e_R):      {np.median(errors_R):.2f}°   (Mean: {np.mean(errors_R):.2f}°)")
    print(f"  • Median ADD-S Error:              {np.median(errors_add):.2f} mm")
    print(f"  • ADD-S (<0.10 diameter) Pass:      {pass_rate:.1f}%")
    print(f"  • Median ICP Fitness:               {np.median(fitnesses):.1f}%")
    print(f"  • Median RANSAC Inliers:            {np.median(inlier_counts):.0f} / 1024")
    print(f"  • Latency Breakdown: AI={np.mean(latencies_ai):.1f}ms | Solver={np.mean(latencies_icp):.1f}ms | Total={np.mean(latencies_tot):.1f}ms")
    print(f"============================================================\n")

if __name__ == "__main__":
    main()
