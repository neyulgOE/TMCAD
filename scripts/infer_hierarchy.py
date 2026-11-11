import os
import sys
import argparse
import time
import json
import copy
import numpy as np
import open3d as o3d
import torch

# Both CadViewer and TMVis invoke this script.  Fix the sampling/mesh RNG so
# the same RGB-D snapshot cannot yield a different pose or Fit value between
# runs merely because RANSAC/mesh sampling happened differently.
np.random.seed(1337)
torch.manual_seed(1337)
o3d.utility.random.seed(1337)

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deep_learning.model_hierarchy import CADHierarchyCorrespondenceModel

try:
    from scripts.build_cad_profile import build_cad_profile
except ImportError:
    build_cad_profile = None

def load_or_create_cad_profile(cad_path, cad_stem, root_dir):
    """
    Tự động đọc <CAD_NAME>_profile.json (hoặc tự tạo nếu chưa có)
    để lấy thông tin hình học, ngưỡng RANSAC/ICP thích ứng tỉ lệ và chế độ đặt.
    """
    candidate_paths = []
    if cad_path and os.path.exists(cad_path):
        cad_dir = os.path.dirname(os.path.abspath(cad_path))
        candidate_paths.append(os.path.join(cad_dir, f"{cad_stem}_profile.json"))
        candidate_paths.append(os.path.join(cad_dir, "profile.json"))
    candidate_paths.append(os.path.join(root_dir, "CAD_DATA", cad_stem, f"{cad_stem}_profile.json"))
    candidate_paths.append(os.path.join(root_dir, "CAD_DATA", f"{cad_stem}_profile.json"))

    for p in candidate_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    prof = json.load(f)
                    print(f"[CAD PROFILE] Loaded profile from {p} (mode={prof.get('placement_mode')}, diag={prof.get('real_bounding_diag_m', 0.0):.3f}m)")
                    return prof
            except Exception as e:
                print(f"[WARN] Failed to read {p}: {e}")

    # Nếu chưa có profile và có thể tạo tự động
    if build_cad_profile is not None:
        try:
            target_dir = os.path.join(root_dir, "CAD_DATA", cad_stem)
            if os.path.exists(target_dir):
                prof = build_cad_profile(target_dir, cad_stem)
                print(f"[CAD PROFILE] Generated profile on the fly for {cad_stem}")
                return prof
        except Exception as e:
            print(f"[WARN] Auto-build profile failed: {e}")

    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Inference: CAD-Aware Hierarchical Correspondence 6DoF Pose (Supports Black-Box .onnx)")
    parser.add_argument("--input", type=str, default=None, help="Input scene point cloud (.ply)")
    parser.add_argument("--onnx", type=str, default="", help="Self-contained Black-Box ONNX model (.onnx) — requires NO .stl or .pth")
    parser.add_argument("--cad", type=str, default="", help="Optional CAD mesh file (.stl/.obj/.ply) or model name")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional PyTorch checkpoint (.pth) or Black-Box (.onnx)")
    parser.add_argument("--output-json", type=str, default="debug/pose_output_ai.json", help="Output JSON result")
    parser.add_argument("--output-cloud", type=str, default="debug/aligned_scene_overlay.ply", help="Output aligned cloud")
    parser.add_argument("--num-obs-points", type=int, default=1024, help="Number of observation points to sample")
    parser.add_argument("--ransac-thresh", type=float, default=0.04, help="RANSAC distance threshold in meters")
    parser.add_argument("--icp-max-corr", type=float, default=0.03, help="ICP max correspondence distance in meters")
    parser.add_argument("--max-depth", type=float, default=3.5, help="Maximum Z depth to filter out background walls (meters)")
    parser.add_argument("--workspace-min-depth", type=float, default=0.0,
                        help="Optional target workspace near-Z bound in meters (0 disables it)")
    parser.add_argument("--workspace-max-depth", type=float, default=0.0,
                        help="Optional target workspace far-Z bound in meters (0 disables it)")
    parser.add_argument("--workspace-max-abs-x", type=float, default=0.0,
                        help="Optional target workspace lateral half-width in meters (0 disables it)")
    parser.add_argument("--fixed-height", type=float, default=-1.0, help="Camera height above floor in meters (AMR constraint), or -1 to disable")
    parser.add_argument("--amr-3dof", action="store_true", default=False, help="Enable AMR 3-DOF manifold constraint (X, Z, Yaw)")
    parser.add_argument("--real-sensor", action="store_true", default=True, help="Apply real sensor preprocessing (floor removal and clutter clustering)")
    parser.add_argument("--fx", type=float, default=386.652, help="Camera focal length fx")
    parser.add_argument("--fy", type=float, default=386.652, help="Camera focal length fy")
    parser.add_argument("--cx", type=float, default=320.0, help="Camera principal point cx")
    parser.add_argument("--cy", type=float, default=240.0, help="Camera principal point cy")
    parser.add_argument("--width", type=int, default=640, help="Camera image width")
    parser.add_argument("--height", type=int, default=480, help="Camera image height")
    parser.add_argument("--daemon", action="store_true", help="Run in persistent daemon mode listening on stdin")
    parser.add_argument("--debug-dump", type=str, default=None, help="Output detailed intermediate pipeline telemetry JSON")
    parser.add_argument("--load-sampled-indices", type=str, default=None, help="Load fixed 1024 observation point indices from file")
    return parser.parse_args()

def kabsch_ransac(pts_obs, pts_cad, confidences, max_iter=1500, distance_threshold=0.03):
    """
    Fast Priority RANSAC Kabsch SVD solver with confidence-weighted sampling
    and early consensus termination.
      pts_obs: (N, 3) in camera space
      pts_cad: (N, 3) in CAD metric object space
      returns: T (4x4), inlier_mask, c_src, c_dst
    """
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
            if count > max(32, int(0.60 * N)): # early stop on strong consensus
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

def run_single_inference(args, model, D_cad, key_centroids, device, blackbox_ctx=None):
    t_start = time.time()
    cad_dir = os.path.dirname(os.path.abspath(args.cad)) if args.cad else project_root
    if args.cad:
        cad_stem = os.path.splitext(os.path.basename(args.cad))[0]
    elif args.onnx:
        cad_stem = os.path.splitext(os.path.basename(args.onnx))[0]
    elif args.checkpoint:
        cad_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    else:
        cad_stem = "TABLE-DEMO"
    if cad_stem.endswith("_mesh"):
        cad_stem = cad_stem[:-5]

    profile = load_or_create_cad_profile(args.cad, cad_stem, project_root)
    placement_mode = profile.get("placement_mode", "FLOOR_STANDING" if "TABLE" in cad_stem.upper() else "FREE_6DOF") if profile else ("FLOOR_STANDING" if "TABLE" in cad_stem.upper() else "FREE_6DOF")
    topology_type = profile.get("topology_type", "FRAME_HOLLOW" if "TABLE" in cad_stem.upper() else "COMPLEX_SOLID") if profile else ("FRAME_HOLLOW" if "TABLE" in cad_stem.upper() else "COMPLEX_SOLID")
    adaptive_params = profile.get("adaptive_solver_params", {}) if profile else {}
    real_cad_diag = float(profile.get("real_bounding_diag_m", 1.07 if "TABLE" in cad_stem.upper() else 0.20)) if profile else (1.07 if "TABLE" in cad_stem.upper() else 0.20)
    is_table = (placement_mode == "FLOOR_STANDING") and ("TABLE" in cad_stem.upper())

    # 4. Load & sample Scene Point Cloud
    scene_pcd = o3d.io.read_point_cloud(args.input)
    if len(scene_pcd.points) < 10:
        raise ValueError("Scene point cloud is empty or has too few points.")

    # This is an installation-level acquisition ROI, not an algorithm-specific
    # TMVis shortcut.  Passing the same explicit values from both front-ends
    # makes them select the same physical workcell before floor/DBSCAN.
    if args.workspace_min_depth > 0 or args.workspace_max_depth > 0 or args.workspace_max_abs_x > 0:
        raw_pts = np.asarray(scene_pcd.points)
        keep = np.ones(len(raw_pts), dtype=bool)
        if args.workspace_min_depth > 0:
            keep &= raw_pts[:, 2] >= args.workspace_min_depth
        if args.workspace_max_depth > 0:
            keep &= raw_pts[:, 2] <= args.workspace_max_depth
        if args.workspace_max_abs_x > 0:
            keep &= np.abs(raw_pts[:, 0]) <= args.workspace_max_abs_x
        if int(np.sum(keep)) >= 128:
            scene_pcd = scene_pcd.select_by_index(np.where(keep)[0])
            print(f"[WORKSPACE] Applied shared target ROI: {len(scene_pcd.points)} points")

    floor_normal = None
    if args.real_sensor and is_table:
        try:
            from scripts.real_sensor_pipeline import remove_ground_and_clutter
            scene_pcd, diag = remove_ground_and_clutter(scene_pcd, camera_nominal_height=args.fixed_height, debug=True)
            print(f"[PREPROCESS REAL] Cleaned scene cloud: {len(scene_pcd.points)} pts remaining. Status: {diag.get('status')}")
            if diag.get("floor_detected") and "floor_normal" in diag and diag["floor_normal"] is not None:
                floor_normal = np.array(diag["floor_normal"], dtype=np.float64)
        except Exception as e:
            print(f"[WARN] Failed to apply real sensor preprocessing: {e}")

    support_plane_normal = None
    support_plane_height = None
    pts_curr = np.asarray(scene_pcd.points)
    scene_extent = pts_curr.max(axis=0) - pts_curr.min(axis=0) if len(pts_curr) > 0 else np.zeros(3)
    scene_diag_m = float(np.linalg.norm(scene_extent))

    # Multi-plane horizontal support filtering (tabletop, conveyor, floor, bench) and background wall removal
    if not is_table and len(scene_pcd.points) > 800 and scene_diag_m > 0.25:
        try:
            # Filter distant background > 1.6m if points are concentrated near
            pts_all_chk = np.asarray(scene_pcd.points)
            near_mask = pts_all_chk[:, 2] <= 1.6
            if np.sum(near_mask) > 300:
                rem_pcd = scene_pcd.select_by_index(np.where(near_mask)[0])
            else:
                rem_pcd = scene_pcd

            detected_planes = []
            for _ in range(6):
                if len(rem_pcd.points) < 500:
                    break
                pl, inls = rem_pcd.segment_plane(distance_threshold=0.012, ransac_n=3, num_iterations=1000)
                if len(inls) < 350:
                    break
                n = np.array(pl[:3], dtype=np.float64)
                n_len = np.linalg.norm(n)
                n /= (n_len + 1e-8)
                d_val = pl[3] / (n_len + 1e-8)
                
                # Check if plane is horizontal (normal aligned with vertical gravity axis in camera frame)
                if abs(n[1]) > 0.65:
                    if n[1] > 0:
                        n = -n
                        d_val = -d_val
                    detected_planes.append({
                        "normal": n,
                        "d": d_val,
                        "count": len(inls),
                        "plane": pl
                    })
                    rem_pcd = rem_pcd.select_by_index(inls, invert=True)
                elif len(inls) > 2000:
                    # Massive vertical wall / partition / background plane -> remove to expose foreground
                    rem_pcd = rem_pcd.select_by_index(inls, invert=True)
                else:
                    # Smaller non-horizontal feature: stop removing to avoid shaving CAD object
                    break

            if len(detected_planes) > 0 and len(rem_pcd.points) >= 30:
                labels = np.array(rem_pcd.cluster_dbscan(eps=0.035, min_points=25))
                valid_labels = [l for l in np.unique(labels) if l >= 0]
                if len(valid_labels) > 0:
                    best_l = -1
                    best_score = -1e9
                    rem_pts = np.asarray(rem_pcd.points)
                    cad_max_val = blackbox_ctx.get("cad_max", None) if blackbox_ctx else None
                    cad_min_val = blackbox_ctx.get("cad_min", None) if blackbox_ctx else None
                    if cad_max_val is not None and cad_min_val is not None:
                        target_cad_diag = float(np.linalg.norm(cad_max_val - cad_min_val))
                    else:
                        target_cad_diag = real_cad_diag
                    for l in valid_labels:
                        l_idx = np.where(labels == l)[0]
                        c_pts = rem_pts[l_idx]
                        c = np.mean(c_pts, axis=0)
                        c_span = np.ptp(c_pts, axis=0)
                        c_diag = float(np.linalg.norm(c_span))
                        
                        # Size similarity
                        size_match = 1.0 / (abs(c_diag - target_cad_diag) + 0.03)
                        
                        # Proximity to a horizontal table
                        table_bonus = 1.0
                        if len(detected_planes) > 0:
                            min_dist_to_table = min(abs(np.dot(t["normal"], c) + t["d"]) for t in detected_planes)
                            if min_dist_to_table < 0.35:
                                table_bonus = 1.0 / (min_dist_to_table + 0.05)
                            else:
                                table_bonus = 0.2
                        
                        # Depth penalty: favor objects within 0.25m - 1.2m
                        dist = float(np.linalg.norm(c))
                        depth_penalty = 1.0 if (0.25 <= c[2] <= 1.25) else 0.1
                        
                        score = (len(l_idx) * size_match * table_bonus * depth_penalty) / (dist + 0.3)
                        if score > best_score:
                            best_score = score
                            best_l = l
                    if best_l >= 0:
                        best_idx = np.where(labels == best_l)[0]
                        scene_pcd = rem_pcd.select_by_index(best_idx)
                        obj_center = np.mean(rem_pts[best_idx], axis=0)

                        best_sup = None
                        min_p_dist = 1e9
                        for p_info in detected_planes:
                            dist = abs(np.dot(p_info["normal"], obj_center) + p_info["d"])
                            if dist < min_p_dist:
                                min_p_dist = dist
                                best_sup = p_info
                        if best_sup is not None:
                            support_plane_normal = best_sup["normal"]
                            support_plane_height = -best_sup["d"]
                            total_plane_pts = sum(p["count"] for p in detected_planes)
                            print(f"[TABLETOP OBJECT] Đã bóc tách vật thể CAD trên mặt bàn: {len(scene_pcd.points)} điểm (Loại bỏ {total_plane_pts} điểm mặt phẳng bàn/sàn, khoảng cách mặt bàn={min_p_dist*1000.0:.1f}mm)")
        except Exception as e:
            print(f"[WARN] Tabletop object extraction: {e}")

    if floor_normal is None:
        try:
            p_all = np.asarray(scene_pcd.points)
            floor_cand = p_all[p_all[:, 1] > 0.12]
            if len(floor_cand) >= 100:
                pcd_fl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(floor_cand))
                plane_m, inls = pcd_fl.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=800)
                if len(inls) >= 60:
                    fn = np.array(plane_m[:3], dtype=np.float64)
                    if fn[1] > 0: fn = -fn
                    fn /= (np.linalg.norm(fn) + 1e-8)
                    if fn[1] < -0.85:
                        floor_normal = fn
        except Exception:
            pass

    scene_pts = np.asarray(scene_pcd.points, dtype=np.float32)
    if args.max_depth > 0:
        valid_z = (scene_pts[:, 2] <= args.max_depth) & (scene_pts[:, 2] > 0.05)
        if np.sum(valid_z) >= 64:
            scene_pts = scene_pts[valid_z]

    N_scene = len(scene_pts)

    # Sample observation points
    if args.load_sampled_indices and os.path.exists(args.load_sampled_indices):
        idx_sample = np.loadtxt(args.load_sampled_indices, dtype=int)
        print(f"[DEBUG] Loaded {len(idx_sample)} pre-fixed observation sample indices from: {args.load_sampled_indices}")
    elif N_scene >= args.num_obs_points:
        idx_sample = np.random.choice(N_scene, args.num_obs_points, replace=False)
    else:
        idx_sample = np.random.choice(N_scene, args.num_obs_points, replace=True)

    obs_sample = scene_pts[idx_sample] # (N_obs, 3)

    t_ai_start = time.time()
    if blackbox_ctx is not None and "ort_sess" in blackbox_ctx:
        ort_out = blackbox_ctx["ort_sess"].run(
            ["D_obs"],
            {"obs_pts": obs_sample[None, ...].astype(np.float32)}
        )
        D_obs_np = ort_out[0][0]  # (N_obs, D)
        D_obs = torch.from_numpy(D_obs_np)
        sim_np = (D_obs_np @ blackbox_ctx["cad_descriptors"].T) / 0.07
        sim_t = torch.from_numpy(sim_np)
        probs = torch.softmax(sim_t, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=3, dim=-1)
    else:
        with torch.no_grad():
            o_pts = torch.from_numpy(obs_sample).unsqueeze(0).to(device)
            D_obs = model.encode_obs(o_pts) # (1, N_obs, D)

            # Cosine similarity matrix S: (1, N_obs, N_cad)
            sim = model.compute_similarity(D_obs, D_cad)[0] # (N_obs, N_cad)
            probs = torch.softmax(sim, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=3, dim=-1)

    t_ai_end = time.time()
    ai_latency_ms = (t_ai_end - t_ai_start) * 1000.0

    pred_idx_np = top_indices[:, 0].cpu().numpy()
    conf_np = top_probs[:, 0].cpu().numpy()
    matched_cad_pts = key_centroids[pred_idx_np] # (N_obs, 3)

    # Cluster NMS: keep top 3 obs points per CAD centroid
    unique_c = np.unique(pred_idx_np)
    keep_mask = np.zeros(len(obs_sample), dtype=bool)
    for c in unique_c:
        c_indices = np.where(pred_idx_np == c)[0]
        top_c = c_indices[np.argsort(-conf_np[c_indices])[:3]]
        keep_mask[top_c] = True

    pts_obs_nms = obs_sample[keep_mask]
    pts_cad_nms = matched_cad_pts[keep_mask]
    conf_nms = conf_np[keep_mask]

    # 5. RANSAC + Kabsch SVD
    ransac_thresh = args.ransac_thresh
    if args.ransac_thresh == 0.04 and "ransac_thresh_m" in adaptive_params:
        ransac_thresh = float(adaptive_params["ransac_thresh_m"])

    t_solver_start = time.time()
    T_coarse, inliers, c_src, c_dst = kabsch_ransac(
        pts_obs_nms, pts_cad_nms, conf_nms,
        max_iter=3000, distance_threshold=ransac_thresh
    )
    num_inliers = int(np.sum(inliers))

    # 6. High-Precision Coarse-to-Fine Refinement with Embedded ONNX Surface Cloud (or CAD Mesh fallback)
    if blackbox_ctx is not None and "cad_surface_pts" in blackbox_ctx:
        cad_pts_np = blackbox_ctx["cad_surface_pts"].astype(np.float64)
        cad_normals_np = blackbox_ctx["cad_surface_normals"].astype(np.float64)
        cad_pcd = o3d.geometry.PointCloud()
        cad_pcd.points = o3d.utility.Vector3dVector(cad_pts_np)
        cad_pcd.normals = o3d.utility.Vector3dVector(cad_normals_np)
        cad_tree_coarse = blackbox_ctx["cad_tree_coarse"]
        cad_min = blackbox_ctx["cad_min"]
        cad_max = blackbox_ctx["cad_max"]
    else:
        cad_mesh_path = os.path.join(cad_dir, f"{cad_stem}_mesh.ply")
        if os.path.exists(cad_mesh_path):
            cad_mesh = o3d.io.read_triangle_mesh(cad_mesh_path)
        else:
            cad_mesh = o3d.io.read_triangle_mesh(args.cad)
            unit_sidecar = args.cad + ".unit"
            unit_scale = 0.001 if os.path.exists(unit_sidecar) else 1.0
            if os.path.exists(unit_sidecar):
                with open(unit_sidecar, 'r') as f:
                    try: unit_scale = float(f.read().strip())
                    except: pass
            if unit_scale != 1.0:
                cad_mesh.scale(unit_scale, center=np.zeros(3))

        cad_mesh.compute_vertex_normals()
        cad_pcd = cad_mesh.sample_points_uniformly(number_of_points=40000)
        cad_pcd.estimate_normals()
        cad_pts_np = np.asarray(cad_pcd.points)
        cad_normals_np = np.asarray(cad_pcd.normals)
        cad_tree_coarse = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cad_pts_np[::4])))
        cad_min = cad_mesh.get_min_bound()
        cad_max = cad_mesh.get_max_bound()

    # Check planar fraction from coarse pose
    sub_eval_check = scene_pts[::4]
    pts_in_cad_check = (np.linalg.inv(T_coarse)[:3, :3] @ sub_eval_check.T).T + np.linalg.inv(T_coarse)[:3, 3]
    top_frac = np.mean(pts_in_cad_check[:, 2] > 0.14)

    # 1. Always-On Discrete C2 Symmetry check for rectangular table (0 deg and 180 deg)
    if is_table:
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
            dists = [np.sqrt(cad_tree_coarse.search_knn_vector_3d(p, 1)[2][0]) for p in sample_sub]
            surf_fit = np.mean(np.array(dists) < 0.025)
            fit_score = in_b * 0.4 + surf_fit * 0.6
            if fit_score > best_fit:
                best_fit = fit_score
                best_R_sym = R_cand
                best_t_sym = t_cand
        T_coarse[:3, :3] = best_R_sym
        T_coarse[:3, 3] = best_t_sym

    T_curr = T_coarse.copy()
    cam_dist = float(np.linalg.norm(T_curr[:3, 3]))

    # 2. Planar Tabletop Boundary Anchor (chỉ áp dụng riêng cho CAD cái bàn TABLE-DEMO)
    if is_table and top_frac > 0.70 and cam_dist < 0.70:
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

                    reg = (dx**2 + dy**2) * 0.005
                    cost = out_bound * 10.0 + reg
                    if cost < best_c:
                        best_c = cost
                        best_sol = (d_th, dx, dy)

        if best_sol != (0, 0, 0):
            th_c, dx_c, dy_c = best_sol
            rad = np.radians(th_c)
            R_z = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
            T_curr[:3, :3] = R_init @ R_z
            T_curr[:3, 3] = t_init + T_curr[:3, :3] @ np.array([dx_c, dy_c, 0.0])

    # 3. Camera Frustum Culling + Visibility Filtering on CAD Points
    # Only keep mesh faces whose normals face TOWARD the camera (dot < 0).
    # Without this, ICP matches scene points on the front surface to both
    # front AND back faces of the aluminium extrusions, causing systematic
    # inward offset of ~half the profile thickness (~15mm).
    p_cam = (T_curr[:3, :3] @ cad_pts_np.T).T + T_curr[:3, 3]
    n_cam = (T_curr[:3, :3] @ cad_normals_np.T).T  # normals in camera space
    z_cam = p_cam[:, 2]
    valid_z = z_cam > 0.05
    u_proj = p_cam[valid_z, 0] * args.fx / z_cam[valid_z] + args.cx
    v_proj = p_cam[valid_z, 1] * args.fy / z_cam[valid_z] + args.cy
    in_frustum = np.zeros(len(cad_pts_np), dtype=bool)
    valid_idx = np.where(valid_z)[0]
    inside = (u_proj >= -35) & (u_proj < args.width + 35) & (v_proj >= -35) & (v_proj < args.height + 35)
    in_frustum[valid_idx[inside]] = True

    # Normal-based visibility: view direction from each mesh point toward camera origin
    # In camera space, camera is at origin. View ray for point p is -p (from p toward origin).
    # Face is visible if dot(normal, -point_dir) > 0, i.e. dot(normal, p) < 0 roughly.
    # More precisely: for each surface point, the ray is (0 - p_cam) = -p_cam.
    # Visible if dot(n_cam, -p_cam) > 0 → dot(n_cam, p_cam) < 0
    view_dot = np.sum(n_cam * p_cam, axis=1)
    facing_camera = view_dot < 0  # normal points toward camera

    visible = in_frustum & facing_camera
    n_visible = np.sum(visible)

    if n_visible >= 500:
        cad_target = o3d.geometry.PointCloud()
        cad_target.points = o3d.utility.Vector3dVector(cad_pts_np[visible])
        cad_target.normals = o3d.utility.Vector3dVector(cad_normals_np[visible])
    elif np.sum(in_frustum) >= 1000:
        cad_target = o3d.geometry.PointCloud()
        cad_target.points = o3d.utility.Vector3dVector(cad_pts_np[in_frustum])
        cad_target.normals = o3d.utility.Vector3dVector(cad_normals_np[in_frustum])
    else:
        cad_target = cad_pcd

    # 4. Fast Foreground Point Extraction in Object Space
    T_inv_curr = np.linalg.inv(T_curr)
    n_sub = min(6000, len(scene_pts))
    sub_scene = scene_pts[np.random.choice(len(scene_pts), n_sub, replace=(len(scene_pts) < n_sub))]
    pts_in_cad = (T_inv_curr[:3, :3] @ sub_scene.T).T + T_inv_curr[:3, 3]

    fg_thresh = float(adaptive_params.get("icp_coarse_corr_m", max(0.015, 0.045 * real_cad_diag)))
    fg_mask = np.array([
        cad_tree_coarse.search_knn_vector_3d(pt, 1)[2][0] < (fg_thresh ** 2)
        for pt in pts_in_cad
    ])

    icp_res = None
    pcd_fg = None
    T_icp = None
    min_fg = 64 if is_table else 16
    if np.sum(fg_mask) >= min_fg:
        pcd_fg = o3d.geometry.PointCloud()
        pcd_fg.points = o3d.utility.Vector3dVector(sub_scene[fg_mask])
        pcd_fg.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(0.012, fg_thresh * 0.8), max_nn=30))

        # 5. Multi-Scale Point-to-Plane ICP with Robust Tukey Loss & Adaptive Early Exit
        coarse_corr = float(adaptive_params.get("icp_coarse_corr_m", max(0.015, 0.045 * real_cad_diag)))
        fine_corr = float(adaptive_params.get("icp_fine_corr_m", max(0.003, 0.018 * real_cad_diag)))
        dist_schedule = [coarse_corr, coarse_corr * 0.55, coarse_corr * 0.30, fine_corr * 1.5, fine_corr]
        T_inv = T_inv_curr.copy()
        for dist_th in dist_schedule:
            loss_k = max(0.002, dist_th / 2.0)
            icp_res = o3d.pipelines.registration.registration_icp(
                pcd_fg, cad_target, dist_th, T_inv,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    o3d.pipelines.registration.TukeyLoss(k=loss_k)
                ),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=35, relative_fitness=1e-6, relative_rmse=1e-6)
            )
            T_inv = icp_res.transformation

        T_icp = np.linalg.inv(T_inv)
        if np.linalg.norm(T_icp[:3, 3] - T_curr[:3, 3]) < 0.08:
            T_final = T_icp
        else:
            T_final = T_curr
    else:
        T_final = T_curr
        T_icp = T_curr
        icp_res = o3d.pipelines.registration.RegistrationResult()
        icp_res.fitness = 0.0
        icp_res.inlier_rmse = 0.05

    # 5.05 Symmetry, Flip & Support-Plane Upright Verification (for ANY generic CAD object)
    # When an object has an occluded head/cap, ambiguous symmetry, or sits on a support surface,
    # coarse pose can suffer a 180° flip or get stuck in a degenerate local minimum.
    # We evaluate candidate orientations against surface consensus and select the optimal fit.
    if cad_tree_coarse is not None and pcd_fg is not None and len(pcd_fg.points) >= 16:
        cad_dz = float(cad_max[2] - cad_min[2])
        cad_dx = float(cad_max[0] - cad_min[0])
        cad_dy = float(cad_max[1] - cad_min[1])
        cad_diag = float(np.linalg.norm(cad_max - cad_min))
        cad_mid = 0.5 * (cad_min + cad_max)
        is_table = "TABLE" in cad_stem.upper()

        if not is_table:
            fg_pts_np = np.asarray(pcd_fg.points)
            eval_pts = fg_pts_np[::max(1, len(fg_pts_np) // 400)]

            # Support plane upright normal in camera coordinates
            v_ref = support_plane_normal if support_plane_normal is not None else floor_normal
            v_up = None
            if v_ref is not None:
                v_up = v_ref.copy()
                if v_up[1] > 0:
                    v_up = -v_up  # In camera frame, Y is down, so upward normal is -Y
                v_up /= (np.linalg.norm(v_up) + 1e-8)

            c_fg = np.mean(fg_pts_np, axis=0)
            view_dir = c_fg / (np.linalg.norm(c_fg) + 1e-8)
            cad_transverse_radius = 0.5 * min(cad_dx, cad_dy)

            if v_up is not None:
                if support_plane_height is not None:
                    base_h = support_plane_height
                else:
                    base_h = float(np.percentile(np.dot(fg_pts_np, v_up), 5))
                mid_above_base = float(cad_mid[2] - cad_min[2])
                target_mid_h = base_h + mid_above_base
                cur_mid_h = float(np.dot(c_fg, v_up))
                c_cam = c_fg + (target_mid_h - cur_mid_h) * v_up + cad_transverse_radius * view_dir
            else:
                c_cam = T_final[:3, :3] @ cad_mid + T_final[:3, 3]

            tol_dist = max(0.010, min(0.025, 0.08 * cad_diag))
            tol_sq = tol_dist ** 2

            def eval_transform_fitness(T_eval):
                T_inv = np.linalg.inv(T_eval)
                pts_c = (T_inv[:3, :3] @ eval_pts.T).T + T_inv[:3, 3]
                dists = np.array([cad_tree_coarse.search_knn_vector_3d(p, 1)[2][0] for p in pts_c])
                return float(np.mean(dists < tol_sq)), np.sqrt(float(np.mean(dists)))

            # Candidate 1: Current pose
            fit1, rmse1 = eval_transform_fitness(T_final)

            # Candidate 2: Flipped 180° around transverse X axis (preserving 3D centroid)
            R_flip_x = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
            T_flip_x = T_final.copy()
            T_flip_x[:3, :3] = T_final[:3, :3] @ R_flip_x
            T_flip_x[:3, 3] = c_cam - T_flip_x[:3, :3] @ cad_mid
            fit2x, rmse2x = eval_transform_fitness(T_flip_x)

            # Candidate 2b: Flipped 180° around transverse Y axis (preserving 3D centroid)
            R_flip_y = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
            T_flip_y = T_final.copy()
            T_flip_y[:3, :3] = T_final[:3, :3] @ R_flip_y
            T_flip_y[:3, 3] = c_cam - T_flip_y[:3, :3] @ cad_mid
            fit2y, rmse2y = eval_transform_fitness(T_flip_y)

            dot_z = float(np.dot(T_final[:3, 2], v_up)) if v_up is not None else 1.0

            # Candidate 3: Upright alignment against tabletop / floor normal with yaw search
            best_up_fit = -1.0
            T_up_cand = None
            if v_up is not None:
                z_ax = v_up
                x_ax = T_final[:3, 0] - np.dot(T_final[:3, 0], z_ax) * z_ax
                if np.linalg.norm(x_ax) < 1e-4:
                    x_ax = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), z_ax) * z_ax
                x_ax /= (np.linalg.norm(x_ax) + 1e-8)
                y_ax = np.cross(z_ax, x_ax)
                R_up_base = np.column_stack([x_ax, y_ax, z_ax])

                for yaw_d in range(-180, 180, 10):
                    rad = np.radians(yaw_d)
                    R_z = np.array([[np.cos(rad), -np.sin(rad), 0],
                                    [np.sin(rad),  np.cos(rad), 0],
                                    [0,            0,           1]])
                    R_cand = R_up_base @ R_z
                    t_cand = c_cam - R_cand @ cad_mid
                    T_cand = np.eye(4)
                    T_cand[:3, :3] = R_cand
                    T_cand[:3, 3] = t_cand
                    fit_u, _ = eval_transform_fitness(T_cand)
                    if fit_u > best_up_fit:
                        best_up_fit = fit_u
                        T_up_cand = T_cand

            print(f"[DEBUG FIT] fit1={fit1:.4f} (dot_z={dot_z:.2f}), fit2x={fit2x:.4f}, fit2y={fit2y:.4f}, fit_up={best_up_fit:.4f}", flush=True)

            is_upside_down = (v_up is not None and dot_z < -0.15)
            best_cand_T = T_final
            cand_name = "Candidate 1 (as-is)"

            # If the object is standing upside down on a support plane, disqualify Candidate 1
            if is_upside_down:
                cand_pool = []
                if T_up_cand is not None:
                    cand_pool.append((best_up_fit, T_up_cand, "Candidate 3 (Upright-Yaw)"))
                cand_pool.append((fit2x, T_flip_x, "Candidate 2 (Flip-X)"))
                cand_pool.append((fit2y, T_flip_y, "Candidate 2b (Flip-Y)"))
                cand_pool.sort(key=lambda x: x[0], reverse=True)
                best_cand_fit, best_cand_T, cand_name = cand_pool[0]
                print(f"[ORIENTATION] Auto-corrected upside-down pose -> {cand_name} (fit: {best_cand_fit*100:.1f}%)")
            else:
                # Upright candidates evaluated on superiority
                if T_up_cand is not None and (best_up_fit > 0.85 or (best_up_fit > fit1 + 0.05 and best_up_fit > max(fit2x, fit2y))):
                    best_cand_T = T_up_cand
                    cand_name = "Candidate 3 (Upright-Yaw)"
                elif max(fit2x, fit2y) > fit1 + 0.05:
                    if fit2x >= fit2y:
                        best_cand_T = T_flip_x
                        cand_name = "Candidate 2 (Flip-X)"
                    else:
                        best_cand_T = T_flip_y
                        cand_name = "Candidate 2b (Flip-Y)"

            # Refine selected candidate with multi-scale Point-to-Plane ICP on full CAD cloud
            if best_cand_T is not T_final or best_up_fit > 0.70:
                try:
                    reg_cand = o3d.pipelines.registration.registration_icp(
                        pcd_fg, cad_pcd, 0.035, np.linalg.inv(best_cand_T),
                        o3d.pipelines.registration.TransformationEstimationPointToPlane(
                            o3d.pipelines.registration.TukeyLoss(k=0.010)
                        ),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                    )
                    T_final = np.linalg.inv(reg_cand.transformation)
                    icp_res = reg_cand
                    print(f"[REFINEMENT] Successfully seated {cand_name} (RMSE: {reg_cand.inlier_rmse*1000:.1f}mm, Fitness: {reg_cand.fitness*100:.1f}%)")
                except Exception as ex:
                    T_final = best_cand_T
                    print(f"[REFINEMENT] Fallback to unrefined {cand_name} ({ex})")

            # Precision contact snap: flush alignment against physical support plane
            if v_up is not None and support_plane_height is not None:
                R_seated = T_final[:3, :3]
                t_seated = T_final[:3, 3]
                if np.dot(R_seated[:, 2], v_up) > 0.70:
                    cad_bot_local = np.array([0.0, 0.0, float(cad_min[2])])
                    bot_world = R_seated @ cad_bot_local + t_seated
                    cur_bot_h = float(np.dot(bot_world, v_up))
                    drift = support_plane_height - cur_bot_h
                    if abs(drift) < 0.015:
                        T_final[:3, 3] += v_up * drift
                        print(f"[CONTACT SNAP] Snapped object base flush onto support plane ({drift*1000.0:+.2f} mm).")

    # 5.1 Gravity Plane Alignment: Enforce CAD Z axis collinear with floor normal
    # Chỉ áp dụng khi vật thể đứng sàn (placement_mode == 'FLOOR_STANDING', ví dụ TABLE-DEMO).
    # Các vật thể FREE_6DOF giữ nguyên tư thế 6 bậc tự do từ ICP.
    is_floor_standing = (placement_mode == "FLOOR_STANDING")
    if is_floor_standing and floor_normal is not None:
        cad_z = T_final[:3, 2].copy()
        up_target = floor_normal / np.linalg.norm(floor_normal)
        if up_target[1] > 0:
            up_target = -up_target  # In camera frame, Y is down, so upward is -Y

        if cad_z[1] > 0:
            cad_z = -cad_z

        cos_th = float(np.dot(cad_z, up_target))
        if cos_th > 0.965:  # Within ~15 deg
            v = np.cross(cad_z, up_target)
            sin_th = float(np.linalg.norm(v))
            if sin_th > 1e-6:
                k = v / sin_th
                K = np.array([
                    [0.0, -k[2], k[1]],
                    [k[2], 0.0, -k[0]],
                    [-k[1], k[0], 0.0]
                ])
                R_align = np.eye(3) + sin_th * K + (1.0 - cos_th) * (K @ K)
                T_final[:3, :3] = R_align @ T_final[:3, :3]
                print(f"[GRAVITY] Aligned CAD Z with floor normal (cos={cos_th:.4f}, corrected: {np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0))):.2f}°)")

    # 5.2 Front-Face Anchoring & Fine Alignment (chỉ bật khi FLOOR_STANDING và enable_2d_frame_anchor == True)
    enable_anchor = bool(adaptive_params.get("enable_2d_frame_anchor", is_table))
    if is_floor_standing and enable_anchor and pcd_fg is not None and len(pcd_fg.points) >= 32:
        cad_half_w = float(profile.get("cad_max_xyz_m", [0.22, 0.415, 0.166])[0]) if profile else 0.220
        R_curr = T_final[:3, :3]
        t_curr = T_final[:3, 3]
        p_pos_x = R_curr @ np.array([cad_half_w, 0.0, 0.0]) + t_curr
        p_neg_x = R_curr @ np.array([-cad_half_w, 0.0, 0.0]) + t_curr
        hx_front = cad_half_w if p_pos_x[2] < p_neg_x[2] else -cad_half_w
        hx_rear = -hx_front
        v_depth_sign = 1.0 if hx_front > 0 else -1.0

        fg_np = np.asarray(pcd_fg.points)
        fg_cad_x = (R_curr.T @ (fg_np - t_curr).T)[0]
        front_region_mask = (fg_cad_x * v_depth_sign > 0.08)
        if np.sum(front_region_mask) >= 16:
            front_pts_cad_x = fg_cad_x[front_region_mask]
            if v_depth_sign > 0:
                obs_front_x = float(np.percentile(front_pts_cad_x, 90))
            else:
                obs_front_x = float(np.percentile(front_pts_cad_x, 10))
            delta_x = obs_front_x - hx_front
            if abs(delta_x) < 0.035:
                T_final[:3, 3] += R_curr[:, 0] * delta_x
                print(f"[FRONT_ANCHOR] Anchored front plane by {delta_x*1000.0:+.1f} mm onto physical front surface.")

    # 6. 3-DOF AMR Manifold Constraint Projection (Fixed Camera Height)
    if args.fixed_height > 0:
        cam_dist = float(np.linalg.norm(T_final[:3, 3]))
        if cam_dist <= 1.5:
            n_table = T_final[:3, 2].copy()
            n_table[0] = 0.0
            n_table /= (np.linalg.norm(n_table) + 1e-8)

            h_curr = float(np.dot(T_final[:3, 3], n_table))
            t_proj = T_final[:3, 3].copy()
            t_proj += (-args.fixed_height - h_curr) * n_table
            t_proj[0] = 0.0

            if np.linalg.norm(t_proj - T_final[:3, 3]) < 0.040:
                T_final[:3, 3] = t_proj

    t_solver_end = time.time()
    solver_latency_ms = (t_solver_end - t_solver_start) * 1000.0
    total_latency_ms = (t_solver_end - t_start) * 1000.0

    R = T_final[:3, :3]
    t = T_final[:3, 3]

    # Physical Table Orientation in Camera Coordinate System:
    # RealSense Camera: X=right, Y=down, Z=forward (optical axis)
    # CAD TABLE-DEMO: X=depth, Y=length, Z=upwards normal (feet to tabletop)
    # This avoids Tait-Bryan gimbal lock (pitch ~ -90 deg) and matches C++ rotationMatrixToEulerDeg
    v_len = R @ np.array([0.0, 1.0, 0.0])  # Col 1: CAD Y (table length beam)
    v_up = R @ np.array([0.0, 0.0, 1.0])   # Col 2: CAD Z (table upright normal)

    roll = float(np.arcsin(np.clip(v_up[0], -1.0, 1.0)) * 180.0 / np.pi)
    pitch = float(np.arcsin(np.clip(v_up[2], -1.0, 1.0)) * 180.0 / np.pi)
    yaw = float(np.arctan2(v_len[2], -v_len[0]) * 180.0 / np.pi)
    while yaw > 90.0: yaw -= 180.0
    while yaw < -90.0: yaw += 180.0
    table_yaw_cam = yaw

    # Camera World Physical parameters
    T_wc = np.linalg.inv(T_final)
    cam_pos = T_wc[:3, 3]
    fwd = T_wc[:3, 2]
    right = T_wc[:3, 0]
    up = -T_wc[:3, 1]

    phys_pitch = np.arcsin(np.clip(fwd[2], -1.0, 1.0)) * 180.0 / np.pi
    phys_roll = np.arctan2(right[2], up[2]) * 180.0 / np.pi
    phys_azimuth = table_yaw_cam
    phys_height = cam_pos[2]
    phys_dist = np.linalg.norm(cam_pos[:2])

    # 7. Generate Aligned Point Cloud for Viewport 3D Overlay
    T_inv = np.linalg.inv(T_final)
    aligned_scene = o3d.geometry.PointCloud(scene_pcd)
    aligned_scene.transform(T_inv)
    # Color aligned points lime green for visibility
    aligned_scene.paint_uniform_color([0.2, 0.95, 0.3])
    os.makedirs(os.path.dirname(os.path.abspath(args.output_cloud)), exist_ok=True)
    o3d.io.write_point_cloud(args.output_cloud, aligned_scene)

    # 8. Compute 3D Bounding Box & 2D Projection for Live HUD Stream
    cad_x0 = float(cad_min[0])
    cad_x1 = float(cad_max[0])
    cad_y0 = float(cad_min[1])
    cad_y1 = float(cad_max[1])
    z_bot = float(cad_min[2])
    z_top = float(cad_max[2])

    if profile and "cad_min_xyz_m" in profile and "cad_max_xyz_m" in profile:
        cad_x0 = float(profile["cad_min_xyz_m"][0])
        cad_x1 = float(profile["cad_max_xyz_m"][0])
        cad_y0 = float(profile["cad_min_xyz_m"][1])
        cad_y1 = float(profile["cad_max_xyz_m"][1])
        z_bot = float(profile.get("bottom_offset_z_m", profile["cad_min_xyz_m"][2]))
        z_top = float(profile.get("top_offset_z_m", profile["cad_max_xyz_m"][2]))
    elif is_table:
        dim_candidates = [
            os.path.join(cad_dir, "table_dimensions.json"),
            os.path.join(cad_dir, "TABLE-DEMO", "table_dimensions.json"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CAD_DATA", "TABLE-DEMO", "table_dimensions.json"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CAD_DATA", "table_dimensions.json"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ObjectDetection_V2.5/models/object_detection/cad_pose/table_cad/v1/cad_data/table_dimensions.json")
        ]
        for dp in dim_candidates:
            if os.path.exists(dp):
                try:
                    with open(dp, "r") as f:
                        dims = json.load(f)
                        if "tabletop_center_offset_z_m" in dims:
                            z_top = float(dims["tabletop_center_offset_z_m"])
                        if "ground_contact_offset_z_m" in dims:
                            z_bot = float(dims["ground_contact_offset_z_m"])
                        break
                except Exception:
                    pass

    # Bottom face (z_bot): 0:(x0,y0,z_bot), 1:(x1,y0,z_bot), 2:(x1,y1,z_bot), 3:(x0,y1,z_bot)
    # Top face (z_top):    4:(x0,y0,z_top), 5:(x1,y0,z_top), 6:(x1,y1,z_top), 7:(x0,y1,z_top)
    corners_3d = np.array([
        [cad_x0, cad_y0, z_bot],  # 0
        [cad_x1, cad_y0, z_bot],  # 1
        [cad_x1, cad_y1, z_bot],  # 2
        [cad_x0, cad_y1, z_bot],  # 3
        [cad_x0, cad_y0, z_top],  # 4
        [cad_x1, cad_y0, z_top],  # 5
        [cad_x1, cad_y1, z_top],  # 6
        [cad_x0, cad_y1, z_top],  # 7
    ], dtype=np.float32)
    # 8. Auto-Refine Camera Intrinsics using true physical CAD scale & project 3D Box
    scale_factor = 1.0
    refined_fx = float(args.fx)
    refined_fy = float(args.fy)
    if pcd_fg is not None and len(pcd_fg.points) >= 64 and not is_table:
        try:
            fg_in_cad = (R.T @ (np.asarray(pcd_fg.points) - t).T).T
            obs_diag = float(np.linalg.norm(np.ptp(fg_in_cad, axis=0)))
            cad_diag = float(np.linalg.norm([cad_x1 - cad_x0, cad_y1 - cad_y0, z_top - z_bot]))
            # Only perform slight fine-tuning if object is substantially fully visible (>88% CAD diag observed)
            if obs_diag > 0.88 * cad_diag and cad_diag > 0.05:
                scale_factor = float(np.clip(obs_diag / cad_diag, 0.97, 1.03))
                refined_fx = round(float(args.fx * scale_factor), 3)
                refined_fy = round(float(args.fy * scale_factor), 3)
        except Exception:
            pass

    corners_cam = (R @ corners_3d.T).T + t
    valid_z = corners_cam[:, 2] > 0.05
    if np.any(valid_z):
        u = corners_cam[:, 0] * refined_fx / np.maximum(corners_cam[:, 2], 0.05) + args.cx
        v = corners_cam[:, 1] * refined_fy / np.maximum(corners_cam[:, 2], 0.05) + args.cy
        corners_2d = np.stack([u, v], axis=1).tolist()
        u_min = max(0.0, float(np.min(u[valid_z])))
        v_min = max(0.0, float(np.min(v[valid_z])))
        u_max = min(float(args.width), float(np.max(u[valid_z])))
        v_max = min(float(args.height), float(np.max(v[valid_z])))
        bbox_2d = [u_min, v_min, max(10.0, u_max - u_min), max(10.0, v_max - v_min)]
    else:
        corners_2d = []
        bbox_2d = [0.0, 0.0, float(args.width), float(args.height)]

    # Mid-height ring (mid-plane slice for 3D box depth perception)
    z_mid = 0.5 * (z_bot + z_top)
    mid_corners_3d = np.array([
        [cad_x0, cad_y0, z_mid],
        [cad_x1, cad_y0, z_mid],
        [cad_x1, cad_y1, z_mid],
        [cad_x0, cad_y1, z_mid],
    ], dtype=np.float32)
    mid_cam = (R @ mid_corners_3d.T).T + t
    if np.all(mid_cam[:, 2] > 0.05):
        mid_u = mid_cam[:, 0] * refined_fx / mid_cam[:, 2] + args.cx
        mid_v = mid_cam[:, 1] * refined_fy / mid_cam[:, 2] + args.cy
        mid_section_2d = np.stack([mid_u, mid_v], axis=1).tolist()
    else:
        mid_section_2d = []

    # 3D Vector Arrow along object canonical orientation (matching benchmark standard)
    cad_mid_3d = np.array([0.5 * (cad_x0 + cad_x1), 0.5 * (cad_y0 + cad_y1), 0.5 * (z_bot + z_top)], dtype=np.float32)
    pt_c_cam = t + R @ cad_mid_3d
    arrow_len = max(0.025, float(0.35 * (z_top - z_bot)))
    pt_arrow_cam = pt_c_cam + R @ np.array([0.0, 0.0, arrow_len], dtype=np.float32)
    if pt_c_cam[2] > 0.05 and pt_arrow_cam[2] > 0.05:
        ac_u = pt_c_cam[0] * refined_fx / pt_c_cam[2] + args.cx
        ac_v = pt_c_cam[1] * refined_fy / pt_c_cam[2] + args.cy
        aa_u = pt_arrow_cam[0] * refined_fx / pt_arrow_cam[2] + args.cx
        aa_v = pt_arrow_cam[1] * refined_fy / pt_arrow_cam[2] + args.cy
        axis_arrow_2d = [[float(ac_u), float(ac_v)], [float(aa_u), float(aa_v)]]
    else:
        axis_arrow_2d = []

    # 9. Honest Scene-to-Mesh fitness at 10mm tolerance.
    # Uses the ICP-aligned foreground points (pcd_fg) — NOT the broadly pre-filtered
    # scene_pts — to avoid self-fulfilling high numbers from the proximity filter.
    try:
        T_inv_final = np.linalg.inv(T_final)
        fg_pts_world = np.asarray(pcd_fg.points) if pcd_fg is not None else scene_pts
        fg_in_cad = (T_inv_final[:3, :3] @ fg_pts_world.T).T + T_inv_final[:3, 3]
        sample_eval = fg_in_cad[::max(1, len(fg_in_cad) // 2000)]
        dists_sq = np.array([cad_tree_coarse.search_knn_vector_3d(p, 1)[2][0] for p in sample_eval])
        consensus_fitness = float(np.mean(dists_sq < (0.010 ** 2)))  # 10mm threshold
    except Exception:
        consensus_fitness = float(icp_res.fitness)

    raw_icp_fitness = float(icp_res.fitness)
    ransac_support = float(num_inliers) / max(1, len(pts_obs_nms))
    # Require independent correspondence support AND ICP support.  RMSE alone
    # is not a validity test: a handful of points can have a tiny RMSE on an
    # entirely incorrect table pose.
    # NOTE: raw ICP fitness is measured over the WHOLE CAD surface (incl. the
    # back/hidden faces embedded in the Black-Box ONNX), so from a single
    # viewpoint it is naturally low (~5%). The scene-side surface consensus
    # (fraction of observed foreground points lying within 10 mm of the CAD
    # surface) is the view-independent measure, so either one may confirm.
    icp_support_ok = (raw_icp_fitness >= 0.10) or (consensus_fitness >= 0.30)
    min_inliers_req = min(40, max(4, int(0.12 * len(pts_obs_nms))))
    is_valid = bool(
        num_inliers >= min_inliers_req and
        ransac_support >= 0.10 and
        icp_support_ok and
        icp_res.inlier_rmse <= 0.025
    )
    err_msg = ""
    if not is_valid:
        if num_inliers < min_inliers_req or ransac_support < 0.10:
            err_msg = (f"RANSAC support không đủ ({num_inliers}/{len(pts_obs_nms)} = "
                       f"{ransac_support*100:.1f}%, cần >= {min_inliers_req} và 10%).")
        elif icp_res.inlier_rmse > 0.025:
            err_msg = f"Sai số khớp mô hình RMSE ({icp_res.inlier_rmse*1000:.1f} mm) vượt ngưỡng 25 mm."
        else:
            err_msg = (f"Độ khớp bề mặt chưa đạt (ICP {raw_icp_fitness*100:.1f}% < 10%, "
                       f"consensus {consensus_fitness*100:.1f}% < 30%).")

    # Mechanical camera offset (Intel RealSense D435 Figure 4-6)
    # Offset +17.5 mm in X from Left IR optical axis to Centerline 1/4-20 front glass center
    cam_offset_x_m = 0.0175
    t_glass = t.copy()
    t_glass[0] += cam_offset_x_m

    # Compute front face (hình bình hành / tứ giác phối cảnh mặt trước) và rear face
    if "hx_front" not in locals():
        p_x0 = R @ np.array([cad_x0, 0.0, 0.0]) + t
        p_x1 = R @ np.array([cad_x1, 0.0, 0.0]) + t
        hx_front = cad_x1 if p_x1[2] < p_x0[2] else cad_x0
        hx_rear = cad_x0 if hx_front == cad_x1 else cad_x1

    c_front_3d = np.array([
        [hx_front, cad_y0, z_top],  # Top-Right
        [hx_front, cad_y1, z_top],  # Top-Left
        [hx_front, cad_y1, z_bot],  # Bottom-Left
        [hx_front, cad_y0, z_bot],  # Bottom-Right
    ], dtype=np.float32)
    c_rear_3d = np.array([
        [hx_rear, cad_y0, z_top],
        [hx_rear, cad_y1, z_top],
        [hx_rear, cad_y1, z_bot],
        [hx_rear, cad_y0, z_bot],
    ], dtype=np.float32)
    c_front_cam = (R @ c_front_3d.T).T + t
    c_rear_cam = (R @ c_rear_3d.T).T + t
    uf = c_front_cam[:, 0] * args.fx / np.maximum(c_front_cam[:, 2], 0.05) + args.cx
    vf = c_front_cam[:, 1] * args.fy / np.maximum(c_front_cam[:, 2], 0.05) + args.cy
    front_face_2d = np.stack([uf, vf], axis=1).tolist()
    ur = c_rear_cam[:, 0] * args.fx / np.maximum(c_rear_cam[:, 2], 0.05) + args.cx
    vr = c_rear_cam[:, 1] * args.fy / np.maximum(c_rear_cam[:, 2], 0.05) + args.cy
    rear_face_2d = np.stack([ur, vr], axis=1).tolist()

    out_dict = {
        "valid": is_valid,
        "error": err_msg,
        "fitness": consensus_fitness,
        "raw_icp_fitness": raw_icp_fitness,
        "surface_consensus": consensus_fitness,
        "ransac_support": ransac_support,
        "ransac_correspondences": int(len(pts_obs_nms)),
        "inlier_rmse_mm": float(icp_res.inlier_rmse * 1000.0),
        "ransac_inliers": int(num_inliers),
        "bbox_2d": bbox_2d,
        "corners_2d": corners_2d,
        "mid_section_2d": mid_section_2d,
        "axis_arrow_2d": axis_arrow_2d,
        "front_face_2d": front_face_2d,
        "rear_face_2d": rear_face_2d,
        "cad_name": str(cad_stem),
        "topology_type": str(topology_type),
        "placement_mode": str(placement_mode),
        "real_bounding_diag_m": float(round(real_cad_diag, 4)),
        "is_table": bool(is_table),
        "cad_min": [float(cad_x0), float(cad_y0), float(z_bot)],
        "cad_max": [float(cad_x1), float(cad_y1), float(z_top)],
        "extrude_depth_m": float(abs(cad_x1 - cad_x0)),
        "refined_intrinsics": {
            "fx": refined_fx,
            "fy": refined_fy,
            "cx": float(args.cx),
            "cy": float(args.cy),
            "width": int(args.width),
            "height": int(args.height),
            "scale_correction": float(scale_factor),
            "status": "auto_refined" if abs(scale_factor - 1.0) > 0.001 else "calibrated"
        },
        "transformation_matrix": T_final.tolist(),
        "translation_m": {"x": float(t_glass[0]), "y": float(t_glass[1]), "z": float(t_glass[2])},
        "optical_translation_m": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
        "pose_frame_id": "camera_front_glass_center",
        "euler_deg": {"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)},
        "camera_physical_world": {
            "height_m": float(phys_height),
            "distance_m": float(phys_dist),
            "pitch_deg": float(phys_pitch),
            "roll_deg": float(phys_roll),
            "azimuth_deg": float(phys_azimuth)
        },
        "latency": {
            "ai_latency_ms": float(ai_latency_ms),
            "icp_latency_ms": float(solver_latency_ms),
            "total_latency_ms": float(total_latency_ms)
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(out_dict, f, indent=2)

    if args.debug_dump:
        os.makedirs(os.path.dirname(os.path.abspath(args.debug_dump)), exist_ok=True)
        debug_dict = {
            "floor_detected": floor_normal is not None,
            "floor_normal": floor_normal.tolist() if floor_normal is not None else None,
            "num_scene_points": int(N_scene),
            "sampled_indices": idx_sample.tolist(),
            "obs_sample": obs_sample.tolist(),
            "D_obs": D_obs[0].cpu().numpy().tolist(),
            "top_indices": pred_idx_np.tolist(),
            "top_confidence": conf_np.tolist(),
            "T_coarse": T_coarse.tolist(),
            "T_icp": T_icp.tolist() if T_icp is not None else T_coarse.tolist(),
            "T_final": T_final.tolist(),
            "num_inliers": int(num_inliers),
            "ransac_support": float(ransac_support),
            "raw_icp_fitness": float(raw_icp_fitness),
            "consensus_fitness": float(consensus_fitness),
            "inlier_rmse_mm": float(icp_res.inlier_rmse * 1000.0),
            "is_valid": bool(is_valid),
            "error": err_msg,
            "corners_2d": corners_2d,
            "front_face_2d": front_face_2d
        }
        with open(args.debug_dump, 'w') as f:
            json.dump(debug_dict, f, indent=2)
        print(f"[DEBUG] Exported intermediate telemetry dump to: {args.debug_dump}")

    print(f"=== [CAD-AWARE HIERARCHY INFERENCE SUCCESS] ===")
    print(f"  Latency: AI={ai_latency_ms:.1f}ms | Solver={solver_latency_ms:.1f}ms | Total={total_latency_ms:.1f}ms")
    print(f"  RANSAC Inliers: {num_inliers}/{len(pts_obs_nms)} ({ransac_support*100:.1f}%) | "
          f"ICP Fitness: {raw_icp_fitness*100:.1f}% | Surface consensus: {consensus_fitness*100:.1f}% | "
          f"RMSE: {icp_res.inlier_rmse*1000:.2f}mm")
    print(f"  Translation (m): X={t[0]:.4f}, Y={t[1]:.4f}, Z={t[2]:.4f}")
    print(f"  Physical Nav: Height={phys_height:.3f}m, Dist={phys_dist:.3f}m, Pitch={phys_pitch:.1f}°, Azimuth={phys_azimuth:.1f}°")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Check if a Self-Contained Black-Box ONNX model is specified or available
    cad_stem = "TABLE-DEMO"
    if args.onnx:
        cad_stem = os.path.splitext(os.path.basename(args.onnx))[0]
    elif args.cad:
        cad_stem = os.path.splitext(os.path.basename(args.cad))[0]
    elif args.checkpoint and args.checkpoint.endswith(".onnx"):
        cad_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]

    onnx_candidates = []
    if args.onnx:
        onnx_candidates.append(args.onnx)
    if args.checkpoint and args.checkpoint.endswith(".onnx"):
        onnx_candidates.append(args.checkpoint)
    if args.cad and args.cad.endswith(".onnx"):
        onnx_candidates.append(args.cad)
    onnx_candidates.extend([
        os.path.join(project_root, "models", f"{cad_stem}.onnx"),
        os.path.join(project_root, "deploy_orange_pi_5b", cad_stem, f"{cad_stem}.onnx"),
        os.path.join(project_root, "deploy_orange_pi_5b", cad_stem, "model_obs_npu.onnx"),
    ])

    onnx_path = ""
    for p in onnx_candidates:
        if p and os.path.exists(p):
            onnx_path = p
            break

    blackbox_ctx = None
    model = None
    D_cad = None
    key_centroids = None

    if onnx_path:
        try:
            import onnxruntime as ort
            ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            if len(ort_sess.get_outputs()) >= 11:
                dummy_in = np.zeros((1, args.num_obs_points, 3), dtype=np.float32)
                outs = ort_sess.run(
                    [
                        "cad_descriptors",
                        "cad_centroids",
                        "cad_surface_pts",
                        "cad_surface_normals",
                        "cad_feature_edges",
                        "cad_top_poly",
                        "cad_foot_centers",
                        "cad_meta",
                    ],
                    {"obs_pts": dummy_in}
                )
                cad_desc, cad_cent, surf_pts, surf_norms, feat_edges, top_poly, foot_centers, cad_meta = outs
                key_centroids = cad_cent.astype(np.float32)
                cad_tree_coarse = o3d.geometry.KDTreeFlann(
                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(surf_pts.astype(np.float64)[::2]))
                )
                blackbox_ctx = {
                    "ort_sess": ort_sess,
                    "onnx_path": onnx_path,
                    "cad_descriptors": cad_desc.astype(np.float32),
                    "cad_centroids": key_centroids,
                    "cad_surface_pts": surf_pts.astype(np.float32),
                    "cad_surface_normals": surf_norms.astype(np.float32),
                    "cad_feature_edges": feat_edges.astype(np.float32),
                    "cad_top_poly": top_poly.astype(np.float32),
                    "cad_foot_centers": foot_centers.astype(np.float32),
                    "cad_meta": cad_meta.astype(np.float32),
                    "cad_min": cad_meta[:3].astype(np.float64),
                    "cad_max": cad_meta[3:6].astype(np.float64),
                    "cad_tree_coarse": cad_tree_coarse,
                }
                try:
                    debug_dir = os.path.join(project_root, "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    cad_pcd = o3d.geometry.PointCloud()
                    cad_pcd.points = o3d.utility.Vector3dVector(surf_pts.astype(np.float64))
                    cad_pcd.normals = o3d.utility.Vector3dVector(surf_norms.astype(np.float64))
                    cad_pcd.paint_uniform_color([0.15, 0.75, 0.90])
                    o3d.io.write_point_cloud(os.path.join(debug_dir, "onnx_cad_surface.ply"), cad_pcd)
                except Exception:
                    pass
                print(f"[BLACK-BOX ONNX] Loaded self-contained model: {onnx_path} (Zero STL/PTH dependencies)")
        except Exception as e:
            print(f"[WARN] Black-Box ONNX load notice: {e}")

    if blackbox_ctx is None:
        if not os.path.exists(args.cad):
            raise FileNotFoundError(f"CAD file not found: {args.cad}")
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        descriptor_dim = ckpt.get("descriptor_dim", 128)
        temperature = ckpt.get("temperature", 0.07)
        cad_key_indices = ckpt.get("cad_key_indices", None)

        cad_dir = os.path.dirname(os.path.abspath(args.cad))
        cad_stem = os.path.splitext(os.path.basename(args.cad))[0]

        tri_path = os.path.join(cad_dir, f"{cad_stem}_triangles.npz")
        patch_path = os.path.join(cad_dir, f"{cad_stem}_patches.npz")

        if not os.path.exists(tri_path) or not os.path.exists(patch_path):
            raise FileNotFoundError(f"Missing canonical hierarchy for {cad_stem} in {cad_dir}")

        tri_data = np.load(tri_path)
        all_features = tri_data['features'].astype(np.float32)
        all_centroids = all_features[:, :3]

        patch_data = np.load(patch_path)
        all_patches = patch_data['patch_indices'].astype(np.int64)
        all_hops = patch_data['patch_hops'].astype(np.int64)

        if cad_key_indices is None:
            cad_key_indices = np.arange(min(512, len(all_features)))

        key_centroids = all_centroids[cad_key_indices] # (N_cad, 3)
        key_patches = all_patches[cad_key_indices, :32]
        patch_features = all_features[key_patches]
        patch_rel_pos = patch_features[:, :, :3] - key_centroids[:, None, :]
        patch_hops = all_hops[cad_key_indices, :32]

        model = CADHierarchyCorrespondenceModel(
            descriptor_dim=descriptor_dim,
            patch_k=32,
            obs_k=20,
            temperature=temperature
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            c_feat = torch.from_numpy(patch_features).unsqueeze(0).to(device)
            c_rel = torch.from_numpy(patch_rel_pos).unsqueeze(0).to(device)
            c_hops = torch.from_numpy(patch_hops).unsqueeze(0).to(device)
            D_cad = model.encode_cad(c_feat, c_rel, c_hops) # (1, N_cad, D)

    if args.daemon:
        print("READY", flush=True)
        default_args = copy.deepcopy(args)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == "QUIT":
                break
            try:
                np.random.seed(1337)
                torch.manual_seed(1337)
                o3d.utility.random.seed(1337)

                req_args = copy.deepcopy(default_args)
                cmd = json.loads(line)
                for k, v in cmd.items():
                    if hasattr(req_args, k):
                        setattr(req_args, k, v)
                run_single_inference(req_args, model, D_cad, key_centroids, device, blackbox_ctx)
                print("DONE", flush=True)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
    else:
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input scene cloud not found: {args.input}")
        run_single_inference(args, model, D_cad, key_centroids, device, blackbox_ctx)

if __name__ == "__main__":
    main()
