#!/usr/bin/env python3
"""
Export Hierarchy Model to a Single Self-Contained Black-Box ONNX File (.onnx).
All CAD descriptors, 3D keypoints, 10k surface points+normals, structural feature edges,
top polygon, floor contact feet, and physical metadata are baked directly inside the .onnx graph.
At runtime (TMVis_Debug, Qt App, or Orange Pi 5B Kit), ONLY this single .onnx file is needed
— zero .stl, .bin, or .npy files are required.
"""
import os
import sys
import struct
import json
import math
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deep_learning.model_hierarchy import CADHierarchyCorrespondenceModel
try:
    from deep_learning.dataset_hierarchy import fps_sampling_np
except ImportError:
    def fps_sampling_np(pts: np.ndarray, num_samples: int) -> np.ndarray:
        n = len(pts)
        if n <= num_samples:
            return np.arange(n, dtype=np.int64)
        selected = np.zeros(num_samples, dtype=np.int64)
        dists = np.full(n, np.inf, dtype=np.float32)
        farthest = 0
        for i in range(num_samples):
            selected[i] = farthest
            diff = pts - pts[farthest]
            d = np.einsum("ij,ij->i", diff, diff)
            dists = np.minimum(dists, d)
            farthest = int(np.argmax(dists))
        return selected


class CADBlackBoxONNXModule(nn.Module):
    """
    Unified Self-Contained Black-Box ONNX Model for 6DoF CAD Pose Estimation.
    Inputs:
      - obs_pts:             (1, 1024, 3)  [float32] camera point cloud in meters
    Outputs:
      - D_obs:               (1, 1024, 128)[float32] L2-normalized point descriptors
      - matched_cad_pts:     (1, 1024, 3)  [float32] best-matching 3D CAD coordinates per obs point
      - match_scores:        (1, 1024)     [float32] cosine similarity score per obs point
      - cad_descriptors:     (512, 128)    [float32] embedded CAD patch descriptors
      - cad_centroids:       (512, 3)      [float32] embedded CAD 3D keypoint centroids
      - cad_surface_pts:     (10000, 3)    [float32] embedded CAD surface point cloud for Mesh ICP
      - cad_surface_normals: (10000, 3)    [float32] embedded CAD surface normals for Mesh ICP
      - cad_feature_edges:   (E, 13)       [float32] embedded 3D wireframe/feature edges [p0(3), p1(3), n0(3), n1(3), len]
      - cad_top_poly:        (P_top, 3)    [float32] embedded top surface convex polygon vertices
      - cad_foot_centers:    (P_foot, 3)   [float32] embedded floor contact foot centers
      - cad_meta:            (16,)         [float32] embedded physical bounds & archetype parameters
    """
    def __init__(self,
                 obs_encoder: nn.Module,
                 cad_descriptors: np.ndarray,
                 cad_centroids: np.ndarray,
                 cad_surface_pts: np.ndarray,
                 cad_surface_normals: np.ndarray,
                 cad_feature_edges: np.ndarray,
                 cad_top_poly: np.ndarray,
                 cad_foot_centers: np.ndarray,
                 cad_meta: np.ndarray):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.register_buffer("cad_descriptors_buf", torch.from_numpy(cad_descriptors.astype(np.float32)))
        self.register_buffer("cad_centroids_buf", torch.from_numpy(cad_centroids.astype(np.float32)))
        self.register_buffer("cad_surface_pts_buf", torch.from_numpy(cad_surface_pts.astype(np.float32)))
        self.register_buffer("cad_surface_normals_buf", torch.from_numpy(cad_surface_normals.astype(np.float32)))
        self.register_buffer("cad_feature_edges_buf", torch.from_numpy(cad_feature_edges.astype(np.float32)))
        self.register_buffer("cad_top_poly_buf", torch.from_numpy(cad_top_poly.astype(np.float32)))
        self.register_buffer("cad_foot_centers_buf", torch.from_numpy(cad_foot_centers.astype(np.float32)))
        self.register_buffer("cad_meta_buf", torch.from_numpy(cad_meta.astype(np.float32)))

    def forward(self, obs_pts: torch.Tensor):
        # 1. Extract 128-D descriptors for observed 3D points: (1, N, 128)
        D_obs = self.obs_encoder(obs_pts)

        # 2. Compute cosine similarity matrix against embedded CAD descriptors: (1, N, 512)
        sim = torch.matmul(D_obs, self.cad_descriptors_buf.t())
        match_scores, best_idx = torch.max(sim, dim=-1)  # (1, N), (1, N)
        matched_cad_pts = self.cad_centroids_buf[best_idx]  # (1, N, 3)

        # Anchor embedded buffers with a zero-valued scalar from obs_pts so ONNX exporter
        # preserves all constant buffers as named graph outputs without pruning.
        zero_anchor = (obs_pts[0, 0, 0] * 0.0)
        cad_descriptors = self.cad_descriptors_buf + zero_anchor
        cad_centroids = self.cad_centroids_buf + zero_anchor
        cad_surface_pts = self.cad_surface_pts_buf + zero_anchor
        cad_surface_normals = self.cad_surface_normals_buf + zero_anchor
        cad_feature_edges = self.cad_feature_edges_buf + zero_anchor
        cad_top_poly = self.cad_top_poly_buf + zero_anchor
        cad_foot_centers = self.cad_foot_centers_buf + zero_anchor
        cad_meta = self.cad_meta_buf + zero_anchor

        return (
            D_obs,
            matched_cad_pts,
            match_scores,
            cad_descriptors,
            cad_centroids,
            cad_surface_pts,
            cad_surface_normals,
            cad_feature_edges,
            cad_top_poly,
            cad_foot_centers,
            cad_meta,
        )


class RKNNStaticObservationEncoder(nn.Module):
    """
    DGCNN Point Cloud Encoder optimized for Rockchip NPU (RK3588S / Orange Pi 5B).
    """
    def __init__(self, original_obs):
        super().__init__()
        self.conv1 = original_obs.conv1
        self.conv2 = original_obs.conv2
        self.conv3 = original_obs.conv3
        self.conv4 = original_obs.conv4
        self.mlp_out = original_obs.mlp_out

    def _get_graph_from_idx(self, x, idx):
        B, C, N = x.shape
        k = idx.shape[1]
        x_t = x[0].transpose(1, 0)
        neighbors = x_t[idx]
        center = x_t.unsqueeze(1).repeat(1, k, 1)
        diff = neighbors - center
        if C == 3:
            dist = torch.norm(diff, dim=-1, keepdim=True)
            feat = torch.cat([center, diff, dist], dim=-1)
        else:
            feat = torch.cat([center, diff], dim=-1)
        feat = feat.permute(2, 0, 1).unsqueeze(0)
        return feat

    def forward(self, obs_pts, idx):
        x1_in = self._get_graph_from_idx(obs_pts, idx)
        x1 = self.conv1(x1_in).max(dim=-1)[0]
        x2_in = self._get_graph_from_idx(x1, idx)
        x2 = self.conv2(x2_in).max(dim=-1)[0]
        x3_in = self._get_graph_from_idx(x2, idx)
        x3 = self.conv3(x3_in).max(dim=-1)[0]
        cat_feat = torch.cat([x1, x2, x3], dim=1)
        point_feat = self.conv4(cat_feat)
        global_feat = point_feat.max(dim=-1, keepdim=True)[0].repeat(1, 1, point_feat.shape[-1])
        fused = torch.cat([point_feat, global_feat], dim=1)
        desc = self.mlp_out(fused).transpose(2, 1)
        return F.normalize(desc, p=2, dim=-1)


def sample_surface_points_from_triangles(vertices: np.ndarray,
                                         normals: np.ndarray,
                                         areas: np.ndarray,
                                         num_samples: int = 10000,
                                         seed: int = 42):
    """
    Uniformly sample `num_samples` 3D points and unit normals on the triangle mesh
    weighted by triangle surface area.
    """
    rng = np.random.default_rng(seed)
    probs = np.maximum(areas.astype(np.float64), 0.0)
    total_area = probs.sum()
    if total_area <= 1e-12:
        probs = np.ones(len(vertices), dtype=np.float64) / max(1, len(vertices))
    else:
        probs /= total_area

    tri_indices = rng.choice(len(vertices), size=num_samples, replace=True, p=probs)
    r1 = np.sqrt(rng.random(num_samples, dtype=np.float32))[:, None]
    r2 = rng.random(num_samples, dtype=np.float32)[:, None]

    v0 = vertices[tri_indices, 0, :]
    v1 = vertices[tri_indices, 1, :]
    v2 = vertices[tri_indices, 2, :]

    pts = (1.0 - r1) * v0 + (r1 * (1.0 - r2)) * v1 + (r1 * r2) * v2
    norms = normals[tri_indices].astype(np.float32)
    norm_len = np.linalg.norm(norms, axis=-1, keepdims=True)
    norms = np.where(norm_len > 1e-8, norms / np.maximum(norm_len, 1e-8), np.array([0.0, 0.0, 1.0], dtype=np.float32))
    return pts.astype(np.float32), norms.astype(np.float32)


def extract_topology_and_edges_from_triangles(vertices: np.ndarray,
                                              normals: np.ndarray,
                                              cad_min: np.ndarray,
                                              cad_max: np.ndarray,
                                              feature_angle_deg: float = 14.0,
                                              min_edge_len_m: float = 0.008):
    """
    Exact Python implementation of TablePoseEstimator::extractStlFeatureEdges so all
    wireframe/feature edges, top polygon, and floor contact feet are baked once into ONNX.
    """
    tri_min = vertices.reshape(-1, 3).min(axis=0)
    tri_max = vertices.reshape(-1, 3).max(axis=0)
    tri_span = np.maximum(tri_max - tri_min, 1e-6)
    cad_span = cad_max - cad_min

    # Align vertices to canonical [cad_min, cad_max] frame if needed
    verts = (vertices - tri_min[None, None, :]) * (cad_span / tri_span)[None, None, :] + cad_min[None, None, :]
    e1 = verts[:, 1, :] - verts[:, 0, :]
    e2 = verts[:, 2, :] - verts[:, 0, :]
    raw_n = np.cross(e1, e2)
    n_len = np.linalg.norm(raw_n, axis=-1, keepdims=True)
    t_normals = np.where(n_len > 1e-12, raw_n / np.maximum(n_len, 1e-12), 0.0).astype(np.float32)

    z_max = float(cad_max[2])
    z_min = float(cad_min[2])

    # 1. Top polygon (convex hull of top-facing triangles near z_max)
    top_mask = (t_normals[:, 2] > 0.85) & (verts[:, :, 2].min(axis=1) >= z_max - 0.005)
    if np.any(top_mask):
        top_pts_2d = verts[top_mask][:, :, :2].reshape(-1, 2).astype(np.float32)
        hull = cv2.convexHull(top_pts_2d)
        approx = cv2.approxPolyDP(hull, 0.003, True).reshape(-1, 2)
        cad_top_poly = np.column_stack([approx, np.full(len(approx), z_max, dtype=np.float32)]).astype(np.float32)
    else:
        hw, hl = 0.5 * (cad_max[0] - cad_min[0]), 0.5 * (cad_max[1] - cad_min[1])
        cx, cy = 0.5 * (cad_min[0] + cad_max[0]), 0.5 * (cad_min[1] + cad_max[1])
        cad_top_poly = np.array([
            [cx - hw, cy - hl, z_max],
            [cx + hw, cy - hl, z_max],
            [cx + hw, cy + hl, z_max],
            [cx - hw, cy + hl, z_max],
        ], dtype=np.float32)

    # 2. Bottom floor contact feet centers
    bot_mask = (t_normals[:, 2] < -0.85) & (verts[:, :, 2].max(axis=1) <= z_min + 0.005)
    cad_foot_centers = []
    if np.any(bot_mask):
        bot_pts = verts[bot_mask].reshape(-1, 3)
        x_mid = 0.5 * (cad_min[0] + cad_max[0])
        y_mid = 0.5 * (cad_min[1] + cad_max[1])
        for q in range(4):
            qx = (bot_pts[:, 0] >= x_mid) if (q & 1) else (bot_pts[:, 0] < x_mid)
            qy = (bot_pts[:, 1] >= y_mid) if (q & 2) else (bot_pts[:, 1] < y_mid)
            q_pts = bot_pts[qx & qy]
            if len(q_pts) > 0:
                avg = q_pts.mean(axis=0)
                cad_foot_centers.append([float(avg[0]), float(avg[1]), z_min])
    if not cad_foot_centers:
        cad_foot_centers = [
            [float(cad_min[0]), float(cad_min[1]), z_min],
            [float(cad_max[0]), float(cad_min[1]), z_min],
            [float(cad_max[0]), float(cad_max[1]), z_min],
            [float(cad_min[0]), float(cad_max[1]), z_min],
        ]
    cad_foot_centers = np.array(cad_foot_centers, dtype=np.float32)

    # 3. Vertex welding (0.2 mm grid) & sharp dihedral edge extraction
    inv_tol = 5000.0
    all_v = verts.reshape(-1, 3)
    quant_v = np.round(all_v * inv_tol).astype(np.int64)
    _, unique_idx, inv_map = np.unique(quant_v, axis=0, return_index=True, return_inverse=True)
    unique_vertices = all_v[unique_idx]
    tri_vids = inv_map.reshape(-1, 3)

    valid_tri = (
        (np.sum(t_normals * t_normals, axis=1) >= 0.5) &
        (tri_vids[:, 0] != tri_vids[:, 1]) &
        (tri_vids[:, 1] != tri_vids[:, 2]) &
        (tri_vids[:, 2] != tri_vids[:, 0])
    )
    tri_vids = tri_vids[valid_tri]
    tri_norms = t_normals[valid_tri]

    cos_thresh = math.cos(feature_angle_deg * math.pi / 180.0)
    edge_dict = {}
    for t_idx in range(len(tri_vids)):
        v0, v1, v2 = int(tri_vids[t_idx, 0]), int(tri_vids[t_idx, 1]), int(tri_vids[t_idx, 2])
        n = tri_norms[t_idx]
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            key = (a, b) if a < b else (b, a)
            info = edge_dict.get(key)
            if info is None:
                edge_dict[key] = [n, n, 1, False]
            else:
                if float(np.dot(info[0], n)) < cos_thresh:
                    info[1] = n
                    info[3] = True
                info[2] += 1

    segs = []
    for (u, v), (n0, n1, count, has_sharp) in edge_dict.items():
        if count != 1 and not has_sharp:
            continue
        p0 = unique_vertices[u]
        p1 = unique_vertices[v]
        length = float(np.linalg.norm(p1 - p0))
        if length < 0.002:
            continue
        segs.append([p0, p1, n0, n1, length])

    segs.sort(key=lambda x: x[4], reverse=True)

    # 4. Collinear line welding
    model_diag = float(np.linalg.norm(cad_max - cad_min))
    main_min_len = max(min_edge_len_m, model_diag * 0.065)
    foot_min_len = max(0.030, model_diag * 0.028)
    foot_zone_z = z_min + (z_max - z_min) * 0.14

    merged_flag = [False] * len(segs)
    welded_edges = []

    for i in range(len(segs)):
        if merged_flag[i]:
            continue
        merged_flag[i] = True
        a = segs[i][0].copy()
        b = segs[i][1].copy()
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-4:
            continue
        d /= length

        grew = True
        while grew:
            grew = False
            for j in range(i + 1, len(segs)):
                if merged_flag[j]:
                    continue
                dj = segs[j][1] - segs[j][0]
                lj = segs[j][4]
                if lj < 1e-4:
                    continue
                dj /= lj
                if abs(float(np.dot(d, dj))) < 0.996:
                    continue
                diff = segs[j][0] - a
                perp = diff - d * float(np.dot(diff, d))
                if float(np.linalg.norm(perp)) > 0.0045:
                    continue
                t0, t1 = 0.0, length
                tj0 = float(np.dot(segs[j][0] - a, d))
                tj1 = float(np.dot(segs[j][1] - a, d))
                if tj0 > tj1:
                    tj0, tj1 = tj1, tj0
                if tj1 >= t0 - 0.015 and tj0 <= t1 + 0.015:
                    new_min = min(t0, tj0)
                    new_max = max(t1, tj1)
                    b = a + d * new_max
                    a = a + d * new_min
                    length = new_max - new_min
                    merged_flag[j] = True
                    grew = True

        is_foot_edge = (min(float(a[2]), float(b[2])) <= foot_zone_z)
        req_len = foot_min_len if is_foot_edge else main_min_len
        if length >= req_len:
            welded_edges.append([a, b, segs[i][2], segs[i][3], length])

    welded_edges.sort(key=lambda x: x[4], reverse=True)

    # 5. 2D Cross-Section Corner Hull filtering
    grouped = [False] * len(welded_edges)
    clean_edges = []
    beam_cluster_radius = model_diag * 0.050

    for i in range(len(welded_edges)):
        if grouped[i]:
            continue
        grouped[i] = True
        seed = welded_edges[i]
        d = (seed[1] - seed[0]) / max(1e-5, seed[4])
        if float(d[0] + d[1] + d[2]) < 0.0:
            d = -d
        origin = 0.5 * (seed[0] + seed[1])
        cluster = [i]

        for j in range(i + 1, len(welded_edges)):
            if grouped[j]:
                continue
            dj = (welded_edges[j][1] - welded_edges[j][0]) / max(1e-5, welded_edges[j][4])
            if abs(float(np.dot(d, dj))) < 0.992:
                continue
            mid_j = 0.5 * (welded_edges[j][0] + welded_edges[j][1])
            diff = mid_j - origin
            axial = abs(float(np.dot(diff, d)))
            perp = diff - d * float(np.dot(diff, d))
            if float(np.linalg.norm(perp)) <= beam_cluster_radius and axial <= 0.60 * (seed[4] + welded_edges[j][4]):
                grouped[j] = True
                cluster.append(j)

        if len(cluster) <= 2:
            for idx in cluster:
                clean_edges.append(welded_edges[idx])
            continue

        ref_up = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(float(d[2])) < 0.85 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        u_axis = np.cross(d, ref_up)
        u_axis /= max(1e-6, float(np.linalg.norm(u_axis)))
        v_axis = np.cross(d, u_axis)
        v_axis /= max(1e-6, float(np.linalg.norm(v_axis)))

        uv_coords = []
        for idx in cluster:
            e = welded_edges[idx]
            mid = 0.5 * (e[0] + e[1]) - origin
            uv_coords.append([float(np.dot(mid, u_axis)), float(np.dot(mid, v_axis))])
        uv_coords = np.array(uv_coords, dtype=np.float32)
        u_min, v_min = uv_coords.min(axis=0)
        u_max, v_max = uv_coords.max(axis=0)
        span_uv = max(float(u_max - u_min), float(v_max - v_min))

        corners_2d = [
            np.array([u_min, v_min], dtype=np.float32),
            np.array([u_max, v_min], dtype=np.float32),
            np.array([u_max, v_max], dtype=np.float32),
            np.array([u_min, v_max], dtype=np.float32),
        ]
        selected_in_cluster = []
        for cor in corners_2d:
            best_c = -1
            best_score = 1e9
            for c_idx, idx in enumerate(cluster):
                dist_corner = float(np.linalg.norm(uv_coords[c_idx] - cor))
                if dist_corner > 0.32 * span_uv + 0.004:
                    continue
                score = dist_corner - 0.015 * welded_edges[idx][4]
                if score < best_score:
                    best_score = score
                    best_c = c_idx
            if best_c >= 0:
                chosen_idx = cluster[best_c]
                dup = False
                for prev_idx in selected_in_cluster:
                    if prev_idx == chosen_idx:
                        dup = True
                        break
                    d_mid = 0.5 * (welded_edges[chosen_idx][0] + welded_edges[chosen_idx][1]) - 0.5 * (welded_edges[prev_idx][0] + welded_edges[prev_idx][1])
                    perp = d_mid - d * float(np.dot(d_mid, d))
                    if float(np.linalg.norm(perp)) < 0.008:
                        dup = True
                        break
                if not dup:
                    selected_in_cluster.append(chosen_idx)

        for idx in selected_in_cluster:
            clean_edges.append(welded_edges[idx])

    if not clean_edges:
        # Fallback: 12 bounding box edges
        x0, y0, z0 = cad_min
        x1, y1, z1 = cad_max
        corners = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
        ]
        pairs = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for u, v in pairs:
            p0 = np.array(corners[u], dtype=np.float32)
            p1 = np.array(corners[v], dtype=np.float32)
            clean_edges.append([p0, p1, np.array([0,0,1], dtype=np.float32), np.array([1,0,0], dtype=np.float32), float(np.linalg.norm(p1 - p0))])

    edge_rows = []
    for a, b, n0, n1, length in clean_edges:
        edge_rows.append([
            float(a[0]), float(a[1]), float(a[2]),
            float(b[0]), float(b[1]), float(b[2]),
            float(n0[0]), float(n0[1]), float(n0[2]),
            float(n1[0]), float(n1[1]), float(n1[2]),
            float(length)
        ])
    cad_feature_edges = np.array(edge_rows, dtype=np.float32)
    return cad_feature_edges, cad_top_poly, cad_foot_centers


def parse_args():
    parser = argparse.ArgumentParser(description="Export Unified Black-Box ONNX Model (Zero STL/BIN/NPY runtime dependencies)")
    parser.add_argument("--cad-name", type=str, default="TABLE-DEMO", help="CAD model name")
    parser.add_argument("--dataset-dir", type=str, default="", help="Path to dataset directory")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to best_hierarchy_model.pth")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory (defaults to deploy_orange_pi_5b/<cad_name>)")
    parser.add_argument("--num-points", type=int, default=1024, help="Number of observation points")
    parser.add_argument("--k", type=int, default=20, help="k-nearest neighbors")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint:
        args.checkpoint = os.path.join(project_root, "deep_learning", "checkpoints", args.cad_name, "best_hierarchy_model.pth")

    if not args.output_dir:
        args.output_dir = os.path.join(project_root, "deploy_orange_pi_5b", args.cad_name)
    elif os.path.basename(os.path.normpath(args.output_dir)) == "deploy_orange_pi_5b":
        args.output_dir = os.path.join(args.output_dir, args.cad_name)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"============================================================")
    print(f" EXPORTING UNIFIED BLACK-BOX ONNX MODEL ({args.cad_name})")
    print(f" Model Checkpoint: {args.checkpoint}")
    print(f" Target Output:    {args.output_dir}")
    print(f"============================================================")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # 1. Load full PyTorch Model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    num_cad_samples = ckpt.get("num_cad_samples", 512)
    desc_dim = ckpt.get("descriptor_dim", 128)

    model = CADHierarchyCorrespondenceModel(
        descriptor_dim=desc_dim,
        patch_k=32,
        obs_k=args.k,
        temperature=ckpt.get("temperature", 0.07)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Loaded PyTorch CADHierarchyCorrespondenceModel successfully.")

    if not args.dataset_dir:
        args.dataset_dir = os.path.join(project_root, "dataset", args.cad_name)

    cad_base_dir = os.path.join(project_root, "CAD_DATA")
    cad_sub_dir = os.path.join(cad_base_dir, args.cad_name)
    active_cad_dir = cad_sub_dir if os.path.isdir(cad_sub_dir) else cad_base_dir

    # 2. Precompute CAD Descriptors (D_cad) and Centroids (P_cad) directly from Canonical Hierarchy
    print("\n[Step 1/4] Precomputing CAD Descriptors & 3D Keypoints...")
    tri_npz_path = os.path.join(active_cad_dir, f"{args.cad_name}_triangles.npz")
    patch_npz_path = os.path.join(active_cad_dir, f"{args.cad_name}_patches.npz")
    sym_json_path = os.path.join(active_cad_dir, f"{args.cad_name}_symmetry.json")

    if not os.path.exists(tri_npz_path) or not os.path.exists(patch_npz_path):
        raise FileNotFoundError(f"Missing canonical hierarchy files ({tri_npz_path}, {patch_npz_path})")

    tri_data = np.load(tri_npz_path)
    all_tri_features = tri_data["features"].astype(np.float32)  # (M, 13)
    all_centroids = all_tri_features[:, :3]                      # (M, 3)
    tri_vertices = tri_data["vertices"].astype(np.float32)       # (M, 3, 3)
    tri_normals = tri_data["normals"].astype(np.float32)         # (M, 3)
    tri_areas = all_tri_features[:, 6].astype(np.float32)        # (M,)

    patch_data = np.load(patch_npz_path)
    all_patch_indices = patch_data["patch_indices"].astype(np.int64)
    all_patch_hops = patch_data["patch_hops"].astype(np.int64)

    symmetry_type = ckpt.get("symmetry_type", "None")
    if os.path.exists(sym_json_path):
        with open(sym_json_path, "r") as f:
            sym_info = json.load(f)
            symmetry_type = sym_info.get("symmetry_type", symmetry_type)

    cache_key_path = os.path.join(active_cad_dir, f"{args.cad_name}_key_triangles_{num_cad_samples}.npy")
    if "cad_key_indices" in ckpt and ckpt["cad_key_indices"] is not None:
        cad_key_indices = np.asarray(ckpt["cad_key_indices"], dtype=np.int64)
    elif os.path.exists(cache_key_path):
        cad_key_indices = np.load(cache_key_path).astype(np.int64)
    else:
        cad_key_indices = fps_sampling_np(all_centroids, num_cad_samples)
        np.save(cache_key_path, cad_key_indices)

    P_cad_np = all_centroids[cad_key_indices].astype(np.float32)
    key_patches = all_patch_indices[cad_key_indices, :32]
    cad_patch_features = all_tri_features[key_patches]
    cad_patch_rel_pos = cad_patch_features[:, :, :3] - P_cad_np[:, None, :]
    cad_patch_hops = all_patch_hops[cad_key_indices, :32]

    cad_patches_t = torch.from_numpy(cad_patch_features).unsqueeze(0).float()
    cad_rel_pos_t = torch.from_numpy(cad_patch_rel_pos).unsqueeze(0).float()
    cad_hops_t = torch.from_numpy(cad_patch_hops).unsqueeze(0)

    with torch.no_grad():
        D_cad = model.encode_cad(cad_patches_t, cad_rel_pos_t, cad_hops_t)[0]
        D_cad_np = D_cad.cpu().numpy().astype(np.float32)

    # Also save sidecar files for legacy compatibility if needed
    cad_npy_path = os.path.join(args.output_dir, f"{args.cad_name}_cad_descriptors.npy")
    cad_bin_path = os.path.join(args.output_dir, f"{args.cad_name}_cad_descriptors.bin")
    cad_centroids_path = os.path.join(args.output_dir, f"{args.cad_name}_key_centroids.npy")
    np.save(cad_npy_path, D_cad_np)
    D_cad_np.tofile(cad_bin_path)
    np.save(cad_centroids_path, P_cad_np)

    # 3. Prepare 10k Surface Points & Normals + Structural Wireframe Edges
    print("\n[Step 2/4] Baking 10k Surface Points, Normals & Structural Wireframe into ONNX...")

    mesh_pts_path = os.path.join(args.output_dir, f"{args.cad_name}_mesh_pts_10k.npy")
    mesh_norms_path = os.path.join(args.output_dir, f"{args.cad_name}_mesh_normals_10k.npy")

    if os.path.exists(mesh_pts_path) and os.path.exists(mesh_norms_path):
        surface_pts = np.load(mesh_pts_path).astype(np.float32)
        surface_normals = np.load(mesh_norms_path).astype(np.float32)
        print(f"  -> Loaded existing 10k surface cloud: {surface_pts.shape}")
    else:
        surface_pts, surface_normals = sample_surface_points_from_triangles(
            tri_vertices, tri_normals, tri_areas, num_samples=10000, seed=42
        )
        np.save(mesh_pts_path, surface_pts)
        np.save(mesh_norms_path, surface_normals)
        print(f"  -> Sampled new 10k surface cloud from mesh triangles: {surface_pts.shape}")

    cad_min = surface_pts.min(axis=0).astype(np.float32)
    cad_max = surface_pts.max(axis=0).astype(np.float32)
    half_w = float((cad_max[0] - cad_min[0]) * 0.5)
    half_l = float((cad_max[1] - cad_min[1]) * 0.5)
    top_z = float(cad_max[2])
    bot_z = float(cad_min[2])

    cad_feature_edges, cad_top_poly, cad_foot_centers = extract_topology_and_edges_from_triangles(
        tri_vertices, tri_normals, cad_min, cad_max, feature_angle_deg=14.0, min_edge_len_m=0.008
    )
    print(f"  -> Extracted {len(cad_feature_edges)} structural feature edges, "
          f"{len(cad_top_poly)} top polygon vertices, {len(cad_foot_centers)} floor feet.")

    sym_map = {"None": 0.0, "NONE": 0.0, "C2": 1.0, "C4": 2.0, "CS": 3.0, "D2": 4.0}
    sym_code = sym_map.get(str(symmetry_type), 0.0)

    # Detect whether this model is a 4-leg rectangular table frame or a general CAD object
    is_table_frame = 1.0 if ("TABLE" in args.cad_name.upper() or (len(cad_foot_centers) == 4 and half_l > 0.25 and half_w > 0.15)) else 0.0
    plate_h = 0.0055 if is_table_frame > 0.5 else 0.0
    beam_h = 0.0455 if is_table_frame > 0.5 else 0.0
    leg_w = 0.0400 if is_table_frame > 0.5 else 0.0
    foot_h = 0.0460 if is_table_frame > 0.5 else 0.0

    cad_meta = np.array([
        float(cad_min[0]), float(cad_min[1]), float(cad_min[2]),
        float(cad_max[0]), float(cad_max[1]), float(cad_max[2]),
        half_w, half_l, top_z, bot_z,
        sym_code, is_table_frame,
        plate_h, beam_h, leg_w, foot_h
    ], dtype=np.float32)

    # 4. Export Unified Black-Box ONNX Model
    print("\n[Step 3/4] Exporting Self-Contained Black-Box ONNX Model...")
    model.obs_encoder.eval()
    blackbox_model = CADBlackBoxONNXModule(
        obs_encoder=model.obs_encoder,
        cad_descriptors=D_cad_np,
        cad_centroids=P_cad_np,
        cad_surface_pts=surface_pts,
        cad_surface_normals=surface_normals,
        cad_feature_edges=cad_feature_edges,
        cad_top_poly=cad_top_poly,
        cad_foot_centers=cad_foot_centers,
        cad_meta=cad_meta,
    )
    blackbox_model.eval()

    dummy_obs = torch.randn(1, args.num_points, 3, dtype=torch.float32)
    output_names = [
        "D_obs",
        "matched_cad_pts",
        "match_scores",
        "cad_descriptors",
        "cad_centroids",
        "cad_surface_pts",
        "cad_surface_normals",
        "cad_feature_edges",
        "cad_top_poly",
        "cad_foot_centers",
        "cad_meta",
    ]

    onnx_path = os.path.join(args.output_dir, "model_obs_npu.onnx")
    named_onnx_path = os.path.join(args.output_dir, f"{args.cad_name}.onnx")
    models_root_dir = os.path.join(project_root, "models")
    os.makedirs(models_root_dir, exist_ok=True)
    global_named_onnx = os.path.join(models_root_dir, f"{args.cad_name}.onnx")

    torch.onnx.export(
        blackbox_model,
        dummy_obs,
        onnx_path,
        input_names=["obs_pts"],
        output_names=output_names,
        opset_version=14,
        dynamo=False
    )

    import shutil
    shutil.copyfile(onnx_path, named_onnx_path)
    shutil.copyfile(onnx_path, global_named_onnx)
    if args.cad_name == "TABLE-DEMO":
        root_deploy_onnx = os.path.join(project_root, "deploy_orange_pi_5b", "model_obs_npu.onnx")
        shutil.copyfile(onnx_path, root_deploy_onnx)

    size_kb = os.path.getsize(onnx_path) / 1024.0
    print(f"  -> Saved Unified Black-Box ONNX: {onnx_path} ({size_kb:.1f} KB)")
    print(f"  -> Saved Named Black-Box ONNX:   {named_onnx_path} ({size_kb:.1f} KB)")
    print(f"  -> Saved in Global Models Store: {global_named_onnx} ({size_kb:.1f} KB)")

    # 4b. Export NPU Static-Graph Backbone for RKNN if needed
    npu_net = RKNNStaticObservationEncoder(model.obs_encoder)
    npu_net.eval()
    obs_dummy = torch.randn(1, 3, args.num_points)
    idx_dummy = torch.randint(0, args.num_points, (args.num_points, args.k), dtype=torch.int64)
    backbone_onnx_path = os.path.join(args.output_dir, "model_obs_backbone.onnx")
    torch.onnx.export(
        npu_net,
        (obs_dummy, idx_dummy),
        backbone_onnx_path,
        input_names=["obs_pts", "idx"],
        output_names=["D_obs"],
        opset_version=14,
        dynamo=False
    )
    print(f"  -> Saved NPU Static-Graph ONNX:  {backbone_onnx_path} ({os.path.getsize(backbone_onnx_path) / 1024:.1f} KB)")

    # 5. Extract calibration point clouds for RKNN INT8 quantization
    print("\n[Step 4/4] Generating calibration samples for INT8 quantization...")
    calib_dir = os.path.join(args.output_dir, "calib_data")
    os.makedirs(calib_dir, exist_ok=True)
    calib_files = []
    rng_calib = np.random.default_rng(123)
    for i in range(20):
        idx_sel = rng_calib.choice(len(surface_pts), size=args.num_points, replace=False)
        obs_pts = surface_pts[idx_sel].copy() + np.array([0.0, 0.0, 1.25], dtype=np.float32)
        sample_path = os.path.join(calib_dir, f"sample_{i:04d}.npy")
        np.save(sample_path, obs_pts)
        calib_files.append(sample_path)

    calib_txt = os.path.join(args.output_dir, "calib_list.txt")
    with open(calib_txt, "w") as f:
        for p in calib_files:
            f.write(p + "\n")

    print("\n============================================================")
    print(" BLACK-BOX ONNX EXPORT COMPLETED SUCCESSFULLY!")
    print(f" Single-File Model Ready: {named_onnx_path}")
    print("============================================================")


if __name__ == "__main__":
    main()
