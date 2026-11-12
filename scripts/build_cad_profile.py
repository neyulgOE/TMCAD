#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TMCAD Universal Profile & 3D Crease Edge Extractor
Tự động trích xuất <CAD_NAME>_profile.json và <CAD_NAME>_crease_edges.npy
từ lưới tam giác (_mesh.ply, _triangles.npz, _adjacency.npz) cho bất kỳ vật thể CAD nào.
"""

import argparse
import json
import os
import numpy as np
import open3d as o3d


def extract_crease_points_from_mesh(mesh: o3d.geometry.TriangleMesh,
                                    dihedral_angle_deg: float = 28.0,
                                    max_points: int = 1200) -> np.ndarray:
    """
    Duyệt qua toàn bộ lưới tam giác để tìm các cạnh sắc (Crease Edges) có góc nhị diện > ngưỡng
    hoặc cạnh biên hở, sau đó lấy mẫu điểm 3D kèm pháp tuyến trung bình phục vụ khớp 2D-3D.
    Trả về mảng (N, 6) gồm [x, y, z, nx, ny, nz].
    """
    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    tri_normals = np.asarray(mesh.triangle_normals, dtype=np.float64)

    if len(triangles) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # Xây dựng từ điển cạnh (v_min, v_max) -> danh sách chỉ số tam giác kề nhau
    edge_to_tris = {}
    for t_idx, tri in enumerate(triangles):
        edges = [
            (min(tri[0], tri[1]), max(tri[0], tri[1])),
            (min(tri[1], tri[2]), max(tri[1], tri[2])),
            (min(tri[2], tri[0]), max(tri[2], tri[0])),
        ]
        for e in edges:
            edge_to_tris.setdefault(e, []).append(t_idx)

    cos_thresh = np.cos(np.deg2rad(dihedral_angle_deg))
    crease_segments = []
    crease_normals = []
    crease_lengths = []

    for (v0, v1), t_list in edge_to_tris.items():
        is_crease = False
        avg_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if len(t_list) == 1:
            # Cạnh biên hở (boundary edge)
            is_crease = True
            avg_n = tri_normals[t_list[0]]
        elif len(t_list) >= 2:
            n1 = tri_normals[t_list[0]]
            n2 = tri_normals[t_list[1]]
            dot_val = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
            if dot_val < cos_thresh:
                is_crease = True
                avg_n = n1 + n2
                norm_val = np.linalg.norm(avg_n)
                if norm_val > 1e-8:
                    avg_n /= norm_val
                else:
                    avg_n = n1

        if is_crease:
            p0, p1 = vertices[v0], vertices[v1]
            seg_len = float(np.linalg.norm(p1 - p0))
            if seg_len > 1e-5:
                crease_segments.append((p0, p1))
                crease_normals.append(avg_n)
                crease_lengths.append(seg_len)

    # Nếu vật thể trơn hoàn toàn (như quả cầu/chai trơn ít góc nhị diện > 28 độ),
    # hạ ngưỡng hoặc lấy mẫu các điểm có độ cong cao trên bề mặt
    if len(crease_segments) == 0:
        pcd = mesh.sample_points_uniformly(number_of_points=max_points)
        pts = np.asarray(pcd.points, dtype=np.float32)
        nrms = np.asarray(pcd.normals, dtype=np.float32)
        return np.hstack([pts, nrms])

    crease_lengths = np.asarray(crease_lengths, dtype=np.float64)
    probs = crease_lengths / np.sum(crease_lengths)
    chosen_indices = np.random.choice(len(crease_segments), size=max_points, replace=True, p=probs)
    t_rand = np.random.uniform(0.0, 1.0, size=(max_points, 1))

    sampled_pts = []
    for idx, alpha in zip(chosen_indices, t_rand):
        p0, p1 = crease_segments[idx]
        pt = (1.0 - alpha[0]) * p0 + alpha[0] * p1
        nrm = crease_normals[idx]
        sampled_pts.append(np.concatenate([pt, nrm]))

    return np.asarray(sampled_pts, dtype=np.float32)


def build_cad_profile(cad_dir: str, cad_name: str, force_floor_standing: str = "auto") -> dict:
    """
    Phân tích tự động hình học từ lưới tam giác CAD và xuất <CAD_NAME>_profile.json.
    Đảm bảo giữ nguyên 100% thông số chuẩn cho TABLE-DEMO và tự động cấu hình cho vật thể mới.
    """
    mesh_path = os.path.join(cad_dir, f"{cad_name}_mesh.ply")
    if not os.path.exists(mesh_path):
        for ext in [".stl", ".obj", ".ply"]:
            cand = os.path.join(cad_dir, f"{cad_name}{ext}")
            if os.path.exists(cand):
                mesh_path = cand
                break

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Không tìm thấy tệp mesh cho {cad_name} trong {cad_dir}")

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int32)

    # Quy đổi đơn vị tự động nếu tệp gốc đang ở mm (đường kính > 10.0)
    unit_file = os.path.join(cad_dir, f"{cad_name}.unit")
    unit_scale = 1.0
    if os.path.exists(unit_file):
        with open(unit_file, "r", encoding="utf-8") as f:
            u_str = f.read().strip().lower()
            if u_str == "mm":
                unit_scale = 0.001
            elif u_str == "cm":
                unit_scale = 0.01

    raw_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    if not os.path.exists(mesh_path.endswith("_mesh.ply") and mesh_path or "") and raw_diag > 10.0 and unit_scale == 1.0:
        unit_scale = 0.001

    if unit_scale != 1.0 and not mesh_path.endswith("_mesh.ply"):
        mesh.scale(unit_scale, center=(0.0, 0.0, 0.0))
        verts = np.asarray(mesh.vertices, dtype=np.float64)

    cad_min = verts.min(axis=0)
    cad_max = verts.max(axis=0)
    extents = cad_max - cad_min
    diag = float(np.linalg.norm(extents))

    # Tương thích ngược với table_dimensions.json nếu có sẵn
    legacy_table_dims_path = os.path.join(cad_dir, "table_dimensions.json")
    z_bot = float(cad_min[2])
    z_top = float(cad_max[2])

    if "TABLE" in cad_name.upper() and os.path.exists(legacy_table_dims_path):
        try:
            with open(legacy_table_dims_path, "r", encoding="utf-8") as f:
                tdims = json.load(f)
            z_top = float(tdims.get("tabletop_center_offset_z_m", z_top))
            z_bot = float(tdims.get("ground_contact_offset_z_m", z_bot))
        except Exception:
            pass

    # Tính diện tích bề mặt và ước lượng độ rỗng (hollowness) từ tam giác
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    tri_cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(tri_cross, axis=1)
    total_surface_area = float(np.sum(tri_areas))
    bbox_volume = float(max(extents[0] * extents[1] * extents[2], 1e-9))

    # Ước lượng thể tích ký hiệu (Signed Volume) của mesh
    signed_vol = float(np.abs(np.sum(np.einsum("ij,ij->i", v0, tri_cross))) / 6.0)
    fill_ratio = float(np.clip(signed_vol / bbox_volume, 0.0, 1.0))

    # Tự động phân loại hình thái vật thể
    is_table_or_frame = ("TABLE" in cad_name.upper()) or (fill_ratio < 0.16 and diag > 0.45)
    if force_floor_standing == "true":
        placement_mode = "FLOOR_STANDING"
    elif force_floor_standing == "false":
        placement_mode = "FREE_6DOF"
    else:
        placement_mode = "FLOOR_STANDING" if is_table_or_frame else "FREE_6DOF"

    topology_type = "FRAME_HOLLOW" if is_table_or_frame else "COMPLEX_SOLID"

    # Trích xuất và lưu đám mây điểm gờ sắc 3D (Crease Edges)
    crease_pts = extract_crease_points_from_mesh(mesh, dihedral_angle_deg=28.0, max_points=1200)
    crease_path = os.path.join(cad_dir, f"{cad_name}_crease_edges.npy")
    np.save(crease_path, crease_pts)

    # Tham số thích ứng theo tỉ lệ (Scale-Adaptive Hyperparameters)
    voxel_size_m = float(np.clip(0.0055 * diag, 0.001, 0.008))
    ransac_thresh_m = float(np.clip(0.035 * diag, 0.004, 0.045))
    icp_coarse_m = float(np.clip(0.045 * diag, 0.006, 0.055))
    icp_fine_m = float(np.clip(0.018 * diag, 0.0025, 0.025))

    profile = {
        "cad_name": cad_name,
        "topology_type": topology_type,
        "placement_mode": placement_mode,
        "real_bounding_diag_m": round(diag, 6),
        "extents_xyz_m": [round(float(x), 6) for x in extents],
        "cad_min_xyz_m": [round(float(cad_min[0]), 6), round(float(cad_min[1]), 6), round(z_bot, 6)],
        "cad_max_xyz_m": [round(float(cad_max[0]), 6), round(float(cad_max[1]), 6), round(z_top, 6)],
        "top_offset_z_m": round(z_top, 6),
        "bottom_offset_z_m": round(z_bot, 6),
        "surface_area_m2": round(total_surface_area, 6),
        "bbox_fill_ratio": round(fill_ratio, 4),
        "num_triangles": int(len(tris)),
        "num_crease_points": int(len(crease_pts)),
        "adaptive_solver_params": {
            "voxel_size_m": round(voxel_size_m, 5),
            "ransac_thresh_m": round(ransac_thresh_m, 5),
            "icp_coarse_corr_m": round(icp_coarse_m, 5),
            "icp_fine_corr_m": round(icp_fine_m, 5),
            "cluster_merge_dx_m": round(float(max(extents[0], extents[1]) * 1.08), 4),
            "cluster_merge_dz_m": round(float(min(extents[0], extents[1]) * 1.25), 4),
            "enable_2d_frame_anchor": bool(is_table_or_frame),
            "enable_2d_crease_refinement": True
        }
    }

    profile_path = os.path.join(cad_dir, f"{cad_name}_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print(f"[build_cad_profile] Đã lưu hồ sơ vật thể tại: {profile_path}")
    print(f"[build_cad_profile] Đã lưu {len(crease_pts)} điểm gờ 3D tại: {crease_path}")
    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xây dựng hồ sơ hình học tự động cho mô hình CAD")
    parser.add_argument("--cad-dir", type=str, default="CAD_DATA", help="Thư mục chứa dữ liệu CAD")
    parser.add_argument("--cad-name", type=str, default="TABLE-DEMO", help="Tên mô hình CAD")
    parser.add_argument("--floor-standing", type=str, default="auto", choices=["auto", "true", "false"])
    args = parser.parse_args()

    cad_target_dir = os.path.join(args.cad_dir, args.cad_name) if not os.path.exists(os.path.join(args.cad_dir, f"{args.cad_name}_mesh.ply")) and os.path.isdir(os.path.join(args.cad_dir, args.cad_name)) else args.cad_dir
    build_cad_profile(cad_target_dir, args.cad_name, args.floor_standing)

