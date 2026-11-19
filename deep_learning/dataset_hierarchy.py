#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Dataset cho TMCAD Single-CAD Hierarchy:
- Tận dụng trực tiếp đặc trưng tam giác (_triangles.npz, _patches.npz, _regions.npz).
- Lấy mẫu ưu tiên độ cong & cạnh sắc tam giác (Curvature-Biased Sampling) cho chi tiết máy.
- Tự động tiêm nhiễu thực tế Sim-to-Real & Che khuất một phần (15%-35% Partial Occlusion)
  để mô hình hoạt động bền bỉ khi vật thể bị che khuất hoặc mất depth trên camera thực.
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CADHierarchyDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        cad_data_dir: str,
        cad_name: str,
        split: str = "train",
        num_obs_points: int = 1024,
        num_cad_samples: int = 512,
        enable_sim2real_noise: bool = True,
        max_occlusion_ratio: float = 0.32,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.cad_data_dir = cad_data_dir
        self.cad_name = cad_name
        self.split = split
        self.num_obs_points = num_obs_points
        self.num_cad_samples = int(num_cad_samples)
        self.is_train = split == "train"
        self.enable_sim2real_noise = enable_sim2real_noise and self.is_train
        self.max_occlusion_ratio = float(np.clip(max_occlusion_ratio, 0.0, 0.45))

        # 1. Nạp dữ liệu phân tầng Tam giác CAD (Phase 1 Canonical Representation)
        self._load_canonical_cad_hierarchy()

        # 2. Nạp danh sách các mẫu từ train.txt / val.txt
        split_file = os.path.join(dataset_root, cad_name, f"{split}.txt")
        self.sample_dirs: List[str] = []
        if os.path.exists(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            for rel_path in lines:
                cand = os.path.join(dataset_root, cad_name, rel_path)
                if not os.path.exists(cand):
                    cand = os.path.join(dataset_root, rel_path)
                if os.path.exists(cand):
                    self.sample_dirs.append(cand)
        else:
            # Quét trực tiếp thư mục con nếu không có file split
            sub_dir = os.path.join(dataset_root, cad_name, split)
            if os.path.exists(sub_dir):
                for name in sorted(os.listdir(sub_dir)):
                    full_p = os.path.join(sub_dir, name)
                    if os.path.isdir(full_p):
                        self.sample_dirs.append(full_p)

    def _load_canonical_cad_hierarchy(self) -> None:
        tri_path = os.path.join(self.cad_data_dir, f"{self.cad_name}_triangles.npz")
        patch_path = os.path.join(self.cad_data_dir, f"{self.cad_name}_patches.npz")
        sym_path = os.path.join(self.cad_data_dir, f"{self.cad_name}_symmetry.json")
        adj_path = os.path.join(self.cad_data_dir, f"{self.cad_name}_adjacency.npz")

        if not os.path.exists(tri_path) or not os.path.exists(patch_path):
            raise FileNotFoundError(
                f"Thiếu tệp phân tầng CAD cho {self.cad_name} tại {self.cad_data_dir}. "
                f"Hãy chạy Build CAD Hierarchy (Phase 1) trước."
            )

        tri_npz = np.load(tri_path)
        patch_npz = np.load(patch_path)

        # Tier 1: Đặc trưng tam giác 13 chiều [c(3), n(3), area(1), lengths(3), angles(3)]
        if "features" in tri_npz:
            self.tri_features = tri_npz["features"].astype(np.float32)
        else:
            centers = tri_npz["centers"].astype(np.float32)
            normals = tri_npz["normals"].astype(np.float32)
            areas = tri_npz["areas"].astype(np.float32).reshape(-1, 1)
            edge_lengths = tri_npz.get("edge_lengths", np.zeros((len(centers), 3), dtype=np.float32))
            angles = tri_npz.get("angles", np.zeros((len(centers), 3), dtype=np.float32))
            self.tri_features = np.concatenate([centers, normals, areas, edge_lengths, angles], axis=1)

        self.tri_centers = self.tri_features[:, 0:3]
        self.tri_normals = self.tri_features[:, 3:6]
        self.tri_areas = self.tri_features[:, 6]

        # Đường kính chuẩn hóa thực của CAD từ tâm các tam giác
        c_min = self.tri_centers.min(axis=0)
        c_max = self.tri_centers.max(axis=0)
        self.cad_diag = float(max(np.linalg.norm(c_max - c_min), 1e-4))

        # Tính điểm độ cong / gờ sắc của từng tam giác từ đồ thị kề _adjacency.npz
        self.tri_sharpness = np.ones(len(self.tri_centers), dtype=np.float32)
        if os.path.exists(adj_path):
            adj_npz = np.load(adj_path)
            if "edges" in adj_npz and "edge_features" in adj_npz:
                edges = adj_npz["edges"]
                ef = adj_npz["edge_features"]
                # Góc nhị diện lớn ->(1 - cos(theta)) lớn -> ưu tiên lấy mẫu cao hơn
                dihedral_weight = np.abs(ef[:, 0]) if ef.shape[1] > 0 else np.ones(len(edges))
                np.add.at(self.tri_sharpness, edges[:, 0], dihedral_weight)
                np.add.at(self.tri_sharpness, edges[:, 1], dihedral_weight)

        # Tier 3: LocalPatchBundle (M_key, K=32, 13)
        M_total = len(self.tri_features)
        key_path = os.path.join(self.cad_data_dir, f"{self.cad_name}_key_triangles_{self.num_cad_samples}.npy")
        if os.path.exists(key_path):
            key_idx = np.load(key_path).astype(np.int64)
            key_idx = np.clip(key_idx, 0, M_total - 1)
        elif M_total >= self.num_cad_samples:
            key_idx = np.linspace(0, M_total - 1, self.num_cad_samples, dtype=np.int64)
        else:
            rng = np.random.default_rng(1337)
            key_idx = rng.choice(M_total, size=self.num_cad_samples, replace=True).astype(np.int64)

        self.cad_key_indices = key_idx

        if "patch_features" in patch_npz:
            pf = patch_npz["patch_features"].astype(np.float32)
            self.patch_features = pf[key_idx] if len(pf) != len(key_idx) else pf
        elif "patch_triangle_indices" in patch_npz or "patch_indices" in patch_npz:
            idx_key = "patch_triangle_indices" if "patch_triangle_indices" in patch_npz else "patch_indices"
            p_idx = patch_npz[idx_key].astype(np.int64)
            if len(p_idx) != len(key_idx):
                p_idx = p_idx[key_idx]
            self.patch_features = self.tri_features[p_idx]
        else:
            raise KeyError("Tệp _patches.npz không chứa 'patch_features' hoặc 'patch_triangle_indices'.")

        self.patch_centers = self.patch_features[:, 0, 0:3]
        self.patch_normals = self.patch_features[:, 0, 3:6]

        if "patch_hops" in patch_npz:
            ph = patch_npz["patch_hops"].astype(np.float32)
            self.patch_hops = ph[key_idx] if len(ph) != len(key_idx) else ph
        else:
            self.patch_hops = np.zeros(self.patch_features.shape[:2], dtype=np.float32)

        # Nhóm đối xứng (Phase 0)
        self.symmetry_matrices = [np.eye(3, dtype=np.float32)]
        if os.path.exists(sym_path):
            with open(sym_path, "r", encoding="utf-8") as f:
                sym_data = json.load(f)
            if "rotation_matrices" in sym_data and len(sym_data["rotation_matrices"]) > 0:
                self.symmetry_matrices = [
                    np.asarray(m, dtype=np.float32).reshape(3, 3) for m in sym_data["rotation_matrices"]
                ]

        self.symmetry_matrices_tensor = torch.from_numpy(np.stack(self.symmetry_matrices, axis=0))

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def _apply_sim2real_occlusion_and_noise(
        self,
        pts_cam: np.ndarray,
        cad_pts: np.ndarray,
        patch_ids: np.ndarray,
        R_gt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Mô phỏng nhiễu thực tế Sim-to-Real & Che khuất cục bộ (15% - 35%):
        1. Grazing-angle dropout dựa trên pháp tuyến của tam giác CAD.
        2. Khoét mảng che khuất cục bộ (Partial Patch Occlusion) nhưng đảm bảo giữ lại >= 68% điểm.
        3. Nhiễu độ sâu phụ thuộc bình phương khoảng cách (Axial + Lateral Sensor Noise) & Lượng tử hóa.
        4. Điểm nhiễu rìa (Outlier/Clutter points) có nhãn inlier_mask = 0.
        """
        N = len(pts_cam)
        inlier_mask = np.ones(N, dtype=np.float32)
        if N < 64:
            return pts_cam, cad_pts, patch_ids, inlier_mask

        keep_mask = np.ones(N, dtype=bool)

        # 1. Mất depth ở góc nghiêng lớn (Grazing-angle / Specular Dropout) từ pháp tuyến tam giác
        valid_patch = np.clip(patch_ids, 0, len(self.patch_normals) - 1)
        normals_cad = self.patch_normals[valid_patch]
        normals_cam = (R_gt @ normals_cad.T).T
        view_dirs = pts_cam / (np.linalg.norm(pts_cam, axis=1, keepdims=True) + 1e-8)
        cos_incidence = np.abs(np.sum(normals_cam * view_dirs, axis=1))

        # Các mặt tam giác nghiêng > 68 độ so với tia nhìn (cos < 0.37) có xác suất rớt depth
        grazing_prob = np.where(cos_incidence < 0.37, 0.55, 0.0)
        grazing_drop = np.random.rand(N) < grazing_prob
        if np.sum(~grazing_drop) >= int(0.75 * N):
            keep_mask &= ~grazing_drop

        # 2. Che khuất cục bộ 1 phần (Structured Partial Occlusion: 15% - 32% diện tích)
        if np.random.rand() < 0.80:
            occ_ratio = np.random.uniform(0.12, self.max_occlusion_ratio)
            anchor_idx = np.random.randint(0, N)
            anchor_pt = pts_cam[anchor_idx]
            dists = np.linalg.norm(pts_cam - anchor_pt, axis=1)
            occ_thresh = np.quantile(dists, occ_ratio)
            occ_drop = dists <= occ_thresh
            if np.sum(keep_mask & (~occ_drop)) >= int(0.65 * N):
                keep_mask &= ~occ_drop

        pts_cam = pts_cam[keep_mask]
        cad_pts = cad_pts[keep_mask]
        patch_ids = patch_ids[keep_mask]
        inlier_mask = np.ones(len(pts_cam), dtype=np.float32)

        # 3. Cộng nhiễu cảm biến chiều sâu RealSense (Axial Z^2 noise + Step quantization)
        z_vals = np.clip(pts_cam[:, 2:3], 0.3, 3.5)
        sigma_z = 0.0015 * (z_vals ** 2)
        sigma_xy = 0.0007 * z_vals
        noise_xyz = np.concatenate(
            [
                np.random.normal(0.0, sigma_xy, size=(len(pts_cam), 2)),
                np.random.normal(0.0, sigma_z, size=(len(pts_cam), 1)),
            ],
            axis=1,
        ).astype(np.float32)
        pts_cam = pts_cam + noise_xyz
        # Lượng tử hóa bước nhảy 1.0 mm
        pts_cam[:, 2] = np.round(pts_cam[:, 2] * 1000.0) / 1000.0

        # 4. Tiêm 3% - 6% điểm nhiễu ngoại lai (Clutter / Outlier) ở vùng biên với inlier_mask = 0
        if np.random.rand() < 0.50 and len(pts_cam) > 32:
            num_outliers = max(4, int(0.04 * len(pts_cam)))
            rand_idx = np.random.choice(len(pts_cam), size=num_outliers, replace=True)
            outlier_pts = pts_cam[rand_idx] + np.random.uniform(
                -0.08 * self.cad_diag, 0.08 * self.cad_diag, size=(num_outliers, 3)
            ).astype(np.float32)
            pts_cam = np.vstack([pts_cam, outlier_pts])
            cad_pts = np.vstack([cad_pts, cad_pts[rand_idx]])
            patch_ids = np.concatenate([patch_ids, patch_ids[rand_idx]])
            inlier_mask = np.concatenate([inlier_mask, np.zeros(num_outliers, dtype=np.float32)])

        return pts_cam, cad_pts, patch_ids, inlier_mask

    def _curvature_biased_sample(
        self,
        pts_cam: np.ndarray,
        cad_pts: np.ndarray,
        patch_ids: np.ndarray,
        inlier_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Lấy mẫu 1.024 điểm ưu tiên các vùng có độ cong cao / cạnh sắc tam giác (50%)
        kết hợp lấy mẫu đều trên bề mặt (50%) để giữ trọn chi tiết cơ khí.
        """
        N = len(pts_cam)
        if N == 0:
            zeros_pts = np.zeros((self.num_obs_points, 3), dtype=np.float32)
            zeros_id = np.zeros(self.num_obs_points, dtype=np.int64)
            zeros_mask = np.zeros(self.num_obs_points, dtype=np.float32)
            return zeros_pts, zeros_pts, zeros_id, zeros_mask

        # Tính trọng số độ sắc của từng điểm dựa trên patch/triangle tương ứng
        valid_pid = np.clip(patch_ids, 0, len(self.patch_centers) - 1)
        # Ước lượng độ biến thiên pháp tuyến trong từng LocalPatchBundle (K=32 tam giác)
        patch_tri_normals = self.patch_features[valid_pid, :, 3:6]  # (N, 32, 3)
        normal_var = np.var(patch_tri_normals, axis=1).sum(axis=1)  # (N,)
        weights = 1.0 + 5.0 * normal_var
        weights /= np.sum(weights)

        half_k = self.num_obs_points // 2
        rem_k = self.num_obs_points - half_k

        replace_flag = N < self.num_obs_points
        idx_sharp = np.random.choice(N, size=half_k, replace=replace_flag, p=weights)
        idx_unif = np.random.choice(N, size=rem_k, replace=replace_flag)
        chosen = np.concatenate([idx_sharp, idx_unif], axis=0)
        np.random.shuffle(chosen)

        return (
            pts_cam[chosen],
            cad_pts[chosen],
            patch_ids[chosen],
            inlier_mask[chosen],
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        pose_path = os.path.join(sample_dir, "pose.json")
        corr_path = os.path.join(sample_dir, "correspondences.csv")

        with open(pose_path, "r", encoding="utf-8") as f:
            pose_meta = json.load(f)

        T_gt = np.asarray(pose_meta.get("cameraFromObject", np.eye(4)), dtype=np.float32).reshape(4, 4)
        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]

        if os.path.exists(corr_path):
            df = pd.read_csv(corr_path)
            obs_cols = ["obs_x", "obs_y", "obs_z"] if "obs_x" in df.columns else ["cam_x", "cam_y", "cam_z"]
            pts_cam = df[obs_cols].values.astype(np.float32)
            if "cad_x" in df.columns:
                cad_pts = df[["cad_x", "cad_y", "cad_z"]].values.astype(np.float32)
            else:
                cad_pts = ((pts_cam - t_gt) @ R_gt).astype(np.float32)

            if "patch_id" in df.columns:
                patch_ids = df["patch_id"].values.astype(np.int64)
            else:
                # Ánh xạ từng điểm CAD về tâm LocalPatchBundle gần nhất
                dists = np.linalg.norm(
                    cad_pts[:, None, :] - self.patch_centers[None, :, :], axis=-1
                )
                patch_ids = np.argmin(dists, axis=1).astype(np.int64)
        else:
            # Dự phòng giải chiếu trực tiếp từ depth.png nếu không có correspondences.csv
            depth_path = os.path.join(sample_dir, "depth.png")
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001
            intr = pose_meta.get("intrinsics", {"fx": 608.0, "fy": 608.0, "cx": 320.0, "cy": 240.0})
            ys, xs = np.where((depth > 0.2) & (depth < 3.0))
            zs = depth[ys, xs]
            x3d = (xs - intr["cx"]) * zs / intr["fx"]
            y3d = (ys - intr["cy"]) * zs / intr["fy"]
            pts_cam = np.stack([x3d, y3d, zs], axis=1).astype(np.float32)
            cad_pts = ((pts_cam - t_gt) @ R_gt).astype(np.float32)
            dists = np.linalg.norm(cad_pts[:, None, :] - self.patch_centers[None, :, :], axis=-1)
            patch_ids = np.argmin(dists, axis=1).astype(np.int64)

        inlier_mask = np.ones(len(pts_cam), dtype=np.float32)

        # Áp dụng nhiễu thực tế Sim-to-Real & Che khuất một phần khi huấn luyện
        if self.enable_sim2real_noise:
            pts_cam, cad_pts, patch_ids, inlier_mask = self._apply_sim2real_occlusion_and_noise(
                pts_cam, cad_pts, patch_ids, R_gt
            )

        # Lấy mẫu ưu tiên độ cong về 1024 điểm
        pts_cam, cad_pts, patch_ids, inlier_mask = self._curvature_biased_sample(
            pts_cam, cad_pts, patch_ids, inlier_mask
        )

        # Chuẩn hóa đám mây điểm quan sát về tâm và chia cho đường kính thực của CAD
        centroid = np.mean(pts_cam, axis=0, keepdims=True)
        pts_cam_norm = (pts_cam - centroid) / self.cad_diag

        return {
            "obs_points_raw": torch.from_numpy(pts_cam),
            "obs_points_norm": torch.from_numpy(pts_cam_norm),
            "gt_cad_points": torch.from_numpy(cad_pts),
            "gt_patch_ids": torch.from_numpy(patch_ids),
            "inlier_mask": torch.from_numpy(inlier_mask),
            "patch_features": torch.from_numpy(self.patch_features),
            "patch_centers": torch.from_numpy(self.patch_centers),
            "patch_normals": torch.from_numpy(self.patch_normals),
            "symmetry_matrices": self.symmetry_matrices_tensor,
            "R_gt": torch.from_numpy(R_gt),
            "t_gt": torch.from_numpy(t_gt),
        }


def _load_mesh_triangles(mesh_path: str):
    """Đọc đỉnh (V, 3) và mặt tam giác (F, 3) từ _mesh.ply khi _triangles.npz không có vertices."""
    if not os.path.isfile(mesh_path):
        return None, None
    try:
        import trimesh
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if hasattr(mesh, "vertices") and hasattr(mesh, "faces") and len(mesh.faces) > 0:
            return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64)
    except Exception:
        pass
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if len(mesh.triangles) > 0:
            return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.triangles, dtype=np.int64)
    except Exception:
        pass
    return None, None


def _random_so3_matrix(rng: np.random.Generator) -> np.ndarray:
    """Sinh ma trận xoay đều trên SO(3) (Shoemake uniform quaternion)."""
    u1, u2, u3 = rng.random(3)
    q0 = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    q1 = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    q2 = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    q3 = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    x, y, z, w = q0, q1, q2, q3
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def fps_sampling_np(pts: np.ndarray, num_samples: int, seed: int = 42) -> np.ndarray:
    """Farthest Point Sampling 3D trên tập tâm tam giác (có lọc ứng viên khi M > 20k)."""
    n = len(pts)
    if n <= num_samples:
        return np.arange(n, dtype=np.int64)
    if n > 20000:
        rng = np.random.default_rng(seed)
        cand_idx = rng.choice(n, size=20000, replace=False)
        sub = pts[cand_idx].astype(np.float32)
    else:
        cand_idx = np.arange(n, dtype=np.int64)
        sub = pts.astype(np.float32)

    m = len(sub)
    selected = np.zeros(num_samples, dtype=np.int64)
    dists = np.full(m, np.inf, dtype=np.float32)
    farthest = 0
    for i in range(num_samples):
        selected[i] = farthest
        diff = sub - sub[farthest]
        d = np.einsum("ij,ij->i", diff, diff)
        dists = np.minimum(dists, d)
        farthest = int(np.argmax(dists))
    return cand_idx[selected].astype(np.int64)


def _sphere_view_with_roll(az: float, el: float, roll_rad: float, dist: float,
                           target: np.ndarray, cad_center: np.ndarray):
    """Sinh pose camera nhìn vào target từ (az, el) trên mặt cầu kèm góc Roll giới hạn quanh trục Z camera."""
    cos_el = math.cos(el)
    cam_dir = np.array([cos_el * math.cos(az), cos_el * math.sin(az), math.sin(el)], dtype=np.float32)
    cam_pos = cad_center + dist * cam_dir
    R0, _ = _look_at_camera_from_object(cam_pos, target, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    if abs(roll_rad) > 1e-6:
        c, s = math.cos(roll_rad), math.sin(roll_rad)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        R = (Rz @ R0).astype(np.float32)
    else:
        R = R0
    t = (-R @ cam_pos).astype(np.float32)
    return R, t


def _look_at_camera_from_object(cam_pos: np.ndarray, target: np.ndarray, up_hint: np.ndarray):
    """Trả về (R_cam_obj, t_cam_obj) chuẩn OpenCV (+X phải, +Y xuống, +Z nhìn vào vật)."""
    z_cam = target - cam_pos
    z_cam = z_cam / (np.linalg.norm(z_cam) + 1e-8)
    x_cam = np.cross(z_cam, up_hint)
    if np.linalg.norm(x_cam) < 1e-5:
        x_cam = np.cross(z_cam, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    x_cam = x_cam / (np.linalg.norm(x_cam) + 1e-8)
    y_cam = np.cross(z_cam, x_cam)
    R_cam_obj = np.stack([x_cam, y_cam, z_cam], axis=0).astype(np.float32)
    t_cam_obj = (-R_cam_obj @ cam_pos).astype(np.float32)
    return R_cam_obj, t_cam_obj


class OnTheFlyTriangleDataset(Dataset):
    """
    Dataset sinh mẫu trực tiếp trên lưới tam giác trong RAM (0 MB ổ cứng):
      - Barycentric Sampling theo trọng số diện tích (khử Tessellation Bias)
      - Backface Culling + Z-buffer 2D vectorized (loại bỏ vách khuất ruột engine_block)
      - Sim-to-Real Dropout (góc nghiêng > 68°, che khuất khối cầu 15-35%, nhiễu Z^2, clutter)
      - Chuẩn hóa (pts - centroid_3d) / real_bounding_diag_m khớp 100% với C++
    """

    def __init__(
        self,
        cad_data_dir: str,
        cad_name: str,
        split: str = "train",
        samples_per_epoch: int = 2400,
        num_obs_points: int = 1024,
        num_cad_samples: int = 256,
        enable_sim2real_noise: bool = True,
        max_occlusion_ratio: float = 0.35,
        zbuffer_grid: int = 64,
    ):
        super().__init__()
        self.cad_data_dir = cad_data_dir
        self.cad_name = cad_name
        self.split = split
        self.samples_per_epoch = int(samples_per_epoch if split == "train" else min(samples_per_epoch, 256))
        self.num_obs_points = int(num_obs_points)
        self.num_cad_samples = int(num_cad_samples)
        self.enable_sim2real_noise = bool(enable_sim2real_noise and split == "train")
        self.max_occlusion_ratio = float(max_occlusion_ratio)
        self.zbuffer_grid = int(zbuffer_grid)

        self._load_canonical_cad_assets()
        self._build_fixed_val_views()

    def _resolve_file(self, suffix: str) -> str:
        cands = [
            os.path.join(self.cad_data_dir, f"{self.cad_name}{suffix}"),
            os.path.join(self.cad_data_dir, self.cad_name, f"{self.cad_name}{suffix}"),
        ]
        for p in cands:
            if os.path.isfile(p):
                return p
        return cands[0]

    def _load_canonical_cad_assets(self):
        tri_npz = np.load(self._resolve_file("_triangles.npz"))
        patch_npz = np.load(self._resolve_file("_patches.npz"))

        if "features" in tri_npz:
            self.tri_features = tri_npz["features"].astype(np.float32)
        else:
            centers = tri_npz["centers"].astype(np.float32)
            normals = tri_npz["normals"].astype(np.float32)
            areas = tri_npz["areas"].astype(np.float32).reshape(-1, 1)
            extra = np.zeros((len(centers), 6), dtype=np.float32)
            self.tri_features = np.concatenate([centers, normals, areas, extra], axis=-1)

        self.tri_centers = tri_npz["centers"].astype(np.float32) if "centers" in tri_npz else self.tri_features[:, 0:3]
        self.tri_normals = tri_npz["normals"].astype(np.float32) if "normals" in tri_npz else self.tri_features[:, 3:6]
        n_norm = np.linalg.norm(self.tri_normals, axis=1, keepdims=True) + 1e-8
        self.tri_normals = self.tri_normals / n_norm

        if "vertices" in tri_npz and tri_npz["vertices"].ndim == 3 and tri_npz["vertices"].shape[1:] == (3, 3):
            verts3 = tri_npz["vertices"].astype(np.float32)
            self.v0 = verts3[:, 0]
            self.v1 = verts3[:, 1]
            self.v2 = verts3[:, 2]
            self.has_exact_mesh = True
        else:
            verts, faces = _load_mesh_triangles(self._resolve_file("_mesh.ply"))
            if verts is not None and faces is not None and len(faces) == len(self.tri_centers):
                self.v0 = verts[faces[:, 0]]
                self.v1 = verts[faces[:, 1]]
                self.v2 = verts[faces[:, 2]]
                self.has_exact_mesh = True
            else:
                self.v0 = self.v1 = self.v2 = None
                self.has_exact_mesh = False

        if "areas" in tri_npz:
            areas = tri_npz["areas"].astype(np.float64).reshape(-1)
        elif self.has_exact_mesh:
            cross_v = np.cross(self.v1 - self.v0, self.v2 - self.v0)
            areas = 0.5 * np.linalg.norm(cross_v, axis=1).astype(np.float64)
        else:
            areas = np.ones(len(self.tri_centers), dtype=np.float64)
        areas = np.clip(areas, 1e-10, None)
        self.tri_area_probs = (areas / np.sum(areas)).astype(np.float64)

        reg_path = self._resolve_file("_regions.npz")
        if os.path.isfile(reg_path):
            reg_npz = np.load(reg_path)
            reg_key = "region_ids" if "region_ids" in reg_npz else (
                "triangle_regions" if "triangle_regions" in reg_npz else list(reg_npz.keys())[0]
            )
            self.tri_regions = reg_npz[reg_key].astype(np.int64).reshape(-1)
        else:
            self.tri_regions = np.zeros(len(self.tri_centers), dtype=np.int64)

        prof_path = self._resolve_file("_profile.json")
        if os.path.isfile(prof_path):
            with open(prof_path, "r", encoding="utf-8") as f:
                prof = json.load(f)
            self.is_floor_standing = (prof.get("placement_mode", "FREE_6DOF") == "FLOOR_STANDING")
            self.real_bounding_diag_m = float(prof.get("real_bounding_diag_m", 0.0))
            bmin_raw = prof.get("cad_min_xyz_m", prof.get("bbox_min", self.tri_centers.min(axis=0)))
            bmax_raw = prof.get("cad_max_xyz_m", prof.get("bbox_max", self.tri_centers.max(axis=0)))
            self.bbox_min = np.asarray(bmin_raw, dtype=np.float32)
            self.bbox_max = np.asarray(bmax_raw, dtype=np.float32)
        else:
            self.is_floor_standing = ("TABLE" in self.cad_name.upper())
            self.bbox_min = self.tri_centers.min(axis=0).astype(np.float32)
            self.bbox_max = self.tri_centers.max(axis=0).astype(np.float32)
            self.real_bounding_diag_m = float(np.linalg.norm(self.bbox_max - self.bbox_min))

        if self.real_bounding_diag_m <= 1e-4:
            self.real_bounding_diag_m = float(np.linalg.norm(self.bbox_max - self.bbox_min))
        self.cad_center = 0.5 * (self.bbox_min + self.bbox_max)

        sym_path = self._resolve_file("_symmetry.json")
        self.symmetry_type = "None"
        sym_list = [np.eye(3, dtype=np.float32)]
        if os.path.isfile(sym_path):
            with open(sym_path, "r", encoding="utf-8") as f:
                sym_j = json.load(f)
            self.symmetry_type = sym_j.get("symmetry_type", "None")
            mats = sym_j.get("symmetry_transforms", [])
            if mats:
                parsed_syms = []
                for m in mats:
                    arr = np.asarray(m, dtype=np.float32)
                    if arr.size == 16:
                        parsed_syms.append(arr.reshape(4, 4)[:3, :3])
                    elif arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 3:
                        parsed_syms.append(arr[:3, :3])
                    elif arr.size == 9:
                        parsed_syms.append(arr.reshape(3, 3))
                if parsed_syms:
                    sym_list = parsed_syms
        self.symmetry_matrices_tensor = torch.from_numpy(np.stack(sym_list, axis=0))

        M_total = len(self.tri_features)
        key_path = os.path.join(self.cad_data_dir, f"{self.cad_name}_key_triangles_{self.num_cad_samples}.npy")
        if os.path.isfile(key_path):
            key_idx = np.clip(np.load(key_path).astype(np.int64), 0, M_total - 1)
        elif M_total >= self.num_cad_samples:
            key_idx = fps_sampling_np(self.tri_centers, self.num_cad_samples, seed=42)
            try:
                np.save(key_path, key_idx)
            except OSError:
                pass
        else:
            rng_k = np.random.default_rng(1337)
            key_idx = rng_k.choice(M_total, size=self.num_cad_samples, replace=True).astype(np.int64)
        self.cad_key_indices = key_idx

        if "patch_features" in patch_npz:
            pf = patch_npz["patch_features"].astype(np.float32)
            self.patch_features = pf[key_idx] if len(pf) != len(key_idx) else pf
        else:
            idx_key = "patch_indices" if "patch_indices" in patch_npz else "patch_triangle_indices"
            p_idx = patch_npz[idx_key].astype(np.int64)
            if len(p_idx) != len(key_idx):
                p_idx = p_idx[key_idx]
            self.patch_features = self.tri_features[np.clip(p_idx, 0, M_total - 1)]

        self.patch_centers = self.patch_features[:, 0, 0:3].copy()
        self.patch_normals = self.patch_features[:, 0, 3:6].copy()

        if "patch_hops" in patch_npz:
            ph = patch_npz["patch_hops"].astype(np.float32)
            self.patch_hops = ph[key_idx] if len(ph) != len(key_idx) else ph
        else:
            self.patch_hops = np.zeros(self.patch_features.shape[:2], dtype=np.float32)

    def _build_fixed_val_views(self):
        self.val_poses = []
        if self.split == "train":
            return
        rng = np.random.default_rng(2026)
        N = self.samples_per_epoch
        golden = math.pi * (3.0 - math.sqrt(5.0))
        for i in range(N):
            if self.is_floor_standing:
                az = 2.0 * math.pi * (i / max(1, N))
                cam_h = 0.25 + 1.55 * ((i % 7) / 6.0)
                dist = self.real_bounding_diag_m * (1.2 + 1.0 * ((i % 5) / 4.0))
                cam_pos = np.array([
                    self.cad_center[0] + dist * math.cos(az),
                    self.cad_center[1] + dist * math.sin(az),
                    self.bbox_min[2] + cam_h,
                ], dtype=np.float32)
                R, t = _look_at_camera_from_object(cam_pos, self.cad_center, np.array([0, 0, 1], dtype=np.float32))
            else:
                az = (golden * i) % (2.0 * math.pi)
                sin_el = math.sin(math.radians(-20.0)) + (i / max(1, N - 1)) * (math.sin(math.radians(85.0)) - math.sin(math.radians(-20.0)))
                el = math.asin(max(-1.0, min(1.0, sin_el)))
                roll_rad = math.radians(float(rng.uniform(-20.0, 20.0)))
                dist = self.real_bounding_diag_m * float(rng.uniform(2.2, 4.5))
                R, t = _sphere_view_with_roll(az, el, roll_rad, dist, self.cad_center, self.cad_center)
            self.val_poses.append((R, t))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_camera_pose(self, rng: np.random.Generator, idx: int):
        if self.split != "train":
            return self.val_poses[idx]
        if self.is_floor_standing:
            az = rng.uniform(0.0, 2.0 * math.pi)
            cam_h = rng.uniform(0.20, 1.85)
            dist = rng.uniform(max(0.65, 1.1 * self.real_bounding_diag_m), max(2.2, 2.6 * self.real_bounding_diag_m))
            cam_pos = np.array([
                self.cad_center[0] + dist * math.cos(az),
                self.cad_center[1] + dist * math.sin(az),
                self.bbox_min[2] + cam_h,
            ], dtype=np.float32)
            target = self.cad_center + rng.normal(0.0, 0.03 * self.real_bounding_diag_m, size=3).astype(np.float32)
            return _look_at_camera_from_object(cam_pos, target, np.array([0.0, 0.0, 1.0], dtype=np.float32))

        az = float(rng.uniform(0.0, 2.0 * math.pi))
        sin_el = float(rng.uniform(math.sin(math.radians(-20.0)), math.sin(math.radians(85.0))))
        el = math.asin(max(-1.0, min(1.0, sin_el)))
        roll_rad = math.radians(float(rng.uniform(-20.0, 20.0)))
        dist = float(rng.uniform(max(0.28, 1.8 * self.real_bounding_diag_m), max(0.85, 5.5 * self.real_bounding_diag_m)))
        target = self.cad_center + rng.normal(0.0, 0.02 * self.real_bounding_diag_m, size=3).astype(np.float32)
        return _sphere_view_with_roll(az, el, roll_rad, dist, target, self.cad_center)

    def _sample_visible_surface(self, R: np.ndarray, t: np.ndarray, rng: np.random.Generator):
        n_raw = max(4096, self.num_obs_points * 4)
        tri_ids = rng.choice(len(self.tri_centers), size=n_raw, replace=True, p=self.tri_area_probs)

        if self.has_exact_mesh:
            r1 = np.sqrt(rng.random(n_raw, dtype=np.float32))[:, None]
            r2 = rng.random(n_raw, dtype=np.float32)[:, None]
            w0 = 1.0 - r1
            w1 = r1 * (1.0 - r2)
            w2 = r1 * r2
            pts_cad = w0 * self.v0[tri_ids] + w1 * self.v1[tri_ids] + w2 * self.v2[tri_ids]
        else:
            pts_cad = self.tri_centers[tri_ids].copy()

        norms_cad = self.tri_normals[tri_ids]
        pts_cam = (pts_cad @ R.T) + t[None, :]
        norms_cam = norms_cad @ R.T

        ray_len = np.linalg.norm(pts_cam, axis=1, keepdims=True) + 1e-8
        view_dir = pts_cam / ray_len
        cos_inc = -np.sum(norms_cam * view_dir, axis=1)
        vis_mask = (pts_cam[:, 2] > 0.10) & (cos_inc > 0.05)
        if np.sum(vis_mask) < 64:
            vis_mask = pts_cam[:, 2] > 0.05

        pts_cam = pts_cam[vis_mask]
        pts_cad = pts_cad[vis_mask]
        tri_ids = tri_ids[vis_mask]
        cos_inc = cos_inc[vis_mask]

        u = pts_cam[:, 0] / np.clip(pts_cam[:, 2], 1e-4, None)
        v = pts_cam[:, 1] / np.clip(pts_cam[:, 2], 1e-4, None)
        u_min, u_max = float(u.min()), float(u.max())
        v_min, v_max = float(v.min()), float(v.max())
        G = self.zbuffer_grid
        bx = np.clip(((u - u_min) / max(1e-6, u_max - u_min) * (G - 1)).astype(np.int32), 0, G - 1)
        by = np.clip(((v - v_min) / max(1e-6, v_max - v_min) * (G - 1)).astype(np.int32), 0, G - 1)
        cell_idx = by * G + bx

        z_min_grid = np.full(G * G, np.inf, dtype=np.float32)
        np.minimum.at(z_min_grid, cell_idx, pts_cam[:, 2])
        z_tol = max(0.006, 0.035 * self.real_bounding_diag_m)
        front_mask = pts_cam[:, 2] <= (z_min_grid[cell_idx] + z_tol)

        if np.sum(front_mask) >= 64:
            pts_cam = pts_cam[front_mask]
            pts_cad = pts_cad[front_mask]
            tri_ids = tri_ids[front_mask]
            cos_inc = cos_inc[front_mask]

        return pts_cam, pts_cad, tri_ids, cos_inc

    def _apply_sim2real_noise(self, pts_cam, pts_cad, tri_ids, cos_inc, rng: np.random.Generator):
        keep_prob = np.where(cos_inc < 0.375, 0.20, 1.0)
        keep = rng.random(len(pts_cam)) < keep_prob
        if np.sum(keep) >= 96:
            pts_cam, pts_cad, tri_ids = pts_cam[keep], pts_cad[keep], tri_ids[keep]

        if rng.random() < 0.75 and len(pts_cam) > 160:
            center_occ = pts_cam[rng.integers(0, len(pts_cam))]
            rad_occ = rng.uniform(0.12, 0.28) * self.real_bounding_diag_m
            dist_occ = np.linalg.norm(pts_cam - center_occ[None, :], axis=1)
            not_occ = dist_occ > rad_occ
            if np.mean(~not_occ) <= self.max_occlusion_ratio and np.sum(not_occ) >= 96:
                pts_cam, pts_cad, tri_ids = pts_cam[not_occ], pts_cad[not_occ], tri_ids[not_occ]

        z = np.clip(pts_cam[:, 2], 0.15, 3.0)
        sigma_z = (0.0015 + 0.0028 * (z ** 2)).astype(np.float32)
        ray_dir = pts_cam / (np.linalg.norm(pts_cam, axis=1, keepdims=True) + 1e-8)
        pts_cam = pts_cam + ray_dir * (rng.normal(0.0, 1.0, size=len(pts_cam)).astype(np.float32) * sigma_z)[:, None]

        n_clutter = int(self.num_obs_points * rng.uniform(0.05, 0.10)) if rng.random() < 0.7 else 0
        n_obj = self.num_obs_points - n_clutter
        sel = rng.choice(len(pts_cam), size=n_obj, replace=(len(pts_cam) < n_obj))

        obj_cam = pts_cam[sel]
        obj_cad = pts_cad[sel]
        obj_tri = tri_ids[sel]
        inlier_mask = np.ones(self.num_obs_points, dtype=np.float32)

        if n_clutter > 0:
            c_mean = obj_cam.mean(axis=0)
            clutter_cam = c_mean[None, :] + rng.uniform(
                -0.75 * self.real_bounding_diag_m, 0.75 * self.real_bounding_diag_m, size=(n_clutter, 3)
            ).astype(np.float32)
            obj_cam = np.concatenate([obj_cam, clutter_cam], axis=0)
            obj_cad = np.concatenate([obj_cad, np.zeros((n_clutter, 3), dtype=np.float32)], axis=0)
            obj_tri = np.concatenate([obj_tri, np.full(n_clutter, -1, dtype=np.int64)], axis=0)
            inlier_mask[n_obj:] = 0.0

        return obj_cam, obj_cad, obj_tri, inlier_mask

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng() if self.split == "train" else np.random.default_rng(10000 + idx)
        R_gt, t_gt = self._sample_camera_pose(rng, idx)
        pts_cam, pts_cad, tri_ids, cos_inc = self._sample_visible_surface(R_gt, t_gt, rng)

        if self.enable_sim2real_noise:
            obs_cam, obs_cad, obs_tri, inlier_mask = self._apply_sim2real_noise(pts_cam, pts_cad, tri_ids, cos_inc, rng)
        else:
            sel = rng.choice(len(pts_cam), size=self.num_obs_points, replace=(len(pts_cam) < self.num_obs_points))
            obs_cam = pts_cam[sel]
            obs_cad = pts_cad[sel]
            obs_tri = tri_ids[sel]
            inlier_mask = np.ones(self.num_obs_points, dtype=np.float32)

        centroid_3d = obs_cam.mean(axis=0, keepdims=True)
        obs_points_norm = ((obs_cam - centroid_3d) / max(1e-4, self.real_bounding_diag_m)).astype(np.float32)

        dists_sq = np.sum((obs_cad[:, None, :] - self.patch_centers[None, :, :]) ** 2, axis=-1)
        patch_targets = np.argmin(dists_sq, axis=1).astype(np.int64)
        patch_targets[inlier_mask < 0.5] = -1

        region_ids = np.where(obs_tri >= 0, self.tri_regions[np.clip(obs_tri, 0, len(self.tri_regions) - 1)], -1).astype(np.int64)

        return {
            "obs_points_norm": torch.from_numpy(obs_points_norm),
            "obs_points_cam": torch.from_numpy(obs_cam.astype(np.float32)),
            "obs_points_cad": torch.from_numpy(obs_cad.astype(np.float32)),
            "obs_tri_ids": torch.from_numpy(obs_tri),
            "obs_patch_targets": torch.from_numpy(patch_targets),
            "obs_region_ids": torch.from_numpy(region_ids),
            "inlier_mask": torch.from_numpy(inlier_mask),
            "patch_features": torch.from_numpy(self.patch_features),
            "patch_centers": torch.from_numpy(self.patch_centers),
            "patch_normals": torch.from_numpy(self.patch_normals),
            "symmetry_matrices": self.symmetry_matrices_tensor,
            "R_gt": torch.from_numpy(R_gt),
            "t_gt": torch.from_numpy(t_gt),
        }
