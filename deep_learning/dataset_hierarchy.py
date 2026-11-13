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