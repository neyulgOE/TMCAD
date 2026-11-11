import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def fps_sampling_np(pts, num_samples):
    """
    Farthest Point Sampling on numpy array (M, 3) -> (num_samples,) indices.
    Uses GPU if available for high speed on large CAD meshes.
    """
    M = len(pts)
    if M <= num_samples:
        return np.arange(M, dtype=np.int64)

    if torch.cuda.is_available():
        pts_t = torch.from_numpy(pts).cuda().float()
        sampled = torch.zeros(num_samples, dtype=torch.long, device='cuda')
        dist = torch.full((M,), 1e10, device='cuda')
        sampled[0] = 0
        cur = pts_t[0]
        for i in range(1, num_samples):
            d = torch.sum((pts_t - cur) ** 2, dim=1)
            dist = torch.minimum(dist, d)
            sampled[i] = torch.argmax(dist)
            cur = pts_t[sampled[i]]
        torch.cuda.synchronize()
        return sampled.cpu().numpy()
    else:
        sampled = np.zeros(num_samples, dtype=np.int64)
        dist = np.full(M, 1e10, dtype=np.float32)
        sampled[0] = 0
        cur = pts[0]
        for i in range(1, num_samples):
            d = np.sum((pts - cur) ** 2, axis=1)
            dist = np.minimum(dist, d)
            sampled[i] = np.argmax(dist)
            cur = pts[sampled[i]]
        return sampled

class CADHierarchyDataset(Dataset):
    """
    PyTorch Dataset linking:
    1. Canonical CAD Hierarchy (13D triangle primitives, K-ring local patches, symmetry group G)
    2. Synthetic RGB-D partial observations (camera point cloud + GT correspondences)
    """
    def __init__(self,
                 dataset_dir: str,
                 cad_dir: str,
                 cad_name: str = "TABLE-DEMO",
                 split: str = "train",
                 num_cad_samples: int = 512,
                 num_obs_samples: int = 1024,
                 patch_k: int = 32):
        super().__init__()
        self.dataset_dir = dataset_dir
        # Auto-resolve dedicated model subfolder if present (e.g. CAD_DATA/TABLE-DEMO)
        model_subfolder = os.path.join(cad_dir, cad_name)
        if os.path.isdir(model_subfolder):
            cad_dir = model_subfolder
        self.cad_dir = cad_dir
        self.cad_name = cad_name
        self.split = split
        self.num_cad_samples = num_cad_samples
        self.num_obs_samples = num_obs_samples
        self.patch_k = patch_k

        # 1. Load Canonical CAD Hierarchy files
        tri_path = os.path.join(cad_dir, f"{cad_name}_triangles.npz")
        patch_path = os.path.join(cad_dir, f"{cad_name}_patches.npz")
        sym_path = os.path.join(cad_dir, f"{cad_name}_symmetry.json")

        if not os.path.exists(tri_path) or not os.path.exists(patch_path):
            raise FileNotFoundError(f"Missing canonical hierarchy files for {cad_name} in {cad_dir}")

        tri_data = np.load(tri_path)
        self.all_tri_features = tri_data['features'].astype(np.float32) # (M, 13)
        self.all_centroids = self.all_tri_features[:, :3]               # (M, 3)

        patch_data = np.load(patch_path)
        self.all_patch_indices = patch_data['patch_indices'].astype(np.int64) # (M, 32)
        self.all_patch_hops = patch_data['patch_hops'].astype(np.int64)       # (M, 32)

        # 2. Load Symmetry Group G
        self.symmetry_transforms = [np.eye(4, dtype=np.float32)]
        self.symmetry_type = "None"
        if os.path.exists(sym_path):
            with open(sym_path, 'r') as f:
                sym_info = json.load(f)
            self.symmetry_type = sym_info.get("symmetry_type", "None")
            raw_xforms = sym_info.get("symmetry_transforms", [])
            if raw_xforms:
                self.symmetry_transforms = [np.array(x, dtype=np.float32).reshape(4, 4) for x in raw_xforms]

        # 3. Select CAD Key Primitives via FPS and cache
        cache_key_path = os.path.join(cad_dir, f"{cad_name}_key_triangles_{num_cad_samples}.npy")
        cache_partner_path = os.path.join(cad_dir, f"{cad_name}_symmetry_partner_indices.npy")
        if os.path.exists(cache_key_path):
            self.cad_key_indices = np.load(cache_key_path)
        else:
            print(f"Sampling {num_cad_samples} key CAD primitives via FPS...")
            self.cad_key_indices = fps_sampling_np(self.all_centroids, num_cad_samples)
            np.save(cache_key_path, self.cad_key_indices)

        self.symmetry_partners = None
        if os.path.exists(cache_partner_path):
            self.symmetry_partners = np.load(cache_partner_path)

        # 4. Precompute CAD Key Patch Tensors
        # key_centroids: (N_cad, 3)
        self.cad_key_centroids = self.all_centroids[self.cad_key_indices].astype(np.float32)
        # key_patches: (N_cad, K)
        key_patches = self.all_patch_indices[self.cad_key_indices, :self.patch_k]
        # patch_features: (N_cad, K, 13)
        self.cad_patch_features = self.all_tri_features[key_patches]
        # rel_centroids: (N_cad, K, 3)
        self.cad_patch_rel_pos = self.cad_patch_features[:, :, :3] - self.cad_key_centroids[:, None, :]
        # patch_hops: (N_cad, K)
        self.cad_patch_hops = self.all_patch_hops[self.cad_key_indices, :self.patch_k]

        # 5. Load observation sample list from split file
        split_file = os.path.join(dataset_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            self.sample_rel_paths = [line.strip() for line in f if line.strip()]

        print(f"CADHierarchyDataset [{split}]: {len(self.sample_rel_paths)} samples, "
              f"CAD key primitives: {self.num_cad_samples}, Patch K: {self.patch_k}, "
              f"Symmetry: {self.symmetry_type} ({len(self.symmetry_transforms)} xforms)")

    def __len__(self):
        return len(self.sample_rel_paths)

    def __getitem__(self, idx):
        rel_path = self.sample_rel_paths[idx]
        sample_dir = os.path.join(self.dataset_dir, rel_path)

        csv_path = os.path.join(sample_dir, "correspondences.csv")
        pose_path = os.path.join(sample_dir, "pose.json")

        # 1. Load GT Pose
        with open(pose_path, 'r') as f:
            pose_meta = json.load(f)
        gt_pose = np.array(pose_meta["cameraFromObject"], dtype=np.float32).reshape(4, 4)

        # 2. Load correspondences: cam_xyz and cad_xyz
        df = pd.read_csv(csv_path, usecols=['cam_x', 'cam_y', 'cam_z', 'cad_x', 'cad_y', 'cad_z'])
        obs_cam = df[['cam_x', 'cam_y', 'cam_z']].values.astype(np.float32)
        obs_cad = df[['cad_x', 'cad_y', 'cad_z']].values.astype(np.float32)

        N = len(obs_cam)
        if N >= self.num_obs_samples:
            choice = np.random.choice(N, self.num_obs_samples, replace=False)
        else:
            choice = np.random.choice(N, self.num_obs_samples, replace=True)

        sampled_obs_cam = obs_cam[choice].copy() # (N_obs, 3)
        sampled_obs_cad = obs_cad[choice].copy() # (N_obs, 3)

        # 2b. Data Augmentation for training: Add Realistic Sensor Depth Jitter (Adaptive to CAD size)
        if self.split == "train":
            cad_span = float(np.max(np.abs(self.cad_key_centroids)))
            max_jitter = min(0.010, max(0.002, cad_span * 0.05))
            noise_std = max_jitter * 0.3
            noise = np.random.normal(0, noise_std, sampled_obs_cam.shape).astype(np.float32)
            noise = np.clip(noise, -max_jitter, max_jitter)
            sampled_obs_cam += noise

        # 3. Compute Positive Matching Targets for all Symmetry Transforms
        diff = sampled_obs_cad[:, None, :] - self.cad_key_centroids[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        best_cad_idx = np.argmin(dist_sq, axis=-1) # (N_obs,)

        num_sym = len(self.symmetry_transforms)
        positive_targets = np.zeros((self.num_obs_samples, num_sym), dtype=np.int64)
        positive_targets[:, 0] = best_cad_idx

        if num_sym > 1:
            if self.symmetry_partners is not None and len(self.symmetry_partners) == self.num_cad_samples:
                # O(1) exact partner assignment with 0.000 mm partner symmetry error!
                positive_targets[:, 1] = self.symmetry_partners[best_cad_idx]
            else:
                for k_idx in range(1, num_sym):
                    S = self.symmetry_transforms[k_idx]
                    R_s = S[:3, :3]
                    t_s = S[:3, 3]
                    p_sym = sampled_obs_cad @ R_s.T + t_s
                    diff_sym = p_sym[:, None, :] - self.cad_key_centroids[None, :, :]
                    positive_targets[:, k_idx] = np.argmin(np.sum(diff_sym ** 2, axis=-1), axis=-1)

        return {
            "obs_cam_pts": torch.from_numpy(sampled_obs_cam),             # (N_obs, 3)
            "obs_cad_pts": torch.from_numpy(sampled_obs_cad),             # (N_obs, 3)
            "positive_targets": torch.from_numpy(positive_targets),       # (N_obs, |G|)
            "gt_pose": torch.from_numpy(gt_pose),                         # (4, 4)
            # CAD representation (shared across batch or indexed)
            "cad_key_centroids": torch.from_numpy(self.cad_key_centroids),       # (N_cad, 3)
            "cad_patch_features": torch.from_numpy(self.cad_patch_features),     # (N_cad, K, 13)
            "cad_patch_rel_pos": torch.from_numpy(self.cad_patch_rel_pos),       # (N_cad, K, 3)
            "cad_patch_hops": torch.from_numpy(self.cad_patch_hops)              # (N_cad, K)
        }
