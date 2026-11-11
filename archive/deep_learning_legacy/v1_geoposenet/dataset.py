import os
import json
import cv2
import numpy as np
import pandas as pd
import open3d as o3d
import torch
from torch.utils.data import Dataset, DataLoader

class TableCADDataset(Dataset):
    def __init__(
        self,
        dataset_dir="/home/huyen1905/TMCAD/dataset/TABLE-DEMO",
        split="train",
        cad_path="/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO.stl",
        num_points=2048,
        num_pos_pairs=512,
        augment=False,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.num_points = num_points
        self.num_pos_pairs = num_pos_pairs
        self.augment = augment

        # 1. Load and prepare CAD model points (rescaled to metric meters and centered)
        assert os.path.exists(cad_path), f"CAD file not found: {cad_path}"
        cache_cad_npy = cad_path + "_sampled_10k.npy"
        if os.path.exists(cache_cad_npy):
            self.cad_full_points = np.load(cache_cad_npy)
        else:
            mesh = o3d.io.read_triangle_mesh(cad_path)
            center = mesh.get_center()
            mesh.translate(-center)
            # Unit scale: mm -> meters
            mesh.scale(0.001, np.zeros(3))
            pcd = mesh.sample_points_poisson_disk(10000)
            self.cad_full_points = np.asarray(pcd.points, dtype=np.float32)
            np.save(cache_cad_npy, self.cad_full_points)

        # 2. Read split file
        split_file = os.path.join(dataset_dir, f"{split}.txt")
        assert os.path.exists(split_file), f"Split file not found: {split_file}"
        with open(split_file, "r") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        
        # Verify samples exist
        self.samples = []
        for line in lines:
            sample_dir = os.path.join(dataset_dir, line)
            if not os.path.isdir(sample_dir):
                sample_dir = os.path.join(dataset_dir, split, line)
            if os.path.isdir(sample_dir):
                self.samples.append(sample_dir)

        print(f"[{split.upper()} DATASET] Loaded {len(self.samples)} samples from {split_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        sample_id = os.path.basename(sample_dir)

        # 1. Load pose.json
        pose_path = os.path.join(sample_dir, "pose.json")
        with open(pose_path, "r") as f:
            pose_data = json.load(f)

        intr = pose_data["intrinsics"]
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]
        T_gt = np.array(pose_data["cameraFromObject"], dtype=np.float32)

        # 2. Load correspondences.csv
        corr_path = os.path.join(sample_dir, "correspondences.csv")
        df = pd.read_csv(corr_path)
        num_corr = len(df)

        if num_corr >= self.num_pos_pairs:
            corr_indices = np.random.choice(num_corr, self.num_pos_pairs, replace=False)
        else:
            corr_indices = np.random.choice(num_corr, self.num_pos_pairs, replace=True)

        cad_pos = df[["cad_x", "cad_y", "cad_z"]].values[corr_indices].astype(np.float32)
        obs_pos = df[["cam_x", "cam_y", "cam_z"]].values[corr_indices].astype(np.float32)

        # 3. Complete CAD points up to self.num_points
        num_extra_cad = self.num_points - self.num_pos_pairs
        extra_cad_idx = np.random.choice(len(self.cad_full_points), num_extra_cad, replace=False)
        extra_cad = self.cad_full_points[extra_cad_idx]
        cad_cloud = np.vstack([cad_pos, extra_cad])  # (num_points, 3)

        # 4. Load depth image to get extra scene observation points
        depth_path = os.path.join(sample_dir, "depth.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint16, mm
        
        # Valid depth mask (filter out background plane at 3.0m)
        v_idx, u_idx = np.where((depth > 200) & (depth < 2850))
        if len(v_idx) > 0:
            z_m = depth[v_idx, u_idx].astype(np.float32) / 1000.0
            x_m = (u_idx.astype(np.float32) - cx) * z_m / fx
            y_m = (v_idx.astype(np.float32) - cy) * z_m / fy
            all_obs = np.stack([x_m, y_m, z_m], axis=1)
        else:
            all_obs = obs_pos

        num_extra_obs = self.num_points - self.num_pos_pairs
        if len(all_obs) >= num_extra_obs:
            extra_obs_idx = np.random.choice(len(all_obs), num_extra_obs, replace=False)
        else:
            extra_obs_idx = np.random.choice(len(all_obs), num_extra_obs, replace=True)
        extra_obs = all_obs[extra_obs_idx]
        obs_cloud = np.vstack([obs_pos, extra_obs])

        # 5. Centering both clouds to their respective centroids
        cad_cloud = cad_cloud - np.mean(cad_cloud, axis=0, keepdims=True)
        obs_cloud = obs_cloud - np.mean(obs_cloud, axis=0, keepdims=True)

        # 6. Data Augmentation (Train only)
        if self.augment:
            # Jitter noise
            jitter = np.random.normal(0, 0.002, obs_cloud.shape).astype(np.float32)
            obs_cloud += jitter

            # Random SO(3) rotation on CAD cloud
            from scipy.spatial.transform import Rotation
            R_cad = Rotation.random().as_matrix().astype(np.float32)
            cad_cloud = cad_cloud @ R_cad.T

            # Random SO(3) rotation on Obs cloud
            R_obs = Rotation.random().as_matrix().astype(np.float32)
            obs_cloud = obs_cloud @ R_obs.T

        return {
            "cad_points": torch.from_numpy(cad_cloud),       # (N, 3)
            "obs_points": torch.from_numpy(obs_cloud),       # (N, 3)
            "T_gt": torch.from_numpy(T_gt),                   # (4, 4)
            "num_pos": self.num_pos_pairs,
            "sample_id": sample_id
        }

if __name__ == "__main__":
    dataset = TableCADDataset(split="train", num_points=2048, num_pos_pairs=512)
    sample = dataset[0]
    print("CAD points shape:", sample["cad_points"].shape)
    print("Obs points shape:", sample["obs_points"].shape)
    print("T_gt shape:", sample["T_gt"].shape)
    print("Positive pairs:", sample["num_pos"])
    print("Sample ID:", sample["sample_id"])

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print("Batch CAD points:", batch["cad_points"].shape)
    print("Batch Obs points:", batch["obs_points"].shape)
    print("Batch T_gt:", batch["T_gt"].shape)
    print("Dataset smoke test PASSED!")
