import os
import sys
import json
import cv2
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset, DataLoader

# Default CAD path and Dataset directory
CAD_PATH_DEFAULT = "/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO.stl"
DATASET_DIR_DEFAULT = "/home/huyen1905/TMCAD/dataset/TABLE-DEMO"


def farthest_point_sample(pts, k):
    """
    Farthest Point Sampling (FPS) on 3D point cloud.
    pts: (N, 3) numpy array
    k: number of samples
    returns: (k, 3) numpy array
    """
    N = pts.shape[0]
    if N <= k:
        return pts
    centroids = np.zeros(k, dtype=np.int64)
    distance = np.ones(N, dtype=np.float32) * 1e10
    farthest = 0
    for i in range(k):
        centroids[i] = farthest
        centroid = pts[farthest, :]
        dist = np.sum((pts - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.argmax(distance))
    return pts[centroids]


def get_cad_keypoints(cad_path=CAD_PATH_DEFAULT, num_keypoints=16):
    """
    Extracts K salient geometric surface keypoints directly from the CAD mesh
    using Farthest Point Sampling (FPS) on the centered, metric-scaled surface.
    Ensures keypoints lie 100% on the real physical surface, corners, and edges
    of the CAD part (NOT floating virtual bounding box corners).
    """
    cache_path = os.path.splitext(cad_path)[0] + f"_surface_kp_{num_keypoints}.npy"
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass

    mesh = o3d.io.read_triangle_mesh(cad_path)
    if not mesh.has_triangles() or len(mesh.vertices) == 0:
        raise ValueError(f"Invalid CAD mesh at {cad_path}")

    # Center at origin
    center = mesh.get_center()
    mesh.translate(-center)

    # Metric scale check (if > 2m, likely in mm)
    bbox_diag = np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound())
    if bbox_diag > 2.0:
        mesh.scale(0.001, np.zeros(3))

    # Sample points on the true CAD surface
    pcd = mesh.sample_points_uniformly(max(4000, num_keypoints * 100))
    pts = np.asarray(pcd.points, dtype=np.float32)

    # Extract K distinct surface points via FPS
    surface_kps = farthest_point_sample(pts, num_keypoints).astype(np.float32)

    # Cache for reuse
    try:
        np.save(cache_path, surface_kps)
    except Exception:
        pass

    return surface_kps


def load_cad_points(cad_path=CAD_PATH_DEFAULT, num_points=1000):
    cache_path = os.path.splitext(cad_path)[0] + f"_cad_pts_{num_points}.npy"
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass
    mesh = o3d.io.read_triangle_mesh(cad_path)
    center = mesh.get_center()
    mesh.translate(-center)
    bbox_diag = np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound())
    if bbox_diag > 2.0:
        mesh.scale(0.001, np.zeros(3))
    pcd = mesh.sample_points_uniformly(num_points)
    pts = np.asarray(pcd.points, dtype=np.float32)
    try:
        np.save(cache_path, pts)
    except Exception:
        pass
    return pts


class CADPoseDataset(Dataset):
    """
    End-to-End 6DoF Pose Dataset for any CAD model with 6D Normal Inputs
    and Geometric Surface Keypoint Targets.
    Outputs:
      - obs_6d: (N, 6) containing (x, y, z, nx, ny, nz)
      - R_gt: (3, 3)
      - t_gt: (3,)
      - keypoints_gt: (K, 3) in camera frame
      - cad_eval_pts: (num_cad, 3)
    """
    def __init__(
        self,
        dataset_dir=DATASET_DIR_DEFAULT,
        split="train",
        cad_path=CAD_PATH_DEFAULT,
        num_points=1024,
        num_keypoints=16,
        num_cad_eval=500,
        augment=False
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.num_points = num_points
        self.num_keypoints = num_keypoints
        self.augment = augment

        # 1. Load CAD Surface Keypoints (FPS) and CAD Evaluation Points
        self.cad_keypoints = get_cad_keypoints(cad_path, num_keypoints=num_keypoints)  # (K, 3)
        self.cad_eval_pts = load_cad_points(cad_path, num_cad_eval)  # (num_cad_eval, 3)

        # 2. Read split file or auto-discover samples
        split_file = os.path.join(dataset_dir, f"{split}.txt")
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        else:
            all_samples = sorted([d for d in os.listdir(dataset_dir) if d.startswith("sample_") and os.path.isdir(os.path.join(dataset_dir, d))])
            rng = np.random.RandomState(1337)
            rng.shuffle(all_samples)
            n_train = max(1, int(len(all_samples) * 0.85))
            lines = all_samples[:n_train] if split == "train" else all_samples[n_train:]
            print(f"[AUTO-SPLIT] {split_file} not found; auto-discovered {len(lines)} samples for {split}")

        self.samples = []
        for line in lines:
            sample_dir = os.path.join(dataset_dir, line)
            if not os.path.isdir(sample_dir):
                sample_dir = os.path.join(dataset_dir, split, line)
            if os.path.isdir(sample_dir):
                self.samples.append(sample_dir)

        print(f"[{split.upper()} E2E DATASET] Loaded {len(self.samples)} samples from {split_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        sample_id = os.path.basename(sample_dir)

        # 1. Load pose
        pose_path = os.path.join(sample_dir, "pose.json")
        with open(pose_path, "r") as f:
            pose_data = json.load(f)

        intr = pose_data["intrinsics"]
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]
        T_gt = np.array(pose_data["cameraFromObject"], dtype=np.float32)
        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]

        # Compute ground truth keypoints in camera frame: K_cam = K_cad @ R_gt^T + t_gt
        keypoints_gt = self.cad_keypoints @ R_gt.T + t_gt.reshape(1, 3)  # (8, 3)

        # 2. Backproject depth to 3D point cloud
        depth_path = os.path.join(sample_dir, "depth.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        v_idx, u_idx = np.where((depth > 200) & (depth < 2850))

        if len(v_idx) >= self.num_points:
            sel = np.random.choice(len(v_idx), self.num_points, replace=False)
        elif len(v_idx) > 0:
            sel = np.random.choice(len(v_idx), self.num_points, replace=True)
        else:
            # Fallback in case of empty depth
            v_idx = np.array([240], dtype=np.int32)
            u_idx = np.array([320], dtype=np.int32)
            sel = np.zeros(self.num_points, dtype=np.int32)

        z_m = depth[v_idx[sel], u_idx[sel]].astype(np.float32) / 1000.0
        x_m = (u_idx[sel].astype(np.float32) - cx) * z_m / fx
        y_m = (v_idx[sel].astype(np.float32) - cy) * z_m / fy
        obs_pts = np.stack([x_m, y_m, z_m], axis=1)  # (N, 3)

        # 3. Estimate surface normals on the sampled points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obs_pts)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=20))
        pcd.orient_normals_towards_camera_location(np.zeros(3))
        obs_normals = np.asarray(pcd.normals, dtype=np.float32)

        # 4. Data augmentation (Train only)
        if self.augment:
            # Jitter noise on points
            jitter = np.random.normal(0, 0.002, obs_pts.shape).astype(np.float32)
            obs_pts += jitter

            # Small random rigid perturbation to the camera frame
            from scipy.spatial.transform import Rotation
            rot_pert = Rotation.from_euler("xyz", np.random.uniform(-10, 10, 3), degrees=True).as_matrix().astype(np.float32)
            trans_pert = np.random.uniform(-0.02, 0.02, (1, 3)).astype(np.float32)

            obs_pts = obs_pts @ rot_pert.T + trans_pert
            obs_normals = obs_normals @ rot_pert.T
            keypoints_gt = keypoints_gt @ rot_pert.T + trans_pert

            R_gt = rot_pert @ R_gt
            t_gt = (rot_pert @ t_gt.reshape(3, 1) + trans_pert.T).reshape(3)

        # 5. Concatenate points and normals into 6D: (N, 6)
        obs_6d = np.concatenate([obs_pts, obs_normals], axis=1)

        return {
            "obs_6d": torch.from_numpy(obs_6d),                   # (N, 6)
            "R_gt": torch.from_numpy(R_gt),                       # (3, 3)
            "t_gt": torch.from_numpy(t_gt),                       # (3,)
            "keypoints_gt": torch.from_numpy(keypoints_gt),       # (K, 3)
            "cad_eval_pts": torch.from_numpy(self.cad_eval_pts),  # (num_cad, 3)
            "sample_id": sample_id
        }

TableE2EDataset = CADPoseDataset


if __name__ == "__main__":
    ds = CADPoseDataset(split="train", num_points=1024, augment=True)
    item = ds[0]
    print("obs_6d shape:", item["obs_6d"].shape)
    print("R_gt shape:", item["R_gt"].shape)
    print("t_gt shape:", item["t_gt"].shape)
    print("keypoints_gt shape:", item["keypoints_gt"].shape)
    print("cad_eval_pts shape:", item["cad_eval_pts"].shape)

    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
    batch = next(iter(loader))
    print("Batch obs_6d:", batch["obs_6d"].shape)
    print("Batch R_gt:", batch["R_gt"].shape)
    print("Batch keypoints_gt:", batch["keypoints_gt"].shape)
    print("Dataset E2E test PASSED!")
