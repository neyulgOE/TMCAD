import os
import sys
import argparse
import numpy as np
import torch
import open3d as o3d
from torch.utils.data import DataLoader

# Add repo root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_learning.dataset import TableCADDataset
from deep_learning.model import GeometricDGCNN

def compute_fpfh_features(points, normal_radius=0.049, fpfh_radius=0.098):
    """
    Compute classical FPFH features for comparison.
    points: (N, 3) numpy array
    returns: (N, 33) FPFH descriptors
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=100)
    )
    feat = np.asarray(fpfh.data, dtype=np.float32).T  # (N, 33)
    norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
    return feat / norm

def evaluate_model(model_path, dataset_dir, split="val", cad_path="/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO.stl", num_pos=512):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load trained model
    assert os.path.exists(model_path), f"Checkpoint not found: {model_path}"
    checkpoint = torch.load(model_path, map_location=device)
    args_saved = checkpoint.get("args", {})
    desc_dim = args_saved.get("desc_dim", 32)
    k_neighbors = args_saved.get("k_neighbors", 20)

    model = GeometricDGCNN(k=k_neighbors, descriptor_dim=desc_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {model_path} (epoch {checkpoint.get('epoch', '?')}, desc_dim={desc_dim})")

    # 2. Dataset
    dataset = TableCADDataset(
        dataset_dir=dataset_dir,
        split=split,
        cad_path=cad_path,
        num_points=2048,
        num_pos_pairs=num_pos,
        augment=False
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"\nEvaluating on {len(dataset)} samples ({split.upper()} set)...")

    learned_recalls_1 = []
    learned_recalls_5 = []
    fpfh_recalls_1 = []
    fpfh_recalls_5 = []

    # Map by visibility
    by_vis_learned_r1 = {0.25: [], 0.50: [], 0.75: [], 1.00: []}
    by_vis_fpfh_r1 = {0.25: [], 0.50: [], 0.75: [], 1.00: []}

    with torch.no_grad():
        for i, batch in enumerate(loader):
            cad_pts = batch["cad_points"].to(device)
            obs_pts = batch["obs_points"].to(device)
            sample_id = batch["sample_id"][0]

            # Read targetVisibleFraction from sample's pose.json
            pose_path = os.path.join(dataset_dir, split, sample_id, "pose.json")
            if not os.path.exists(pose_path):
                pose_path = os.path.join(dataset_dir, sample_id, "pose.json")
            
            import json
            with open(pose_path, "r") as f:
                pdata = json.load(f)
            vis = round(pdata.get("targetVisibleFraction", 1.0), 2)

            # A. Learned Model
            cad_desc = model(cad_pts)[0, :num_pos, :]  # (K, D)
            obs_desc = model(obs_pts)[0, :num_pos, :]  # (K, D)

            # Cosine similarity matrix: (K, K)
            sim = torch.matmul(cad_desc, obs_desc.t())
            targets = torch.arange(num_pos, device=device)

            pred1 = sim.argmax(dim=1)
            r1_learned = (pred1 == targets).float().mean().item()

            top5 = sim.topk(k=5, dim=1)[1]
            r5_learned = (top5 == targets.unsqueeze(1)).any(dim=1).float().mean().item()

            learned_recalls_1.append(r1_learned)
            learned_recalls_5.append(r5_learned)
            if vis in by_vis_learned_r1:
                by_vis_learned_r1[vis].append(r1_learned)

            # B. Classical FPFH (evaluate every 5th sample to keep eval fast)
            if i % 5 == 0:
                cad_np = cad_pts[0, :num_pos, :].cpu().numpy()
                obs_np = obs_pts[0, :num_pos, :].cpu().numpy()

                fpfh_cad = compute_fpfh_features(cad_np)
                fpfh_obs = compute_fpfh_features(obs_np)

                sim_fpfh = np.dot(fpfh_cad, fpfh_obs.T)
                pred1_fpfh = np.argmax(sim_fpfh, axis=1)
                r1_fpfh = np.mean(pred1_fpfh == np.arange(num_pos))

                top5_fpfh = np.argsort(-sim_fpfh, axis=1)[:, :5]
                r5_fpfh = np.mean([i in top5_fpfh[i] for i in range(num_pos)])

                fpfh_recalls_1.append(r1_fpfh)
                fpfh_recalls_5.append(r5_fpfh)
                if vis in by_vis_fpfh_r1:
                    by_vis_fpfh_r1[vis].append(r1_fpfh)

            if (i + 1) % 50 == 0 or (i + 1) == len(dataset):
                print(f"[{i+1}/{len(dataset)}] Learned Recall@1: {np.mean(learned_recalls_1)*100:.2f}% | Recall@5: {np.mean(learned_recalls_5)*100:.2f}%")

    print("\n" + "=" * 70)
    print(" EVALUATION RESULTS: LEARNED DESCRIPTOR (PHASE 3) vs FPFH (CLASSICAL)")
    print("=" * 70)
    print(f"Overall Recall@1 (Learned): {np.mean(learned_recalls_1)*100:.2f}%")
    print(f"Overall Recall@5 (Learned): {np.mean(learned_recalls_5)*100:.2f}%")
    if fpfh_recalls_1:
        print(f"Overall Recall@1 (FPFH):    {np.mean(fpfh_recalls_1)*100:.2f}%")
        print(f"Overall Recall@5 (FPFH):    {np.mean(fpfh_recalls_5)*100:.2f}%")
    print("-" * 70)
    print(f"{'Visibility':<15} | {'Learned Recall@1':<20} | {'FPFH Recall@1':<20}")
    print("-" * 70)
    for v in [1.00, 0.75, 0.50, 0.25]:
        lr1 = np.mean(by_vis_learned_r1[v])*100 if by_vis_learned_r1[v] else 0.0
        fr1 = np.mean(by_vis_fpfh_r1[v])*100 if by_vis_fpfh_r1[v] else 0.0
        print(f"Vis = {v:<10.2f} | {lr1:<18.2f}% | {fr1:<18.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/huyen1905/TMCAD/deep_learning/checkpoints/best_geometric_model.pth")
    parser.add_argument("--dataset-dir", type=str, default="/home/huyen1905/TMCAD/dataset/TABLE-DEMO")
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()
    evaluate_model(args.model_path, args.dataset_dir, split=args.split)
