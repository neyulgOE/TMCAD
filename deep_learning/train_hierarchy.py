import os
import sys
import argparse
import time
import json

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader

from deep_learning.dataset_hierarchy import CADHierarchyDataset
from deep_learning.model_hierarchy import CADHierarchyCorrespondenceModel
from deep_learning.loss_hierarchy import SymmetryAwareInfoNCELoss, compute_matching_recall

def parse_args():
    parser = argparse.ArgumentParser(description="Train CAD-Aware Hierarchical Correspondence Model")
    parser.add_argument("--cad-name", type=str, default="TABLE-DEMO", help="CAD model name")
    parser.add_argument("--dataset-dir", type=str, default="", help="Path to dataset directory")
    parser.add_argument("--cad-dir", type=str, default="", help="Path to CAD data directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num-cad-samples", type=int, default=512, help="Number of CAD key primitives")
    parser.add_argument("--num-obs-samples", type=int, default=1024, help="Number of observation points")
    parser.add_argument("--descriptor-dim", type=int, default=128, help="Descriptor dimensionality")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    parser.add_argument("--checkpoint-dir", type=str, default="", help="Output checkpoint directory")
    parser.add_argument("--reset-data", action="store_true", help="Reset existing checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if not args.dataset_dir:
        args.dataset_dir = os.path.join(project_root, "dataset", args.cad_name)
    if not args.cad_dir:
        args.cad_dir = os.path.join(project_root, "CAD_DATA")
    if os.path.isdir(os.path.join(args.cad_dir, args.cad_name)):
        args.cad_dir = os.path.join(args.cad_dir, args.cad_name)
    if not args.checkpoint_dir:
        args.checkpoint_dir = os.path.join(project_root, "deep_learning", "checkpoints", args.cad_name)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(args.checkpoint_dir, "best_hierarchy_model.pth")

    if args.reset_data and os.path.exists(best_model_path):
        os.remove(best_model_path)
        print(f"[RESET] Removed previous checkpoint: {best_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f" TRAINING CAD-AWARE HIERARCHICAL CORRESPONDENCE MODEL")
    print(f" Object: {args.cad_name} | Device: {device}")
    print(f" Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")
    print(f" CAD key primitives: {args.num_cad_samples} | Obs points: {args.num_obs_samples}")
    print(f"============================================================")

    # 1. Datasets & DataLoaders
    norm_ds = os.path.normpath(args.dataset_dir)
    dataset_root = os.path.dirname(norm_ds) if os.path.basename(norm_ds) == args.cad_name else norm_ds

    train_dataset = CADHierarchyDataset(
        dataset_root=dataset_root,
        cad_data_dir=args.cad_dir,
        cad_name=args.cad_name,
        split="train",
        num_obs_points=args.num_obs_samples,
        enable_sim2real_noise=True,
    )

    val_dataset = CADHierarchyDataset(
        dataset_root=dataset_root,
        cad_data_dir=args.cad_dir,
        cad_name=args.cad_name,
        split="val",
        num_obs_points=args.num_obs_samples,
        enable_sim2real_noise=False,
    )

    import numpy as np
    patch_npz_path = os.path.join(args.cad_dir, f"{args.cad_name}_patches.npz")
    patch_npz = np.load(patch_npz_path)
    patch_rel_np = patch_npz["patch_rel_pos"].astype(np.float32) if "patch_rel_pos" in patch_npz else None
    patch_hops_np = patch_npz["patch_hops"].astype(np.float32) if "patch_hops" in patch_npz else None
    key_indices_np = np.arange(args.num_cad_samples, dtype=np.int64)
    key_path = os.path.join(args.cad_dir, f"{args.cad_name}_key_triangles_512.npy")
    if os.path.exists(key_path):
        key_indices_np = np.load(key_path).astype(np.int64)
    if patch_hops_np is not None and len(patch_hops_np) > args.num_cad_samples:
        if os.path.exists(key_path):
            patch_hops_np = patch_hops_np[key_indices_np]
        else:
            patch_hops_np = patch_hops_np[np.linspace(0, len(patch_hops_np) - 1, args.num_cad_samples, dtype=np.int64)]
    patch_rel_const = torch.from_numpy(patch_rel_np).to(device) if patch_rel_np is not None else None
    patch_hops_const = torch.from_numpy(patch_hops_np).to(device) if patch_hops_np is not None else None

    def prepare_batch(batch):
        obs_cam = batch["obs_points_norm"].to(device, non_blocking=True)
        patch_feat = batch["patch_features"].to(device, non_blocking=True)
        patch_centers = batch["patch_centers"].to(device, non_blocking=True)
        B = obs_cam.shape[0]
        if patch_rel_const is not None:
            patch_rel = patch_rel_const.unsqueeze(0).expand(B, -1, -1, -1)
        else:
            patch_rel = patch_feat[..., 0:3] - patch_centers.unsqueeze(2)
        if patch_hops_const is not None:
            patch_hops = patch_hops_const.unsqueeze(0).expand(B, -1, -1)
        else:
            patch_hops = torch.zeros(patch_feat.shape[:3], device=device, dtype=patch_feat.dtype)

        gt_cad = batch["gt_cad_points"].to(device, non_blocking=True)
        sym_mats = batch["symmetry_matrices"].to(device, non_blocking=True)
        K_sym = sym_mats.shape[1]
        pos_list = []
        for s_idx in range(K_sym):
            rot_cad = torch.bmm(gt_cad, sym_mats[:, s_idx].transpose(1, 2))
            dists = torch.cdist(rot_cad, patch_centers)
            pos_list.append(dists.argmin(dim=-1))
        targets = torch.stack(pos_list, dim=-1)
        return obs_cam, patch_feat, patch_rel, patch_hops, targets

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # 2. Model, Loss, Optimizer
    model = CADHierarchyCorrespondenceModel(
        descriptor_dim=args.descriptor_dim,
        patch_k=32,
        obs_k=20,
        temperature=args.temperature
    ).to(device)

    criterion = SymmetryAwareInfoNCELoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_recall = -1.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_recall1_sum = 0.0
        train_recall5_sum = 0.0
        num_train_batches = len(train_loader)

        for batch in train_loader:
            obs_cam, patch_feat, patch_rel, patch_hops, targets = prepare_batch(batch)

            optimizer.zero_grad()
            sim, _, _ = model(obs_cam, patch_feat, patch_rel, patch_hops)
            loss = criterion(sim, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_recall1_sum += compute_matching_recall(sim, targets, topk=1)
            train_recall5_sum += compute_matching_recall(sim, targets, topk=5)

        scheduler.step()

        avg_train_loss = total_train_loss / max(1, num_train_batches)
        avg_train_recall1 = train_recall1_sum / max(1, num_train_batches)
        avg_train_recall5 = train_recall5_sum / max(1, num_train_batches)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_recall1_sum = 0.0
        val_recall5_sum = 0.0
        num_val_batches = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                obs_cam, patch_feat, patch_rel, patch_hops, targets = prepare_batch(batch)

                sim, _, _ = model(obs_cam, patch_feat, patch_rel, patch_hops)
                val_loss = criterion(sim, targets)

                total_val_loss += val_loss.item()
                val_recall1_sum += compute_matching_recall(sim, targets, topk=1)
                val_recall5_sum += compute_matching_recall(sim, targets, topk=5)

        avg_val_loss = total_val_loss / max(1, num_val_batches)
        avg_val_recall1 = val_recall1_sum / max(1, num_val_batches)
        avg_val_recall5 = val_recall5_sum / max(1, num_val_batches)

        # Print structured progress for GUI & log
        print(f"PROGRESS:epoch={epoch},total_epochs={args.epochs},"
              f"loss={avg_train_loss:.4f},val_loss={avg_val_loss:.4f},"
              f"recall1={avg_val_recall1:.2f},recall5={avg_val_recall5:.2f}")
        sys.stdout.flush()

        # Save best model
        if avg_val_recall1 > best_val_recall:
            best_val_recall = avg_val_recall1
            torch.save({
                "epoch": epoch,
                "cad_name": args.cad_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_recall1": avg_val_recall1,
                "val_recall5": avg_val_recall5,
                "num_cad_samples": args.num_cad_samples,
                "descriptor_dim": args.descriptor_dim,
                "temperature": args.temperature,
                "symmetry_type": getattr(train_dataset, "symmetry_type", "auto"),
                "cad_key_indices": getattr(train_dataset, "cad_key_indices", key_indices_np)
            }, best_model_path)
            print(f"  --> Best model saved to {best_model_path} (Val Recall@1: {best_val_recall:.2f}%)")
            sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"============================================================")
    print(f" TRAINING COMPLETED in {elapsed:.1f}s")
    print(f" Best Validation Recall@1: {best_val_recall:.2f}%")
    print(f" Model saved at: {best_model_path}")
    print(f"============================================================")

if __name__ == "__main__":
    main()
