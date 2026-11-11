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
    train_dataset = CADHierarchyDataset(
        dataset_dir=args.dataset_dir,
        cad_dir=args.cad_dir,
        cad_name=args.cad_name,
        split="train",
        num_cad_samples=args.num_cad_samples,
        num_obs_samples=args.num_obs_samples,
        patch_k=32
    )

    val_dataset = CADHierarchyDataset(
        dataset_dir=args.dataset_dir,
        cad_dir=args.cad_dir,
        cad_name=args.cad_name,
        split="val",
        num_cad_samples=args.num_cad_samples,
        num_obs_samples=args.num_obs_samples,
        patch_k=32
    )

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
            obs_cam = batch["obs_cam_pts"].to(device, non_blocking=True)
            patch_feat = batch["cad_patch_features"].to(device, non_blocking=True)
            patch_rel = batch["cad_patch_rel_pos"].to(device, non_blocking=True)
            patch_hops = batch["cad_patch_hops"].to(device, non_blocking=True)
            targets = batch["positive_targets"].to(device, non_blocking=True)

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
                obs_cam = batch["obs_cam_pts"].to(device, non_blocking=True)
                patch_feat = batch["cad_patch_features"].to(device, non_blocking=True)
                patch_rel = batch["cad_patch_rel_pos"].to(device, non_blocking=True)
                patch_hops = batch["cad_patch_hops"].to(device, non_blocking=True)
                targets = batch["positive_targets"].to(device, non_blocking=True)

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
                "symmetry_type": train_dataset.symmetry_type,
                "cad_key_indices": train_dataset.cad_key_indices
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
