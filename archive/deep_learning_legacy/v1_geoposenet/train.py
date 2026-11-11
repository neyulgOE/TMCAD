import os
import sys
import time
import argparse

# Add repo root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from deep_learning.dataset import TableCADDataset
from deep_learning.model import GeometricDGCNN
from deep_learning.loss import SymmetricInfoNCELoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deep Geometric Representation v2")
    parser.add_argument("--dataset-dir",    type=str,   default="/home/huyen1905/TMCAD/dataset/TABLE-DEMO")
    parser.add_argument("--cad-path",       type=str,   default="/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO.stl")
    parser.add_argument("--batch-size",     type=int,   default=16)
    parser.add_argument("--epochs",         type=int,   default=60)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--weight-decay",   type=float, default=1e-4)
    parser.add_argument("--num-points",     type=int,   default=2048)
    parser.add_argument("--num-pos",        type=int,   default=256)
    parser.add_argument("--k-neighbors",    type=int,   default=20)
    parser.add_argument("--desc-dim",       type=int,   default=64)
    parser.add_argument("--temperature",    type=float, default=0.07)
    parser.add_argument("--hard-neg-k",     type=int,   default=64,
                        help="Number of hard negatives to mine per anchor (0 to disable)")
    parser.add_argument("--checkpoint-dir", type=str,   default="/home/huyen1905/TMCAD/deep_learning/checkpoints")
    parser.add_argument("--resume",         type=str,   default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Run 1 epoch with limited steps to test pipeline")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, num_pos, max_steps=None):
    model.train()
    total_loss, total_top1, total_top5, count = 0.0, 0.0, 0.0, 0

    for step, batch in enumerate(loader):
        cad_pts = batch["cad_points"].to(device, non_blocking=True)
        obs_pts = batch["obs_points"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        cad_desc = model(cad_pts)
        obs_desc = model(obs_pts)

        loss, top1, top5 = loss_fn(cad_desc, obs_desc, num_pos=num_pos)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = cad_pts.size(0)
        total_loss += loss.item() * bs
        total_top1 += top1.item() * bs
        total_top5 += top5.item() * bs
        count += bs

        if max_steps and step + 1 >= max_steps:
            break

    return total_loss / count, total_top1 / count, total_top5 / count


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, num_pos, max_steps=None):
    model.eval()
    total_loss, total_top1, total_top5, count = 0.0, 0.0, 0.0, 0

    for step, batch in enumerate(loader):
        cad_pts = batch["cad_points"].to(device, non_blocking=True)
        obs_pts = batch["obs_points"].to(device, non_blocking=True)

        cad_desc = model(cad_pts)
        obs_desc = model(obs_pts)

        loss, top1, top5 = loss_fn(cad_desc, obs_desc, num_pos=num_pos)

        bs = cad_pts.size(0)
        total_loss += loss.item() * bs
        total_top1 += top1.item() * bs
        total_top5 += top5.item() * bs
        count += bs

        if max_steps and step + 1 >= max_steps:
            break

    return total_loss / count, total_top1 / count, total_top5 / count


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print(" PHASE 3 v2: DEEP GEOMETRIC REPRESENTATION TRAINING")
    print(f" Device:          {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Batch size:      {args.batch_size}")
    print(f" Epochs:          {args.epochs}")
    print(f" LR (max):        {args.lr}")
    print(f" Pos pairs:       {args.num_pos}")
    print(f" Descriptor dim:  {args.desc_dim}")
    print(f" Hard neg K:      {args.hard_neg_k}")
    print(f" Temperature:     {args.temperature}")
    print(f" Checkpoints:     {args.checkpoint_dir}")
    print("=" * 65 + "\n")

    # ── Datasets & Loaders ────────────────────────────────────────────────────
    train_dataset = TableCADDataset(
        dataset_dir=args.dataset_dir,
        split="train",
        cad_path=args.cad_path,
        num_points=args.num_points,
        num_pos_pairs=args.num_pos,
        augment=True
    )
    val_dataset = TableCADDataset(
        dataset_dir=args.dataset_dir,
        split="val",
        cad_path=args.cad_path,
        num_points=args.num_points,
        num_pos_pairs=args.num_pos,
        augment=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False
    )

    # ── Model, Loss, Optimizer, Scheduler ────────────────────────────────────
    model = GeometricDGCNN(k=args.k_neighbors, descriptor_dim=args.desc_dim).to(device)
    loss_fn = SymmetricInfoNCELoss(
        temperature=args.temperature,
        hard_negative_k=args.hard_neg_k
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # OneCycleLR: warm-up + cosine decay in one pass — very effective for geometric learning
    steps_per_epoch = len(train_loader) if not args.dry_run else 5
    num_epochs = 1 if args.dry_run else args.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.1,          # 10% warmup
        anneal_strategy="cos",
        div_factor=10.0,        # start lr = max_lr / 10
        final_div_factor=1000.0 # end lr = start_lr / 1000
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_val_top1 = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"[Resume] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"  [WARNING] Could not restore optimizer state: {e}")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_top1 = ckpt.get("val_top1", 0.0)
        print(f"  Resumed from epoch {start_epoch - 1}, best val_top1 = {best_val_top1*100:.2f}%\n")

    max_train_steps = 5 if args.dry_run else None
    max_val_steps   = 3 if args.dry_run else None

    # ── Training Loop ─────────────────────────────────────────────────────────
    print(f"{'Epoch':<7} {'TrLoss':<9} {'TrTop1':<9} {'ValLoss':<9} {'ValTop1':<9} {'ValTop5':<9} {'LR':<10} {'Time':<6}")
    print("-" * 72)

    total_start = time.time()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        t0 = time.time()

        train_loss, train_top1, _ = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn,
            device, args.num_pos, max_steps=max_train_steps
        )
        val_loss, val_top1, val_top5 = evaluate(
            model, val_loader, loss_fn, device,
            args.num_pos, max_steps=max_val_steps
        )
        epoch_time = time.time() - t0
        curr_lr = optimizer.param_groups[0]["lr"]

        is_best = val_top1 > best_val_top1
        marker = " ★" if is_best else ""
        if is_best:
            best_val_top1 = val_top1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_top1": val_top1,
                "val_top5": val_top5,
                "val_loss": val_loss,
                "args": vars(args)
            }, os.path.join(args.checkpoint_dir, "best_geometric_model.pth"))

        # Always save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_top1": val_top1,
            "args": vars(args)
        }, os.path.join(args.checkpoint_dir, "latest_model.pth"))

        print(f"{epoch:<7} {train_loss:<9.4f} {train_top1*100:<8.2f}% "
              f"{val_loss:<9.4f} {val_top1*100:<8.2f}%{marker} "
              f"{val_top5*100:<8.2f}% {curr_lr:<10.2e} {epoch_time:<5.1f}s")

    total_time = time.time() - total_start
    print("-" * 72)
    print(f"Done in {total_time:.0f}s. Best Val Top-1: {best_val_top1*100:.2f}%")
    print(f"Best model: {os.path.join(args.checkpoint_dir, 'best_geometric_model.pth')}")


if __name__ == "__main__":
    main()
