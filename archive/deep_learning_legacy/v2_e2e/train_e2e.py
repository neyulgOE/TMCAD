import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from deep_learning.dataset_e2e import CADPoseDataset, TableE2EDataset, get_cad_keypoints
from deep_learning.model_e2e import GeoPoseNet
from deep_learning.procrustes import EndToEndPoseLoss, geodesic_rotation_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train End-to-End GeoPoseNet (6D Normals + Keypoint Voting + Procrustes)")
    parser.add_argument("--dataset-dir",    type=str,   default="")
    parser.add_argument("--cad-path",       type=str,   default="/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO.stl")
    parser.add_argument("--cad-name",       type=str,   default="")
    parser.add_argument("--num-keypoints",  type=int,   default=16)
    parser.add_argument("--batch-size",     type=int,   default=16)
    parser.add_argument("--epochs",         type=int,   default=40)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--weight-decay",   type=float, default=1e-4)
    parser.add_argument("--num-points",     type=int,   default=1024)
    parser.add_argument("--k-neighbors",    type=int,   default=20)
    parser.add_argument("--feat-dim",       type=int,   default=128)
    parser.add_argument("--checkpoint-dir", type=str,   default="/home/huyen1905/TMCAD/deep_learning/checkpoints")
    parser.add_argument("--reset-data",     action="store_true", help="Wipe existing checkpoint before training")
    parser.add_argument("--dry-run",        action="store_true")
    return parser.parse_args()


def compute_metrics(R_pred, t_pred, R_gt, t_gt, cad_points):
    """
    Computes translation error (mm), rotation error (deg), and ADD error (mm).
    """
    B = R_pred.shape[0]
    e_t = torch.norm(t_pred - t_gt, dim=-1) * 1000.0  # (B,) mm

    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
    e_R = torch.rad2deg(torch.acos(cos_theta))  # (B,) deg

    # ADD error: (B, N, 3)
    pts_pred = torch.bmm(cad_points, R_pred.transpose(1, 2)) + t_pred.unsqueeze(1)
    pts_gt   = torch.bmm(cad_points, R_gt.transpose(1, 2))   + t_gt.unsqueeze(1)
    e_add = torch.norm(pts_pred - pts_gt, dim=-1).mean(dim=-1) * 1000.0  # (B,) mm

    relaxed_pass = ((e_t <= 20.0) & (e_R <= 5.0)).float()
    strict_pass  = ((e_t <= 5.0)  & (e_R <= 1.0)).float()

    return {
        "e_t": e_t.detach().cpu().numpy(),
        "e_R": e_R.detach().cpu().numpy(),
        "e_add": e_add.detach().cpu().numpy(),
        "relaxed": relaxed_pass.detach().cpu().numpy(),
        "strict": strict_pass.detach().cpu().numpy()
    }


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, max_steps=None):
    model.train()
    total_loss = 0.0
    count = 0

    for step, batch in enumerate(loader):
        obs_6d = batch["obs_6d"].to(device, non_blocking=True)
        R_gt   = batch["R_gt"].to(device, non_blocking=True)
        t_gt   = batch["t_gt"].to(device, non_blocking=True)
        kp_gt  = batch["keypoints_gt"].to(device, non_blocking=True)
        cad_eval = batch["cad_eval_pts"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(obs_6d)
        R_pred = outputs["R"]
        t_pred = outputs["t"]
        kp_pred = outputs["voted_keypoints"]

        loss, metrics = loss_fn(R_pred, t_pred, kp_pred, R_gt, t_gt, kp_gt, cad_eval)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = obs_6d.size(0)
        total_loss += loss.item() * bs
        count += bs

        if max_steps and step + 1 >= max_steps:
            break

    return total_loss / count


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, max_steps=None):
    model.eval()
    total_loss = 0.0
    count = 0
    all_et, all_er, all_add, all_relaxed, all_strict = [], [], [], [], []

    for step, batch in enumerate(loader):
        obs_6d = batch["obs_6d"].to(device, non_blocking=True)
        R_gt   = batch["R_gt"].to(device, non_blocking=True)
        t_gt   = batch["t_gt"].to(device, non_blocking=True)
        kp_gt  = batch["keypoints_gt"].to(device, non_blocking=True)
        cad_eval = batch["cad_eval_pts"].to(device, non_blocking=True)

        outputs = model(obs_6d)
        R_pred = outputs["R"]
        t_pred = outputs["t"]
        kp_pred = outputs["voted_keypoints"]

        loss, metrics = loss_fn(R_pred, t_pred, kp_pred, R_gt, t_gt, kp_gt, cad_eval)
        bs = obs_6d.size(0)
        total_loss += loss.item() * bs
        count += bs

        m = compute_metrics(R_pred, t_pred, R_gt, t_gt, cad_eval)
        all_et.extend(m["e_t"])
        all_er.extend(m["e_R"])
        all_add.extend(m["e_add"])
        all_relaxed.extend(m["relaxed"])
        all_strict.extend(m["strict"])

        if max_steps and step + 1 >= max_steps:
            break

    return {
        "val_loss": total_loss / count,
        "mean_et": np.mean(all_et),
        "median_et": np.median(all_et),
        "mean_er": np.mean(all_er),
        "median_er": np.median(all_er),
        "mean_add": np.mean(all_add),
        "relaxed_pass_pct": np.mean(all_relaxed) * 100.0,
        "strict_pass_pct": np.mean(all_strict) * 100.0
    }


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print(" END-TO-END GEO-POSE-NET TRAINING (6D Normals + Keypoint Voting + SVD)")
    print(f" Device:         {device}")
    if torch.cuda.is_available():
        print(f" GPU:            {torch.cuda.get_device_name(0)}")
    print(f" Epochs:         {args.epochs}")
    print(f" Batch size:     {args.batch_size}")
    print(f" Initial LR:     {args.lr}")
    print(f" Points:         {args.num_points}")
    print(f" K-Neighbors:    {args.k_neighbors}")
    print("=" * 75)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    cad_name = args.cad_name if args.cad_name else os.path.splitext(os.path.basename(args.cad_path))[0]
    if not args.dataset_dir:
        tmcad_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.dataset_dir = os.path.join(tmcad_root, "dataset", cad_name)

    cad_ckpt_dir = os.path.join(args.checkpoint_dir, cad_name)
    os.makedirs(cad_ckpt_dir, exist_ok=True)
    best_checkpoint = os.path.join(cad_ckpt_dir, "best_model.pth")
    legacy_best_checkpoint = os.path.join(args.checkpoint_dir, "best_e2e_pose_model.pth")

    if args.reset_data:
        if os.path.exists(best_checkpoint):
            os.remove(best_checkpoint)
            print(f"[RESET] Cleared previous checkpoint: {best_checkpoint}")

    # 1. Dataset & Keypoints
    cad_kp = get_cad_keypoints(args.cad_path, num_keypoints=args.num_keypoints)
    print(f"[CAD SURFACE KEYPOINTS] Extracted {len(cad_kp)} surface points via FPS on {cad_name}")

    train_dataset = CADPoseDataset(
        dataset_dir=args.dataset_dir, split="train", cad_path=args.cad_path,
        num_points=args.num_points, num_keypoints=args.num_keypoints, augment=True
    )
    val_dataset = CADPoseDataset(
        dataset_dir=args.dataset_dir, split="val", cad_path=args.cad_path,
        num_points=args.num_points, num_keypoints=args.num_keypoints, augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # 2. Model & Loss
    model = GeoPoseNet(
        cad_keypoints=cad_kp, k_neighbors=args.k_neighbors, feat_dim=args.feat_dim
    ).to(device)
    loss_fn = EndToEndPoseLoss(w_add=2.0, w_trans=5.0, w_rot=2.0, w_kp=10.0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader) if not args.dry_run else 3
    total_steps = steps_per_epoch * args.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0
    )

    best_add = float("inf")
    best_relaxed = 0.0

    print(f"\n{'Epoch':<7} {'TrainLoss':<11} {'ValLoss':<11} {'ValADD(mm)':<13} {'MedEt(mm)':<12} {'MedEr(deg)':<12} {'RlxPass%':<10} {'LR':<10} {'Time'}")
    print("-" * 92)

    start_total = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        max_steps = 3 if args.dry_run else None

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, max_steps)
        val_res = evaluate(model, val_loader, loss_fn, device, max_steps)
        epoch_time = time.time() - t0

        curr_lr = scheduler.get_last_lr()[0]

        is_best = val_res["mean_add"] < best_add
        star = "★" if is_best else " "
        if is_best:
            best_add = val_res["mean_add"]
            best_relaxed = val_res["relaxed_pass_pct"]
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cad_keypoints": cad_kp,
                "best_add": best_add,
                "val_res": val_res,
                "args": vars(args)
            }
            torch.save(ckpt_dict, best_checkpoint)
            torch.save(ckpt_dict, legacy_best_checkpoint)

        print(
            f"{epoch:<7} "
            f"{train_loss:<11.4f} "
            f"{val_res['val_loss']:<11.4f} "
            f"{val_res['mean_add']:<8.2f}{star:<5} "
            f"{val_res['median_et']:<12.1f} "
            f"{val_res['median_er']:<12.1f} "
            f"{val_res['relaxed_pass_pct']:<10.2f} "
            f"{curr_lr:<10.2e} "
            f"{epoch_time:.1f} s"
        )
        # Machine-readable output for Qt QProcess
        print(
            f"PROGRESS:epoch={epoch},total_epochs={args.epochs},"
            f"loss={train_loss:.4f},val_loss={val_res['val_loss']:.4f},"
            f"add_mm={val_res['mean_add']:.1f},med_et_mm={val_res['median_et']:.1f},"
            f"med_er_deg={val_res['median_er']:.1f},pass_pct={val_res['relaxed_pass_pct']:.1f}",
            flush=True
        )

        if args.dry_run and epoch >= 2:
            print("\n[Dry-run completed successfully!]")
            break

    total_time = time.time() - start_total
    print("-" * 92)
    print(f"Training completed in {total_time:.1f}s. Best Val ADD: {best_add:.2f} mm | Relaxed: {best_relaxed:.2f}%")
    print(f"Best model saved to: {best_checkpoint}")


if __name__ == "__main__":
    main()
