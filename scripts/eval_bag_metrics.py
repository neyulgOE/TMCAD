#!/usr/bin/env python3
"""
Compute Precision / Recall / F1 / FP-per-frame, RMSE, translation jitter, and p50/p95 latency
from `eval_bag_pose` CSV output (with optional ground-truth bbox CSV).
"""
import argparse
import csv
import os
import subprocess
import sys
import numpy as np


def compute_iou(b1, b2):
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[0] + b1[2], b2[0] + b2[2])
    y1 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = max(1e-6, b1[2] * b1[3] + b2[2] * b2[3] - inter)
    return inter / union


def main():
    ap = argparse.ArgumentParser(description="Evaluate .bag pose CSV metrics (Precision/Recall/FP/Jitter)")
    ap.add_argument("--csv", type=str, default="", help="Path to bag_eval.csv (if empty, runs --bag + --onnx)")
    ap.add_argument("--bag", type=str, default="", help="Optional .bag path to run eval_bag_pose")
    ap.add_argument("--onnx", type=str, default="", help="Optional .onnx path to run eval_bag_pose")
    ap.add_argument("--out-dir", type=str, default="bag_eval_out", help="Output directory for eval_bag_pose")
    ap.add_argument("--gt-csv", type=str, default="", help="Optional GT CSV: frame_idx,has_object,bbox_x,bbox_y,bbox_w,bbox_h")
    ap.add_argument("--iou-thresh", type=float, default=0.5, help="2D bbox IoU threshold for TP")
    ap.add_argument("--max-frames", type=int, default=-1)
    ap.add_argument("--temporal", action="store_true")
    ap.add_argument("--multi", action="store_true")
    args = ap.parse_args()

    csv_path = args.csv
    if not csv_path:
        if not args.bag or not args.onnx:
            print("Provide either --csv or both --bag and --onnx", file=sys.stderr)
            sys.exit(1)
        exe = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "eval_bag_pose")
        cmd = [exe, args.bag, args.onnx, args.out_dir]
        if args.max_frames > 0:
            cmd += ["--max-frames", str(args.max_frames)]
        if args.temporal:
            cmd.append("--temporal")
        if args.multi:
            cmd.append("--multi")
        subprocess.run(cmd, check=True)
        csv_path = os.path.join(args.out_dir, "bag_eval.csv")

    rows = []
    with open(csv_path, "r", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if not rows:
        print("No rows in CSV:", csv_path)
        return

    acc_rows = [r for r in rows if int(r["accepted"]) == 1]
    lats = [float(r["total_ms"]) for r in rows]
    confs = [float(r["confidence"]) for r in acc_rows]
    rmses = [float(r["rmse_mm"]) for r in acc_rows]

    jumps_mm = []
    for i in range(1, len(rows)):
        if int(rows[i]["accepted"]) == 1 and int(rows[i - 1]["accepted"]) == 1:
            t0 = np.array([float(rows[i - 1]["tx"]), float(rows[i - 1]["ty"]), float(rows[i - 1]["tz"])])
            t1 = np.array([float(rows[i]["tx"]), float(rows[i]["ty"]), float(rows[i]["tz"])])
            jumps_mm.append(float(np.linalg.norm(t1 - t0) * 1000.0))

    print(f"=== Bag Evaluation Summary ({csv_path}) ===")
    print(f"Frames: {len(rows)} | Accepted: {len(acc_rows)} ({100.0 * len(acc_rows) / len(rows):.1f}%)")
    if confs:
        print(f"Confidence median: {np.median(confs):.3f} | RMSE median: {np.median(rmses):.2f} mm")
    if jumps_mm:
        print(f"Consecutive frame jitter median: {np.median(jumps_mm):.2f} mm | p95: {np.percentile(jumps_mm, 95):.2f} mm")
    print(f"Latency p50 / p95: {np.percentile(lats, 50):.1f} / {np.percentile(lats, 95):.1f} ms")

    if args.gt_csv and os.path.exists(args.gt_csv):
        gt_map = {}
        with open(args.gt_csv, "r", newline="") as f:
            for r in csv.DictReader(f):
                gt_map[int(r["frame_idx"])] = r
        tp = fp = fn = 0
        for r in rows:
            idx = int(r["frame_idx"])
            if idx not in gt_map:
                continue
            gt = gt_map[idx]
            has_obj = int(gt["has_object"]) == 1
            pred_acc = int(r["accepted"]) == 1
            if not has_obj and not pred_acc:
                continue
            if not has_obj and pred_acc:
                fp += 1
            elif has_obj and not pred_acc:
                fn += 1
            else:
                pb = (float(r["bbox_x"]), float(r["bbox_y"]), float(r["bbox_w"]), float(r["bbox_h"]))
                gb = (float(gt["bbox_x"]), float(gt["bbox_y"]), float(gt["bbox_w"]), float(gt["bbox_h"]))
                if compute_iou(pb, gb) >= args.iou_thresh:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        n_eval = len(gt_map)
        print(f"GT Eval ({n_eval} frames, IoU>={args.iou_thresh:.2f}): TP={tp} FP={fp} FN={fn}")
        print(f"Precision: {100.0 * prec:.1f}% | Recall: {100.0 * rec:.1f}% | F1: {100.0 * f1:.1f}% | FP/frame: {fp / max(1, n_eval):.3f}")


if __name__ == "__main__":
    main()

