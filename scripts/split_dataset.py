import os
import json
import random
import numpy as np
from collections import defaultdict

def split_dataset(
    dataset_dir="/home/huyen1905/TMCAD/dataset/TABLE-DEMO",
    n_train=None,
    n_val=None,
    n_test=None,
    train_ratio=0.80,
    val_ratio=0.10,
    test_ratio=0.10,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)
    
    assert os.path.isdir(dataset_dir), f"Directory {dataset_dir} does not exist!"
    samples = sorted([d for d in os.listdir(dataset_dir) if d.startswith("sample_") and os.path.isdir(os.path.join(dataset_dir, d))])
    total = len(samples)
    if n_train is None or n_val is None or n_test is None:
        n_val = int(round(total * val_ratio))
        n_test = int(round(total * test_ratio))
        n_train = total - n_val - n_test
    assert total == n_train + n_val + n_test, f"Found {total} samples, but expected {n_train + n_val + n_test}"

    # Group samples by targetVisibleFraction for stratified splitting
    by_occlusion = defaultdict(list)
    metadata = {}
    
    for s in samples:
        s_path = os.path.join(dataset_dir, s)
        pose_path = os.path.join(s_path, "pose.json")
        with open(pose_path, "r") as f:
            meta = json.load(f)
        vis = meta.get("targetVisibleFraction", 1.0)
        by_occlusion[vis].append(s)
        metadata[s] = meta

    train_list = []
    val_list = []
    test_list = []

    # Sort keys for determinism
    sorted_levels = sorted(by_occlusion.keys())
    
    # Calculate exact counts per category
    # Ratio: 1500/2000 = 0.75, 250/2000 = 0.125, 250/2000 = 0.125
    for vis in sorted_levels:
        group = by_occlusion[vis]
        random.shuffle(group)
        n_group = len(group)
        n_g_val = int(round(n_group * (n_val / total)))
        n_g_test = int(round(n_group * (n_test / total)))
        n_g_train = n_group - n_g_val - n_g_test
        
        train_list.extend(group[:n_g_train])
        val_list.extend(group[n_g_train:n_g_train + n_g_val])
        test_list.extend(group[n_g_train + n_g_val:])

    # Adjust any slight rounding discrepancies so sums are exact (1500, 250, 250)
    while len(val_list) > n_val:
        train_list.append(val_list.pop())
    while len(val_list) < n_val:
        val_list.append(train_list.pop())
        
    while len(test_list) > n_test:
        train_list.append(test_list.pop())
    while len(test_list) < n_test:
        test_list.append(train_list.pop())

    train_list.sort()
    val_list.sort()
    test_list.sort()

    assert len(train_list) == n_train, f"Train count mismatch: {len(train_list)}"
    assert len(val_list) == n_val, f"Val count mismatch: {len(val_list)}"
    assert len(test_list) == n_test, f"Test count mismatch: {len(test_list)}"
    
    # Check disjointness
    set_train, set_val, set_test = set(train_list), set(val_list), set(test_list)
    assert len(set_train.intersection(set_val)) == 0, "Train and Val overlap!"
    assert len(set_train.intersection(set_test)) == 0, "Train and Test overlap!"
    assert len(set_val.intersection(set_test)) == 0, "Val and Test overlap!"

    # Move sample folders into train/val/test subdirectories if currently in root
    import shutil
    for folder, s_list in [("train", train_list), ("val", val_list), ("test", test_list)]:
        target_dir = os.path.join(dataset_dir, folder)
        os.makedirs(target_dir, exist_ok=True)
        for s in s_list:
            src = os.path.join(dataset_dir, s)
            dst = os.path.join(target_dir, s)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)

    # Save to files with relative paths matching directory layout
    train_txt = os.path.join(dataset_dir, "train.txt")
    val_txt = os.path.join(dataset_dir, "val.txt")
    test_txt = os.path.join(dataset_dir, "test.txt")
    splits_json = os.path.join(dataset_dir, "splits.json")

    with open(train_txt, "w") as f:
        f.write("\n".join(f"train/{s}" for s in train_list) + "\n")
    with open(val_txt, "w") as f:
        f.write("\n".join(f"val/{s}" for s in val_list) + "\n")
    with open(test_txt, "w") as f:
        f.write("\n".join(f"test/{s}" for s in test_list) + "\n")

    splits_data = {
        "train": [f"train/{s}" for s in train_list],
        "val": [f"val/{s}" for s in val_list],
        "test": [f"test/{s}" for s in test_list],
        "counts": {
            "total": total,
            "train": len(train_list),
            "val": len(val_list),
            "test": len(test_list)
        }
    }
    with open(splits_json, "w") as f:
        json.dump(splits_data, f, indent=2)

    print("=" * 60)
    print("DATASET SPLIT SUMMARY (Stratified by Occlusion Level)")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Train: {len(train_list)} (saved to {train_txt})")
    print(f"Val:   {len(val_list)} (saved to {val_txt})")
    print(f"Test:  {len(test_list)} (saved to {test_txt})")
    print(f"JSON:  {splits_json}")
    print("-" * 60)
    print(f"{'Occlusion (Vis)':<18} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print("-" * 60)
    for vis in sorted_levels:
        c_tr = sum(1 for s in train_list if metadata[s].get("targetVisibleFraction") == vis)
        c_va = sum(1 for s in val_list if metadata[s].get("targetVisibleFraction") == vis)
        c_te = sum(1 for s in test_list if metadata[s].get("targetVisibleFraction") == vis)
        print(f"{vis:<18} | {c_tr:<8} | {c_va:<8} | {c_te:<8} | {c_tr + c_va + c_te:<8}")
    print("=" * 60)

if __name__ == "__main__":
    split_dataset()
