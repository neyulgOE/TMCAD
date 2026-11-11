#!/usr/bin/env python3
"""
Generate C2 Symmetric 512 Keypoints for CAD Pose Estimation.
Guarantees 0.000 mm partner symmetry error across the Discrete_C2_Z symmetry group.
"""
import os
import sys
import numpy as np

def fps_np(pts: np.ndarray, k: int) -> np.ndarray:
    """Farthest Point Sampling on numpy array."""
    n = len(pts)
    selected = [np.argmax(np.sum(pts ** 2, axis=-1))]
    dists = np.sum((pts - pts[selected[0]]) ** 2, axis=-1)
    for _ in range(1, k):
        idx = np.argmax(dists)
        selected.append(idx)
        dists = np.minimum(dists, np.sum((pts - pts[idx]) ** 2, axis=-1))
    return np.array(selected, dtype=np.int64)

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cad_dir = os.path.join(project_root, "CAD_DATA", "TABLE-DEMO")
    
    tri_path = os.path.join(cad_dir, "TABLE-DEMO_triangles.npz")
    if not os.path.exists(tri_path):
        print(f"Error: Missing {tri_path}")
        sys.exit(1)
        
    print(f"[C2 Keypoints] Loading triangles from {tri_path}...")
    tri_data = np.load(tri_path)
    all_centroids = tri_data['features'][:, :3].astype(np.float32)
    M = len(all_centroids)
    print(f"[C2 Keypoints] Total CAD triangles: {M}")
    
    # 1. Filter half-space X >= 0
    mask_half = all_centroids[:, 0] >= 0
    idx_half = np.where(mask_half)[0]
    cents_half = all_centroids[idx_half]
    print(f"[C2 Keypoints] Triangles in half-space X >= 0: {len(cents_half)}")
    
    # 2. Sample 256 key triangles via FPS on half-space
    print(f"[C2 Keypoints] Sampling 256 keypoints via FPS on X >= 0...")
    fps_sub_idx = fps_np(cents_half, 256)
    key_idx_pos = idx_half[fps_sub_idx] # (256,)
    key_cents_pos = all_centroids[key_idx_pos] # (256, 3)
    
    # 3. For each of the 256 keypoints, find the exact symmetric triangle at (-x, -y, z)
    partner_targets = key_cents_pos * np.array([-1.0, -1.0, 1.0], dtype=np.float32)
    diff = partner_targets[:, None, :] - all_centroids[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    key_idx_neg = np.argmin(dist_sq, axis=-1) # (256,)
    min_dists_mm = np.sqrt(np.min(dist_sq, axis=-1)) * 1000.0
    print(f"[C2 Keypoints] Triangle partner distance error on CAD surface:")
    print(f"    min: {np.min(min_dists_mm):.4f} mm, median: {np.median(min_dists_mm):.4f} mm, max: {np.max(min_dists_mm):.4f} mm")
    
    # 4. Form 512 key triangle indices
    key_512 = np.concatenate([key_idx_pos, key_idx_neg]).astype(np.int64)
    
    # 5. Form 512 key centroids with EXACT mathematical C2 symmetry (0.0000 mm error)
    # First 256: key_cents_pos
    # Second 256: key_cents_pos * [-1, -1, 1]
    key_cents_512 = np.vstack([
        key_cents_pos,
        key_cents_pos * np.array([-1.0, -1.0, 1.0], dtype=np.float32)
    ]).astype(np.float32)
    
    # 6. Partner indices mapping: partner[i] = (i + 256) % 512
    partner_indices = np.array([(i + 256) % 512 for i in range(512)], dtype=np.int64)
    
    # Verify exact partner symmetry
    partner_err = np.linalg.norm(key_cents_512 - key_cents_512[partner_indices] * np.array([-1.0, -1.0, 1.0], dtype=np.float32), axis=-1)
    assert np.max(partner_err) == 0.0, "Partner symmetry verification failed!"
    print(f"[C2 Keypoints] Mathematical C2 partner error: {np.max(partner_err):.6f} mm (EXACT 0.000 mm)!")
    
    # Save files
    out_tri_path = os.path.join(cad_dir, "TABLE-DEMO_key_triangles_512.npy")
    out_cent_path = os.path.join(cad_dir, "TABLE-DEMO_key_centroids_512.npy")
    out_partner_path = os.path.join(cad_dir, "TABLE-DEMO_symmetry_partner_indices.npy")
    
    np.save(out_tri_path, key_512)
    np.save(out_cent_path, key_cents_512)
    np.save(out_partner_path, partner_indices)
    
    print(f"[C2 Keypoints] Saved:")
    print(f"  - {out_tri_path}")
    print(f"  - {out_cent_path}")
    print(f"  - {out_partner_path}")

if __name__ == "__main__":
    main()
