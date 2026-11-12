#!/usr/bin/env python3
"""Build the same metric RGB-D point cloud used by the hierarchy worker."""

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


ROOT = Path("/home/huyen1905/TMCAD")
DEPTH = ROOT / "dump_depth.png"
OUT = ROOT / "debug" / "dump_hierarchy_input.ply"

FX, FY, PPX, PPY = 608.215, 608.128, 341.473, 228.438
DEPTH_SCALE = 0.001
VOXEL = 0.005


def main() -> None:
    depth = cv2.imread(str(DEPTH), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise RuntimeError(f"Cannot load depth image: {DEPTH}")

    v, u = np.indices(depth.shape, dtype=np.float32)
    z = depth.astype(np.float32) * DEPTH_SCALE
    valid = (z > 0.0) & (z <= 3.0)
    z = z[valid]
    x = (u[valid] - PPX) * z / FX
    y = (v[valid] - PPY) * z / FY
    points = np.column_stack((x, y, z)).astype(np.float64)

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(VOXEL)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(OUT), cloud, write_ascii=True):
        raise RuntimeError(f"Cannot write point cloud: {OUT}")
    print(f"wrote {len(cloud.points)} points to {OUT}")


if __name__ == "__main__":
    main()
