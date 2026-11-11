#!/usr/bin/env python3
"""
bake_cad_templates.py
Offline tool to pre-render and pre-bake 36-angle CAD templates into binary format (.bin).
Placed outside of ObjectDetection_V2.4 to keep the plugin clean.
"""

import os
import sys
import struct
import json
import math
import numpy as np
import cv2

MAGIC_HEADER = b"CADTPL01"  # 8 bytes

def rotation_x(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [1.0, 0.0,  0.0],
        [0.0,   c,   -s],
        [0.0,   s,    c]
    ], dtype=np.float64)

def rotation_z(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [  c,  -s,  0.0],
        [  s,   c,  0.0],
        [0.0, 0.0,  1.0]
    ], dtype=np.float64)

def bake_templates(cad_dimensions_path, centroids_path, output_bin_path, num_templates=36):
    print(f"[*] Reading CAD dimensions: {cad_dimensions_path}")
    half_w = 0.220
    half_l = 0.415
    top_z = 0.1665
    bot_z = -0.3355

    if os.path.exists(cad_dimensions_path):
        with open(cad_dimensions_path, "r") as f:
            dims = json.load(f)
            half_w = float(dims.get("width_m", 0.44)) * 0.5
            half_l = float(dims.get("length_m", 0.83)) * 0.5
            top_z = float(dims.get("tabletop_center_offset_z_m", 0.1665))
            bot_z = float(dims.get("ground_contact_offset_z_m", -0.3355))

    print(f"[*] Geometry: half_w={half_w:.3f}m, half_l={half_l:.3f}m, top_z={top_z:.4f}m, bot_z={bot_z:.4f}m")

    # Load 3D centroids if present
    pts_3d = []
    if os.path.exists(centroids_path):
        print(f"[*] Loading 3D centroids: {centroids_path}")
        centroids = np.load(centroids_path)
        for pt in centroids:
            pts_3d.append([float(pt[0]), float(pt[1]), float(pt[2])])

    # Add 8 structural corners
    corners = [
        [-half_w, -half_l, top_z],
        [ half_w, -half_l, top_z],
        [ half_w,  half_l, top_z],
        [-half_w,  half_l, top_z],
        [-half_w, -half_l, bot_z],
        [ half_w, -half_l, bot_z],
        [ half_w,  half_l, bot_z],
        [-half_w,  half_l, bot_z],
    ]
    pts_3d.extend(corners)

    # Perimeter dense sampling
    n_perim = 20
    for i in range(n_perim + 1):
        alpha = float(i) / n_perim
        x = -half_w + 2.0 * half_w * alpha
        y = -half_l + 2.0 * half_l * alpha
        pts_3d.append([x, -half_l, top_z])
        pts_3d.append([x,  half_l, top_z])
        pts_3d.append([-half_w, y, top_z])
        pts_3d.append([ half_w, y, top_z])

    pts_3d = np.array(pts_3d, dtype=np.float64)

    # Standard RealSense D435 Intrinsics
    K = np.array([
        [605.0,   0.0, 320.0],
        [  0.0, 605.0, 240.0],
        [  0.0,   0.0,   1.0]
    ], dtype=np.float64)

    orb = cv2.ORB_create(
        nfeatures=400,
        scaleFactor=1.2,
        nlevels=4,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=15
    )

    templates_data = []
    pitch_rad = 20.0 * math.pi / 180.0
    Rx = rotation_x(pitch_rad)
    t = np.array([0.0, 0.25, 1.25], dtype=np.float64)

    print(f"[*] Baking {num_templates} CAD templates (every {360 // num_templates}°)...")

    for i in range(num_templates):
        yaw_deg = float(i * (360 // num_templates))
        yaw_rad = yaw_deg * math.pi / 180.0
        Rz = rotation_z(yaw_rad)
        R = Rx @ Rz

        # Project points to 640x480 canvas
        canvas = np.zeros((480, 640), dtype=np.uint8)

        def proj_point(p3):
            pc = R @ p3 + t
            if pc[2] <= 0.1:
                return None
            u = int(K[0, 0] * pc[0] / pc[2] + K[0, 2])
            v = int(K[1, 1] * pc[1] / pc[2] + K[1, 2])
            return (u, v)

        # Corners
        c_proj = [proj_point(np.array(c, dtype=np.float64)) for c in corners]

        all_proj_u = []
        all_proj_v = []
        for pt in pts_3d:
            p2 = proj_point(pt)
            if p2 and 0 <= p2[0] < 640 and 0 <= p2[1] < 480:
                all_proj_u.append(p2[0])
                all_proj_v.append(p2[1])

        if len(all_proj_u) < 10 or any(c is None for c in c_proj):
            print(f"[-] Warning: skipping template yaw={yaw_deg}")
            continue

        min_u, max_u = min(all_proj_u), max(all_proj_u)
        min_v, max_v = min(all_proj_v), max(all_proj_v)
        roi_bbox = (float(min_u), float(min_v), float(max_u - min_u), float(max_v - min_v))

        # Draw wireframe & polygon
        top_poly = np.array([c_proj[0], c_proj[1], c_proj[2], c_proj[3]], dtype=np.int32)
        cv2.fillConvexPoly(canvas, top_poly, 180)
        cv2.polylines(canvas, [top_poly], True, 255, 2)

        # 4 Legs
        cv2.line(canvas, c_proj[0], c_proj[4], 220, 2)
        cv2.line(canvas, c_proj[1], c_proj[5], 220, 2)
        cv2.line(canvas, c_proj[2], c_proj[6], 220, 2)
        cv2.line(canvas, c_proj[3], c_proj[7], 220, 2)

        # Internal texture dots
        for u, v in zip(all_proj_u, all_proj_v):
            cv2.circle(canvas, (u, v), 1, 240, -1)

        # Detect ORB
        kpts, desc = orb.detectAndCompute(canvas, None)
        if desc is None or len(kpts) < 8:
            print(f"[-] Warning: low features at yaw={yaw_deg}")
            continue

        templates_data.append({
            "yaw_deg": yaw_deg,
            "bbox": roi_bbox,
            "kpts": kpts,
            "desc": desc
        })

    print(f"[+] Successfully generated {len(templates_data)} valid templates")

    # Serialize to Binary Format
    os.makedirs(os.path.dirname(output_bin_path), exist_ok=True)
    with open(output_bin_path, "wb") as f:
        # Header: magic(8 bytes) + num_templates(uint32) + desc_bytes(uint32 = 32)
        f.write(MAGIC_HEADER)
        f.write(struct.pack("<II", len(templates_data), 32))

        for tmpl in templates_data:
            yaw = tmpl["yaw_deg"]
            bx, by, bw, bh = tmpl["bbox"]
            kpts = tmpl["kpts"]
            desc = tmpl["desc"]
            n_kpts = len(kpts)

            # Per template header: yaw(f), bbox(4f), n_kpts(I)
            f.write(struct.pack("<fffffI", yaw, bx, by, bw, bh, n_kpts))

            # Keypoints: pt.x(f), pt.y(f), size(f), angle(f), response(f)
            for kp in kpts:
                f.write(struct.pack("<fffff", kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response))

            # Descriptors: n_kpts * 32 bytes
            f.write(desc.tobytes())

    file_size_kb = os.path.getsize(output_bin_path) / 1024.0
    print(f"[+] Baked templates saved to: {output_bin_path} ({file_size_kb:.1f} KB)")
    return output_bin_path

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dims_file = os.path.join(base_dir, "CAD_DATA/TABLE-DEMO/table_dimensions.json")
    if not os.path.exists(dims_file):
        dims_file = os.path.join(base_dir, "CAD_DATA/table_dimensions.json")

    centroids_file = os.path.join(base_dir, "CAD_DATA/TABLE-DEMO/TABLE-DEMO_key_centroids.npy")
    if not os.path.exists(centroids_file):
        centroids_file = os.path.join(base_dir, "CAD_DATA/TABLE-DEMO_key_centroids.npy")
        if not os.path.exists(centroids_file):
            centroids_file = os.path.join(base_dir, "ObjectDetection_V2.4/models/object_detection/cad_pose/table_cad/v1/cad_data/TABLE-DEMO_key_centroids.npy")
    
    # Primary output: directly into V2.4 model store
    out_file = os.path.join(
        base_dir,
        "ObjectDetection_V2.4/models/object_detection/cad_pose/table_cad/v1/cad_data/TABLE-DEMO_cad_templates.bin"
    )
    
    # Also save a copy in CAD_DATA/TABLE-DEMO
    out_file_caddata = os.path.join(base_dir, "CAD_DATA/TABLE-DEMO/TABLE-DEMO_cad_templates.bin")

    bake_templates(dims_file, centroids_file, out_file)
    import shutil
    os.makedirs(os.path.dirname(out_file_caddata), exist_ok=True)
    shutil.copyfile(out_file, out_file_caddata)
    print(f"[+] Copied to: {out_file_caddata}")
