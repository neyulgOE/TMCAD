#!/usr/bin/env python3
"""
Bake 168-view (42 icosphere x 4 in-plane rotations) DINOv2 + binary silhouette templates
directly from any CAD mesh (.stl/.obj/.ply) into <CAD>_templates.bin (CADMV01 format).
"""
import argparse
import glob
import json
import os
import struct
import time
import cv2
import numpy as np
import onnxruntime as ort
import open3d as o3d
import trimesh

MAGIC = b"CADMV01\x00"


def letterbox(img: np.ndarray, size: int = 224, bg_val: int = 128) -> np.ndarray:
    h, w = img.shape[:2]
    scale = (size * 0.892857) / max(1, max(h, w))  # ~200/224
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    if img.ndim == 2:
        out = np.full((size, size), bg_val, dtype=np.uint8)
    else:
        out = np.full((size, size, img.shape[2]), bg_val, dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def find_cad_mesh(cad_name: str, project_root: str) -> str:
    cad_dir = os.path.join(project_root, "CAD_DATA", cad_name)
    canon_ply = os.path.join(cad_dir, f"{cad_name}_mesh.ply")
    if os.path.exists(canon_ply):
        return canon_ply
    for ext in ("*.ply", "*.obj", "*.stl", "*.PLY", "*.OBJ", "*.STL"):
        matches = sorted(glob.glob(os.path.join(cad_dir, ext)))
        for m in matches:
            if os.path.splitext(os.path.basename(m))[0] == cad_name:
                return m
        if matches:
            return matches[0]
    hope_dir = os.path.join(project_root, "data", "hope_sample", "cad_models")
    for ext in (".obj", ".stl"):
        p = os.path.join(hope_dir, cad_name + ext)
        if os.path.exists(p):
            return p
    return ""


def compute_area_weighted_centroid(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return verts.mean(axis=0)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total_area = float(np.sum(areas))
    if total_area > 1e-12:
        tri_centers = (v0 + v1 + v2) / 3.0
        return np.sum(tri_centers * areas[:, None], axis=0) / total_area
    return verts.mean(axis=0)


def render_168_views(mesh_path: str, img_size: int = 224, mask_size: int = 64):
    loaded = trimesh.load(mesh_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    verts = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if not os.path.basename(mesh_path).endswith("_mesh.ply"):
        center = compute_area_weighted_centroid(verts, faces)
        verts = verts - center
    radius = float(np.linalg.norm(verts, axis=1).max())
    if radius < 1e-6:
        radius = 1.0
    verts /= radius

    has_vc = (hasattr(loaded.visual, "vertex_colors") and
              loaded.visual.vertex_colors is not None and
              len(loaded.visual.vertex_colors) == len(verts))
    if len(faces) > 12000:
        m_o3d = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(faces)
        )
        m_dec = m_o3d.simplify_quadric_decimation(10000)
        verts = np.asarray(m_dec.vertices, dtype=np.float64)
        faces = np.asarray(m_dec.triangles, dtype=np.int64)
        has_vc = False

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fnorm = np.cross(v1 - v0, v2 - v0)
    fnorm /= np.maximum(np.linalg.norm(fnorm, axis=1, keepdims=True), 1e-9)
    fcenter = (v0 + v1 + v2) / 3.0

    if has_vc:
        vc = np.asarray(loaded.visual.vertex_colors[:, :3], dtype=np.float64) / 255.0
        fcolor = (vc[faces[:, 0]] + vc[faces[:, 1]] + vc[faces[:, 2]]) / 3.0
        if np.std(fcolor) < 0.02:
            fcolor = np.full_like(fcolor, 0.78)
    else:
        fcolor = np.full((len(faces), 3), 0.78, dtype=np.float64)

    ico = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    dirs = np.asarray(ico.vertices, dtype=np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    focal = img_size * 1.25
    cam_dist = 3.0
    cx = cy = 0.5 * img_size
    light_dir = np.array([0.35, -0.55, 0.75], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)

    views = []
    for d in dirs:
        z_axis = -d / np.linalg.norm(d)
        world_up = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.92 else np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(z_axis, world_up)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R_cw = np.stack([x_axis, y_axis, z_axis], axis=0)

        fc_cam = fcenter @ R_cw.T
        fc_cam[:, 2] += cam_dist
        fn_cam = fnorm @ R_cw.T
        vis = (fn_cam[:, 2] < 0.0) & (fc_cam[:, 2] > 0.2)
        if not np.any(vis):
            continue
        vis_idx = np.where(vis)[0]
        order = vis_idx[np.argsort(-fc_cam[vis_idx, 2])]

        pts_cam = verts @ R_cw.T
        pts_cam[:, 2] += cam_dist
        u = focal * (pts_cam[:, 0] / np.maximum(pts_cam[:, 2], 0.1)) + cx
        v = focal * (pts_cam[:, 1] / np.maximum(pts_cam[:, 2], 0.1)) + cy

        canvas = np.full((img_size, img_size, 3), 128, dtype=np.uint8)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        ndotl = np.clip(-np.sum(fn_cam * light_dir[None, :], axis=1), 0.0, 1.0)
        shade = 0.35 + 0.65 * ndotl

        tris_2d = np.stack([
            np.stack([u[faces[order, 0]], v[faces[order, 0]]], axis=1),
            np.stack([u[faces[order, 1]], v[faces[order, 1]]], axis=1),
            np.stack([u[faces[order, 2]], v[faces[order, 2]]], axis=1),
        ], axis=1).astype(np.int32)
        cols_bgr = np.clip(fcolor[order] * shade[order, None] * 255.0, 0, 255).astype(np.uint8)
        for tri, col in zip(tris_2d, cols_bgr):
            cv2.fillConvexPoly(canvas, tri, (int(col[2]), int(col[1]), int(col[0])))
            cv2.fillConvexPoly(mask, tri, 255)

        ys, xs = np.where(mask > 0)
        if len(xs) < 20:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        c224 = letterbox(canvas[y0:y1 + 1, x0:x1 + 1], size=img_size, bg_val=128)
        m224 = letterbox(mask[y0:y1 + 1, x0:x1 + 1], size=img_size, bg_val=0)
        m64 = (cv2.resize(m224, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)

        for k_rot in range(4):
            rad = k_rot * np.pi * 0.5
            cs, sn = np.cos(rad), np.sin(rad)
            R_in = np.array([[cs, sn, 0.0], [-sn, cs, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            views.append((np.rot90(c224, k_rot).copy(), np.rot90(m64, k_rot).copy(), (R_in @ R_cw).astype(np.float32)))
    return views


def encode_with_onnx(dino_onnx_path: str, imgs_bgr) -> np.ndarray:
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    sess = ort.InferenceSession(dino_onnx_path, providers=providers)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    feats = []
    batch_sz = 42
    for i in range(0, len(imgs_bgr), batch_sz):
        chunk = imgs_bgr[i:i + batch_sz]
        arr = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in chunk], axis=0).astype(np.float32) / 255.0
        arr = (np.transpose(arr, (0, 3, 1, 2)) - mean) / std
        out = sess.run(None, {"images": arr})[0].astype(np.float32)
        feats.append(out)
    return np.concatenate(feats, axis=0)


def write_templates_bin(out_path: str, views, feats: np.ndarray, mask_size: int = 64):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    num_views = len(views)
    feat_dim = int(feats.shape[1])
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", num_views, feat_dim, mask_size))
        for idx, (_, m64, R_cw) in enumerate(views):
            f.write(R_cw.astype(np.float32).tobytes())
            f.write(feats[idx].astype(np.float32).tobytes())
            f.write(m64.astype(np.uint8).tobytes())


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cad-name", default="")
    parser.add_argument("--cad-mesh", default="")
    parser.add_argument("--out-bin", default="")
    parser.add_argument("--dino-onnx", default=os.path.join(project_root, "models", "shared", "dinov2_vits14.onnx"))
    parser.add_argument("--profile-json", default="")
    args = parser.parse_args()

    mesh_path = args.cad_mesh
    if not mesh_path and args.cad_name:
        mesh_path = find_cad_mesh(args.cad_name, project_root)
    if not mesh_path or not os.path.exists(mesh_path):
        raise FileNotFoundError(f"CAD mesh not found (cad_name={args.cad_name}, cad_mesh={args.cad_mesh})")

    cad_stem = args.cad_name if args.cad_name else os.path.splitext(os.path.basename(mesh_path))[0]
    out_bin = args.out_bin if args.out_bin else os.path.join(project_root, "models", f"{cad_stem}_templates.bin")

    t0 = time.perf_counter()
    views = render_168_views(mesh_path, img_size=224, mask_size=64)
    feats = encode_with_onnx(args.dino_onnx, [v[0] for v in views])
    write_templates_bin(out_bin, views, feats, mask_size=64)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[bake_cad_multiview_templates] {cad_stem}: {len(views)} views -> {out_bin} "
          f"({os.path.getsize(out_bin)/1024:.1f} KB) in {dt_ms:.1f} ms")

    if args.profile_json and os.path.exists(args.profile_json):
        with open(args.profile_json, "r") as f:
            prof = json.load(f)
        prof["multiview_templates_bin"] = os.path.abspath(out_bin)
        with open(args.profile_json, "w") as f:
            json.dump(prof, f, indent=2)


if __name__ == "__main__":
    main()

