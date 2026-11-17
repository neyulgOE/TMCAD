#!/usr/bin/env python3
"""
Export shared 2D vision backbones for Universal Zero-Shot CAD Mask Matching:
  1. models/shared/fastsam_s.onnx      (input: images [1,3,640,640] float32 0..1 -> FastSAM-s outputs)
  2. models/shared/dinov2_vits14.onnx  (input: images [N,3,224,224] float32 ImageNet-norm -> [N,384] L2-normed)
"""
import argparse
import os
import shutil
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from ultralytics import FastSAM


class DinoV2NormalizedWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return torch.nn.functional.normalize(feat, p=2, dim=-1)


def export_fastsam(out_path: str, weights_path: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"FastSAM weights not found at {weights_path}")
    print(f"[1/2] Exporting FastSAM-s from {weights_path} -> {out_path}")
    model = FastSAM(weights_path)
    exported = model.export(format="onnx", imgsz=640, opset=17, simplify=True, half=False)
    exported_str = str(exported)
    if os.path.abspath(exported_str) != os.path.abspath(out_path):
        shutil.move(exported_str, out_path)

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    outs = sess.run(None, {in_name: dummy})
    print(f"      Verified FastSAM ONNX: input={in_name} {sess.get_inputs()[0].shape}, "
          f"outputs={[o.shape for o in outs]}")


def export_dinov2(out_path: str):
    print(f"[2/2] Exporting DINOv2-vits14 -> {out_path}")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cpu().eval()
    wrapper = DinoV2NormalizedWrapper(backbone).eval()
    dummy = torch.randn(2, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
    )
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"images": dummy.numpy()})
    print(f"      Verified DINOv2 ONNX: output={outs[0].shape}, norm={np.linalg.norm(outs[0][0]):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/home/huyen1905/TMCAD/models/shared")
    parser.add_argument("--fastsam-pt", default="/tmp/FastSAM-s.pt")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    export_fastsam(os.path.join(args.out_dir, "fastsam_s.onnx"), args.fastsam_pt)
    export_dinov2(os.path.join(args.out_dir, "dinov2_vits14.onnx"))
    print("Done exporting shared vision ONNX models.")


if __name__ == "__main__":
    main()

