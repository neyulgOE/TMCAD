import os
import sys
import argparse
import torch

# Add repo root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_learning.model import GeometricDGCNN

def export_model(model_path, out_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    assert os.path.exists(model_path), f"Checkpoint not found: {model_path}"
    checkpoint = torch.load(model_path, map_location=device)
    args_saved = checkpoint.get("args", {})
    desc_dim = args_saved.get("desc_dim", 32)
    k_neighbors = args_saved.get("k_neighbors", 20)

    model = GeometricDGCNN(k=k_neighbors, descriptor_dim=desc_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 2048, 3, device=device)

    # 1. Export to TorchScript
    ts_path = os.path.join(out_dir, "geometric_dgcnn.pt")
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(ts_path)
    print(f"[TorchScript] Model successfully exported to: {ts_path}")

    # 2. Export to ONNX
    onnx_path = os.path.join(out_dir, "geometric_dgcnn.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["points"],
        output_names=["descriptors"],
        dynamic_axes={"points": {0: "batch_size", 1: "num_points"}, "descriptors": {0: "batch_size", 1: "num_points"}}
    )
    print(f"[ONNX] Model successfully exported to: {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/huyen1905/TMCAD/deep_learning/checkpoints/best_geometric_model.pth")
    parser.add_argument("--out-dir", type=str, default="/home/huyen1905/TMCAD/deep_learning/checkpoints")
    args = parser.parse_args()
    export_model(args.model_path, args.out_dir)
