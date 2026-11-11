#!/usr/bin/env python3
"""
Convert ONNX Model to Rockchip RKNN Format for Orange Pi 5B (RK3588S).
"""
import os
import sys
import argparse
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX to RKNN for Orange Pi 5B")
    parser.add_argument("--cad-name", type=str, default="", help="CAD model name (e.g. TABLE-DEMO)")
    parser.add_argument("--onnx", type=str, default="", help="Path to ONNX model")
    parser.add_argument("--target", type=str, default="rk3588", help="Target platform (rk3588 for Orange Pi 5B)")
    parser.add_argument("--quantize", "--dtype", dest="quantize", choices=["fp16", "int8"], default="fp16", help="Quantization mode (fp16 or int8)")
    parser.add_argument("--output", type=str, default="", help="Output RKNN file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Auto-resolve paths based on cad_name
    cad_subfolder = os.path.join(project_root, "deploy_orange_pi_5b", args.cad_name) if args.cad_name else ""
    if not args.onnx:
        if cad_subfolder and os.path.exists(os.path.join(cad_subfolder, "model_obs_backbone.onnx")):
            args.onnx = os.path.join(cad_subfolder, "model_obs_backbone.onnx")
        elif cad_subfolder and os.path.exists(os.path.join(cad_subfolder, "model_obs_npu.onnx")):
            args.onnx = os.path.join(cad_subfolder, "model_obs_npu.onnx")
        else:
            args.onnx = os.path.join(project_root, "deploy_orange_pi_5b", "model_obs_backbone.onnx")
    elif os.path.isdir(args.onnx) and args.cad_name:
        if os.path.exists(os.path.join(args.onnx, args.cad_name, "model_obs_backbone.onnx")):
            args.onnx = os.path.join(args.onnx, args.cad_name, "model_obs_backbone.onnx")
        else:
            args.onnx = os.path.join(args.onnx, args.cad_name, "model_obs_npu.onnx")
        
    if not args.output:
        if cad_subfolder:
            args.output = os.path.join(cad_subfolder, "model_obs_rk3588.rknn")
        else:
            args.output = os.path.join(project_root, "deploy_orange_pi_5b", "model_obs_rk3588.rknn")
    elif os.path.isdir(args.output) and args.cad_name:
        args.output = os.path.join(args.output, args.cad_name, "model_obs_rk3588.rknn")

    print("============================================================")
    print(" CONVERTING MODEL TO RKNN FOR ORANGE PI 5B")
    if args.cad_name:
        print(f" CAD Model:       {args.cad_name}")
    print(f" Source ONNX:     {args.onnx}")
    print(f" Target Platform: {args.target}")
    print(f" Quantization:    {args.quantize.upper()}")
    print(f" Output RKNN:     {args.output}")
    print("============================================================")

    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")

    try:
        from rknn.api import RKNN
    except ImportError:
        print("[ERROR] rknn-toolkit2 is not installed. Please run: pip install rknn-toolkit2")
        sys.exit(1)

    rknn = RKNN(verbose=False)

    # 1. Configure RKNN for RK3588
    print("\n[1/4] Configuring RKNN target platform...")
    rknn.config(
        target_platform=args.target,
        optimization_level=3
    )

    # 2. Load ONNX model
    print(f"\n[2/4] Loading ONNX model ({args.onnx})...")
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print(f"[ERROR] Failed to load ONNX model (error code: {ret})")
        sys.exit(ret)
    print("  -> Loaded ONNX successfully.")

    # 3. Build RKNN Model
    print(f"\n[3/4] Building RKNN model (Quantization: {args.quantize.upper()})...")
    do_quant = (args.quantize == "int8")
    dataset_file = None

    if do_quant:
        # Prepare calibration dataset if needed
        calib_dir = os.path.join(cad_subfolder, "calib_data") if (cad_subfolder and os.path.exists(os.path.join(cad_subfolder, "calib_data"))) else os.path.join(project_root, "deploy_orange_pi_5b/calib_data")
        if os.path.exists(calib_dir):
            calib_rknn_list = os.path.join(os.path.dirname(calib_dir), "calib_rknn.txt")
            # generate multi-input calibration list
            sample_files = [os.path.join(calib_dir, f) for f in os.listdir(calib_dir) if f.endswith(".npy")]
            if sample_files:
                dataset_file = calib_rknn_list

    ret = rknn.build(do_quantization=do_quant, dataset=dataset_file)
    if ret != 0:
        print(f"[ERROR] RKNN build failed (error code: {ret})")
        sys.exit(ret)
    print("  -> RKNN build completed successfully.")

    # 4. Export RKNN Model
    print(f"\n[4/4] Exporting RKNN to: {args.output}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print(f"[ERROR] Export RKNN failed (error code: {ret})")
        sys.exit(ret)

    model_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  -> Exported RKNN model: {args.output} ({model_size_mb:.2f} MB)")

    print("\n============================================================")
    print(" RKNN CONVERSION COMPLETED SUCCESSFULLY!")
    print(f" Target: Orange Pi 5B (SoC RK3588S)")
    print(f" Binary Model: {args.output}")
    print("============================================================")

    rknn.release()

if __name__ == "__main__":
    main()
