#!/usr/bin/env python3
"""
Parity Evaluation Harness: Compares intermediate & final outputs of TMVis (Python)
and ObjectDetection_V2.5 (C++) stage-by-stage against strict tolerance gates.
"""
import sys
import json
import argparse
import numpy as np

def rotation_error_deg(R1, R2):
    """Computes geodesic rotation difference in degrees: arccos((trace(R1.T @ R2) - 1) / 2)"""
    R_diff = R1.T @ R2
    trace = np.trace(R_diff)
    cos_val = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_val) * 180.0 / np.pi)

def main():
    parser = argparse.ArgumentParser(description="Evaluate TMVis (Python) vs V2.5 (C++) Parity")
    parser.add_argument("--py-dump", type=str, required=True, help="Path to Python telemetry JSON dump")
    parser.add_argument("--cpp-dump", type=str, required=True, help="Path to C++ telemetry JSON dump")
    args = parser.parse_args()

    with open(args.py_dump) as f:
        py_data = json.load(f)
    with open(args.cpp_dump) as f:
        cpp_data = json.load(f)

    print("================================================================================")
    print("           PARITY EVALUATION: TMVIS (PYTHON) vs V2.5 (C++)                      ")
    print("================================================================================")

    all_passed = True

    # 1. Floor Plane Check
    print("\n[TẦNG 1: MẶT SÀN & TIỀN XỬ LÝ]")
    py_fn = py_data.get("floor_normal")
    cpp_fn = cpp_data.get("floor_normal")
    if py_fn and cpp_fn:
        n1 = np.array(py_fn, dtype=np.float64)
        n2 = np.array(cpp_fn, dtype=np.float64)
        cos_fn = np.clip(np.dot(n1, n2), -1.0, 1.0)
        deg_fn = float(np.arccos(cos_fn) * 180.0 / np.pi)
        status = "PASSED" if deg_fn <= 1.0 else "FAILED"
        if deg_fn > 1.0: all_passed = False
        print(f"  • Floor Normal Angular Error: {deg_fn:.3f}° (Ngưỡng: <= 1.0°) -> [{status}]")
    else:
        print("  • Floor Normal: Một trong 2 bên không phát hiện sàn.")

    py_npts = py_data.get("num_scene_points", 0)
    cpp_npts = cpp_data.get("num_scene_points", 0)
    diff_pts = abs(py_npts - cpp_npts)
    ratio_pts = min(py_npts, cpp_npts) / max(1, max(py_npts, cpp_npts))
    status_pts = "PASSED" if ratio_pts >= 0.85 else "FAILED"
    if ratio_pts < 0.85: all_passed = False
    print(f"  • Số điểm cụm bàn: Python={py_npts} pts | C++={cpp_npts} pts (Tỉ lệ: {ratio_pts*100:.1f}%) -> [{status_pts}]")

    # 2. Backbone Descriptors Check
    print("\n[TẦNG 2: MẠNG NƠ-RON & ĐẶC TRƯNG ĐIỂM (BACKBONE)]")
    py_d = py_data.get("D_obs")
    cpp_d = cpp_data.get("D_obs")
    if py_d and cpp_d:
        d1 = np.array(py_d, dtype=np.float32)
        d2 = np.array(cpp_d, dtype=np.float32)
        cos_d = np.sum(d1 * d2, axis=-1)
        mean_cos = float(np.mean(cos_d))
        min_cos = float(np.min(cos_d))
        status_cos = "PASSED" if mean_cos >= 0.98 else "FAILED"
        if mean_cos < 0.98: all_passed = False
        print(f"  • Cosine Similarity Đặc trưng: Trung bình={mean_cos:.4f}, Thấp nhất={min_cos:.4f} (Ngưỡng: >= 0.98) -> [{status_cos}]")

        py_top1 = np.array(py_data.get("top_indices", []))
        cpp_top1 = np.array(cpp_data.get("top_indices", []))
        if len(py_top1) == len(cpp_top1) and len(py_top1) > 0:
            top1_agree = float(np.mean(py_top1 == cpp_top1))
            status_top1 = "PASSED" if top1_agree >= 0.90 else "FAILED"
            if top1_agree < 0.90: all_passed = False
            print(f"  • Tỉ lệ trùng khớp Top-1 CAD Centroid: {top1_agree*100:.2f}% (Ngưỡng: >= 90.0%) -> [{status_top1}]")

    # 3. Coarse Pose RANSAC Check
    print("\n[TẦNG 3: BỘ GIẢI COARSE POSE RANSAC KABSCH]")
    py_Tc = np.array(py_data.get("T_coarse", np.eye(4)))
    cpp_Tc = np.array(cpp_data.get("T_coarse", np.eye(4)))
    dt_c = float(np.linalg.norm(py_Tc[:3, 3] - cpp_Tc[:3, 3]) * 1000.0) # mm
    dR_c = rotation_error_deg(py_Tc[:3, :3], cpp_Tc[:3, :3])
    status_tc = "PASSED" if (dt_c <= 20.0 and dR_c <= 2.0) else "FAILED"
    if dt_c > 20.0 or dR_c > 2.0: all_passed = False
    print(f"  • Sai số T_coarse: Delta_t = {dt_c:.2f} mm (Ngưỡng: <= 20 mm) | Delta_R = {dR_c:.2f}° (Ngưỡng: <= 2.0°) -> [{status_tc}]")

    # 4. ICP Refined Pose Check
    print("\n[TẦNG 4: MESH ICP REFINEMENT]")
    py_Ticp = np.array(py_data.get("T_icp", np.eye(4)))
    cpp_Ticp = np.array(cpp_data.get("T_icp", np.eye(4)))
    dt_icp = float(np.linalg.norm(py_Ticp[:3, 3] - cpp_Ticp[:3, 3]) * 1000.0)
    dR_icp = rotation_error_deg(py_Ticp[:3, :3], cpp_Ticp[:3, :3])
    status_ticp = "PASSED" if (dt_icp <= 5.0 and dR_icp <= 0.5) else "FAILED"
    if dt_icp > 5.0 or dR_icp > 0.5: all_passed = False
    print(f"  • Sai số T_icp:    Delta_t = {dt_icp:.2f} mm (Ngưỡng: <= 5.0 mm) | Delta_R = {dR_icp:.2f}° (Ngưỡng: <= 0.5°) -> [{status_ticp}]")

    # 5. Final Pose Check
    print("\n[TẦNG 5: HẬU XỬ LÝ (GRAVITY + FRONT ANCHOR)]")
    py_Tf = np.array(py_data.get("T_final", np.eye(4)))
    cpp_Tf = np.array(cpp_data.get("T_final", np.eye(4)))
    dt_f = float(np.linalg.norm(py_Tf[:3, 3] - cpp_Tf[:3, 3]) * 1000.0)
    dR_f = rotation_error_deg(py_Tf[:3, :3], cpp_Tf[:3, :3])
    status_tf = "PASSED" if (dt_f <= 3.0 and dR_f <= 0.3) else "FAILED"
    if dt_f > 3.0 or dR_f > 0.3: all_passed = False
    print(f"  • Sai số T_final:  Delta_t = {dt_f:.2f} mm (Ngưỡng: <= 3.0 mm) | Delta_R = {dR_f:.2f}° (Ngưỡng: <= 0.3°) -> [{status_tf}]")

    # 6. Validation Gate Agreement
    print("\n[TẦNG 6: VALIDATION GATE & CHỐNG DƯƠNG TÍNH GIẢ]")
    py_val = py_data.get("is_valid", False)
    cpp_val = cpp_data.get("is_valid", False)
    val_status = "PASSED" if (py_val == cpp_val) else "FAILED"
    if py_val != cpp_val: all_passed = False
    print(f"  • Phán quyết Pose hợp lệ: Python={py_val} | C++={cpp_val} -> [{val_status}]")
    print(f"  • Inlier count:           Python={py_data.get('num_inliers')} | C++={cpp_data.get('num_inliers')}")
    print(f"  • RANSAC support:         Python={py_data.get('ransac_support', 0)*100:.1f}% | C++={cpp_data.get('ransac_support', 0)*100:.1f}%")
    print(f"  • Surface Fitness:        Python={py_data.get('consensus_fitness', 0)*100:.1f}% | C++={cpp_data.get('consensus_fitness', 0)*100:.1f}%")

    # 7. 2D Corners Overlay Error
    print("\n[TẦNG 7: PHÉP CHIẾU 2D OVERLAY (8 GÓC)]")
    py_c2d = np.array(py_data.get("corners_2d", []))
    cpp_c2d = np.array(cpp_data.get("corners_2d", []))
    if len(py_c2d) == 8 and len(cpp_c2d) == 8:
        corner_errs = np.linalg.norm(py_c2d - cpp_c2d, axis=-1)
        mean_px_err = float(np.mean(corner_errs))
        max_px_err = float(np.max(corner_errs))
        status_c2d = "PASSED" if mean_px_err <= 3.0 else "FAILED"
        if mean_px_err > 3.0: all_passed = False
        print(f"  • Sai số tọa độ pixel 8 góc: Trung bình = {mean_px_err:.2f} px, Lớn nhất = {max_px_err:.2f} px (Ngưỡng: <= 3.0 px) -> [{status_c2d}]")

    print("\n================================================================================")
    if all_passed:
        print(" TỔNG KẾT PARITY HARNESS: [THÀNH CÔNG RỰC RỠ] MỌI TẦNG NẰM TRONG DUNG SAI!  ")
    else:
        print(" TỔNG KẾT PARITY HARNESS: [CHƯA ĐẠT CHUẨN] CẦN HIỆU CHỈNH CÁC TẦNG BỊ FAILED ")
    print("================================================================================")
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
