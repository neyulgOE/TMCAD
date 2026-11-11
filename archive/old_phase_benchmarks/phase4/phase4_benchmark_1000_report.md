# Báo cáo Đánh giá Phase 4: CAD-Aware Hierarchical 6DoF Pose Solver

- **Tập dữ liệu:** `benchmark_1000` (1000 samples)
- **Mô hình AI:** CAD-Aware Hierarchical Model (`best_hierarchy_model.pth`)
- **Bộ giải Pose:** Confidence-weighted RANSAC Kabsch SVD + Point-to-Plane ICP + Boundary Optimization
- **Đường kính CAD:** 1089.1 mm | **Ngưỡng ADD-S (10%):** 108.9 mm

## 1. Kết Quả Tổng Thể

| Tiêu Chí | Giá Trị |
| :--- | :---: |
| **Tỷ Lệ Mẫu Hợp Lệ (Valid Rate)** | **99.8%** (998/1000) |
| **Tỷ Lệ Đạt Mục Tiêu Tối Ưu ($e_t \le 10\text{mm}, e_R \le 1.0^\circ$)** | **61.4%** |
| **Tỷ Lệ Đạt Chuẩn ADD-S (< 0.10d)** | **98.9%** |
| **Tỷ Lệ Đạt Chuẩn Relaxed ($e_t \le 30\text{mm}, e_R \le 10^\circ$)** | **89.2%** |
| **Tỷ Lệ Đạt Chuẩn Strict ($e_t \le 10\text{mm}, e_R \le 5^\circ$)** | **65.8%** |
| **Phân Bố Sai Số Tịnh Tiến ($e_t$)** | **Min: 0.07 mm \| Med: 5.82 mm \| P90: 29.94 mm \| P95: 56.56 mm \| Max: 321.80 mm** |
| **Phân Bố Sai Số Góc Xoay ($e_R^{\text{sym}}$)** | **Min: 0.01° \| Med: 0.42° \| P90: 2.61° \| P95: 6.27° \| Max: 89.92°** |
| **Sai Số ADD-S Trung Vị** | **11.78 mm** |
| **Thời Gian Xử Lý Trung Bình** | **107.1 ms** |

## 2. Kết Quả Theo Từng Mức Độ Che Khuất (Visibility)

| Mức Độ Che Khuất | N | Hợp Lệ % | <10mm & <1° Pass % | ADD-S Pass % | Relaxed Pass % | Trung vị $e_t$ (mm) | Trung vị $e_R^{\text{sym}}$ (deg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Full (100%) | 275 | 100.0% | 66.9% | 99.3% | 90.9% | 4.65 | 0.27 |
| Mild (75%) | 249 | 100.0% | 66.3% | 99.6% | 90.4% | 5.29 | 0.29 |
| Heavy (50%) | 222 | 100.0% | 63.5% | 100.0% | 94.6% | 5.49 | 0.38 |
| Severe (25%) | 254 | 99.2% | 48.8% | 96.9% | 81.5% | 8.32 | 0.79 |
