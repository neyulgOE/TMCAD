# Báo cáo Đánh giá Phase 4: CAD-Aware Hierarchical 6DoF Pose Solver

- **Tập dữ liệu:** `test` (1 samples)
- **Mô hình AI:** CAD-Aware Hierarchical Model (`best_hierarchy_model.pth`)
- **Bộ giải Pose:** Confidence-weighted RANSAC Kabsch SVD + Point-to-Plane ICP + Boundary Optimization
- **Đường kính CAD:** 1089.1 mm | **Ngưỡng ADD-S (10%):** 108.9 mm

## 1. Kết Quả Tổng Thể

| Tiêu Chí | Giá Trị |
| :--- | :---: |
| **Tỷ Lệ Mẫu Hợp Lệ (Valid Rate)** | **100.0%** (1/1) |
| **Tỷ Lệ Đạt Mục Tiêu Tối Ưu ($e_t \le 10\text{mm}, e_R \le 1.0^\circ$)** | **100.0%** |
| **Tỷ Lệ Đạt Chuẩn ADD-S (< 0.10d)** | **100.0%** |
| **Tỷ Lệ Đạt Chuẩn Relaxed ($e_t \le 30\text{mm}, e_R \le 10^\circ$)** | **100.0%** |
| **Tỷ Lệ Đạt Chuẩn Strict ($e_t \le 10\text{mm}, e_R \le 5^\circ$)** | **100.0%** |
| **Phân Bố Sai Số Tịnh Tiến ($e_t$)** | **Min: 4.94 mm \| Med: 4.94 mm \| P90: 4.94 mm \| P95: 4.94 mm \| Max: 4.94 mm** |
| **Phân Bố Sai Số Góc Xoay ($e_R^{\text{sym}}$)** | **Min: 0.19° \| Med: 0.19° \| P90: 0.19° \| P95: 0.19° \| Max: 0.19°** |
| **Sai Số ADD-S Trung Vị** | **11.99 mm** |
| **Thời Gian Xử Lý Trung Bình** | **222.9 ms** |

## 2. Kết Quả Theo Từng Mức Độ Che Khuất (Visibility)

| Mức Độ Che Khuất | N | Hợp Lệ % | <10mm & <1° Pass % | ADD-S Pass % | Relaxed Pass % | Trung vị $e_t$ (mm) | Trung vị $e_R^{\text{sym}}$ (deg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Full (100%) | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 4.94 | 0.19 |
