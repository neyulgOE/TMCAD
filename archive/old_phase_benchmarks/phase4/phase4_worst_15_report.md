# Báo cáo Đánh giá Phase 4: CAD-Aware Hierarchical 6DoF Pose Solver

- **Tập dữ liệu:** `worst_15` (15 samples)
- **Mô hình AI:** CAD-Aware Hierarchical Model (`best_hierarchy_model.pth`)
- **Bộ giải Pose:** Confidence-weighted RANSAC Kabsch SVD + Point-to-Plane ICP + Boundary Optimization
- **Đường kính CAD:** 1089.1 mm | **Ngưỡng ADD-S (10%):** 108.9 mm

## 1. Kết Quả Tổng Thể

| Tiêu Chí | Giá Trị |
| :--- | :---: |
| **Tỷ Lệ Mẫu Hợp Lệ (Valid Rate)** | **93.3%** (14/15) |
| **Tỷ Lệ Đạt Mục Tiêu Tối Ưu ($e_t \le 10\text{mm}, e_R \le 1.0^\circ$)** | **0.0%** |
| **Tỷ Lệ Đạt Chuẩn ADD-S (< 0.10d)** | **80.0%** |
| **Tỷ Lệ Đạt Chuẩn Relaxed ($e_t \le 30\text{mm}, e_R \le 10^\circ$)** | **33.3%** |
| **Tỷ Lệ Đạt Chuẩn Strict ($e_t \le 10\text{mm}, e_R \le 5^\circ$)** | **6.7%** |
| **Phân Bố Sai Số Tịnh Tiến ($e_t$)** | **Min: 8.38 mm \| Med: 39.22 mm \| P90: 199.24 mm \| P95: 444.95 mm \| Max: 793.21 mm** |
| **Phân Bố Sai Số Góc Xoay ($e_R^{\text{sym}}$)** | **Min: 0.12° \| Med: 3.32° \| P90: 10.77° \| P95: 37.86° \| Max: 87.37°** |
| **Sai Số ADD-S Trung Vị** | **23.76 mm** |
| **Thời Gian Xử Lý Trung Bình** | **698.9 ms** |

## 2. Kết Quả Theo Từng Mức Độ Che Khuất (Visibility)

| Mức Độ Che Khuất | N | Hợp Lệ % | <10mm & <1° Pass % | ADD-S Pass % | Relaxed Pass % | Trung vị $e_t$ (mm) | Trung vị $e_R^{\text{sym}}$ (deg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Heavy (50%) | 1 | 100.0% | 0.0% | 100.0% | 100.0% | 25.20 | 0.12 |
| Severe (25%) | 14 | 92.9% | 0.0% | 78.6% | 28.6% | 42.44 | 3.52 |
