# Báo cáo Đánh giá Phase 4: CAD-Aware Hierarchical 6DoF Pose Solver

- **Tập dữ liệu:** `test` (250 samples)
- **Mô hình AI:** CAD-Aware Hierarchical Model (`best_hierarchy_model.pth`)
- **Bộ giải Pose:** Confidence-weighted RANSAC Kabsch SVD + Point-to-Plane ICP + Boundary Optimization
- **Đường kính CAD:** 1089.1 mm | **Ngưỡng ADD-S (10%):** 108.9 mm

## 1. Kết Quả Tổng Thể

| Tiêu Chí | Giá Trị |
| :--- | :---: |
| **Tỷ Lệ Mẫu Hợp Lệ (Valid Rate)** | **100.0%** (250/250) |
| **Tỷ Lệ Đạt Mục Tiêu Tối Ưu ($e_t \le 10\text{mm}, e_R \le 1.0^\circ$)** | **47.6%** |
| **Tỷ Lệ Đạt Chuẩn ADD-S (< 0.10d)** | **98.0%** |
| **Tỷ Lệ Đạt Chuẩn Relaxed ($e_t \le 30\text{mm}, e_R \le 10^\circ$)** | **92.0%** |
| **Tỷ Lệ Đạt Chuẩn Strict ($e_t \le 10\text{mm}, e_R \le 5^\circ$)** | **51.6%** |
| **Phân Bố Sai Số Tịnh Tiến ($e_t$)** | **Min: 0.25 mm \| Med: 9.78 mm \| P90: 20.59 mm \| P95: 51.87 mm \| Max: 427.26 mm** |
| **Phân Bố Sai Số Góc Xoay ($e_R^{\text{sym}}$)** | **Min: 0.00° \| Med: 0.55° \| P90: 2.15° \| P95: 4.41° \| Max: 89.85°** |
| **Sai Số ADD-S Trung Vị** | **11.93 mm** |
| **Thời Gian Xử Lý Trung Bình** | **108.8 ms** |

## 2. Kết Quả Theo Từng Mức Độ Che Khuất (Visibility)

| Mức Độ Che Khuất | N | Hợp Lệ % | <10mm & <1° Pass % | ADD-S Pass % | Relaxed Pass % | Trung vị $e_t$ (mm) | Trung vị $e_R^{\text{sym}}$ (deg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Full (100%) | 65 | 100.0% | 61.5% | 100.0% | 100.0% | 7.26 | 0.40 |
| Mild (75%) | 62 | 100.0% | 50.0% | 100.0% | 98.4% | 9.58 | 0.57 |
| Heavy (50%) | 59 | 100.0% | 42.4% | 100.0% | 96.6% | 10.58 | 0.53 |
| Severe (25%) | 64 | 100.0% | 35.9% | 92.2% | 73.4% | 11.88 | 0.96 |
