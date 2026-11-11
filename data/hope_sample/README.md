# HOPE Dataset Sample (Household & Grocery Objects)
Dữ liệu mẫu thu thập từ camera **Intel RealSense D415** (RGB + 16-bit Depth) và mô hình **CAD 3D** gốc chuẩn BOP Benchmark.

---

## 1. Cấu Trúc Thư Mục Mẫu
```text
/home/huyen1905/TMCAD/data/hope_sample/
├── cad_models/                        # File CAD 3D gốc kèm texture dùng để huấn luyện
│   ├── obj_000001.ply & .png          # Alphabet Soup (Hộp súp bảng chữ cái)
│   ├── obj_000002.ply & .png          # BBQ Sauce (Chai sốt BBQ)
│   ├── obj_000004.ply & .png          # Cherries (Hộp quả anh đào)
│   ├── obj_000010.ply & .png          # Green Beans (Hộp đậu cô ve)
│   ├── obj_000011.ply & .png          # Ketchup (Chai tương cà chua)
│   ├── obj_000016.ply & .png          # Mustard (Chai mù tạt vàng)
│   └── models_info.json               # Kích thước 3D chuẩn mm, hộp giới hạn và tâm vật thể
│
├── test_scenes/scene_000001/          # Cảnh kiểm thử thực tế từ RealSense D415
│   ├── rgb/
│   │   ├── 000000.png, 000001.png, 000002.png (1920x1080 uint8)
│   ├── depth/
│   │   ├── 000000.png, 000001.png, 000002.png (1920x1080 uint16, đơn vị mm)
│   ├── scene_camera.json              # Thông số nội tại camera RealSense D415 (fx, fy, cx, cy)
│   ├── scene_gt.json                  # Tọa độ & tư thế 6DoF chuẩn (Ground Truth R, t)
│   └── scene_gt_info.json             # 2D Bounding Box và mức độ che khuất (occlusion)
```

---

## 2. Thông Số Camera RealSense D415
* **Độ phân giải:** 1920 x 1080
* **Tiêu cự:** $f_x = 1390.53$, $f_y = 1386.99$
* **Tâm quang học:** $c_x = 964.957$, $c_y = 522.586$
* **Thang đo Depth (`depth_scale`):** `1.0` (giá trị pixel 16-bit bằng đúng khoảng cách milimét: $1000 = 1.0\text{ m}$)

---

## 3. Cách Dùng Trong TMCAD

### Cách 1: Test Trực Tiếp Trên Giao Diện TMCAD
1. Mở ứng dụng **TMCAD** (`./build/CadViewer`).
2. Vào mục **Bước 2: Huấn Luyện & Nhận Diện 6DoF**.
3. Bấm nút **📂 Từ File Ảnh/Depth**.
4. Chọn:
   - **Ảnh màu RGB:** `data/hope_sample/test_scenes/scene_000001/rgb/000000.png`
   - **Ảnh độ sâu Depth:** `data/hope_sample/test_scenes/scene_000001/depth/000000.png`
   - **fx, fy:** `1390.53`, `1386.99`
   - **cx, cy:** `964.957`, `522.586`
   - **Depth Scale:** `1.0`
5. Bấm **Chấp nhận**: TMCAD sẽ nạp frame RGB-D, chạy mô hình và vẽ khung 3D bounding box lên giao diện.

### Cách 2: Nạp Mô Hình CAD Vào TMCAD Để Render / Huấn Luyện
Các mô hình CAD đã được đồng bộ sẵn vào thư mục:
`/home/huyen1905/TMCAD/CAD_DATA/<obj_name>/`
Người dùng có thể kéo thả trực tiếp file `.ply` hoặc `.stl` vào cửa sổ 3D của TMCAD để xem lưới 3D, đo đạc kích thước hoặc tạo tập dữ liệu huấn luyện.

