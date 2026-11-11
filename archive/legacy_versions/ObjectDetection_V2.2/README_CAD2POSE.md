# CAD-Native 6DoF Pose Estimation for ObjectDetection Plugin

Tài liệu hướng dẫn tích hợp và vận hành module **CAD-Native 6DoF Pose Estimation** trên kiến trúc C++ của `ObjectDetection` cho bo mạch **Orange Pi 5B (Rockchip RK3588S, NPU 6 TOPS)**.

---

## 1. Nguyên Lý Thiết Kế (Design Principles)

1. **100% CAD-Native (Không dùng 2D YOLO / Detector 2D)**:
   - Hệ thống trực tiếp trích xuất đám mây điểm 3D từ cảm biến chiều sâu (`depth_z16`) trong khoảng làm việc $0.35\text{ m} \le Z \le 2.50\text{ m}$.
   - Dùng mô hình NPU RKNN (`model_obs_rk3588.rknn`) sinh 128-dim geometric descriptors cho 1024 điểm quan sát (~3.5 ms).
   - Nhận diện và ước lượng tư thế đồng thời thông qua matching với `TABLE-DEMO_cad_descriptors.bin` (512 điểm then chốt của CAD) bằng RANSAC Kabsch SVD (~2.5 ms).
   - Tỷ lệ inlier $\ge 35\%$ khẳng định sự hiện diện của vật thể và cung cấp ma trận biến đổi $[R \mid t]$ chính xác.

2. **Đầy Đủ 6DoF (Full 6 Degrees of Freedom)**:
   - Tính toán và trích xuất trọn vẹn cả 6 bậc tự do: $[X, Y, Z, \text{Roll}, \text{Pitch}, \text{Yaw}]$.
   - Không giới hạn hay ép cứng về 3DoF. Các ứng dụng downstream (hệ thống điều khiển, lập kế hoạch gắp/thao tác) tự quyết định trường dữ liệu cần sử dụng.

3. **Độc Lập & Không Phụ Thuộc Robot Base (Zero AMR Assumption)**:
   - Module hoạt động hoàn toàn ở tầng thị giác máy tính và đo lường chuẩn xác (metrology).
   - Không chứa bất kỳ giả định hay ràng buộc nào về di chuyển mặt đất, bánh xe hay nâng hạ thủy lực.

---

## 2. Hệ Tọa Độ & Chuẩn Đo Lường (Metrology & Coordinate Frames)

### 2.1. Gốc Tọa Độ Camera (Camera Housing Centerline Origin)
Theo tài liệu chuẩn kỹ thuật **Intel RealSense D435 Figure 4-6 (Mechanical Dimensions)**:
- Gốc đo lường mặc định của hệ thống được đặt tại **Tâm mặt kính camera** (trùng với tâm ốc vít cơ khí 1/4"-20 ở đáy vỏ camera).
- Cảm biến hồng ngoại trái (Left IR / Depth Origin) nằm lệch $+17.5\text{ mm}$ ($+0.0175\text{ m}$) theo trục $+X$ so với tâm vỏ camera.
- Công thức hiệu chỉnh cơ khí từ tâm Depth sang tâm vỏ kính camera:
  $$\begin{cases}
  X_{\text{camera\_front\_glass}} = X_{\text{depth}} + 0.0175\text{ m} \\
  Y_{\text{camera\_front\_glass}} = Y_{\text{depth}} \\
  Z_{\text{camera\_front\_glass}} = Z_{\text{depth}}
  \end{cases}$$

### 2.2. Gốc Tọa Độ CAD Vật Thể (TABLE-DEMO Origin)
Từ phân tích file hình học 3D chuẩn `TABLE-DEMO_mesh.ply`:
- Gốc $(X=0, Y=0)$ nằm trên **trục đối xứng thẳng đứng đi qua tâm mặt bàn thẳng xuống**.
- Đỉnh mặt bàn: $Z = +0.1665\text{ m}$ ($+166.5\text{ mm}$).
- Đáy chân bàn: $Z = -0.3355\text{ m}$ ($-335.5\text{ mm}$).
- Tổng chiều cao bàn: $0.502\text{ m}$ ($502\text{ mm}$).
- Ma trận biến đổi $[R \mid t]$ ánh xạ từ gốc CAD này trực tiếp sang hệ tọa độ tâm kính camera.

---

## 3. Cấu Trúc Gói Mô Hình (Model Package Schema v2)

Gói mô hình đặt tại: `models/object_detection/cad_pose/table_cad/v1/`

```text
models/object_detection/cad_pose/table_cad/v1/
├── model.json                     # Schema v2 manifest
├── labels.txt                     # Nhãn lớp ("table")
├── cad_data/
│   ├── TABLE-DEMO_cad_descriptors.bin   # 512 x 128 float32 (256 KB)
│   ├── TABLE-DEMO_key_centroids.npy     # 512 x 3 float32 keypoints (6.2 KB)
│   └── table_dimensions.json            # Thông số hình học CAD
└── artifacts/
    └── orangepi5b_rk3588/
        └── model_obs_rk3588.rknn         # Binary NPU RK3588 (1.63 MB)
```

---

## 4. Chuẩn Đầu Ra Dữ Liệu (Protobuf Output Contract)

Kết quả trả về qua protobuf message `Message_ObjectDetections` với từng phần tử `DetectionObject`:

| Trường Protobuf | Kiểu Dữ Liệu | Ý Nghĩa / Giá Trị |
| :--- | :--- | :--- |
| `class_id` | `int32` | `0` (lớp bàn) |
| `label` | `string` | `"table"` |
| `confidence` | `float` | Tỷ lệ inlier RANSAC (0.35 - 1.00) |
| `has_3d` | `bool` | `true` |
| `position_x_m` | `float` | Tọa độ $X$ từ tâm mặt kính camera (mét) |
| `position_y_m` | `float` | Tọa độ $Y$ từ tâm mặt kính camera (mét) |
| `position_z_m` | `float` | Tọa độ $Z$ từ tâm mặt kính camera (mét) |
| `depth_m` | `float` | Khoảng cách $Z$ trực tiếp (mét) |
| `has_pose_6d` | `bool` | `true` |
| `roll_deg` | `float` | Góc Roll quanh trục $X$ camera (độ, $[-180^\circ, +180^\circ]$) |
| `pitch_deg` | `float` | Góc Pitch quanh trục $Y$ camera (độ, $[-90^\circ, +90^\circ]$) |
| `yaw_deg` | `float` | Góc Yaw quanh trục $Z$ camera (độ, $[-180^\circ, +180^\circ]$) |
| `pose_frame_id`| `string` | `"camera_front_glass_center"` |
| `x`, `y`, `w`, `h` | `float` | Hộp bao 2D trên ảnh RGB (chiếu từ bounding box 3D của CAD) |

---

## 5. Cấu Hình Runtime (Runtime Configuration)

Tệp cấu hình: `config/orangepi5b/cad_table.json`

```json
{
  "ObjectDetection": {
    "enabled": true,
    "enable_perf_log": true,
    "perf_log_every_n": 300,
    "debug_publish_raw_objects": false,
    "loop_sleep_ms": 20,
    "backend": "rknn",
    "platform_profile": "orangepi5b_rk3588",
    "strict_config_validation": true,
    "model_root": "/home/orangepi/Desktop/TMVisionary/models",
    "detectors": "cad_table_primary",
    "source_name": "CameraRGBD",
    "image_encoding": "bgr8",
    "default_interval_ms": 50,
    "default_confidence_threshold": 0.35,
    "default_nms_threshold": 0.45,
    "default_max_detections": 10
  },
  "ObjectDetection_cad_table_primary": {
    "enabled": true,
    "model_package": "object_detection/cad_pose/table_cad/v1",
    "model_variant": "fp16",
    "enabled_classes": "table",
    "application": "cad_pose",
    "interval_ms": 50,
    "confidence_threshold": 0.35,
    "nms_threshold": 0.45,
    "max_detections": 10
  }
}
```

---

## 6. Hiệu Năng Thực Tế (Benchmark trên RK3588S)

- **Trích xuất đám mây điểm 3D**: ~1.2 ms
- **Xây dựng k-NN Graph ($k=20$)**: ~1.8 ms
- **NPU RKNN Forward**: ~3.5 ms
- **RANSAC Kabsch SVD Matching**: ~2.5 ms
- **Chiếu Bounding Box 3D sang 2D**: ~0.2 ms
- **Tổng chu kỳ (Latency)**: **~9.2 ms / frame** (đáp ứng tốc độ xử lý > 100 FPS, cấu hình chạy mượt mà ở 20 FPS / `interval_ms: 50`).
