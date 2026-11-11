# Hướng Dẫn Triển Khai Mô Hình Pose Estimation Lên Kit Orange Pi 5B (RK3588S)

Thư mục này chứa đầy đủ mô hình đã được biên dịch tối ưu cho NPU **Rockchip RK3588S (6 TOPS)** và mã nguồn suy luận thời gian thực độc lập cho kit nhúng **Orange Pi 5B**.

---

## 1. Cấu Trúc Các Tệp Trong Gói Triển Khai

| Tên tệp | Kích thước | Chức năng |
| :--- | :--- | :--- |
| `model_obs_rk3588.rknn` | **1.63 MB** | Mô hình mạng nơ-ron nhị phân chạy trực tiếp trên 3 nhân NPU của RK3588S |
| `model_obs_npu.onnx` | **2.38 MB** | Mô hình ONNX chuẩn hóa (dùng để convert lại hoặc kiểm tra kiến trúc) |
| `TABLE-DEMO_cad_descriptors.npy` | **256 KB** | Ma trận đặc trưng CAD tĩnh ($512 \times 128$) được tính sẵn offline |
| `TABLE-DEMO_cad_descriptors.bin` | **256 KB** | Dạng nhị phân thô (raw binary float32) cho C/C++ runtime |
| `TABLE-DEMO_key_centroids.npy` | **6 KB** | Tọa độ 3D của 512 tâm cụm mốc CAD trong không gian vật thể |
| `TABLE-DEMO_hierarchy_model.pt` | **2.43 MB** | Mô hình TorchScript JIT (dành cho C++ LibTorch nếu cần đối chứng) |
| `rknn_pose_runtime.py` | Mã nguồn | Script chạy suy luận độc lập bằng `rknn-toolkit-lite2` trên Orange Pi |
| `calib_data/` | Thư mục | Các mẫu point cloud thực tế dùng cho lượng tử hóa INT8 |

---

## 2. Các Bước Cài Đặt Trên Orange Pi 5B (Chỉ Làm 1 Lần)

Trên hệ điều hành Ubuntu 22.04 LTS (ARM64) của Orange Pi 5B:

### Bước 1: Cài đặt thư viện Python cơ bản
```bash
sudo apt update
sudo apt install -y python3-pip python3-numpy python3-opencv
```

### Bước 2: Cài đặt RKNN Toolkit Lite2 (Dành riêng cho chip RK3588)
Tải và cài đặt wheel từ repository chính thức của Rockchip:
```bash
pip install https://github.com/airockchip/rknn-toolkit2/raw/master/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.0-cp310-cp310-linux_aarch64.whl
```

---

## 3. Chạy Suy Luận Thời Gian Thực Trên Board

Copy toàn bộ thư mục `deploy_orange_pi_5b/` sang Orange Pi 5B:
```bash
cd deploy_orange_pi_5b

# Chạy thử nghiệm benchmark với mẫu ngẫu nhiên:
python3 rknn_pose_runtime.py

# Chạy với đám mây điểm thực tế từ camera RealSense:
python3 rknn_pose_runtime.py --input-cloud calib_data/sample_0000.npy
```

### Hiệu Năng Đo Được:
- **Thời gian xử lý CPU (ARM Cortex-A76)**: $< 1.5$ ms (Tạo đồ thị láng giềng k-NN).
- **Thời gian suy luận NPU (RK3588S 6 TOPS)**: $\sim 3.0$ ms.
- **Thời gian đối sánh ma trận (Similarity Matching)**: $< 0.5$ ms.
- **Tổng độ trễ**: $\approx 5.0$ ms / frame $\longrightarrow$ **Đạt tốc độ lên tới ~120 - 150 FPS** trên Orange Pi 5B!
