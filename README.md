# TMCAD — Hệ Thống Ước Lượng 6DoF Pose Từ CAD-Aware Hierarchy & RGB-D

> **Nguyên tắc cốt lõi:** *Học biểu diễn từ cấu trúc hình học và tô-pô tam giác phân tầng của CAD (CAD-Aware Hierarchy) thay vì point-cloud ngẫu nhiên hay bounding box. Tận dụng dữ liệu Synthetic Raycasting C++ để có Ground-Truth chính xác 100%, hỗ trợ triệt để vật thể đối xứng và chạy thời gian thực (< 250 ms).*

---

## 📑 Mục Lục
1. [Giới Thiệu Chung](#1-giới-thiệu-chung)
2. [Yêu Cầu Hệ Thống & Cài Đặt](#2-yêu-cầu-hệ-thống--cài-đặt)
3. [Hướng Dẫn Sử Dụng Giao Diện GUI (`CadViewer`)](#3-hướng-dẫn-sử-dụng-giao-diện-gui-cadviewer)
   - [Quy trình 4 bước chuẩn cho một chi tiết CAD mới](#quy-trình-4-bước-chuẩn-cho-một-chi-tiết-cad-mới)
   - [Chi tiết từng Tab chức năng trong GUI](#chi-tiết-từng-tab-chức-năng-trong-gui)
4. [Sử Dụng Qua Dòng Lệnh (CLI Workflow)](#4-sử-dụng-qua-dòng-lệnh-cli-workflow)
5. [Cấu Trúc Thư Mục Dự Án](#5-cấu-trúc-thư-mục-dự-án)
6. [Kết Quả Thực Nghiệm & Đo Đạc Kiểm Chuẩn](#6-kết-quả-thực-nghiệm--đo-đạc-kiểm-chuẩn)
7. [Khắc Phục Sự Cố (Troubleshooting)](#7-khắc-phục-sự-cố-troubleshooting)

---

## 1. Giới Thiệu Chung

Trong các bài toán gắp đặt công nghiệp, robot AMR tiếp cận pallet/kệ hàng, camera quan sát thường chỉ thấy một phần bề mặt bị che khuất (partial observation), trong khi mô hình CAD 3D chính xác đã được biết trước.

Khác với các phương pháp truyền thống (biến CAD thành một đám mây điểm vô định hình hoặc dự đoán bounding-box 3D rời rạc), hệ thống **TMCAD** triển khai:
- **Cấu trúc phân tầng CAD 4 cấp (Phase 1):** 
  - *Tier 1:* Vector đặc trưng tam giác 13 chiều $[\mathbf{c}, \mathbf{n}, A, \mathbf{l}_{0..2}, \boldsymbol{\theta}_{0..2}]$.
  - *Tier 2:* Đồ thị kề tam giác chung cạnh kèm đặc trưng cạnh 6D.
  - *Tier 3:* Vùng lân cận tô-pô $K=32$ tam giác gần nhất (Local Patches) theo BFS.
  - *Tier 4:* Phân đoạn mặt phẳng/cong đồng nhất theo góc nhị diện Dihedral $< 15^\circ$.
- **Học liên kết đối xứng (Symmetry-Aware InfoNCE - Phase 3):** Xử lý nhóm đối xứng hình học $G = \{S_k\}$ ($C_2, C_4, D_2$), triệt tiêu mâu thuẫn gradient trên các vật thể đối xứng.
- **Bộ giải 6DoF Pose lai (Hybrid Solver - Phase 4):** So khớp ma trận tương đồng cosine sinh các cặp tương ứng $(p_{\text{obs}}, c_{\text{cad}}) \rightarrow$ Giải pose thô qua RANSAC Kabsch SVD $\rightarrow$ Tinh chỉnh bằng Coarse-to-Fine Point-to-Plane ICP trực tiếp trên lưới CAD canonical.

---

## 2. Yêu Cầu Hệ Thống & Cài Đặt

### 2.1. Yêu cầu môi trường
- **Hệ điều hành:** Linux Ubuntu 20.04 / 22.04 LTS.
- **Phần cứng:** CPU $\ge$ 4 cores, RAM $\ge$ 8GB, GPU NVIDIA có CUDA (khuyến nghị RTX 3060/4060 trở lên để huấn luyện và suy luận thời gian thực).
- **Phần mềm & Thư viện C++:**
  - `cmake` ($\ge 3.16$), `g++` hỗ trợ C++17, `make`.
  - `Qt5` (`qtbase5-dev`), OpenGL, GLEW (`libglew-dev`), Open3D C++.
- **Môi trường Python:** Python 3.8+ (khuyến nghị Python 3.10+):
  ```bash
  pip install torch torchvision numpy scipy pandas opencv-python open3d
  ```

### 2.2. Biên dịch ứng dụng C++ (`CadViewer`)
```bash
# Clone và mở thư mục dự án
git clone https://github.com/neyulgOE/TMCAD.git
cd TMCAD

# Tạo thư mục build và biên dịch toàn bộ target (bao gồm CadViewer & thư viện dùng chung tmcad_pose_core)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

Sau khi biên dịch thành công, file thực thi nằm tại: `build/CadViewer`.

---

## 3. Hướng Dẫn Sử Dụng Giao Diện GUI (`CadViewer`)

Khởi động ứng dụng từ thư mục gốc hoặc thư mục `build`:
```bash
./build/CadViewer
```

Giao diện ứng dụng được tổ chức khoa học thành **4 Tab chức năng** ở bảng điều khiển bên phải và khung nhìn **3D Viewport OpenGL** tương tác ở bên trái:
- **Tab 1: 📐 CAD** (Quản lý CAD, hệ tọa độ, đối xứng Phase 0 & Hierarchy Phase 1).
- **Tab 2: 🎲 Sim / Oracle** (Hạ tầng sinh dữ liệu giả lập & kiểm tra tính toàn vẹn hệ tọa độ).
- **Tab 3: 🧠 Train** (Kiểm chuẩn Classical Baseline FPFH & Huấn luyện mô hình AI).
- **Tab 4: 🎯 6DoF Pose** (Ước lượng 6DoF pose thời gian thực từ AI / Classical kèm lớp phủ 3D).

---

### Quy trình 4 bước chuẩn cho một chi tiết CAD mới

Khi bạn muốn import một chi tiết CAD mới (ví dụ `NEW_PART.stl` hoặc `TABLE-DEMO.stl`), hãy tuân thủ tuần tự 4 bước sau:

#### 🔹 Bước 1: Nạp CAD & Trích xuất Canonical Hierarchy (Tab 1: 📐 CAD)
1. Bấm **File $\rightarrow$ Open...** (hoặc kéo thả file `.stl` / `.obj` vào khung nhìn 3D).
2. Chi tiết CAD sẽ xuất hiện trong danh sách **Scene Objects**. Nhấp chuột chọn chi tiết đó.
3. Trong **Tab 1 (📐 CAD)**:
   - Kiểm tra hướng mô hình (Orientation Fix). Nếu chi tiết bị lật trục Z/Y, xoay về đúng chiều thẳng đứng và bấm **Save Orientation**.
   - Tại mục **Phase 0: Symmetry Definition**, chọn nhóm đối xứng tương ứng:
     - `None`: Vật thể bất đối xứng.
     - `Discrete_C2_Z`: Đối xứng quay $180^\circ$ quanh trục Z (như bàn `TABLE-DEMO`, kệ pallet đối xứng 2 đầu).
     - `Discrete_C4_Z`: Đối xứng quay $90^\circ$ quanh trục Z (vật thể hình vuông).
     - `Discrete_D2`: Đối xứng gương/lật qua 2 trục.
   - Bấm **"⚡ Build CAD Hierarchy (Phase 1)"**:
     - Hệ thống C++ tự động chuẩn hóa về mét, trích xuất 13D Triangle Primitives, Adjacency Graph, Local Patches ($K=32$) và phân đoạn Dihedral Regions.
     - Toàn bộ file canonical được lưu vào thư mục `CAD_DATA/`: `_mesh.ply`, `_triangles.npz`, `_adjacency.npz`, `_patches.npz`, `_regions.npz`, `_symmetry.json`.
   - Bấm **"🎨 Color by Regions (3D View)"** để xem trực quan các mặt phẳng/cong của CAD được tô màu động trên GPU.

#### 🔹 Bước 2: Sinh Dữ Liệu Huấn Luyện Tự Động (Tab 2: 🎲 Sim / Oracle)
1. Chuyển sang **Tab 2: 🎲 Sim / Oracle**.
2. Tại mục **Phase 1.5: Synthetic Dataset Engine**:
   - Nhập số lượng mẫu cần sinh (khuyến nghị: **1,000 - 2,000 mẫu** cho huấn luyện).
   - Thiết lập phạm vi khoảng cách camera (ví dụ $0.6 - 1.5\text{ m}$).
   - Chọn dải che khuất (Occlusion: $0.25 - 1.0$) để mô hình học cách thích ứng khi bị vật thể khác che một phần.
   - Bật tùy chọn mặt phẳng nền (Add background plane) để mô phỏng sàn nhà/tường thực tế.
3. Bấm **"🎲 Generate Dataset..."**:
   - Thanh tiến trình sẽ hiển thị tiến độ Raycasting C++ thời gian thực.
   - Dữ liệu được ghi vào `dataset/<CAD_NAME>/` gồm: ảnh `depth.png` (uint16 mét), ảnh `color.png`, siêu dữ liệu `pose.json`, và ground-truth tương ứng từng điểm `correspondences.csv`.
   - Hệ thống tự động phân chia tập `train.txt` (85%) và `val.txt` (15%).
4. *(Tùy chọn kiểm tra)* Bấm **"Run Oracle Sanity Check"** để xác thực hệ tọa độ: Bộ giải SVD toán học thuần túy sẽ kiểm định trên ground-truth, đảm bảo sai số $e_t = 0.00\text{ mm}, e_R = 0.00^\circ$ (Zero coordinate drift).

#### 🔹 Bước 3: Huấn Luyện AI CAD-Aware Correspondence (Tab 3: 🧠 Train)
1. Chuyển sang **Tab 3: 🧠 Train**.
2. Tại mục **Phase 4 & 6: AI Model Training**, bấm **"🧠 Train AI Model..."**:
   - Hộp thoại cấu hình xuất hiện:
     - **Epochs:** 15 – 25 epochs.
     - **Batch Size:** 8 (hoặc 16 nếu VRAM $\ge$ 8GB).
     - **Learning Rate:** 0.001.
     - **Tùy chọn:** Đánh dấu tích vào ☑ **"Xóa sạch trọng số đã học cũ (Train from scratch)"** nếu bạn muốn mô hình xóa bỏ hoàn toàn dữ liệu đã học trước đó để chỉ học duy nhất chi tiết CAD này.
3. Bấm **"Bắt Đầu Huấn Luyện (Start)"**:
   - Hộp thoại tiến trình `QProgressDialog` sẽ cập nhật trực tiếp theo thời gian thực:
     $$\text{Epoch } X/Y \quad|\quad \text{Train Loss} \quad|\quad \text{Val Loss} \quad|\quad \text{Val Recall@1} \quad|\quad \text{Val Recall@5}$$
   - Pipeline Log bên dưới sẽ in chi tiết từng epoch.
4. Khi hoàn thành, mô hình tốt nhất được lưu tại `deep_learning/checkpoints/<CAD_NAME>/best_hierarchy_model.pth`.
5. Trạng thái trong **Tab 1** sẽ tự động chuyển sang màu xanh: **Model AI: Đã huấn luyện**.

#### 🔹 Bước 4: Ước Lượng 6DoF Pose & Đánh Giá Lớp Phủ 3D (Tab 4: 🎯 6DoF Pose)
1. Chuyển sang **Tab 4: 🎯 6DoF Pose**.
2. Chuẩn bị đám mây điểm quan sát (chọn 1 trong 3 cách):
   - **Cách A (Mô phỏng tức thời):** Bấm **"🎲 Synthesize Virtual RGB-D Scene..."** để tạo một góc nhìn camera giả lập tùy ý (chọn khoảng cách, góc xoay, mức che khuất).
   - **Cách B (Từ tệp ảnh):** Bấm **"📁 Load RGB-D Files..."** nạp cặp ảnh `color.png` và `depth.png`.
   - **Cách C (Từ Camera thực):** Bấm **"📷 Capture RealSense RGB-D..."** (yêu cầu cắm Intel RealSense D435/D435F).
3. Tại phần phương pháp: Chọn radio button **"🧠 AI (GeoPoseNet / Hierarchy + ICP)"**.
4. Bấm nút lớn **"🎯 Estimate Pose (AI + ICP)"**:
   - Ứng dụng thực hiện suy luận chỉ trong **~220 ms**!
   - Kết quả xuất hiện chi tiết trong cửa sổ **Pipeline Log**:
     - Ma trận biến đổi $T_{\text{camera} \leftarrow \text{object}}$ và vector dịch chuyển $X, Y, Z$ (m).
     - Góc quay cảm biến Euler: Roll, Pitch, Yaw.
     - **Bộ thông số điều hướng vật lý cho Xe Tự Hành / AMR:**
       - *Chiều cao đặt camera:* `Height (m)`
       - *Khoảng cách mặt đất đến vật:* `Distance (m)`
       - *Góc ngẩng/chúi:* `Pitch (tilt deg)`
       - *Góc xoay hướng tâm:* `Azimuth (deg)`
     - Chỉ số chất lượng: **ICP Fitness %**, **Inlier RMSE (mm)**, số lượng **RANSAC Inliers**.
     - Sai số đối chứng so với Ground Truth: $e_t\text{ (mm)}$ và $e_R\text{ (deg)}$.
   - **Trực quan hóa lớp phủ 3D (Viewport):**
     - Đám mây điểm quan sát đã được biến đổi về hệ tọa độ CAD xuất hiện dưới dạng đối tượng **`Aligned Scene Points` (màu xanh lá lục bảo)**, phủ khớp chính xác trên mô hình CAD để kiểm tra trực quan độ khít.
     - Khung nón camera (FOV Frustum) màu xanh lá mô phỏng góc đặt của camera trong không gian.

---

## 4. Sử Dụng Qua Dòng Lệnh (CLI Workflow)

Toàn bộ các tác vụ huấn luyện, suy luận và kiểm chuẩn đều có thể chạy độc lập qua script Python hoặc tích hợp vào hệ thống ROS/AMR.

### 4.1. Huấn luyện mô hình CAD-Aware Hierarchy
```bash
python3 deep_learning/train_hierarchy.py \
    --cad-name TABLE-DEMO \
    --epochs 15 \
    --batch-size 8 \
    --lr 0.001 \
    --reset-data
```
*Ghi chú:* Thêm `--reset-data` để xóa checkpoint cũ và train lại từ đầu.

### 4.2. Chạy suy luận 6DoF Pose cho một đám mây điểm quan sát
```bash
python3 scripts/infer_hierarchy.py \
    --input debug/temp_scene_ai.ply \
    --cad CAD_DATA/TABLE-DEMO.stl \
    --checkpoint deep_learning/checkpoints/TABLE-DEMO/best_hierarchy_model.pth \
    --output-json debug/pose_output_ai.json \
    --output-cloud debug/aligned_scene_overlay.ply \
    --ransac-thresh 0.04 \
    --icp-max-corr 0.03 \
    --max-depth 2.5
```
Kết quả trả về file `pose_output_ai.json` chứa:
- `transformation_matrix`: Ma trận biến đổi 4x4.
- `camera_physical_world`: Thông số điều hướng xe tự hành AMR (`height_m`, `distance_m`, `pitch_deg`, `azimuth_deg`).
- `fitness`, `inlier_rmse_mm`, `ransac_inliers`.
- `latency`: Thời gian xử lý chi tiết (AI latency, Solver latency, Total latency).

### 4.3. Đánh giá tự động Benchmark & Ablation trên tập kiểm định
```bash
python3 scripts/eval_hierarchy_benchmark.py \
    --cad-name TABLE-DEMO \
    --num-samples 50 \
    --max-depth 2.5
```
Script sẽ tự động chạy qua 50 mẫu validation, tính toán tỷ lệ đạt chuẩn **ADD-S Pass Rate (<0.10 diameter)**, sai số trung vị $e_t$, $e_R$, số điểm inliers và phân rã độ trễ.

---

## 5. Cấu Trúc Thư Mục Dự Án

```text
TMCAD/
├── CAD_DATA/                       # Chứa mô hình CAD gốc & dữ liệu phân tầng canonical
│   ├── TABLE-DEMO.stl              # File CAD gốc (STL/OBJ)
│   ├── TABLE-DEMO_mesh.ply         # CAD mesh chuẩn hóa đơn vị mét (Phase 1)
│   ├── TABLE-DEMO_triangles.npz    # Đặc trưng 13D của từng tam giác (Tier 1)
│   ├── TABLE-DEMO_adjacency.npz    # Đồ thị kề tam giác kèm đặc trưng cạnh 6D (Tier 2)
│   ├── TABLE-DEMO_patches.npz      # Vùng lân cận tô-pô K=32 (Tier 3)
│   ├── TABLE-DEMO_regions.npz      # Phân đoạn vùng Dihedral đồng nhất (Tier 4)
│   └── TABLE-DEMO_symmetry.json    # Nhóm biến đổi đối xứng hình học (Phase 0)
│
├── dataset/                        # Tập dữ liệu sinh tự động bằng Raycasting C++
│   └── TABLE-DEMO/
│       ├── train.txt / val.txt     # Danh sách mẫu phân chia train/val
│       └── train/sample_00002/
│           ├── depth.png           # Ảnh depth 16-bit (mm)
│           ├── color.png           # Ảnh RGB
│           ├── pose.json           # Ma trận GT cameraFromObject & intrinsics
│           └── correspondences.csv # Ground-truth liên kết từng điểm ảnh với CAD
│
├── deep_learning/                  # Pipeline học sâu CAD-Aware Hierarchy
│   ├── dataset_hierarchy.py        # PyTorch Dataset nạp dữ liệu phân tầng & nhóm đối xứng
│   ├── model_hierarchy.py          # CADPatchEncoder, ObservationEncoder (DGCNN), Similarity Matrix
│   ├── loss_hierarchy.py           # Symmetry-Aware Multi-Positive InfoNCE Loss & Recall metrics
│   └── train_hierarchy.py          # Script huấn luyện với tiến trình phát tín hiệu cho GUI
│
├── models/                         # Các Black-Box ONNX tự chứa (<CAD_NAME>.onnx) chạy trực tiếp
│
├── scripts/                        # Pipeline cấu hình CAD, xuất ONNX, suy luận & kiểm chuẩn
│   ├── build_cad_profile.py        # Tự động trích xuất profile hình học & gờ sắc 3D cho CAD mới
│   ├── export_hierarchy_onnx.py    # Đóng gói Black-Box ONNX tự chứa (descriptors, meta, bounds, edges)
│   ├── infer_hierarchy.py          # Suy luận 6DoF đa vật thể / đa cá thể + Auto-Intrinsics
│   ├── real_sensor_pipeline.py     # Lọc mặt phẳng & nhiễu cảm biến thực
│   ├── eval_hierarchy_benchmark.py # Đánh giá benchmark tự động
│   ├── eval_v25_tmvis_parity.py    # Kiểm tra đồng bộ kết quả với TMVis_Debug
│   ├── convert_to_rknn.py          # Chuyển đổi ONNX sang RKNN cho NPU Orange Pi 5B
│   ├── simulate_rknn_embedded.py   # Mô phỏng suy luận nhúng RKNN
│   ├── extract_bag.py              # Trích xuất RGB-D từ file .bag RealSense
│   ├── split_dataset.py            # Chia tập train/val
│   └── legacy_debug/               # Lưu trữ các script debug / render thử nghiệm cũ
│
├── libs/                           # Thư viện bên thứ ba & ONNX Runtime headers (include/)
│
├── src/                            # Nguồn duy nhất (Single Source of Truth) cho CadViewer & lõi tmcad_pose_core
│   ├── RgbdFrame.h / RGBDTypes.h   # Kiểu CameraIntrinsics & RGB-D chuẩn hóa dùng chung OpenCV/Open3D
│   ├── CADSceneSegmenter.h/.cpp    # Phân đoạn đa cụm 3D, bóc mặt phẳng nền thích ứng kích thước CAD
│   ├── CADTemplateDetector.h/.cpp  # Phát hiện khớp mẫu 2D đa góc nhìn (ORB + RANSAC)
│   ├── UniversalPoseEstimator.*    # Bộ giải 6DoF Universal (FREE_6DOF & FLOOR_STANDING, OBB/Z-buffer)
│   ├── TablePoseEstimator.h        # Alias tương thích ngược tới UniversalPoseEstimator
│   ├── CADHierarchy.h/.cpp         # Pipeline trích xuất 4 tầng Canonical CAD Representation
│   ├── SymmetryDefinition.h/.cpp   # Định nghĩa nhóm đối xứng & công thức sai số ADD-S
│   ├── MuJoCoD435SimSource.h/.cpp  # Mô phỏng cảm biến RGB-D D435if thời gian thực trên MuJoCo
│   ├── MuJoCoSimDialog.h/.cpp      # Hộp thoại cấu hình MuJoCo đồng bộ tự động với Active ONNX
│   ├── StudioUxControls.h          # Thanh công cụ hot-swap .onnx, 1-Click Auto-Build & Traffic-Light pills
│   ├── CameraStreamWidget.h/.cpp   # Stream RealSense, 2D ROI tương tác & lớp phủ HUD/OBB 3D
│   ├── MainWindow.h/.cpp           # Giao diện Qt5 chính của CadViewer
│   └── Viewport3D.h/.cpp           # Khung nhìn OpenGL hiển thị CAD, đám mây điểm, FOV frustum
│
├── TMVis_Debug_v1.0.0/TMVis_Debug/ # Ứng dụng C++ kiểm thử độc lập, link trực tiếp lõi tmcad_pose_core từ src/
│
├── CMakeLists.txt                  # Cấu hình biên dịch CMake C++17 (CadViewer + tmcad_pose_core)
├── ROADMAP.md                      # Lộ trình nghiên cứu chi tiết 10 Phase
└── README.md                       # Tài liệu hướng dẫn sử dụng này
```

---

## 6. Kết Quả Thực Nghiệm & Đo Đạc Kiểm Chuẩn

Kết quả so sánh đối chứng giữa **Baseline Cổ Điển (Phase 2 - FPFH)** và **CAD-Aware Hierarchical Correspondence (Phase 3 & 4 Tối Ưu Hóa Cao Cấp)** trên toàn bộ **250 mẫu test set** độc lập:

| Tiêu Chí Đánh Giá | Classical Baseline (FPFH) | CAD-Aware Hierarchy (Ours) | Mức Độ Cải Thiện / Đạt Chuẩn |
| :--- | :---: | :---: | :---: |
| **Sai số tịnh tiến trung vị ($e_t$)**| 39.14 mm | **9.54 mm** | 🎯 **Đạt chuẩn < 10 mm (Giảm 4.1 lần)** |
| **Sai số góc xoay trung vị ($e_R^{\text{sym}}$)**| 18.20° | **0.41°** | 🎯 **Đạt chuẩn < 1.0° (Giảm 44 lần)** |
| **Tỷ lệ đạt chuẩn kép ($e_t \le 10\text{mm}, e_R \le 1.0^\circ$)** | 2.4% | **51.2%** (Full: **70.8%**) | **Tăng hơn 21 lần** |
| **Tỷ Lệ Đạt ADD-S (< 0.10d)** | 35.2% | **98.4%** | **Gần như tuyệt đối (+63.2%)** |
| **Tỷ Lệ Đạt Chuẩn Relaxed ($e_t \le 30\text{mm}, e_R \le 10^\circ$)** | 7.6% | **92.4%** | **Tăng gấp 12 lần (+84.8%)** |
| **Thời gian xử lý trung bình** | ~2,500 ms | **71.8 ms** | **Đạt tốc độ ~14 FPS Real-Time** |

> **Ghi chú về phân rã khoảng cách:** Ở cự ly làm việc tiêu chuẩn $0.8 - 1.4\text{ m}$ (khoảng cách gắp đặt thực tế của robot AMR), sai số đạt mức xuất sắc: **$e_t = 5.29\text{ mm}$** và **$e_R = 0.076^\circ$**. Với cự ly $1.4 - 2.0\text{ m}$, sai số duy trì ở **$e_t = 8.05\text{ mm}$** và **$e_R = 0.251^\circ$**.

---

## 7. Khắc Phục Sự Cố (Troubleshooting)

1. **Lỗi `Missing canonical hierarchy for <CAD_NAME>` khi bấm Train hoặc Estimate Pose:**
   - *Nguyên nhân:* Bạn chưa trích xuất đặc trưng phân tầng cho chi tiết CAD đó.
   - *Cách xử lý:* Vào **Tab 1 (📐 CAD)**, chọn chi tiết CAD trong danh sách và bấm nút **"⚡ Build CAD Hierarchy (Phase 1)"**.
2. **Khi nạp một vật thể CAD mới, làm sao để xóa dữ liệu đã học của vật thể cũ?**
   - *Cách xử lý:* Vào **Tab 3 (🧠 Train)** $\rightarrow$ Bấm **"Train AI Model..."** $\rightarrow$ Đánh dấu tích vào ô ☑ **"Xóa sạch trọng số đã học cũ (Train from scratch)"**. Hệ thống sẽ tự động xóa sạch checkpoint cũ của chi tiết này và huấn luyện lại từ đầu.
3. **Mô hình bị lật $180^\circ$ khi ước lượng pose:**
   - *Nguyên nhân:* Vật thể có tính đối xứng quay nhưng chưa khai báo nhóm đối xứng.
   - *Cách xử lý:* Vào **Tab 1 (📐 CAD)** $\rightarrow$ Tại mục **Phase 0 Symmetry**, chọn đúng nhóm đối xứng (ví dụ `Discrete_C2_Z`) và bấm **Save Orientation**. Sau đó chạy lại suy luận.
4. **Không kết nối được Camera RealSense:**
   - *Kiểm tra:* Cắm camera vào cổng USB 3.0 (cổng màu xanh dương). Kiểm tra bằng lệnh `rs-enumerate-devices`.
   - Nếu không có camera thực, bạn hoàn toàn có thể sử dụng tính năng **Synthesize Virtual RGB-D Scene** trong Tab 4 để kiểm nghiệm trên môi trường giả lập với độ chính xác tương đương.

---
*Dự án phát triển bởi nhóm nghiên cứu TMCAD — Mọi ý kiến đóng góp xin cập nhật qua file `ROADMAP.md`.*
