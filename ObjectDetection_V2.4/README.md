# ObjectDetection V2.4 (Hybrid RGB-Guided 6DoF Release)

Bản phát hành hợp nhất **ObjectDetection V2.4** kết hợp đầy đủ:
1. **Thuật toán Pallet Detection V2.0 Stateless + Ray-Plane Intersection**: Kiến trúc không trạng thái (khớp kiến trúc Request-Response đồng bộ của native service), tiền lọc outlier Z bằng MAD (`madFilterByZ`), chặn góc nghiêng mặt phẳng $>60^\circ$ trong RANSAC, weighted least-squares refit $Z = aX + bY + c$ và xác định tọa độ 3D từng block bằng giao điểm tia camera với mặt phẳng (`planeDepthAtPixel`), loại bỏ hoàn toàn sai lệch foreshortening khi pallet xoay góc lớn.
2. **CAD-Guided 6DoF Pose Estimation (v2.4 Production Hardened)**: Nhận diện và khoanh vùng đối tượng trực tiếp từ ảnh RGB bằng mô hình CAD (Multi-view ORB Template Matching), crop ROI đám mây điểm Depth, loại bỏ 100% tạp âm môi trường (sàn, tường), Priority RANSAC Kabsch SVD trên NPU RKNN (Orange Pi 5B - RK3588S) đạt inlier >65-90%, trích xuất đủ 6 bậc tự do $[X, Y, Z, \text{Roll}, \text{Pitch}, \text{Yaw}]$, chuẩn hóa tâm đo lường tại **Tâm mặt kính camera (Front Glass Centerline)** lệch $+17.5\text{ mm}$ từ Left IR.

---

## Cấu Trúc Thư Mục (Directory Layout)

```text
ObjectDetection_V2.4/
├── ObjectDetection/                     # Mã nguồn C++ Plugin
│   ├── adapters/
│   │   ├── CxCyWhClassScoresAdapter.*
│   │   ├── DflMultiScaleClassScoresAdapter.*
│   │   └── CADHierarchyAdapter.*         # CAD-Native model adapter
│   ├── applications/
│   │   ├── pallet/
│   │   │   └── PalletApplication.*       # [v2.0 Stateless] Ray-Plane Intersection & MAD Filter
│   │   └── cad_pose/
│   │       ├── CADTemplateDetector.*     # [v2.4] Multi-view CAD Template & Feature Matching
│   │       └── CADPoseApplication.*      # [v2.4] RGB-Guided ROI + Priority RANSAC Kabsch SVD
│   ├── backends/
│   │   ├── rknn/
│   │   └── tensorrt/
│   ├── config/
│   │   └── orangepi5b/
│   │       ├── person.json
│   │       ├── pallet.json
│   │       ├── pallet_debug.json
│   │       ├── person_and_pallet.json
│   │       ├── cad_table.json            # Runtime config for CAD table
│   │       ├── cad_table_debug.json      # Debug config for CAD table
│   │       └── person_pallet_and_cad.json # [v2.3] Combined 3-detector config
│   ├── ApplicationFactory.cpp            # Đăng ký cả pallet & cad_pose
│   ├── ModelAdapterFactory.cpp           # Đăng ký cả dfl, cxcywh, cad_hierarchy_v1
│   ├── ModelManifest.cpp                 # Hỗ trợ bbox & cad_6dof
│   ├── changelog_version.txt             # Lịch sử phiên bản từ v1.0 -> v2.3
│   └── README.md
├── models/                               # Kho mô hình hợp nhất
│   └── object_detection/                 # Toàn bộ model ngang hàng bên trong object_detection
│       ├── cad_pose/
│       │   └── table_cad/v1/             # CAD 6DoF NPU RKNN & Descriptors
│       ├── pallet/
│       │   └── v3/                       # Model Pallet DFL YOLO
│       └── person/
│           └── v1/                       # Model Person YOLO
├── README_CAD2POSE.md                    # Tài liệu kỹ thuật chi tiết CAD 6DoF
└── README.md                             # Tài liệu tổng quan bản 2.3
```

---

## Chi Tiết Kỹ Thuật

### 1. Pallet Detection (Cải tiến v1.5)
- **RANSAC 600 iterations, 30 min points**: Tăng độ hội tụ mặt phẳng block pallet ngay cả trong điều kiện điểm depth thưa hoặc nhiễu biên.
- **Centroid-fix**: Tính trung bình tọa độ 3D của toàn bộ inliers mặt phẳng block thay vì lấy depth tại 1 pixel tâm 2D bbox. Triệt tiêu sai lệch khi pallet quan sát ở góc nghiêng (foreshortening).
- **RawJumpDetector (Auto-reset)**: Theo dõi độ lệch chuẩn thực tế của vị trí và góc xoay thô. Tự động reset lịch sử làm mượt khi robot đổi vị trí hoặc góc quan sát.
- **Settle Detection & NormalWindowAverager**: Nhận diện khi camera/robot đã dừng ổn định ($\text{std}_{\text{pos}} \le 3\text{ mm}$, $\text{std}_{\text{yaw}} \le 0.30^\circ$) để chuyển sang trung bình trượt không trễ (zero-lag).
- **Camera Optical Frame Euler Fix**: Sửa công thức Roll/Pitch tương thích với hệ tọa độ quang học camera ($X$-phải, $Y$-xuống, $Z$-tiến).

### 2. CAD-Native 6DoF Pose Estimation (v2.0)
- **Không dùng 2D YOLO**: Trực tiếp chuyển đám mây điểm quan sát ($0.35\text{ m} \le Z \le 2.50\text{ m}$) thành đồ thị $k\text{-NN}$ ($k=20$), chạy suy luận qua RKNN NPU sinh vector đặc trưng 128 chiều.
- **RANSAC Kabsch SVD**: Khớp đối sánh với 512 điểm then chốt hình học của CAD (`TABLE-DEMO_cad_descriptors.bin`), inlier ratio $\ge 35\%$ khẳng định nhận diện và trả về ma trận $[R \mid t]$.
- **Camera Metrology**: Bù sai lệch $+17.5\text{ mm}$ từ Left IR sang tâm mặt kính camera (Front Glass Centerline / 1/4"-20 screw center).
- **Trích xuất đủ 6DoF**: Cung cấp đầy đủ $[X, Y, Z, \text{Roll}, \text{Pitch}, \text{Yaw}]$ và bounding box 2D tái chiếu từ mô hình 3D.
- **Zero AMR Dependency**: Hoàn toàn độc lập với hệ thống điều khiển gầm/bánh xe của robot.
