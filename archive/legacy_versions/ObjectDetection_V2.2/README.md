# ObjectDetection V2.1 (Unified Release)

Bản phát hành hợp nhất **ObjectDetection V2.1** kết hợp đầy đủ:
1. **Thuật toán Pallet Detection mới nhất (từ v1.5)**: Cải tiến toàn diện thuật toán RANSAC, bộ lọc nhảy tức thời (JumpDetector), nhận diện trạng thái dừng yên (Settle Detector), sửa tâm inlier (Centroid-fix), bộ lọc trung vị không gian (VectorMedianFilter) và sửa góc Euler theo trục quang học camera thực tế.
2. **CAD-Native 6DoF Pose Estimation (từ v2.0)**: Khả năng nhận diện và ước lượng 6DoF trực tiếp từ file CAD mà không qua 2D YOLO, chạy trên NPU RKNN (Orange Pi 5B - RK3588S), trích xuất đủ 6 bậc tự do $[X, Y, Z, \text{Roll}, \text{Pitch}, \text{Yaw}]$, chuẩn hóa tâm đo lường tại **Tâm mặt kính camera (Front Glass Centerline)** lệch $+17.5\text{ mm}$ từ Left IR.

---

## Cấu Trúc Thư Mục (Directory Layout)

```text
ObjectDetection_V2.1/
├── ObjectDetection/                     # Mã nguồn C++ Plugin
│   ├── adapters/
│   │   ├── CxCyWhClassScoresAdapter.*
│   │   ├── DflMultiScaleClassScoresAdapter.*
│   │   └── CADHierarchyAdapter.*         # [v2.0] CAD-Native model adapter
│   ├── applications/
│   │   ├── pallet/
│   │   │   └── PalletApplication.*       # [v1.5] Latest Pallet algorithms
│   │   └── cad_pose/
│   │       └── CADPoseApplication.*      # [v2.0] CAD 6DoF RANSAC Kabsch SVD
│   ├── backends/
│   │   ├── rknn/
│   │   └── tensorrt/
│   ├── config/
│   │   └── orangepi5b/
│   │       ├── person.json
│   │       ├── pallet.json
│   │       ├── pallet_debug.json
│   │       ├── person_and_pallet.json
│   │       ├── cad_table.json            # [v2.0] Runtime config for CAD table
│   │       ├── cad_table_debug.json      # [v2.0] Debug config for CAD table
│   │       └── person_pallet_and_cad.json # [v2.1] Combined 3-detector config
│   ├── ApplicationFactory.cpp            # Đăng ký cả pallet & cad_pose
│   ├── ModelAdapterFactory.cpp           # Đăng ký cả dfl, cxcywh, cad_hierarchy_v1
│   ├── ModelManifest.cpp                 # Hỗ trợ bbox & cad_6dof
│   ├── changelog_version.txt             # Lịch sử phiên bản từ v1.0 -> v2.1
│   └── README.md
├── models/                               # Kho mô hình hợp nhất
│   └── object_detection/                 # Toàn bộ model ngang hàng bên trong object_detection
│       ├── cad_pose/
│       │   └── table_cad/v1/             # [v2.0] CAD 6DoF NPU RKNN & Descriptors
│       ├── pallet/
│       │   └── v3/                       # Model Pallet DFL YOLO
│       └── person/
│           └── v1/                       # Model Person YOLO
├── README_CAD2POSE.md                    # Tài liệu kỹ thuật chi tiết CAD 6DoF
└── README.md                             # Tài liệu tổng quan bản 2.1
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
