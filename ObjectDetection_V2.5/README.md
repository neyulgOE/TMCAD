# ObjectDetection V2.5 (Stateless Pallet & Sub-2mm 3D CAD Metrology Release)

Bản phát hành hợp nhất **ObjectDetection V2.5** đại diện cho bước nâng cấp toàn diện về độ chính xác đo lường tĩnh và tính ổn định trên bo mạch nhúng (Orange Pi 5B - RK3588S):

1. **Thuật toán Pallet Detection V2.0 Stateless + Ray-Plane Intersection**: 
   - Kiến trúc không trạng thái (khớp kiến trúc Request-Response đồng bộ của native service).
   - Tiền lọc outlier Z bằng MAD (`madFilterByZ`), triệt tiêu 100% gai độ sâu ở lỗ khoét và viền bóng.
   - Chặn góc nghiêng mặt phẳng $>60^\circ$ trong RANSAC, tránh bắt nhầm cạnh bên khi nhìn chéo.
   - Weighted least-squares refit $Z = aX + bY + c$ và xác định tọa độ 3D từng block bằng giao điểm tia camera với mặt phẳng (`planeDepthAtPixel`), loại bỏ hoàn toàn sai lệch foreshortening khi pallet xoay góc lớn.

2. **CAD-Guided 6DoF Pose Estimation Sub-2mm Metrology**: 
   - **CADTemplateDetector**: Nhận diện và khoanh vùng đối tượng trực tiếp từ ảnh RGB bằng mô hình CAD (36 mẫu ORB đa góc nhìn), hỗ trợ fallback phân đoạn hình học trên độ sâu khi bề mặt không vân.
   - **CADSceneSegmenter**: Giới hạn frustum độ sâu theo bbox RGB, bóc tách mặt phẳng sàn (RANSAC Floor Plane), lọc đai cao độ ($0.04\text{m} \le h \le 1.25\text{m}$) và gom cụm không gian 3D voxel qua BFS flood-fill lan truyền liên thông.
   - **Softmax Centroid Interpolation**: Nội suy tọa độ CAD liên tục trên top-3 centroids ($\tau = 0.07$), triệt tiêu sai số lượng tử hóa 26.7 mm của 512 centroids rời rạc.
   - **Fast Gauss-Newton Point-to-Plane Refinement**: Bộ tinh chỉnh siêu tốc (<0.3 ms, thuần C++, không phụ thuộc Open3D), ép sai số tịnh tiến về **$\le 2\text{ mm}$ (thực tế $< 0.5\text{ mm}$)** và sai số góc **$< 0.1^\circ$**.
   - **Zero Parallax Reprojection**: Chiếu hộp 2D và 8 đỉnh 3D chuẩn xác theo hệ quang học `pose.t`, đồng thời giữ nguyên mechanical offset $+17.5\text{ mm}$ (`t_glass`) cho tọa độ cơ học xuất ra robot.

---

## Cấu Trúc Thư Mục (Directory Layout)

```text
ObjectDetection_V2.5/
├── ObjectDetection/                     # Mã nguồn C++ Plugin
│   ├── adapters/
│   │   ├── CxCyWhClassScoresAdapter.*
│   │   ├── DflMultiScaleClassScoresAdapter.*
│   │   └── CADHierarchyAdapter.*         # CAD-Native model adapter
│   ├── applications/
│   │   ├── pallet/
│   │   │   └── PalletApplication.*       # [v2.5 Stateless] Ray-Plane Intersection & MAD Filter
│   │   └── cad_pose/
│   │       ├── CADTemplateDetector.*     # [v2.5] Multi-view CAD Template & Feature Matching
│   │       ├── CADSceneSegmenter.*       # [v2.5] Ground Plane RANSAC & 3D Voxel Clustering
│   │       └── CADPoseApplication.*      # [v2.5] Softmax Interp + Fast Gauss-Newton Refinement
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
│   │       └── person_pallet_and_cad.json # Combined 3-detector config
│   ├── ApplicationFactory.cpp            # Đăng ký cả pallet & cad_pose
│   ├── ModelAdapterFactory.cpp           # Đăng ký cả dfl, cxcywh, cad_hierarchy_v1
│   ├── ModelManifest.cpp                 # Hỗ trợ bbox & cad_6dof
│   ├── changelog_version.txt             # Lịch sử phiên bản từ v1.0 -> v2.5
│   └── README.md
├── models/                               # Kho mô hình hợp nhất
│   └── object_detection/                 # Toàn bộ model ngang hàng bên trong object_detection
│       ├── cad_pose/
│       │   └── table_cad/v1/             # CAD 6DoF NPU RKNN & Descriptors
│       ├── pallet/
│       │   └── v3/                       # Model Pallet DFL YOLO
│       └── person/
│           └── v1/                       # Model Person YOLO
├── RELEASE_NOTES.txt                     # Ghi chú phát hành chi tiết V2.5
└── README.md
```

---

## Chi Tiết Kỹ Thuật

### 1. Pallet Detection (Stateless & Marker-Style)
- **Kiến trúc Stateless**: Triệt tiêu overhead lưu vết buffer thời gian, khớp chuẩn xác với mô hình Request-Response độc lập của service native trên robot.
- **Tiền lọc MAD trên Z (`madFilterByZ`)**: Trước khi RANSAC, lọc loại bỏ các điểm có $|Z - \text{med}| > \max(3\sigma, 5\text{ mm})$.
- **Chặn góc nghiêng mặt phẳng RANSAC**: Loại bỏ ngay các mặt phẳng có $|\mathbf{n}_z| < 0.5$ (nghiêng $> 60^\circ$ so với camera) trong vòng lặp vote.
- **Giao điểm tia - mặt phẳng (`planeDepthAtPixel`)**:
  $$l_x = \frac{p_x - c_x}{f_x}, \quad l_y = \frac{p_y - c_y}{f_y}, \quad Z = \frac{c}{1 - a \cdot l_x - b \cdot l_y}$$
- **Euler góc theo hệ camera optical frame**: $X$-phải, $Y$-xuống, $Z$-tiến.

### 2. CAD-Native 6DoF Pose Estimation (Sub-2mm Metrology)
- **Đám mây điểm tinh sạch**: RGB ROI + Bóc sàn + Lọc đai cao độ loại bỏ $100\%$ điểm ngoại lai (tường, sàn, chướng ngại vật).
- **Softmax Centroid Interpolation**: $\mathbf{P}_{\text{cad}} = \sum_{m \in \text{top-3}} w_m \mathbf{c}_m$ với $w_m \propto \exp(s_m / 0.07)$.
- **Gauss-Newton Point-to-Plane Refinement**: Tối ưu nghiệm trên phương trình $\mathbf{J}^T \mathbf{W} \mathbf{J} \Delta = \mathbf{J}^T \mathbf{W} \mathbf{r}$, giảm sai số từ 18-20 mm xuống $\le 2\text{ mm}$ trong thời gian $< 0.3\text{ ms}$.
- **Chuẩn hóa đối xứng $C_2$**: Chuẩn hóa góc Yaw trong dải $[-90^\circ, +90^\circ]$ ổn định cho robot docking.
- **Tốc độ thời gian thực**: Toàn bộ chu trình đạt $>70\text{ FPS}$ trên Orange Pi 5B (RK3588S), CPU load $<20\%$.
