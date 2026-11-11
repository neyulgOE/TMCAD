# ObjectDetection Architecture (Version 2.1)

```text
CameraRGBD callback
    -> new-data state
    -> existing detection scheduler
    -> DetectionManager
    -> Detector
    -> ModelAdapter + Backend
    -> optional Application
    -> Message_ObjectDetections
```

Model versioning belongs to model packages, not core source names:

```text
models/object_detection/person/v1
models/object_detection/pallet/v3
models/object_detection/cad_pose/table_cad/v1
```

1. **Person**: Uses `cxcywh_class_scores_v1` with no application (`application: "none"`).
2. **Pallet**: Uses `dfl_multiscale_class_scores_v1` and the `pallet` application (v1.5 production algorithms with Settle detector, JumpDetector, Centroid-fix, VectorMedianFilter).
3. **CAD 6DoF Table**: Uses `cad_hierarchy_v1` and the `cad_pose` application (v2.0 100% CAD-native pose estimation, RANSAC Kabsch SVD on RKNN NPU, +17.5 mm Front Glass Centerline offset).

