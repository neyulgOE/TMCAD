# ObjectDetection Architecture

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

Model versioning belongs to model packages, not core source names.

```text
object_detection/person/v1
object_detection/pallet/v3
```

Person uses `cxcywh_class_scores_v1` with no application.
Pallet uses `dfl_multiscale_class_scores_v1` and the `pallet` application.
