# Model Package Schema

Required package form:

```text
object_detection/<task_or_object>/...
(e.g., object_detection/pallet/v3/, object_detection/person/v1/, object_detection/cad_pose/table_cad/v1/)
├── model.json
├── labels.txt
└── artifacts/
    └── <platform_profile>/
        └── <model artifact>
```

Required manifest fields (BBox Detection):

```json
{
  "schema_version": 2,
  "model_id": "person_v1",
  "model_name": "Person Detector V1",
  "version": "v1",
  "task_type": "bbox",
  "object_type": "person",
  "adapter": "cxcywh_class_scores_v1",
  "preprocess": {},
  "output": {},
  "labels": "labels.txt",
  "artifacts": {
    "profiles": {}
  }
}
```

Required manifest fields (CAD-Native 6DoF Pose):

```json
{
  "schema_version": 2,
  "model_id": "table_cad_v1",
  "model_name": "CAD-Native 6DoF Pose Estimator",
  "version": "v1",
  "task_type": "cad_6dof",
  "object_type": "table",
  "adapter": "cad_hierarchy_v1",
  "cad_metadata": {},
  "input": {},
  "output": {
    "type": "cad_hierarchy_v1",
    "num_classes": 1
  },
  "labels": "labels.txt",
  "artifacts": {
    "profiles": {}
  }
}
```

Supported `task_type`: `"bbox"`, `"cad_6dof"`.
Supported `adapter`: `"cxcywh_class_scores_v1"`, `"dfl_multiscale_class_scores_v1"`, `"cad_hierarchy_v1"`.
Legacy top-level `backend`, `target_platform`, `model`, and `postprocess` forms are not supported.
