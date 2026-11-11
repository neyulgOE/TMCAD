# Model Package Schema

Required package form:

```text
object_detection/<object>/v<N>/
├── model.json
├── labels.txt
└── artifacts/
    └── <platform_profile>/
        └── <model artifact>
```

Required manifest fields:

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

Legacy top-level `backend`, `target_platform`, `model`, and `postprocess` forms are not supported.
