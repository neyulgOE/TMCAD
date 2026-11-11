# ObjectDetection (Version 2.3)

Production model layout:

```text
models/
└── object_detection/
    ├── cad_pose/
    │   └── table_cad/v1/
    ├── pallet/
    │   └── v3/
    └── person/
        └── v1/
```

All packages use manifest schema 2 and platform-specific artifact profiles.

Canonical adapters:

- `cxcywh_class_scores_v1`
- `dfl_multiscale_class_scores_v1`
- `cad_hierarchy_v1`

Canonical applications:

- `none`
- `pallet` (v1.5 production algorithms: Settle detector, JumpDetector, Centroid-fix, VectorMedianFilter)
- `cad_pose` (v2.3 Priority RANSAC Kabsch SVD, distance-adaptive point extraction, multi-input zero-allocation RKNN)

Orange Pi 5B configs are in `config/orangepi5b/` (`person.json`, `pallet.json`, `pallet_debug.json`, `cad_table.json`, `cad_table_debug.json`, `person_and_pallet.json`, `person_pallet_and_cad.json`).

Production logging uses:

```json
"enable_perf_log": true,
"perf_log_every_n": 300,
"debug_publish_raw_objects": false
```

`enable_perf_log` controls only periodic performance lines. Errors, warnings, initialization status and the first valid RGBD metadata line are not suppressed by it.

`pallet_debug.json` and `cad_table_debug.json` keep `debug_publish_raw_objects=true` for raw visualization. Raw visualization does not enable tensor or geometry debug logging.
