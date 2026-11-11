# ObjectDetection

Production model layout:

```text
models/object_detection/
├── person/v1/
└── pallet/v3/
```

Both packages use manifest schema 2 and platform-specific artifact profiles.

Canonical adapters:

- `cxcywh_class_scores_v1`
- `dfl_multiscale_class_scores_v1`

Canonical applications:

- `none`
- `pallet`

Orange Pi 5B configs are in `config/orangepi5b/`.

Production logging uses:

```json
"enable_perf_log": true,
"perf_log_every_n": 300,
"debug_publish_raw_objects": false
```

`enable_perf_log` controls only periodic performance lines. Errors, warnings, initialization status and the first valid RGBD metadata line are not suppressed by it.

`pallet_debug.json` keeps `debug_publish_raw_objects=true` for raw block/front visualization. Raw visualization does not enable tensor or geometry debug logging.
