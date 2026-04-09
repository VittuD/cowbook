# Configuration

Cowbook uses a JSON runtime config plus a small CLI override surface. The config is the primary contract; the CLI only tweaks a few operational settings.

The fields that matter most in practice are `model_path`, `calibration_file`, and `video_groups`. The remaining fields mostly control run layout, [masking](cli.md), export behavior, worker counts, and optional progress logging.

Minimal example:

```json
{
  "model_path": "models/yolov11_best.pt",
  "calibration_file": "assets/calibration/camera_system.json",
  "runtime_root": "var",
  "run_name": "demo",
  "fps": 6,
  "mask_videos": false,
  "create_projection_video": true,
  "clean_frames_after_video": false,
  "convert_to_csv": true,
  "num_plot_workers": 0,
  "tracking_concurrency": 1,
  "log_progress": false,
  "video_groups": [
    [
      { "path": "sample_data/videos/Ch1_60.mp4", "camera_nr": 1 }
    ]
  ]
}
```

Input paths may be either videos or precomputed tracking JSON files. `tracking_concurrency` defaults to `1` as the conservative baseline and is clamped to the number of trackable videos in the group. The public contract is unchanged across values: effective concurrency `1` runs tracking inline, while higher effective values use worker processes. Both paths preserve the same event flow and result shape. The inline path reuses one YOLO model instance per `(model_path, tracking mode)` within the group; the pooled path reuses one model instance per worker and per `(model_path, tracking mode)` instead of reloading it for every video. Direct tracking and cleanup tracking do not share a model instance, so tracker callbacks and tracking IDs do not leak across modes. `log_progress` is off by default and only affects human-readable console output. If masking is enabled, masks default to files under `assets/masks`, and masked-video reuse is cached under `var/cache/masked_videos`.

Config loading is strict:

- file-based loading raises `FileNotFoundError` for missing files
- file-based loading raises `json.JSONDecodeError` for invalid JSON
- file-based and in-memory normalization raise `ValueError` for invalid config values
- in-memory config loading does not mutate the caller-provided object

Optional `tracking_cleanup` adds an alternate tracking path that:

- preprocesses detections before tracking
- preserves detection lineage with `det_idx`
- can prune short-lived tracks using a gap-tolerant consecutive streak rule
- can smooth final output boxes after tracking

For short-track pruning, `min_track_length` now means the minimum surviving streak length rather than the total observation count. Small gaps are tolerated via `short_track_gap_tolerance`, which defaults to `6` frames. With the defaults, a track survives pruning when it reaches a streak of at least `min_track_length` observations while allowing up to `6` missing frames between consecutive observations in that streak.
