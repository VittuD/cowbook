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

Input paths may be either videos or precomputed tracking JSON files. `tracking_concurrency` defaults to `1` intentionally to avoid GPU contention. It caps how many video inputs in a group may be tracked at the same time and is clamped to the number of trackable videos in that group. `log_progress` is off by default and only affects human-readable console output. If masking is enabled, masks default to files under `assets/masks`, and masked-video reuse is cached under `var/cache/masked_videos`.

Config loading is strict:

- file-based loading raises `FileNotFoundError` for missing files
- file-based loading raises `json.JSONDecodeError` for invalid JSON
- file-based and in-memory normalization raise `ValueError` for invalid config values
- in-memory config loading does not mutate the caller-provided object

Optional `tracking_cleanup` adds an alternate tracking path that:

- preprocesses detections before tracking
- preserves detection lineage with `det_idx`
- can prune short-lived tracks with a second tracking pass
- can smooth final output boxes after tracking
