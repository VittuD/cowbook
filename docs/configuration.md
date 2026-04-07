# Configuration

Cowbook uses a JSON runtime config plus a small CLI override surface. The config is the primary contract; the CLI only tweaks a few operational settings.

The fields that matter most in practice are `model_path`, `calibration_file`, and `video_groups`. The remaining fields mostly control run layout, [masking](cli.md), export behavior, and worker counts.

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
  "num_tracking_workers": 1,
  "video_groups": [
    [
      { "path": "sample_data/videos/Ch1_60.mp4", "camera_nr": 1 }
    ]
  ]
}
```

Input paths may be either videos or precomputed tracking JSON files. `num_tracking_workers` defaults to `1` intentionally to avoid GPU contention. If masking is enabled, masks default to files under `assets/masks`, and masked-video reuse is cached under `var/cache/masked_videos`.
