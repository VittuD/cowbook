# Multi-Camera Video Tracking & Ground Projection

Detect objects with Ultralytics YOLO (v8 → v11), undistort image coordinates with a calibrated camera model, project them onto a ground plane (barn coordinates), and render a combined top-down sequence across multiple cameras.

## Features

* 🔎 YOLO tracking (Ultralytics) with per-frame JSON outputs
* 📷 Multi-camera grouping (1–4 per group), with uniqueness validation on `camera_nr`
* 🎯 Camera undistortion & ground projection using pre-measured 2D↔3D correspondences
* 🖼️ Combined per-frame images of projected points (across cameras)
* 🎞️ Video assembly from projected images
* ⚙️ Config-driven workflow + command-line overrides
* 🧹 Optional cleanup of rendered frames after the final video (ON by default)

---

## Requirements

* Python 3.10+
* `pip install -r requirements.txt`
* A working PyTorch + CUDA/CPU stack compatible with `ultralytics` (tested with YOLO v8–v11).

### Python dependencies (from `requirements.txt`)

* `ultralytics`
* `opencv-python-headless`
* `numpy`
* `tqdm`

> YOLO will also create `runs/track/...` outputs **only if** tracking-video saving is enabled (see flags below).

---

## Directory Structure

```
├── config_loader.py
├── config.json
├── csv_converter.py
├── directory_manager.py
├── frame_processor.py
├── group_processor.py
├── json_merger.py
├── legacy/
│   ├── calibration_matrix.json
│   ├── image_utils.py
│   ├── points_data.py
│   └── real_world_points.json
├── main.py
├── mask/
│   ├── combined_mask_ch1.png
│   ├── combined_mask_ch4.png
│   ├── combined_mask_ch6.png
│   └── combined_mask_ch8.png
├── models/
│   ├── yolov8_best.pt
│   └── yolov11_best.pt
├── output_frames/
├── output_json/
├── output_videos/
├── preprocess_video.py
├── processing.py
├── tracking.py
├── video_processor.py
└── videos/
```

`legacy/barn.png` (optional): background image used when drawing projected points. If missing, a blank canvas is used.

---

## Configuration (`config.json`)

Example:

```
{
  "model_path": "models/yolov11_best.pt",
  "calibration_file": "legacy/calibration_matrix.json",

  "mask_videos": true,
  "masked_video_folder": "masked_videos/demo_videos_masking/",
  "num_mask_workers": 4,
  "mask_strict_half_rule": true,

  "output_image_folder": "output_frames/demo_videos_masking/",
  "output_video_folder": "output_videos/demo_videos_masking/",
  "output_json_folder": "output_json/demo_videos_masking/",
  "output_video_filename": "combined_projection.mp4",
  "output_image_format": "jpg",

  "save_tracking_video": true,
  "create_projection_video": true,
  "clean_frames_after_video": true,         // NEW: delete frames after video (default true)
  "convert_to_csv": true,
  "fps": 6,

  "num_plot_workers": 8,
  "num_tracking_workers": 1,                // NEW default: 1 to avoid GPU OOM

  "masks": {
    "Ch1": "mask/combined_mask_ch1.png",
    "Ch4": "mask/combined_mask_ch4.png",
    "Ch6": "mask/combined_mask_ch6.png",
    "Ch8": "mask/combined_mask_ch8.png"
  },

  "camera_to_mask_map": {
    "1": "Ch1",
    "4": "Ch4",
    "6": "Ch6",
    "8": "Ch8"
  },

  "video_groups": [
    [
      { "path": "videos/demo_videos_masking/Ch1_60.mp4", "camera_nr": 1 },
      { "path": "videos/demo_videos_masking/Ch4_60.mp4", "camera_nr": 4 },
      { "path": "videos/demo_videos_masking/Ch6_60.mp4", "camera_nr": 6 },
      { "path": "videos/demo_videos_masking/Ch8_60.mp4", "camera_nr": 8 }
    ]
  ]
}
```

**Notes**

* `video_groups` is a list of groups; each group contains 1–4 entries with unique `camera_nr`.
* You may pass **JSON files** as inputs too (skips YOLO and uses the provided tracking data).
* Set `save_tracking_video` to `true` to make YOLO write annotated track videos under `runs/track/...`.
* `num_plot_workers` controls parallel rendering of per-frame images (0 = sequential).
* `num_tracking_workers` defaults to 1. Increase cautiously if your GPU has plenty of VRAM.
* `output_image_format` can be `png` or `jpg` (jpg is smaller/faster to write).
* Masking:
  * A mask is chosen per input either by `camera_to_mask_map` or by filename heuristic (`Ch1|Ch4|Ch6|Ch8` in the path).
  * Masks are applied if their size **matches** the video frame, or if the video is **exactly half** the mask size (then it is NEAREST-resized). Otherwise frames are left unmodified.

---

## Usage

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Run

```bash
# uses config.json by default
python -m cowbook

# explicit config (positional or flag)
python -m cowbook config.my.json
python -m cowbook --config config.my.json
```

---

## Outputs

* `output_json/*.json` — tracking data per input video and per processed/merged step
* `output_frames/combined_projected_centroids_frame_XXX.png` — projected points across cameras for each frame index
* `output_videos/combined_projection.mp4` — single video built from the frames (if `create_projection_video=true`)

> By default, `output_frames` are **deleted** after the final video is created. Use `--no-clean-frames` (or set `clean_frames_after_video: false`) to keep them.

---

## CSV conversion & JSON merging

**Merge processed JSONs (per group):**

```bash
python json_merger.py --inputs output_json/cam1_processed.json output_json/cam4_processed.json \
                      --output output_json/group_1_merged_processed.json
```

**Convert JSON → CSV:**

Single file:

```bash
python csv_converter.py --input output_json/group_1_merged_processed.json --output group1.csv
```

Multiple files → one CSV, with a `source` column:

```bash
python csv_converter.py \
  --inputs output_json/group_1_merged_processed.json output_json/group_2_merged_processed.json \
  --output all_groups.csv --source-col source
```

---

## How it works (pipeline)

1. Load config, ensure output folders exist (and `output_frames` is cleared).
2. For each `video_group`:
   * Run YOLO tracking for each video (unless the input is already a JSON).
   * Undistort detections using camera intrinsics & distortion coefficients.
   * Project centroids to ground plane (PnP with pre-measured correspondences).
   * Merge projected points across the group’s cameras and render per-frame images.
   * Optionally convert processed/merged JSONs to CSV.
3. Optionally stitch images → `combined_projection.mp4` at configured FPS.
4. Optionally delete frames (default true).

---

## Masking (optional)

If `mask_videos=true`, videos are preprocessed into `masked_videos/...` with static masks:

* If mask resolution equals video resolution → apply directly.
* If video is exactly half of mask resolution → mask is NEAREST-resized and applied.
* Otherwise (mismatch) → frames are left unmodified (warning is logged).  
Channel is chosen via `camera_to_mask_map` or `ChX` in the filename.

---

## Troubleshooting

* **Torch / CUDA not found:** install a compatible PyTorch build for your system.
* **`calibration_matrix.json` missing:** update `calibration_file` path or provide the file.
* **No images in `output_frames`:** ensure `video_groups` points to existing videos/JSONs.
* **YOLO video outputs in `runs/track/...`:** use `--no-save-tracking-video` or set `save_tracking_video: false` in config.
* **Frames disappeared:** default cleanup is ON → use `--no-clean-frames`.

---

## Warnings & gotchas

* **Calibration & resolution coupling:** projection assumes your camera intrinsics and frame size match the calibration. In `legacy/image_utils.py`, constants are tuned for `2688×1520`. Mismatches will degrade or break projection.
* **Frame alignment:** merging uses integer `frame_id` only. Ensure videos are time-aligned (fps & offsets), or projections across cameras won’t represent the same moment.
* **Frame numbering/padding:** rendered filenames pad based on total frames processed, not on max `frame_id`. If your `frame_id`s are sparse, the zero-padding may not match the highest `frame_id` magnitude (this is cosmetic).
* **Barn background:** if `legacy/barn.png` is missing, a blank canvas is used.
* **OpenCV headless vs GUI:** repo uses `opencv-python-headless`. Some `legacy` helpers (`cv.imshow`) are present but not used in the pipeline; avoid calling them in headless environments.
* **GPU VRAM & parallelism:** `num_tracking_workers` defaults to 1 to avoid OOM. Increase only if your GPU can handle multiple concurrent models.

---

## Known issues / TODOs

* **CSV `--source-col` header mismatch:** currently only works correctly when `--source-col source`. (Fix: align header naming in `csv_converter.py`.)
* **`mask_strict_half_rule` not wired:** config key exists; logic currently enforces only the exact-half rule regardless. (Plumb the flag or remove it.)
* **Ultralytics API evolution:** the `track` API may change; consider pinning `ultralytics` version or adding adapters.
* **External JSON schema assumptions:** `processing.extract_data` assumes the raw schema written by this repo; add schema checks if ingesting third-party JSONs.

---

## Docker (example)

```bash
docker run -it --rm \
  -p 8888:8888 \
  --name=yolo_cow \
  --ipc=host \
  --gpus all \
  -v ~/COW:/ultralytics/COW \
  davidevitturini/ultralytics_jupyter
```

Mount your project into the container and run the same commands inside.

---

## License

GPL-3.0. See `LICENSE`.
