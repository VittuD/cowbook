# Multi‑Camera Video Tracking & Ground Projection

Detect objects in videos with YOLOv8, undistort their image coordinates with a calibrated camera model, project them onto a ground plane (barn coordinates), and render a combined top‑down sequence (images + video) across multiple cameras.

## Features

* 🔎 **YOLOv8 tracking** with per‑frame JSON outputs
* 📷 **Multi‑camera** grouping (1–4 per group), with uniqueness validation on `camera_nr`
* 🎯 **Camera undistortion** & **ground projection** using pre‑measured 2D↔3D correspondences
* 🖼️ **Combined per‑frame images** of projected points (across cameras)
* 🎞️ **Video assembly** from projected images
* ⚙️ Config‑driven workflow + command‑line overrides

---

## Requirements

* Python 3.10+
* `pip install -r requirements.txt`
* A working PyTorch + CUDA/CPU stack compatible with `ultralytics` (YOLOv8). See PyTorch’s install guide for your GPU/OS if needed.

### Python dependencies (from `requirements.txt`)

* `ultralytics`
* `opencv-python-headless`
* `numpy`
* `tqdm`

> **Note:** YOLOv8 will also create `runs/track/...` outputs **only if** tracking‑video saving is enabled (see flags below).

---

## Directory Structure

```
├── config_loader.py
├── config.json
├── directory_manager.py
├── frame_processor.py
├── legacy/
│   ├── __init__.py
│   ├── calibration_matrix.json
│   ├── image_utils.py
│   ├── points_data.py
│   └── real_world_points.json
├── main.py
├── models/
│   └── best.pt                # your YOLO weights
├── output_frames/             # generated per‑frame projected images
├── output_json/               # tracking JSONs
├── output_videos/             # final combined video
├── processing.py
├── requirements.txt
├── tracking.py
├── video_processor.py
└── videos/
    └── ...                    # your input videos (e.g. Ch1_60.mp4)
```

`legacy/barn.png` (optional): background image used for drawing projected points. If missing, a blank canvas is used.

---

## Configuration (`config.json`)

Example:

```json
{
  "model_path": "models/best.pt",
  "calibration_file": "legacy/calibration_matrix.json",
  "output_image_folder": "output_frames",
  "output_video_folder": "output_videos",
  "output_json_folder": "output_json",
  "video_groups": [
    [
      { "path": "videos/Ch1_60.mp4", "camera_nr": 1 },
      { "path": "videos/Ch4_60.mp4", "camera_nr": 4 },
      { "path": "videos/Ch6_60.mp4", "camera_nr": 6 },
      { "path": "videos/Ch8_60.mp4", "camera_nr": 8 }
    ]
  ],
  "num_plot_workers": 4,
  "output_image_format": "jpg",
  "save_tracking_video": false,
  "create_projection_video": true,
  "fps": 6
}
```

**Notes**

* `video_groups` is a list of groups; each group contains 1–4 entries with unique `camera_nr`.
* You may pass **JSON files** as inputs too (skips YOLO and uses the provided tracking data).
* Set `save_tracking_video` to `true` to make YOLO write annotated track videos under `runs/track/...`.
* Set `num_plot_workers` to control the number of parallel workers for rendering images (0 = sequential).
* Set `output_image_format` to choose between "png" or "jpg" for the output images.

* REMINDER, include the json_merger and csv in the readme: python3 csv_converter.py --input output_json/group_1_merged_processed.json --output group1.csv
* python csv_converter.py \
  --inputs output_json/group_1_merged_processed.json output_json/group_2_merged_processed.json \
  --output all_groups.csv --source-col source

---

## Usage

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Run

```bash
# uses config.json by default
python main.py

# explicit config (positional or flag)
python main.py config.my.json
python main.py --config config.my.json
```

### Command‑line flags

You can override whether YOLO writes annotated tracking videos:

```bash
# force ON (YOLO saves annotated videos)
python main.py --save-tracking-video

# force OFF (skip saving annotated videos)
python main.py --no-save-tracking-video
```

If neither flag is provided, the value in `config.json` (`save_tracking_video`) is used.

---

## Outputs

* **`output_json/*.json`** — tracking data per input video
* **`output_frames/combined_projected_centroids_frame_XXX.png`** — projected points across cameras for each frame index
* **`output_videos/combined_projection.mp4`** — single video built from the frames (if `create_projection_video=true`)

---

## How it works (pipeline)

1. Load config, model, and ensure output folders exist.
2. For each `video_group`:

   * Run YOLOv8 `track` (unless inputs are already JSON) → write per‑frame detections.
   * Undistort detections using camera intrinsics & distortion coefficients.
   * Project centroids to ground plane using PnP with pre‑measured correspondences.
   * Merge projected points across the group’s cameras and render a frame image.
3. Optionally stitch images → `combined_projection.mp4` at configured FPS.

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

## Troubleshooting

* **Torch / CUDA not found:** install a compatible PyTorch build for your system.
* **`calibration_matrix.json` missing:** update `calibration_file` path or provide the file.
* **No images in `output_frames`:** ensure `video_groups` points to existing videos/JSONs.
* **YOLO video outputs in `runs/track/...`:** use `--no-save-tracking-video` or set `save_tracking_video: false` in config.

---

## License

AGPL-3.0
