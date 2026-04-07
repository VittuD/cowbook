# Cowbook

Cowbook is a packaged Python pipeline for multi-camera cow tracking and ground-plane projection in a fixed barn setup.

It can:
- run YOLO tracking on one or more camera videos
- convert detections into per-frame centroids
- project those centroids into barn coordinates using a calibrated camera model
- merge camera outputs at group level
- render a combined top-down sequence
- export JSON, CSV, and final MP4 artifacts

The current supported entrypoint is:

```bash
python -m cowbook
```

## Current Layout

```text
.
├── assets/
│   ├── calibration/
│   ├── images/
│   ├── masks/
│   └── trackers/
├── config.json
├── configs/
├── models/
├── sample_data/
│   └── videos/
├── scripts/
├── src/cowbook/
│   ├── app/
│   ├── core/
│   ├── execution/
│   ├── io/
│   ├── vision/
│   └── workflows/
├── tests/
└── var/
```

Directory intent:
- `assets/`: persistent non-code assets such as calibration, masks, tracker config, and barn background
- `configs/`: example run configs
- `sample_data/`: local sample inputs for smoke/full runs
- `src/cowbook/`: packaged application code
- `var/`: runtime outputs and cache

## Install

Runtime install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Dev install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

The repo expects a working PyTorch environment that is compatible with `ultralytics`. CUDA setup is external to this project.

## Run

Use the default config:

```bash
python -m cowbook
```

Use an explicit config:

```bash
python -m cowbook --config configs/smoke.json
python -m cowbook configs/full.cpu.json
```

Common CLI overrides:

```bash
python -m cowbook --config configs/full.cpu.json --fps 12 --output-video-filename run.mp4
python -m cowbook --config configs/full.cpu.json --mask-videos
python -m cowbook --config configs/full.cpu.json --no-clean-frames-after-video
```

Supported CLI overrides:
- `--fps`
- `--output-video-filename`
- `--output-image-format`
- `--num-plot-workers`
- `--num-tracking-workers`
- `--create-projection-video` / `--no-create-projection-video`
- `--clean-frames-after-video` / `--no-clean-frames-after-video`
- `--mask-videos` / `--no-mask-videos`

## Docker

One Docker image is included:
- `docker/Dockerfile`: runtime based on the official Ultralytics image

The image:
- uses `ultralytics/ultralytics:8.4.34` as the base
- copies the repo into `/app`
- installs the package in editable mode
- includes the current `assets/`, `configs/`, `models/`, and `sample_data/`
- defaults to `python -m cowbook --config config.json`

Build the image:

```bash
docker build -f docker/Dockerfile -t cowbook .
```

Run it on CPU and persist outputs on the host:

```bash
docker run --rm -it \
  -v "$(pwd)/var:/app/var" \
  cowbook
```

Run a specific config on CPU:

```bash
docker run --rm -it \
  -v "$(pwd)/var:/app/var" \
  cowbook \
  --config configs/full.cpu.json
```

Run the same image with GPU access:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd)/var:/app/var" \
  cowbook \
  --config configs/full.gpu.json
```

Notes:
- the host needs a working NVIDIA driver for GPU runs
- Docker needs NVIDIA Container Toolkit support for `--gpus all`
- the same image can run on CPU-only hosts or on NVIDIA GPU hosts
- pinning the Ultralytics base tag keeps the runtime reproducible

If you want to override configs or assets from the host instead of using the copies baked into the image, mount them into `/app`.

## Config Model

The runtime contract is a JSON config plus a small CLI override surface.

Important fields:
- `model_path`: YOLO weights path
- `calibration_file`: calibration JSON, default `assets/calibration/calibration_matrix.json`
- `video_groups`: list of groups; each group contains 1 to 4 inputs with unique `camera_nr`
- `runtime_root`: base runtime folder, default `var`
- `run_name`: run-scoped output namespace, default `default`
- `mask_videos`: whether to preprocess videos through static masks before inference
- `create_projection_video`: whether to assemble the final MP4
- `clean_frames_after_video`: whether to delete rendered frames after video assembly
- `convert_to_csv`: whether to export CSV beside processed and merged JSON

Minimal example:

```json
{
  "model_path": "models/yolov11_best.pt",
  "calibration_file": "assets/calibration/calibration_matrix.json",
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

Notes:
- input paths may be videos or precomputed tracking JSON files
- `num_tracking_workers` defaults to `1` intentionally to avoid GPU contention
- masks default to `assets/masks/*.png`
- masked-video cache defaults to `var/cache/masked_videos`

## Output Layout

By default, each run lives under:

```text
var/runs/<run_name>/
├── frames/
├── json/
└── videos/
```

Typical artifacts:
- `var/runs/<run_name>/json/<input>_tracking.json`
- `var/runs/<run_name>/json/<input>_tracking_processed.json`
- `var/runs/<run_name>/json/group_<n>_merged_processed.json`
- `var/runs/<run_name>/json/*.csv`
- `var/runs/<run_name>/frames/combined_projected_centroids_frame_XXX.jpg`
- `var/runs/<run_name>/videos/<output_video_filename>`
- `var/cache/masked_videos/...`

## Pipeline Stages

For each group, the pipeline does this:

1. Load and normalize config.
2. Prepare run-scoped output directories.
3. Optionally preprocess videos with masks.
4. For video inputs, run YOLO tracking and emit raw tracking JSON.
5. For each camera JSON, compute centroids and projected centroids.
6. Render combined projected frames across the surviving cameras in the group.
7. Merge processed JSONs into one group-level document.
8. Export CSV files when enabled.
9. Assemble the final MP4 when enabled.

If one camera in a group fails, the group continues with surviving cameras instead of aborting the whole group.

## Data Contracts

High-level JSON flow:
- raw tracking JSON: `frames`, `frame_id`, `detections.xyxy`, `labels`
- processed JSON: adds `centroids` and `projected_centroids`
- merged JSON: group-level merged processed output

Merged identity semantics:
- `camera_nr`: source camera
- `local_track_id`: camera-local tracking identity
- `global_id`: reserved for future cross-camera identity association and currently `null`

Cowbook does not currently perform true barn-wide identity association. Merged outputs are structurally merged, but not globally re-identified across cameras.

## Observability

The pipeline now emits structured execution events rather than baking status directly into a specific interface.

Current shape:
- job lifecycle events
- stage events such as config, masking, tracking, processing, merge, export, and video
- artifact events for generated JSON, CSV, directories, and videos

This is implemented under [src/cowbook/execution](/home/davide/Desktop/cowbook/src/cowbook/execution).

Design intent:
- the pipeline publishes structured events
- the CLI consumes them as logs today
- a future FastAPI/background-job layer can attach its own observer without changing the pipeline core

## Assets

Important asset locations:
- calibration: [assets/calibration](/home/davide/Desktop/cowbook/assets/calibration)
- masks: [assets/masks](/home/davide/Desktop/cowbook/assets/masks)
- tracker config: [assets/trackers/cows_botsort.yaml](/home/davide/Desktop/cowbook/assets/trackers/cows_botsort.yaml)
- barn background: [assets/images/barn.png](/home/davide/Desktop/cowbook/assets/images/barn.png)

If `assets/images/barn.png` is missing, frame rendering falls back to a blank background.

## Example Configs

Included examples:
- [config.json](/home/davide/Desktop/cowbook/config.json): default local run config
- [configs/smoke.json](/home/davide/Desktop/cowbook/configs/smoke.json): small CPU smoke run
- [configs/smoke.gpu.json](/home/davide/Desktop/cowbook/configs/smoke.gpu.json): small GPU smoke run
- [configs/full.cpu.json](/home/davide/Desktop/cowbook/configs/full.cpu.json): full sample CPU run
- [configs/full.gpu.json](/home/davide/Desktop/cowbook/configs/full.gpu.json): full sample GPU run

## Caveats

- Calibration is specific to this barn/camera setup. Projection quality depends on matching the expected geometry and resolution.
- Frame merging uses `frame_id`; inputs must already be time-aligned.
- `num_tracking_workers > 1` can increase GPU memory pressure significantly.
- Some projection logic still relies on the legacy geometry implementation in [src/cowbook/vision/legacy_impl](/home/davide/Desktop/cowbook/src/cowbook/vision/legacy_impl).
- YOLO API behavior can drift across `ultralytics` releases.

## Development

Run the checks:

```bash
pytest
ruff check src/cowbook tests
```

The repo currently has a passing baseline regression suite around config loading, pipeline behavior, masking, merging, CSV export, and smoke flows.

## License

GPL-3.0. See [LICENSE](/home/davide/Desktop/cowbook/LICENSE).
