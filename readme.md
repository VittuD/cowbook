# Cowbook

Cowbook is a packaged Python pipeline for multi-camera cow tracking and [ground-plane projection](docs/calibration.md) in a fixed barn setup.

It runs a batch [pipeline](docs/pipeline.md): YOLO-based tracking on one or more camera streams, centroid extraction, camera-space to barn-space projection, [group-level merge](docs/pipeline.md), top-down rendering, and export to JSON, CSV, and MP4 artifacts.

Supported entrypoint:

```bash
python -m cowbook
```

For non-CLI embedding, `cowbook` also exposes a small stable Python [runtime surface](docs/package-boundaries.md):

```python
from cowbook import (
    PipelineRunner,
    RunRequest,
    load_pipeline_config,
    load_pipeline_config_object,
    materialize_pipeline_config,
    run_pipeline,
    run_pipeline_request,
)
```

## Documentation

The package documentation is organized as a MkDocs site under [docs](docs).

Install docs tooling only:

```bash
pip install -e ".[docs]"
```

Serve docs locally:

```bash
mkdocs serve
```

Build the static docs site:

```bash
mkdocs build
```

Stable documentation targets are the CLI entrypoint, the public package [runtime surface](docs/package-boundaries.md) in [runtime.py](src/cowbook/runtime.py), and the architecture and package-boundary guidance for the package itself.

Deep internal modules are documented only when they become stable extension points.

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

Directory intent is simple: `assets/` stores persistent non-code project assets such as calibration, masks, tracker config, and the barn background; `configs/` stores example run configs; `sample_data/` stores local inputs for smoke and full runs; `src/cowbook/` contains the packaged code; and `var/` is reserved for runtime outputs and cache data.

The [`scripts/`](scripts) directory contains optional repository utilities. [`group_videos.sh`](scripts/group_videos.sh) is a helper for reorganizing flat raw camera drops into grouped `videos/<group>/ChX.mp4` directories before config creation.

## Install

Runtime install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Development checks only:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Full contributor install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
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
python -m cowbook --config configs/full.cpu.json --log-progress
python -m cowbook --config configs/full.cpu.json --no-clean-frames-after-video
```

Supported overrides are intentionally small: frame rate, output filename, output image format, plot workers, tracking concurrency, progress logging, projection-video creation, frame cleanup, and whether video masking runs before inference.

## Python Embedding

`cowbook` is usable in two ways: as a standalone CLI/engine, and as a Python package imported directly.

The stable import surface is the package root or [runtime.py](src/cowbook/runtime.py), not deep internal modules.

Example:

```python
from cowbook import (
    RunRequest,
    load_pipeline_config,
    load_pipeline_config_object,
    materialize_pipeline_config,
    run_pipeline,
    run_pipeline_request,
)

config = load_pipeline_config("configs/smoke.json")
result = run_pipeline("configs/smoke.json")
tracking_jsons = result.tracking_json_paths

config_object = load_pipeline_config_object(
    {
        "model_path": "models/best.pt",
        "video_groups": [[{"path": "sample_data/videos/Ch1_60.mp4", "camera_nr": 1}]],
    }
)

request = RunRequest(config=config_object, overrides={"run_name": "demo"})
result = run_pipeline_request(request)
materialized = materialize_pipeline_config(config_object, "var/tmp/demo.json")
```

Request-based runtime entrypoints return a normalized `RunResult`, which wraps the underlying `JobRun` plus summarized artifact paths.

Config loading is strict: file-based loading raises `FileNotFoundError` for missing files, `json.JSONDecodeError` for invalid JSON, and both file-based and in-memory config normalization raise `ValueError` for invalid runtime values. In-memory config loading does not mutate the caller-provided object.

For GPU-oriented runs, `tracking_concurrency=1` is the intended baseline on smaller cards. Cowbook keeps the same execution events and output contract in that mode, but bypasses multiprocessing and reuses one YOLO model instance per `(model_path, tracking mode)` inside a group to avoid unnecessary startup cost.

Package-facing exports are `PipelineRunner`, `PipelineConfig`, `RunRequest`, `RunResult`, `JobRun`, `JobEvent`, `JobArtifact`, `CancellationToken`, `JobCancelledError`, `load_pipeline_config()`, `load_pipeline_config_object()`, `materialize_pipeline_config()`, `run_pipeline()`, and `run_pipeline_request()`.

## Docker

Docker images included:

- `docker/Dockerfile`: runtime based on the official Ultralytics image
- `docker/Dockerfile.a40-cleanup`: cleanup-focused GPU benchmark/runtime image
- `docker/Dockerfile.backend-bench`: backend A/B benchmark image for `.pt` vs exported `onnx` / `engine` artifacts
- `docker/Dockerfile.tensorrt-bench`: TensorRT concurrency sweep image for `.pt` vs `.engine` at tracking concurrency `1 2 3 4`

The image:

- uses `ultralytics/ultralytics:8.4.34` as the base
- copies the repo into `/app`
- installs the package from `pyproject.toml` in editable mode
- includes the current `assets/`, `configs/`, `models/`, and `sample_data/`
- defaults to `python -m cowbook --config config.json`

Build the image:

```bash
docker build -f docker/Dockerfile -t cowbook .
```

Build the cleanup benchmark image:

```bash
docker build -f docker/Dockerfile.a40-cleanup -t cowbook-a40-cleanup .
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

To override configs or assets from the host instead of using the copies baked into the image, mount them into `/app`.

The cleanup benchmark image runs the optional `tracking_cleanup` path on prepared long videos, saves tracking JSON and annotated tracking videos, renders projected barn frames, and assembles a combined projection video. Its current defaults target `/scratch/vet/var/...` and enable `--log-progress`.

The backend benchmark image runs `tools.benchmark_tracking_backends` against the four sample videos, supports both sequential shared-model runs and `process_parallel_models` runs such as `--process-workers 2`, exports `onnx` and `engine` candidates from the baseline `.pt` model when the environment supports that, and writes a JSON summary under `var/benchmarks/`. The same tool can also benchmark prebuilt artifacts through `--onnx-artifact-path` and `--engine-artifact-path`.

The TensorRT concurrency image runs `tools.benchmark_tensorrt_concurrency`, exports or reuses one TensorRT engine, then benchmarks both `.pt` and `.engine` across the requested tracking concurrencies. Concurrency `1` uses the single-model sequential path; higher values use `process_parallel_models` with the matching worker count. Its defaults now follow the same folder layout as the cleanup image under `/scratch/vet/var/...`.

Build the TensorRT concurrency image:

```bash
docker build -f docker/Dockerfile.tensorrt-bench -t cowbook-tensorrt-bench .
```

Run the default `1 2 3 4` sweep on a GPU host:

```bash
docker run --rm -it \
  --gpus all \
  -v /scratch/vet:/scratch/vet \
  cowbook-tensorrt-bench
```

Run the same image on a remote A40 box with an explicit output path:

```bash
docker run --rm -it \
  --gpus all \
  -v /scratch/vet:/scratch/vet \
  cowbook-tensorrt-bench \
  --concurrency-values 1 2 3 4 \
  --output /scratch/vet/var/benchmarks/tensorrt_a40_1_4.json
```

## Config Model

The runtime contract is a JSON config plus a small CLI override surface.

Core fields are `model_path`, `calibration_file`, and `video_groups`, because they define the detector, the camera geometry, and the actual inputs. Runtime layout, [masking](docs/cli.md), CSV export, and cleanup settings control execution and retention behavior.

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

Notes:

- input paths may be videos or precomputed tracking JSON files
- `tracking_concurrency` defaults to `1` intentionally to avoid GPU contention
- when effective tracking concurrency is `1`, tracking runs inline instead of through a worker process
- the inline single-worker path may reuse one YOLO model instance per `(model_path, tracking mode)` within a group
- direct tracking and cleanup tracking do not share a reused model instance, so tracker state does not bleed across modes
- `log_progress` enables human-readable milestone logs for long tracking stages
- masks default to `assets/masks/*.png`
- masked-video cache defaults to `var/cache/masked_videos`
- optional tracking cleanup lives under `tracking_cleanup`

Optional tracking cleanup can preprocess detections before tracking, preserve detection lineage, prune short-lived tracks with a second tracking pass, and smooth final output boxes. It is off by default.

Example cleanup block:

```json
"tracking_cleanup": {
  "enabled": true,
  "conf_threshold": 0.15,
  "nms_mode": "hybrid_nms",
  "two_pass_prune_short_tracks": true,
  "min_track_length": 30,
  "postprocess_smoothing": true
}
```

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
- processed JSON: adds [`centroids`](docs/pipeline.md) and [`projected_centroids`](docs/pipeline.md)
- merged JSON: group-level merged processed output

Merged identity semantics:

- `camera_nr`: source camera
- `local_track_id`: camera-local tracking identity
- `global_id`: reserved for future cross-camera identity association and currently `null`

Cowbook does not currently perform true barn-wide identity association. Merged outputs are structurally merged, but not globally re-identified across cameras.

## Observability

The pipeline now emits [structured execution events](docs/job-execution.md) rather than baking status directly into a specific interface.

Current shape:

- job lifecycle events
- stage events such as config, masking, tracking, processing, merge, export, and video
- tracking-internal stage events such as `direct`, `detect`, `cleanup_pass1`, `cleanup_pass2`, `preprocess`, `prune`, and `postprocess`
- artifact events for generated JSON, CSV, directories, and videos

This is implemented under [src/cowbook/execution](src/cowbook/execution).

Design intent:

- the pipeline publishes structured events
- `--log-progress` adds human-readable milestone logs for long tracking stages
- another caller can attach its own observer without changing the pipeline core

## Assets

Important asset locations:

- calibration: [assets/calibration](assets/calibration)
- masks: [assets/masks](assets/masks)
- tracker config: [assets/trackers/cows_botsort.yaml](assets/trackers/cows_botsort.yaml)
- barn background: [assets/images/barn.png](assets/images/barn.png)

If `assets/images/barn.png` is missing, frame rendering falls back to a blank background.

## Example Configs

Included examples:

- [config.json](config.json): default local run config
- [configs/smoke.json](configs/smoke.json): small CPU smoke run
- [configs/smoke.gpu.json](configs/smoke.gpu.json): small GPU smoke run
- [configs/full.cpu.json](configs/full.cpu.json): full sample CPU run
- [configs/full.gpu.json](configs/full.gpu.json): full sample GPU run

## Caveats

- Calibration is specific to this barn/camera setup. Projection quality depends on matching the expected geometry and resolution.
- Frame merging uses `frame_id`; inputs must already be time-aligned.
- `tracking_concurrency > 1` can increase GPU memory pressure significantly.
- Camera calibration and ground-plane projection now live in [src/cowbook/vision/calibration.py](src/cowbook/vision/calibration.py), with fixed correspondences stored in [assets/calibration/camera_correspondences.json](assets/calibration/camera_correspondences.json).
- YOLO API behavior can drift across `ultralytics` releases.

## Development

Run the checks:

```bash
pytest
ruff check src/cowbook tests
```

The repo currently has a passing baseline regression suite around config loading, pipeline behavior, masking, merging, CSV export, and smoke flows.

## License

GPL-3.0. See [LICENSE](LICENSE).
