# Docker

Docker images included:

- `docker/Dockerfile`: runtime based on the official Ultralytics image
- `docker/Dockerfile.a40-bench`: direct tracking benchmark/runtime image
- `docker/Dockerfile.a40-cleanup`: cleanup-focused GPU benchmark/runtime image
- `docker/Dockerfile.backend-bench`: backend A/B benchmark image for `.pt` vs exported `onnx` / `engine` artifacts
- `docker/Dockerfile.deva-base`: heavy cached DEVA + Grounded-Segment-Anything dependency image
- `docker/Dockerfile.deva-bench`: DEVA text-prompted video segmentation benchmark image
- `docker/Dockerfile.sam3-bench`: SAM3 all-instance semantic video tracking benchmark image
- `docker/Dockerfile.sam3-chunk-run`: persistent multi-GPU SAM3 batch runner for a full camera-channel folder
- `docker/Dockerfile.tensorrt-bench`: runtime TensorRT concurrency sweep image for `.pt` vs `.engine` at tracking concurrency `1 2 3 4`

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

The backend benchmark image runs `tools.benchmark_tracking_backends` against the four sample videos, supports both sequential shared-model runs and `process_parallel_models` runs such as `--process-workers 2`, exports `onnx` and `engine` candidates from the baseline `.pt` model when the environment supports that, and writes a JSON summary under `var/benchmarks/`. Use this image for backend export comparison, not as the primary source of truth for runtime concurrency decisions. The same tool can also benchmark prebuilt artifacts through `--onnx-artifact-path` and `--engine-artifact-path`.

The TensorRT concurrency image runs `tools.benchmark_runtime_tracking_concurrency`, exports or reuses one TensorRT engine, then benchmarks Cowbook's real `group_processor` tracking path across the requested tracking concurrencies. Concurrency `1` uses the runtime inline path; higher values use the runtime multiprocessing path. This is the benchmark to use when deciding how runtime concurrency should behave on a target machine. Its defaults follow the same folder layout as the cleanup image under `/scratch/vet/var/...`.

The SAM3 benchmark image runs `tools.benchmark_sam3_semantic_tracking` for text-only all-instance concept tracking, writes per-video JSON summaries, and produces annotated overlay videos for visual inspection. Its defaults target `/scratch/vet/var/benchmarks/sam3_semantic_tracking_300s`. Per the Ultralytics SAM3 docs, the `sam3.pt` weights are not auto-downloaded and must be provided explicitly in the image or working directory. The image also preinstalls the extra SAM3 runtime dependencies that Ultralytics otherwise attempts to install dynamically at runtime, including the Ultralytics `CLIP` package and `timm`.

The SAM3 chunk-run image runs `tools.run_sam3_multi_gpu --launch` against a whole camera-channel folder, unlike the dev container (`docker/Dockerfile.sam3-dev`) it's meant to replace for real batch runs, which doesn't survive the remote session closing. It defaults to `--input-dir /scratch/vet/vanzetti18032026_07_13 --channels Ch1 Ch4 Ch6 Ch8`, dispatches `tools.run_sam3_windowed` (fixed-duration windows, bounded GPU memory, transient per-window chunks -- see [Utilities](utilities.md)) on one worker per detected GPU, and writes outputs under `/scratch/vet/var/batches/sam3_ch1468`. GPU count auto-detects via `nvidia-smi` inside the container, so the same image adapts to however many GPUs `--gpus` exposes at `docker run` time (4, 8, or otherwise) without an image rebuild; override with `--num-gpus` if a fixed count is preferred. Mount `/scratch/vet` from the host so both the source videos and the batch output survive the container exiting.

The DEVA benchmark path is split into two images:

- `docker/Dockerfile.deva-base`: the heavy cached layer with upstream DEVA, the `hkchengrex/Grounded-Segment-Anything` fork, Python dependencies, and downloaded checkpoints
- `docker/Dockerfile.deva-bench`: the thin Cowbook wrapper image that installs the local package and benchmark tool on top of that base

This split keeps normal Cowbook changes from invalidating the expensive DEVA/GSA install and checkpoint-download layers, so rebuilds and registry pushes stay much smaller after the base image is established. The DEVA benchmark image runs `tools.benchmark_deva_text_tracking` as a thin wrapper around upstream `Tracking-Anything-with-DEVA` text mode and defaults to `/scratch/vet/var/benchmarks/deva_text_tracking_300s`.

Build the TensorRT concurrency image:

```bash
docker build -f docker/Dockerfile.tensorrt-bench -t cowbook-tensorrt-bench .
```

Build the DEVA base image:

```bash
docker build -f docker/Dockerfile.deva-base -t cowbook-deva-base .
```

Build the DEVA benchmark image on top of that base:

```bash
docker build -f docker/Dockerfile.deva-bench -t cowbook-deva-bench .
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
