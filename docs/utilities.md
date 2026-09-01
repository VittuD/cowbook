# Utilities

The top-level `scripts/` and `tools/` directories contain repository utilities. They are optional helpers, not part of the `cowbook` package runtime surface.

`tools/` contains benchmark harnesses and operational experiments. `scripts/` contains smaller one-off helpers such as video regrouping.

## `group_videos.sh`

`scripts/group_videos.sh` reorganizes flat camera files named like:

```text
Ch1_<group>.mp4
Ch4_<group>.mp4
```

into grouped directories like:

```text
videos/<group>/Ch1.mp4
videos/<group>/Ch4.mp4
```

This is useful when a raw video drop arrives as one flat folder and needs to be rearranged into a grouped layout before writing configs.

Dry-run example:

```bash
scripts/group_videos.sh --src raw_drop --dest videos --dry-run
```

Move files:

```bash
scripts/group_videos.sh --src raw_drop --dest videos
```

Copy files instead:

```bash
scripts/group_videos.sh --src raw_drop --dest videos --copy
```

Overwrite existing targets only when `--overwrite` is passed. Without it, existing destination files are preserved.

## `undistort_videos.py`

`tools/undistort_videos.py` rectifies every matching video in a folder using a selected camera calibration:

```bash
python -m tools.undistort_videos \
  --input-dir sample_data/videos \
  --camera-nr 1 \
  --output-dir var/undistorted
```

The tool writes rectified MP4 files and a `summary.json`. It reads the actual input video resolution and scales the camera intrinsics to that output resolution before remapping. This is correct for videos that are resized versions of the calibrated image; it does not account for crop or padding offsets.

## `project_sam3_tracking.py`

`tools/project_sam3_tracking.py` converts exported SAM3 tracking detections into Cowbook projected outputs. New SAM3 export JSON declares `source_image_size` directly. For older exports, the projector can recover the source size from the paired export summary before passing data through the same resolution-aware processing path used by the normal pipeline.

## `run_sam3_windowed.py`

`tools/run_sam3_windowed.py` runs SAM3 semantic video tracking in fixed-duration windows instead of one pass over the whole video:

```bash
python -m tools.run_sam3_windowed \
  --videos sample_data/videos/Ch1_60.mp4 \
  --prompts cow \
  --model-path sam3.pt \
  --window-seconds 600 \
  --output-root var/windowed/sam3_semantic_tracking
```

Each window is written as a transient chunk -- seeked directly from the source video, deleted immediately after that window's SAM3 pass finishes -- so a multi-hour video never grows disk usage beyond one window's worth. This also resets SAM3's video-tracker memory on a fixed cadence, which matters because that memory bank's GPU footprint grows with tracked-object count over a video's duration; an unbroken pass over a long recording risks running out of memory well before the end. Track IDs do not carry across a window boundary. Internally it re-uses `tools.benchmark_sam3_semantic_tracking`'s per-video processing unchanged, so every window gets the same cleanup, mask handling, and output layout as a normal run.

## `run_sam3_multi_gpu.py`

`tools/run_sam3_multi_gpu.py` partitions a folder's videos across the available GPUs and, by default, dispatches `tools.run_sam3_windowed` on each one:

```bash
python -m tools.run_sam3_multi_gpu \
  --input-dir /scratch/vet/vanzetti18032026_07_13 \
  --channels Ch1 Ch4 Ch6 Ch8 \
  --launch
```

GPU count auto-detects via `nvidia-smi` unless `--num-gpus` is passed explicitly, so the same command adapts to however many GPUs are actually available. Videos are bin-packed with a longest-processing-time-first heuristic (SAM3 video tracking can't be split within a single video, so the only axis that parallelizes is across videos, one full model instance per GPU); a video's total frame count is used as its weight, since windowing's per-window reload cost is small relative to actual inference time. Without `--launch`, it only prints the assignment plan and a rough per-GPU time estimate -- pass `--plan-path` to also save that plan as JSON. Pass `--no-windowed` to dispatch `tools.benchmark_sam3_semantic_tracking` (single pass per video) instead.
