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
