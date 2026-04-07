from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

from cowbook.vision.preprocess_video import preprocess_videos


def _write_tiny_video(path: Path, frames: int = 2) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6, (32, 24))
    for i in range(frames):
        frame = np.full((24, 32, 3), 30 + i, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_preprocess_videos_rewrites_paths_and_reuses_up_to_date_outputs(tmp_path):
    src_video = tmp_path / "Ch1_clip.mp4"
    mask_path = tmp_path / "mask.png"
    _write_tiny_video(src_video)
    cv2.imwrite(str(mask_path), np.full((24, 32), 255, dtype=np.uint8))

    config = {
        "masked_video_folder": str(tmp_path / "masked"),
        "video_groups": [[{"path": str(src_video), "camera_nr": 1}]],
        "masks": {"Ch1": str(mask_path)},
        "num_mask_workers": 1,
        "mask_strict_half_rule": True,
    }

    first_groups = preprocess_videos(config)
    first_masked_path = Path(first_groups[0][0]["path"])
    first_mtime = first_masked_path.stat().st_mtime

    second_groups = preprocess_videos(config)
    second_masked_path = Path(second_groups[0][0]["path"])
    second_mtime = second_masked_path.stat().st_mtime

    assert first_masked_path.exists()
    assert first_masked_path != src_video
    assert second_masked_path == first_masked_path
    assert second_mtime == first_mtime


def test_preprocess_videos_rebuilds_when_mask_changes(tmp_path):
    src_video = tmp_path / "Ch1_clip.mp4"
    mask_path = tmp_path / "mask.png"
    _write_tiny_video(src_video)
    cv2.imwrite(str(mask_path), np.full((24, 32), 255, dtype=np.uint8))

    config = {
        "masked_video_folder": str(tmp_path / "masked"),
        "video_groups": [[{"path": str(src_video), "camera_nr": 1}]],
        "masks": {"Ch1": str(mask_path)},
        "num_mask_workers": 1,
        "mask_strict_half_rule": True,
    }

    first_groups = preprocess_videos(config)
    first_masked_path = Path(first_groups[0][0]["path"])
    first_mtime_ns = first_masked_path.stat().st_mtime_ns

    time.sleep(0.01)
    cv2.imwrite(str(mask_path), np.full((24, 32), 0, dtype=np.uint8))
    os.utime(mask_path, ns=(first_mtime_ns + 2_000_000_000, first_mtime_ns + 2_000_000_000))

    second_groups = preprocess_videos(config)
    second_masked_path = Path(second_groups[0][0]["path"])

    assert second_masked_path == first_masked_path
    assert second_masked_path.stat().st_mtime_ns > first_mtime_ns


def test_preprocess_videos_rebuilds_when_mask_mode_changes(tmp_path):
    src_video = tmp_path / "Ch1_clip.mp4"
    mask_path = tmp_path / "mask.png"
    _write_tiny_video(src_video)
    cv2.imwrite(str(mask_path), np.full((24, 32), 255, dtype=np.uint8))

    config = {
        "masked_video_folder": str(tmp_path / "masked"),
        "video_groups": [[{"path": str(src_video), "camera_nr": 1}]],
        "masks": {"Ch1": str(mask_path)},
        "num_mask_workers": 1,
        "mask_strict_half_rule": True,
    }

    first_groups = preprocess_videos(config)
    first_masked_path = Path(first_groups[0][0]["path"])
    first_mtime_ns = first_masked_path.stat().st_mtime_ns

    time.sleep(0.01)
    second_groups = preprocess_videos({**config, "mask_strict_half_rule": False})
    second_masked_path = Path(second_groups[0][0]["path"])

    assert second_masked_path == first_masked_path
    assert second_masked_path.stat().st_mtime_ns > first_mtime_ns
