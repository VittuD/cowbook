from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from video_processor import create_video_from_images, extract_frame_number


def test_extract_frame_number_returns_minus_one_when_missing():
    assert extract_frame_number("combined_projected_centroids_frame_002.jpg") == 2
    assert extract_frame_number("no_frame_number_here.jpg") == -1


def test_create_video_from_images_builds_mp4_from_sorted_images(tmp_path):
    image_dir = tmp_path / "frames"
    image_dir.mkdir()

    for frame_no in [2, 0, 1]:
        img = np.full((60, 80, 3), frame_no * 40, dtype=np.uint8)
        cv2.imwrite(str(image_dir / f"combined_frame_{frame_no}_frame_{frame_no:03d}.jpg"), img)

    output_video = tmp_path / "assembled.mp4"
    create_video_from_images(str(image_dir), str(output_video), fps=6)

    assert output_video.exists()
    assert output_video.stat().st_size > 0


def test_create_video_from_images_raises_for_empty_folder(tmp_path):
    image_dir = tmp_path / "empty"
    image_dir.mkdir()

    with pytest.raises(ValueError, match="No images found"):
        create_video_from_images(str(image_dir), str(tmp_path / "out.mp4"), fps=6)
