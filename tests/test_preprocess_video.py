from __future__ import annotations

import numpy as np
from preprocess_video import (
    _derive_masked_path,
    _ensure_mask_size,
    _infer_channel_from_name,
)


def test_infer_channel_from_name_uses_filename_and_dataset_fallback():
    assert _infer_channel_from_name("/tmp/Ch1_clip.mp4") == "Ch1"
    assert _infer_channel_from_name("/tmp/clip_Ch6_segment.mp4") == "Ch6"
    assert _infer_channel_from_name("/tmp/frame_001_anything_jpg") == "Ch8"
    assert _infer_channel_from_name("/tmp/no_channel_here.mp4") is None


def test_ensure_mask_size_resizes_only_exact_half():
    mask = np.ones((8, 10), dtype=np.uint8)

    resized, state = _ensure_mask_size(mask, (5, 4))
    assert state == "half"
    assert resized.shape == (4, 5)

    unchanged, mismatch_state = _ensure_mask_size(mask, (7, 4))
    assert mismatch_state == "mismatch"
    assert unchanged.shape == (8, 10)


def test_derive_masked_path_keeps_name_and_adds_source_hash(tmp_path):
    first = _derive_masked_path(str(tmp_path), "/a/b/clip.mp4")
    second = _derive_masked_path(str(tmp_path), "/x/y/clip.mp4")

    assert first.endswith(".mp4")
    assert second.endswith(".mp4")
    assert first != second
