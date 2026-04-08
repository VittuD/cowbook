from __future__ import annotations

from concurrent.futures import Future

import numpy as np

from cowbook.execution import InMemoryJobStore, JobReporter
from cowbook.vision.preprocess_video import (
    _derive_masked_path,
    _ensure_mask_size,
    _infer_channel_from_name,
    preprocess_videos,
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


def test_preprocess_videos_emits_masking_progress(monkeypatch, tmp_path):
    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            future = Future()
            future.set_result(fn(*args, **kwargs))
            return future

    def fake_process_one_video(src, dst, mask_path, strict_half_rule):
        dst_path = tmp_path / dst.split("/")[-1]
        dst_path.write_text("ok")
        return (src, str(dst_path), True)

    monkeypatch.setattr("cowbook.vision.preprocess_video._process_one_video", fake_process_one_video)
    monkeypatch.setattr("cowbook.vision.preprocess_video.futures.ProcessPoolExecutor", FakeExecutor)

    store = InMemoryJobStore()
    reporter = JobReporter(job_id="job-mask", config_path="config.json", observer=store)

    preprocess_videos(
        {
            "masked_video_folder": str(tmp_path / "masked"),
            "video_groups": [
                [{"path": "videos/a.mp4", "camera_nr": 1}],
                [{"path": "videos/b.mp4", "camera_nr": 4}],
            ],
            "num_mask_workers": 2,
        },
        reporter=reporter,
    )

    snapshot = store.get("job-mask")
    assert snapshot is not None
    event_types = [event.event_type for event in snapshot.events]
    assert event_types == [
        "masking_stage_started",
        "masking_stage_progress",
        "masking_stage_progress",
        "masking_stage_completed",
    ]
