from __future__ import annotations

from pathlib import Path

import numpy as np
from tools import preview_sam3_samples as module


def test_select_equally_spaced_frame_indices_includes_both_endpoints():
    indices = module._select_equally_spaced_frame_indices(frame_count=100, sample_count=5)
    assert indices[0] == 0
    assert indices[-1] == 99
    assert indices == sorted(indices)
    assert len(indices) == len(set(indices))


def test_select_equally_spaced_frame_indices_handles_edge_cases():
    assert module._select_equally_spaced_frame_indices(frame_count=0, sample_count=5) == []
    assert module._select_equally_spaced_frame_indices(frame_count=10, sample_count=1) == [0]
    assert module._select_equally_spaced_frame_indices(frame_count=10, sample_count=0) == [0]
    # Requesting more samples than frames still dedupes to real, unique indices.
    indices = module._select_equally_spaced_frame_indices(frame_count=3, sample_count=50)
    assert indices == [0, 1, 2]


def test_resolve_sample_count_prefers_explicit_count():
    resolved = module._resolve_sample_count(frame_count=3600, fps=6.0, sample_count=12, interval_seconds=60.0)
    assert resolved == 12


def test_resolve_sample_count_derives_from_interval_seconds():
    # A 10-minute clip sampled once per minute should yield 11 markers:
    # one at t=0 and one at the top of every following minute.
    resolved = module._resolve_sample_count(frame_count=3600, fps=6.0, sample_count=0, interval_seconds=60.0)
    assert resolved == 11


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], *, snap_to_even: bool = False):
        self._frames = frames
        self._pos = 0
        self._snap_to_even = snap_to_even
        self.read_count = 0

    def isOpened(self):
        return True

    def set(self, prop, value):
        requested = int(value)
        self._pos = (requested // 2) * 2 if self._snap_to_even else requested
        return True

    def read(self):
        self.read_count += 1
        if self._pos < 0 or self._pos >= len(self._frames):
            return False, None
        frame = self._frames[self._pos]
        self._pos += 1
        return True, frame

    def get(self, prop):
        return float(self._pos)

    def release(self):
        return None


def _make_frames(count: int, size: tuple[int, int] = (24, 32)) -> list[np.ndarray]:
    return [np.full((size[0], size[1], 3), fill_value=index % 256, dtype=np.uint8) for index in range(count)]


def test_seek_frame_reports_exact_landing():
    capture = FakeCapture(_make_frames(10))
    frame, landed_index = module._seek_frame(capture, 4)
    assert landed_index == 4
    assert int(frame[0, 0, 0]) == 4


def test_seek_frame_reports_drift_when_seek_is_inexact():
    capture = FakeCapture(_make_frames(10), snap_to_even=True)
    frame, landed_index = module._seek_frame(capture, 5)
    assert landed_index == 4
    assert int(frame[0, 0, 0]) == 4


def test_sparse_sample_color_varies_by_index_not_track_or_class():
    color_a = module._sparse_sample_color(None, None, 0)
    color_b = module._sparse_sample_color(None, None, 1)
    color_c = module._sparse_sample_color(99, 7, 0)
    assert color_a != color_b
    assert color_a == color_c


def test_run_sparse_preview_for_video_samples_without_full_decode(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"placeholder")
    frame_count = 3600
    frames = _make_frames(frame_count, size=(24, 32))
    capture = FakeCapture(frames)

    class FakeArray:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float32)

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    class FakeBoxes:
        def __init__(self):
            self.xyxy = FakeArray([[1.0, 1.0, 10.0, 10.0], [12.0, 12.0, 20.0, 20.0]])
            self.id = None  # independent single-image calls carry no track identity
            self.conf = FakeArray([0.9, 0.85])
            self.cls = FakeArray([0, 0])

        def __len__(self):
            return 2

    class FakeMasks:
        def __init__(self):
            self.data = FakeArray(np.ones((2, 24, 32), dtype=np.uint8))

    class FakeResult:
        def __init__(self, frame):
            self.boxes = FakeBoxes()
            self.masks = FakeMasks()
            self.names = {0: "cow"}
            self.orig_img = frame
            self.path = str(video_path)

    call_count = {"n": 0}

    class FakePredictor:
        def __call__(self, *, source, text):
            call_count["n"] += 1
            assert text == ["cow"]
            return [FakeResult(source)]

    monkeypatch.setattr(
        module,
        "_probe_video_metadata",
        lambda _path: {"fps": 6.0, "width": 32, "height": 24, "frame_count": frame_count},
    )
    monkeypatch.setattr(module, "_open_capture", lambda _path: capture)

    result = module._run_sparse_preview_for_video(
        video_path=str(video_path),
        output_root=tmp_path / "out",
        predictor=FakePredictor(),
        model_path="sam3.pt",
        prompts=["cow"],
        sample_count=10,
        interval_seconds=0.0,
        imgsz=640,
        cleanup_config=module._default_cleanup_config(),
        log_progress=False,
    )

    assert result.sample_count == 10
    assert result.requested_sample_count == 10
    assert call_count["n"] == 10
    # The whole point of sparse sampling: touch only the requested frames,
    # not every frame of a 3600-frame video.
    assert capture.read_count == 10
    assert len(result.sample_image_paths) == 10
    for image_path in result.sample_image_paths:
        assert Path(image_path).exists()
    assert Path(result.tracking_json_path).exists()
    assert Path(result.summary_json_path).exists()
