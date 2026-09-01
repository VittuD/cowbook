from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tools import benchmark_sam3_semantic_tracking as tracking_module
from tools import run_sam3_windowed as module


def test_compute_window_bounds_splits_into_fixed_size_windows_with_short_tail():
    bounds = module._compute_window_bounds(frame_count=25, fps=5.0, window_seconds=2.0)

    assert [(b.start_frame, b.end_frame) for b in bounds] == [(0, 10), (10, 20), (20, 25)]
    assert [b.window_index for b in bounds] == [0, 1, 2]


def test_compute_window_bounds_handles_edge_cases():
    assert module._compute_window_bounds(frame_count=0, fps=5.0, window_seconds=2.0) == []
    assert module._compute_window_bounds(frame_count=10, fps=0.0, window_seconds=2.0) == []
    assert module._compute_window_bounds(frame_count=10, fps=5.0, window_seconds=0.0) == []
    # A window shorter than one frame still advances by at least one frame.
    bounds = module._compute_window_bounds(frame_count=3, fps=5.0, window_seconds=0.001)
    assert [b.end_frame - b.start_frame for b in bounds] == [1, 1, 1]


def _write_real_video(path: Path, frame_count: int, fps: float, size: tuple[int, int] = (32, 24)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for index in range(frame_count):
        frame = np.full((size[1], size[0], 3), fill_value=index % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_write_window_chunk_writes_exactly_the_requested_frame_range(tmp_path: Path):
    source_path = tmp_path / "source.mp4"
    _write_real_video(source_path, frame_count=20, fps=5.0)

    bounds = module.WindowBounds(window_index=0, start_frame=5, end_frame=12)
    output_path = tmp_path / "chunk.mp4"
    written = module._write_window_chunk(
        str(source_path), bounds, output_path, fps=5.0, frame_size=(32, 24)
    )

    assert written == 7
    metadata = tracking_module._probe_video_metadata(str(output_path))
    assert metadata["frame_count"] == 7


def test_write_window_chunk_reports_fewer_frames_when_source_runs_out_early(tmp_path: Path):
    source_path = tmp_path / "source.mp4"
    _write_real_video(source_path, frame_count=10, fps=5.0)

    # Request a range that extends past the end of the source.
    bounds = module.WindowBounds(window_index=0, start_frame=6, end_frame=20)
    output_path = tmp_path / "chunk.mp4"
    written = module._write_window_chunk(
        str(source_path), bounds, output_path, fps=5.0, frame_size=(32, 24)
    )

    assert written == 4


def test_run_windowed_semantic_tracking_processes_each_window_and_cleans_up_chunks(
    monkeypatch, tmp_path: Path
):
    source_path = tmp_path / "source.mp4"
    frame_count = 25
    fps = 5.0
    _write_real_video(source_path, frame_count=frame_count, fps=fps)

    class FakeArray:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float32)

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self._values, dtype=np.float32)

    class FakeBoxes:
        def __init__(self):
            self.xyxy = FakeArray([[0.0, 0.0, 5.0, 5.0]])
            self.id = FakeTensor([1])
            self.conf = FakeTensor([0.9])
            self.cls = FakeTensor([0])

        def __len__(self):
            return 1

    class FakeMasks:
        def __init__(self):
            self.data = FakeArray(np.ones((1, 24, 32), dtype=np.uint8))

    class FakeResult:
        def __init__(self, path):
            self.boxes = FakeBoxes()
            self.masks = FakeMasks()
            self.names = {0: "cow"}
            self.orig_img = np.zeros((24, 32, 3), dtype=np.uint8)
            self.path = path

        def plot(self, **_kwargs):
            return np.zeros((24, 32, 3), dtype=np.uint8)

    predictor_calls: list[str] = []

    class FakePredictor:
        def __init__(self, overrides):
            self.overrides = overrides

        def __call__(self, *, source, text, stream):
            predictor_calls.append(source)
            metadata = tracking_module._probe_video_metadata(source)
            return iter([FakeResult(source) for _ in range(int(metadata["frame_count"]))])

    monkeypatch.setattr(tracking_module, "SAM3VideoSemanticPredictor", FakePredictor)

    chunk_tmp_dir = tmp_path / "chunks"
    results = module._run_windowed_semantic_tracking_for_video(
        video_path=str(source_path),
        output_root=tmp_path / "out",
        chunk_tmp_dir=chunk_tmp_dir,
        window_seconds=2.0,  # -> 10-frame windows at 5fps: [0,10) [10,20) [20,25)
        prompts=["cow"],
        model_path="sam3.pt",
        conf_threshold=0.25,
        imgsz=512,
        device=None,
        half=True,
        render_mode="none",
        max_render_frames=0,
        dump_frame_metadata=False,
        log_progress=False,
        log_every_frames=25,
    )

    assert len(results) == 3
    assert [result.frame_count for result in results] == [10, 10, 5]
    assert len(predictor_calls) == 3
    # Chunks are transient: nothing should be left behind once processing
    # finishes, regardless of how many windows a video was split into.
    assert list(chunk_tmp_dir.glob("*.mp4")) == []
    for result in results:
        assert Path(result.summary_json_path).exists()
        assert Path(result.tracking_json_path).exists()


def test_run_windowed_semantic_tracking_composes_target_fps_with_windowing(monkeypatch, tmp_path: Path):
    source_path = tmp_path / "source.mp4"
    frame_count = 25
    fps = 5.0
    _write_real_video(source_path, frame_count=frame_count, fps=fps)

    class FakeArray:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float32)

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self._values, dtype=np.float32)

    class FakeBoxes:
        def __init__(self):
            self.xyxy = FakeArray(np.zeros((0, 4), dtype=np.float32))
            self.id = FakeTensor([])
            self.conf = FakeTensor([])
            self.cls = FakeTensor([])

        def __len__(self):
            return 0

    class FakeMasks:
        def __init__(self):
            self.data = FakeArray(np.zeros((0, 24, 32), dtype=np.uint8))

    class FakeResult:
        def __init__(self, path):
            self.boxes = FakeBoxes()
            self.masks = FakeMasks()
            self.names = {0: "cow"}
            self.orig_img = np.zeros((24, 32, 3), dtype=np.uint8)
            self.path = path

        def plot(self, **_kwargs):
            return np.zeros((24, 32, 3), dtype=np.uint8)

    predictor_calls: list[str] = []

    class FakePredictor:
        def __init__(self, overrides):
            self.overrides = overrides

        def __call__(self, *, source, text, stream):
            predictor_calls.append(source)
            metadata = tracking_module._probe_video_metadata(source)
            return iter([FakeResult(source) for _ in range(int(metadata["frame_count"]))])

    monkeypatch.setattr(tracking_module, "SAM3VideoSemanticPredictor", FakePredictor)

    chunk_tmp_dir = tmp_path / "chunks"
    results = module._run_windowed_semantic_tracking_for_video(
        video_path=str(source_path),
        output_root=tmp_path / "out",
        chunk_tmp_dir=chunk_tmp_dir,
        window_seconds=2.0,  # -> 10-frame windows at 5fps: [0,10) [10,20) [20,25)
        prompts=["cow"],
        model_path="sam3.pt",
        conf_threshold=0.25,
        imgsz=512,
        device=None,
        half=True,
        render_mode="none",
        max_render_frames=0,
        dump_frame_metadata=False,
        log_progress=False,
        log_every_frames=25,
        target_fps=1.0,  # 5fps windows decimated to ~1fps -> stride 5
    )

    # Each window is thinned independently: 10, 10, and 5 source frames at
    # stride 5 become 2, 2, and 1 processed frames respectively.
    assert [result.frame_count for result in results] == [2, 2, 1]
    for result in results:
        assert result.target_fps == 1.0
        assert result.frame_stride == 5
    # Neither the window chunks nor the decimated copies survive.
    assert list(chunk_tmp_dir.glob("*.mp4")) == []
