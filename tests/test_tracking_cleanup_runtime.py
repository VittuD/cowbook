from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cowbook.vision import tracking_cleanup as cleanup_module
from cowbook.vision.cleanup import DetectionFrame


class _FakeArray:
    def __init__(self, array):
        self._array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def test_read_video_meta_and_open_writer_cover_cv2_failures(monkeypatch, tmp_path):
    class ClosedCapture:
        def __init__(self, _path):
            return None

        def isOpened(self):
            return False

    class ClosedWriter:
        def __init__(self, *args, **kwargs):
            return None

        def isOpened(self):
            return False

    monkeypatch.setattr(cleanup_module.cv2, "VideoCapture", ClosedCapture)
    with pytest.raises(RuntimeError, match="Could not open video"):
        cleanup_module._read_video_meta("missing.mp4")

    monkeypatch.setattr(cleanup_module.cv2, "VideoWriter", lambda *args, **kwargs: ClosedWriter())
    with pytest.raises(RuntimeError, match="Could not open VideoWriter"):
        cleanup_module._open_writer(tmp_path / "out.mp4", fps=6.0, width=32, height=24)


def test_color_for_id_and_make_ultralytics_boxes_cover_helpers():
    color = cleanup_module._color_for_id(7)
    assert all(64 <= channel <= 255 for channel in color)

    empty_boxes = cleanup_module.make_ultralytics_boxes(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int32),
        (48, 64),
    )
    non_empty_boxes = cleanup_module.make_ultralytics_boxes(
        np.asarray([[1, 2, 10, 20]], dtype=np.float32),
        np.asarray([0.9], dtype=np.float32),
        np.asarray([3], dtype=np.int32),
        (48, 64),
    )

    assert empty_boxes.data.shape == (0, 6)
    assert tuple(non_empty_boxes.orig_shape) == (48, 64)
    assert non_empty_boxes.data.shape == (1, 6)


def test_build_tracker_selects_tracker_backend(tmp_path, monkeypatch):
    tracker_yaml = tmp_path / "tracker.yaml"
    tracker_yaml.write_text("tracker_type: bytetrack\n", encoding="utf-8")

    monkeypatch.setattr(cleanup_module, "BYTETracker", lambda args, frame_rate: ("byte", args.tracker_type, frame_rate))
    monkeypatch.setattr(cleanup_module, "BOTSORT", lambda args, frame_rate: ("bot", args.tracker_type, frame_rate))

    byte_tracker = cleanup_module.build_tracker(tracker_yaml, fps=5.8)
    tracker_yaml.write_text("tracker_type: botsort\n", encoding="utf-8")
    bot_tracker = cleanup_module.build_tracker(tracker_yaml, fps=6.2)

    assert byte_tracker == ("byte", "bytetrack", 6)
    assert bot_tracker == ("bot", "botsort", 6)


def test_detect_video_to_frames_collects_empty_and_nonempty_results(monkeypatch):
    class FakeBoxes:
        xyxy = _FakeArray([[1, 2, 10, 20]])
        conf = _FakeArray([0.9])
        cls = _FakeArray([3])

        def __len__(self):
            return 1

    class FakeResult:
        def __init__(self, boxes, orig_shape=(48, 64)):
            self.boxes = boxes
            self.orig_shape = orig_shape

    class FakeModel:
        def predict(self, **kwargs):
            assert kwargs["stream"] is True
            return iter([FakeResult(FakeBoxes()), FakeResult(None)])

    progress_calls: list[tuple[int, int | None]] = []

    monkeypatch.setattr(cleanup_module, "YOLO", lambda _path, task=None: FakeModel())
    monkeypatch.setattr(cleanup_module, "_read_video_meta", lambda _path: (6.0, 64, 48, 2))

    frames = cleanup_module.detect_video_to_frames(
        "video.mp4",
        "model.pt",
        cleanup_config=type("Cfg", (), {"conf_threshold": 0.15})(),
        progress_reporter=type("Reporter", (), {"frame_progress": lambda self, cur, total: progress_calls.append((cur, total))})(),
    )

    assert progress_calls == [(1, 2), (2, 2)]
    assert frames[0].xyxy.tolist() == [[1.0, 2.0, 10.0, 20.0]]
    assert frames[0].conf.tolist() == pytest.approx([0.9])
    assert frames[0].cls.tolist() == [3]
    assert frames[1].xyxy.shape == (0, 4)


def test_track_from_detection_frames_writes_video_and_det_idx(monkeypatch, tmp_path):
    detection_frames = [
        DetectionFrame(
            frame_idx=0,
            shape=(48, 64),
            xyxy=np.asarray([[1, 2, 10, 20]], dtype=np.float32),
            conf=np.asarray([0.9], dtype=np.float32),
            cls=np.asarray([3], dtype=np.int32),
        ),
        DetectionFrame(
            frame_idx=1,
            shape=(48, 64),
            xyxy=np.asarray([[2, 3, 11, 21]], dtype=np.float32),
            conf=np.asarray([0.8], dtype=np.float32),
            cls=np.asarray([3], dtype=np.int32),
        ),
    ]

    class FakeTracker:
        def __init__(self):
            self.reset_called = False
            self.calls = 0

        def reset(self):
            self.reset_called = True

        def update(self, _boxes, img):
            assert img.shape == (48, 64, 3)
            self.calls += 1
            if self.calls == 1:
                return []
            return np.asarray([[2, 3, 11, 21, 7, 0.8, 3, 5]], dtype=np.float32)

    class FakeCapture:
        def __init__(self, _path):
            self.frames = [
                np.zeros((48, 64, 3), dtype=np.uint8),
                np.ones((48, 64, 3), dtype=np.uint8),
            ]
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def release(self):
            self.released = True

    class FakeWriter:
        def __init__(self):
            self.frames = []
            self.released = False

        def write(self, frame):
            self.frames.append(frame.copy())

        def release(self):
            self.released = True

    tracker = FakeTracker()
    writer = FakeWriter()
    progress_calls: list[tuple[int, int | None]] = []

    monkeypatch.setattr(cleanup_module, "_read_video_meta", lambda _path: (6.0, 64, 48, 2))
    monkeypatch.setattr(cleanup_module, "build_tracker", lambda _path, fps: tracker)
    monkeypatch.setattr(cleanup_module.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(cleanup_module, "_open_writer", lambda path, fps, width, height: writer)
    monkeypatch.setattr(cleanup_module, "make_ultralytics_boxes", lambda *args, **kwargs: "boxes")
    monkeypatch.setattr(cleanup_module, "_render_tracked_frame", lambda frame, *_args, **_kwargs: frame)

    document = cleanup_module.track_from_detection_frames(
        "video.mp4",
        detection_frames,
        Path("tracker.yaml"),
        save_video_path=str(tmp_path / "tracked.mp4"),
        progress_reporter=type("Reporter", (), {"frame_progress": lambda self, cur, total: progress_calls.append((cur, total))})(),
    )

    assert tracker.reset_called is True
    assert progress_calls == [(1, 2), (2, 2)]
    assert len(document.frames) == 2
    assert document.frames[0].labels == []
    assert document.frames[1].labels[0].id == 7
    assert document.frames[1].labels[0].det_idx == 5
    assert document.frames[1].labels[0].src == "tracker"
    assert len(writer.frames) == 2
    assert writer.released is True


def test_render_tracked_frame_draws_on_frame():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    rendered = cleanup_module._render_tracked_frame(
        frame.copy(),
        np.asarray([[5, 5, 20, 20]], dtype=np.float32),
        np.asarray([3], dtype=np.int64),
        np.asarray([0.75], dtype=np.float32),
    )

    assert rendered.sum() > 0
