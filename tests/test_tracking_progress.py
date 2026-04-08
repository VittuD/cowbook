from __future__ import annotations

import json
from types import SimpleNamespace

from cowbook.vision import tracking as tracking_module


def test_direct_tracking_emits_shared_progress_events(tmp_path, monkeypatch):
    output_json = tmp_path / "tracking.json"
    events: list[tuple[str, dict]] = []

    class FakeCap:
        def get(self, _prop):
            return 2

        def release(self):
            return None

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class FakeBox:
        def __init__(self, coords, class_id, track_id):
            self.xyxy = [coords]
            self.cls = FakeTensor(class_id)
            self.id = FakeTensor(track_id)

    class FakeModel:
        def track(self, **_kwargs):
            yield SimpleNamespace(boxes=[FakeBox([1.0, 2.0, 3.0, 4.0], 0, 11)])
            yield SimpleNamespace(boxes=[FakeBox([5.0, 6.0, 7.0, 8.0], 0, 11)])

    monkeypatch.setattr(tracking_module, "load_yolo_model", lambda _path: FakeModel())
    monkeypatch.setattr(
        tracking_module.cv2,
        "VideoCapture",
        lambda _path: FakeCap(),
    )

    def sink(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))

    tracking_module.track_video_with_yolo(
        "video.mp4",
        str(output_json),
        "model.pt",
        progress_event_sink=sink,
        camera_nr=1,
    )

    assert [event_type for event_type, _payload in events] == [
        "tracking_stage_started",
        "tracking_stage_progress",
        "tracking_stage_progress",
        "tracking_stage_completed",
    ]
    assert events[1][1]["stage_name"] == "direct"
    assert events[1][1]["tracking_mode"] == "direct"
    assert events[1][1]["camera_nr"] == 1
    assert events[1][1]["frame_current"] == 1
    assert events[2][1]["frame_current"] == 2
    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(saved["frames"]) == 2


def test_cleanup_tracking_emits_shared_stage_events(tmp_path, monkeypatch):
    output_json = tmp_path / "tracking.json"
    events: list[tuple[str, dict]] = []

    fake_frames = [SimpleNamespace(frame_idx=0), SimpleNamespace(frame_idx=1)]
    fake_doc = tracking_module.TrackingDocument(frames=[])

    def fake_detect(*_args, progress_reporter=None, **_kwargs):
        progress_reporter.frame_progress(1, 2)
        progress_reporter.frame_progress(2, 2)
        return fake_frames

    def fake_track(*_args, progress_reporter=None, **_kwargs):
        progress_reporter.frame_progress(1, 2)
        progress_reporter.frame_progress(2, 2)
        return fake_doc

    monkeypatch.setattr(tracking_module, "detect_video_to_frames", fake_detect)
    monkeypatch.setattr(
        tracking_module,
        "preprocess_detection_frames",
        lambda frames, *_args, **_kwargs: frames,
    )
    monkeypatch.setattr(tracking_module, "track_from_detection_frames", fake_track)
    monkeypatch.setattr(tracking_module, "compute_short_track_ids", lambda *_args, **_kwargs: {1})
    monkeypatch.setattr(
        tracking_module,
        "prune_detection_frames_by_track_ids",
        lambda frames, *_args, **_kwargs: frames,
    )
    monkeypatch.setattr(
        tracking_module,
        "postprocess_tracking_document",
        lambda doc, *_args, **_kwargs: doc,
    )

    def sink(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))

    tracking_module.track_video_with_yolo(
        "video.mp4",
        str(output_json),
        "model.pt",
        tracking_cleanup={
            "enabled": True,
            "two_pass_prune_short_tracks": True,
            "postprocess_smoothing": True,
        },
        progress_event_sink=sink,
        camera_nr=4,
    )

    stage_names = [payload["stage_name"] for _event_type, payload in events]
    assert "detect" in stage_names
    assert "preprocess" in stage_names
    assert "cleanup_pass1" in stage_names
    assert "prune" in stage_names
    assert "cleanup_pass2" in stage_names
    assert "postprocess" in stage_names
    progress_events = [
        payload for event_type, payload in events if event_type == "tracking_stage_progress"
    ]
    assert any(payload["stage_name"] == "detect" for payload in progress_events)
    assert any(payload["stage_name"] == "cleanup_pass1" for payload in progress_events)
    assert any(payload["stage_name"] == "cleanup_pass2" for payload in progress_events)
