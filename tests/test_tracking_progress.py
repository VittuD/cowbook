from __future__ import annotations

import json
from types import SimpleNamespace

from cowbook.execution import InMemoryJobStore, JobReporter, StageProgressReporter
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


def test_stage_progress_reporter_uses_event_sink_and_unknown_total_milestones(capsys):
    events = []
    reporter = StageProgressReporter(
        event_prefix="processing",
        reporter_stage="processing",
        stage_name="custom",
        path_value="target",
        event_sink=lambda event_type, payload: events.append((event_type, payload.copy())),
        log_progress=True,
    )

    reporter.stage_started()
    reporter.step_progress(1)
    reporter.step_progress(5)
    reporter.step_progress(11)
    reporter.stage_completed()

    assert [event_type for event_type, _ in events] == [
        "processing_stage_started",
        "processing_stage_progress",
        "processing_stage_progress",
        "processing_stage_completed",
    ]
    assert events[1][1]["current"] == 1
    assert events[2][1]["current"] == 11
    assert "[processing] custom: target started" in capsys.readouterr().out


def test_stage_progress_reporter_emits_to_job_reporter_with_camera_and_fraction():
    store = InMemoryJobStore()
    job_reporter = JobReporter(job_id="job-stage", config_path="config.json", observer=store)
    reporter = StageProgressReporter(
        event_prefix="masking",
        reporter_stage="masking",
        stage_name="mask_videos",
        path_value="masked",
        camera_nr=4,
        total=20,
        reporter=job_reporter,
        group_idx=2,
        extra_payload={"kind": "demo"},
    )

    reporter.stage_started()
    reporter.step_progress(1)
    reporter.step_progress(20)
    reporter.stage_completed()

    snapshot = store.get("job-stage")
    assert snapshot is not None
    progress_payloads = [event.payload for event in snapshot.events if event.event_type == "masking_stage_progress"]
    assert progress_payloads[0]["camera_nr"] == 4
    assert progress_payloads[0]["kind"] == "demo"
    assert progress_payloads[-1]["progress_fraction"] == 1.0
