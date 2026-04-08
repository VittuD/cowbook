from __future__ import annotations

from queue import Queue

import pytest

from cowbook.execution import InMemoryJobStore, JobCancelledError, JobReporter
from cowbook.workflows import group_processor as group_processor_module


def test_tracking_worker_passes_cleanup_and_progress_sink(monkeypatch, tmp_path):
    recorded = {}

    def fake_track(video_path, output_json, model_ref, **kwargs):
        recorded["args"] = (video_path, output_json, model_ref)
        recorded["kwargs"] = kwargs
        kwargs["progress_event_sink"]("tracking_stage_progress", {"frame_current": 1})

    monkeypatch.setattr(group_processor_module, "track_video_with_yolo", fake_track)

    progress_queue = Queue()
    output_json, err = group_processor_module._tracking_worker(
        "video.mp4",
        str(tmp_path / "tracking.json"),
        "model.pt",
        save=True,
        tracking_cleanup={"enabled": True},
        camera_nr=4,
        log_progress=True,
        group_idx=2,
        progress_queue=progress_queue,
    )

    assert err is None
    assert output_json is not None
    assert recorded["kwargs"]["tracking_cleanup"] == {"enabled": True}
    assert recorded["kwargs"]["camera_nr"] == 4
    assert progress_queue.get_nowait()["event_type"] == "tracking_stage_progress"


def test_drain_tracking_progress_queue_emits_events():
    progress_queue = Queue()
    progress_queue.put({"event_type": "tracking_stage_progress", "group_idx": 2, "payload": {"frame_current": 3}})

    store = InMemoryJobStore()
    reporter = JobReporter(job_id="job-progress-queue", config_path="config.json", observer=store)

    group_processor_module._drain_tracking_progress_queue(progress_queue, reporter)

    snapshot = store.get("job-progress-queue")
    assert snapshot is not None
    assert snapshot.events[0].event_type == "tracking_stage_progress"
    assert snapshot.events[0].group_idx == 2


def test_process_video_group_raises_when_tracking_produces_no_jsons(monkeypatch, tmp_path):
    class FakePool:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starmap(self, fn, items):
            return [(None, "tracking boom") for _ in items]

    class FakeContext:
        def Pool(self, processes):
            return FakePool()

    monkeypatch.setattr(group_processor_module.mp, "get_context", lambda _method: FakeContext())

    with pytest.raises(RuntimeError, match="No JSONs produced"):
        group_processor_module.process_video_group(
            1,
            [{"path": "video.mp4", "camera_nr": 1}],
            "model.pt",
            {"tracking_concurrency": 1, "calibration_file": "calibration.json"},
            str(tmp_path),
            str(tmp_path / "frames"),
        )


def test_process_video_group_emits_merge_failed(monkeypatch, tmp_path):
    input_json = tmp_path / "input_tracking.json"
    input_json.write_text('{"frames": []}', encoding="utf-8")
    processed_json = tmp_path / "input_tracking_processed.json"
    processed_json.write_text('{"frames": []}', encoding="utf-8")

    monkeypatch.setattr(group_processor_module, "process_and_save_frames", lambda *args, **kwargs: [str(processed_json)])
    monkeypatch.setattr(group_processor_module, "merge_json_files", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("merge boom")))
    monkeypatch.setattr(group_processor_module, "_json_to_csv", lambda _path: None)

    store = InMemoryJobStore()
    reporter = JobReporter(job_id="job-merge-fail", config_path="config.json", observer=store)

    group_processor_module.process_video_group(
        1,
        [{"path": str(input_json), "camera_nr": 1}],
        "model.pt",
        {
            "calibration_file": "calibration.json",
            "convert_to_csv": True,
            "num_plot_workers": 0,
            "output_image_format": "jpg",
        },
        str(tmp_path),
        str(tmp_path / "frames"),
        reporter=reporter,
    )

    snapshot = store.get("job-merge-fail")
    assert snapshot is not None
    merge_failed = [event for event in snapshot.events if event.event_type == "merge_failed"][0]
    assert merge_failed.payload["error_code"] == "merge_failed"


def test_process_video_group_propagates_cancellation_during_export(monkeypatch, tmp_path):
    input_json = tmp_path / "input_tracking.json"
    input_json.write_text('{"frames": []}', encoding="utf-8")
    processed_json = tmp_path / "input_tracking_processed.json"
    processed_json.write_text('{"frames": []}', encoding="utf-8")
    merged_json = tmp_path / "group_1_merged_processed.json"
    merged_json.write_text('{"frames": []}', encoding="utf-8")

    monkeypatch.setattr(group_processor_module, "process_and_save_frames", lambda *args, **kwargs: [str(processed_json)])
    monkeypatch.setattr(group_processor_module, "merge_json_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(group_processor_module.os.path, "exists", lambda path: True)
    monkeypatch.setattr(group_processor_module, "_json_to_csv", lambda _path: "out.csv")

    class FakeCancellationToken:
        def __init__(self):
            self.calls = 0
            self.cancelled = False

        def raise_if_cancelled(self):
            self.calls += 1
            if self.calls >= 4:
                self.cancelled = True
            if self.cancelled:
                raise JobCancelledError("cancelled")

    cancellation_token = FakeCancellationToken()

    with pytest.raises(JobCancelledError):
        group_processor_module.process_video_group(
            1,
            [{"path": str(input_json), "camera_nr": 1}],
            "model.pt",
            {
                "calibration_file": "calibration.json",
                "convert_to_csv": True,
                "num_plot_workers": 0,
                "output_image_format": "jpg",
            },
            str(tmp_path),
            str(tmp_path / "frames"),
            cancellation_token=cancellation_token,
        )
