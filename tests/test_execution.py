from __future__ import annotations

from pathlib import Path

from cowbook.execution import InMemoryJobStore, JobReporter
from cowbook.workflows.group_processor import process_video_group


def test_in_memory_job_store_tracks_status_errors_and_artifacts():
    store = InMemoryJobStore()
    reporter = JobReporter(job_id="job-1", config_path="config.json", observer=store)

    reporter.emit("job_started", status="running", stage="config")
    reporter.emit("groups_discovered", stage="groups", payload={"count": 2})
    reporter.artifact("merged_json", "var/runs/demo/json/group_1.json", group_idx=1)
    reporter.emit(
        "group_failed",
        stage="group",
        group_idx=2,
        message="Group 2 failed.",
        payload={"error": "boom"},
    )
    reporter.emit("job_completed", status="completed", stage="pipeline")

    snapshot = store.get("job-1")

    assert snapshot is not None
    assert snapshot.status == "completed"
    assert snapshot.groups_total == 2
    assert snapshot.groups_failed == 1
    assert snapshot.error_count == 1
    assert snapshot.errors == ["boom"]
    assert snapshot.artifacts[0].kind == "merged_json"


def test_in_memory_job_store_prefers_error_detail_for_errors():
    store = InMemoryJobStore()
    reporter = JobReporter(job_id="job-2", config_path="config.json", observer=store)

    reporter.emit(
        "processing_failed",
        stage="processing",
        payload={"error_code": "processing_failed", "error_detail": "detail"},
    )

    snapshot = store.get("job-2")
    assert snapshot is not None
    assert snapshot.errors == ["detail"]


def test_group_processor_emits_structured_events_for_precomputed_json_group(
    tmp_path: Path,
    fixtures_dir: Path,
    monkeypatch,
):
    input_json = tmp_path / "input_tracking.json"
    input_json.write_text((fixtures_dir / "smoke_tracking_ch1_short.json").read_text())
    processed_json = tmp_path / "input_tracking_processed.json"
    merged_json = tmp_path / "group_1_merged_processed.json"

    def fake_process_and_save_frames(*args, **kwargs):
        processed_json.write_text('{"frames": []}')
        return [str(processed_json)]

    def fake_merge_json_files(json_paths, output_path, camera_nrs):
        merged_json.write_text('{"frames": []}')

    def fake_json_to_csv(json_path):
        csv_path = Path(json_path).with_suffix(".csv")
        csv_path.write_text("frame_id\n")
        return str(csv_path)

    monkeypatch.setattr("cowbook.workflows.group_processor.process_and_save_frames", fake_process_and_save_frames)
    monkeypatch.setattr("cowbook.workflows.group_processor.merge_json_files", fake_merge_json_files)
    monkeypatch.setattr("cowbook.workflows.group_processor._json_to_csv", fake_json_to_csv)

    store = InMemoryJobStore()
    reporter = JobReporter(job_id="job-group", config_path="config.json", observer=store)

    process_video_group(
        1,
        [{"path": str(input_json), "camera_nr": 1}],
        "models/yolo.pt",
        {
            "calibration_file": "assets/calibration/calibration_matrix.json",
            "convert_to_csv": True,
            "num_plot_workers": 0,
            "output_image_format": "jpg",
        },
        str(tmp_path),
        str(tmp_path / "frames"),
        reporter=reporter,
    )

    snapshot = store.get("job-group")

    assert snapshot is not None
    assert snapshot.groups_completed == 1
    assert [event.event_type for event in snapshot.events] == [
        "group_started",
        "artifact_created",
        "tracking_skipped",
        "processing_started",
        "artifact_created",
        "processing_completed",
        "merge_started",
        "artifact_created",
        "merge_completed",
        "artifact_created",
        "artifact_created",
        "csv_export_completed",
        "group_completed",
    ]
    assert [artifact.kind for artifact in snapshot.artifacts] == [
        "tracking_json",
        "processed_json",
        "merged_json",
        "csv",
        "csv",
    ]


def test_group_processor_emits_error_codes_for_tracking_processing_and_merge_failures(
    tmp_path: Path,
    monkeypatch,
):
    input_json = tmp_path / "input_tracking.json"
    input_json.write_text('{"frames": []}')

    def fake_process_and_save_frames(*args, **kwargs):
        raise RuntimeError("processing boom")

    def fake_merge_json_files(*args, **kwargs):
        raise RuntimeError("merge boom")

    monkeypatch.setattr(
        "cowbook.workflows.group_processor.process_and_save_frames",
        fake_process_and_save_frames,
    )
    monkeypatch.setattr("cowbook.workflows.group_processor.merge_json_files", fake_merge_json_files)

    store = InMemoryJobStore()
    reporter = JobReporter(job_id="job-errors", config_path="config.json", observer=store)

    process_video_group(
        1,
        [{"path": str(input_json), "camera_nr": 1}],
        "models/yolo.pt",
        {
            "calibration_file": "assets/calibration/calibration_matrix.json",
            "convert_to_csv": False,
            "num_plot_workers": 0,
            "output_image_format": "jpg",
        },
        str(tmp_path),
        str(tmp_path / "frames"),
        reporter=reporter,
    )

    snapshot = store.get("job-errors")
    assert snapshot is not None
    by_type = {event.event_type: event.payload for event in snapshot.events}
    assert by_type["processing_failed"]["error_code"] == "processing_failed"
