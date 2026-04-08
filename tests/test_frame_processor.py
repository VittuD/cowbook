from __future__ import annotations

import json

from cowbook.execution import InMemoryJobStore, JobReporter
from cowbook.vision.frame_processor import process_and_save_frames
from cowbook.vision.processing import extract_projected_centroids_from_files


def test_extract_projected_centroids_from_files_merges_multiple_sources_by_frame(tmp_path):
    first = tmp_path / "first_processed.json"
    second = tmp_path / "second_processed.json"
    first.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": {"projected_centroids": [[1.0, 2.0, 100.0]]},
                    },
                    {
                        "frame_id": 1,
                        "detections": {"projected_centroids": [[3.0, 4.0, 100.0]]},
                    },
                ]
            }
        )
    )
    second.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_id": 1,
                        "detections": {"projected_centroids": [[5.0, 6.0, 100.0]]},
                    }
                ]
            }
        )
    )

    result = extract_projected_centroids_from_files([str(first), str(second)])

    assert result == {
        0: [[1.0, 2.0, 100.0]],
        1: [[3.0, 4.0, 100.0], [5.0, 6.0, 100.0]],
    }


def test_process_and_save_frames_emits_processing_progress(tmp_path, monkeypatch):
    input_a = tmp_path / "a.json"
    input_b = tmp_path / "b.json"
    input_a.write_text("{}")
    input_b.write_text("{}")
    output_dir = tmp_path / "frames"
    output_dir.mkdir()

    monkeypatch.setattr("cowbook.vision.frame_processor.os.cpu_count", lambda: 1)
    monkeypatch.setattr(
        "cowbook.vision.frame_processor.process_centroids",
        lambda *args, **kwargs: [{"frame_id": 0, "detections": [{"projected_centroid": [1.0, 2.0, 100.0]}]}],
    )
    monkeypatch.setattr(
        "cowbook.vision.frame_processor.save_frame_data_json",
        lambda _frames_data, output_json_path: open(output_json_path, "w", encoding="utf-8").write('{"frames": []}'),
    )
    monkeypatch.setattr(
        "cowbook.vision.frame_processor.extract_projected_centroids_from_files",
        lambda _paths: {0: [[1.0, 2.0, 100.0]], 1: [[3.0, 4.0, 100.0]]},
    )
    monkeypatch.setattr("cowbook.vision.frame_processor.default_barn_image_path", lambda: "barn.png")
    monkeypatch.setattr("cowbook.vision.frame_processor.load_barn_image", lambda _path: object())
    monkeypatch.setattr("cowbook.vision.frame_processor.render_projection_frame", lambda *args, **kwargs: None)

    store = InMemoryJobStore()
    reporter = JobReporter(job_id="job-processing", config_path="config.json", observer=store)

    processed_paths = process_and_save_frames(
        [str(input_a), str(input_b)],
        [1, 4],
        str(output_dir),
        "assets/calibration/camera_system.json",
        num_plot_workers=0,
        output_image_format="jpg",
        reporter=reporter,
        group_idx=1,
    )

    assert len(processed_paths) == 2
    snapshot = store.get("job-processing")
    assert snapshot is not None
    stage_names = [event.payload.get("stage_name") for event in snapshot.events]
    assert "process_centroids" in stage_names
    assert "render_frames" in stage_names
    assert any(event.event_type == "processing_stage_progress" for event in snapshot.events)
