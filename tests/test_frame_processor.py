from __future__ import annotations

import json
import sys
from concurrent.futures import Future

from cowbook.execution import InMemoryJobStore, JobReporter
from cowbook.vision import frame_processor as frame_processor_module
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


def test_render_frame_worker_reuses_cached_barn_image(monkeypatch, tmp_path):
    calls = []
    frame_processor_module._BARN_IMG = None
    monkeypatch.setattr(frame_processor_module.os.path, "exists", lambda _path: True)
    monkeypatch.setitem(sys.modules, "cv2", type("FakeCv2", (), {"imread": staticmethod(lambda _path: "barn-image")})())
    monkeypatch.setattr(
        frame_processor_module,
        "render_projection_frame",
        lambda projected_centroids, frame_id, frame_output_path, barn_image_path, barn_image: calls.append(
            (frame_id, frame_output_path, barn_image)
        ),
    )

    first = frame_processor_module._render_frame_worker((1, [[1.0, 2.0, 3.0]], str(tmp_path / "a.jpg"), "barn.png"))
    second = frame_processor_module._render_frame_worker((2, [[4.0, 5.0, 6.0]], str(tmp_path / "b.jpg"), "barn.png"))

    assert first.endswith("a.jpg")
    assert second.endswith("b.jpg")
    assert calls[0][2] == "barn-image"
    assert calls[1][2] == "barn-image"


def test_process_centroids_worker_saves_processed_json(monkeypatch, tmp_path):
    source = tmp_path / "input.json"
    source.write_text("{}", encoding="utf-8")
    saved = {}
    monkeypatch.setattr(
        frame_processor_module,
        "process_centroids",
        lambda *args, **kwargs: [{"frame_id": 0, "detections": []}],
    )
    monkeypatch.setattr(
        frame_processor_module,
        "save_frame_data_json",
        lambda frames_data, output_json_path: saved.setdefault("call", (frames_data, output_json_path)),
    )

    output_path, camera_nr = frame_processor_module._process_centroids_worker((str(source), 4, "calibration.json"))

    assert camera_nr == 4
    assert output_path.endswith("_processed.json")
    assert saved["call"][1] == output_path


def test_process_and_save_frames_parallel_branch_preserves_input_order(tmp_path, monkeypatch):
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

    input_a = tmp_path / "a.json"
    input_b = tmp_path / "b.json"
    input_a.write_text("{}", encoding="utf-8")
    input_b.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "frames"
    output_dir.mkdir()

    monkeypatch.setattr(frame_processor_module.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(frame_processor_module._fut, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(frame_processor_module, "_process_centroids_worker", lambda args: (args[0].replace(".json", "_processed.json"), args[1]))
    monkeypatch.setattr(frame_processor_module, "plot_combined_projected_centroids", lambda *args, **kwargs: None)

    processed_paths = frame_processor_module.process_and_save_frames(
        [str(input_a), str(input_b)],
        [1, 4],
        str(output_dir),
        "calibration.json",
    )

    assert processed_paths == [
        str(input_a).replace(".json", "_processed.json"),
        str(input_b).replace(".json", "_processed.json"),
    ]


def test_plot_combined_projected_centroids_parallel_branch_emits_progress(tmp_path, monkeypatch):
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

    written = []
    monkeypatch.setattr(frame_processor_module, "extract_projected_centroids_from_files", lambda _paths: {0: [[1, 2, 3]], 1: [[4, 5, 6]]})
    monkeypatch.setattr(frame_processor_module, "default_barn_image_path", lambda: "barn.png")
    monkeypatch.setattr(frame_processor_module._fut, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(frame_processor_module, "_render_frame_worker", lambda item: written.append(item[2]) or item[2])

    store = InMemoryJobStore()
    reporter = JobReporter(job_id="job-render-parallel", config_path="config.json", observer=store)

    frame_processor_module.plot_combined_projected_centroids(
        ["a.json"],
        str(tmp_path / "combined"),
        num_workers=2,
        image_format="jpg",
        reporter=reporter,
    )

    snapshot = store.get("job-render-parallel")
    assert snapshot is not None
    assert any(event.payload.get("stage_name") == "render_frames" for event in snapshot.events)
    assert len(written) == 2


def test_process_centroids_projects_centroids(monkeypatch, tmp_path):
    input_json = tmp_path / "tracking.json"
    input_json.write_text('{"frames": []}', encoding="utf-8")

    monkeypatch.setattr(frame_processor_module, "load_camera_setup", lambda *_args, **_kwargs: "camera-model")
    monkeypatch.setattr(
        frame_processor_module,
        "load_projection_context",
        lambda *_args, **_kwargs: type(
            "ProjectionContext",
            (),
            {"project_points_to_ground": lambda self, centroids: [[c[0], c[1], 100.0] for c in centroids]},
        )(),
    )
    monkeypatch.setattr(frame_processor_module, "parse_json", lambda _path: {"frames": []})
    monkeypatch.setattr(
        frame_processor_module,
        "extract_data",
        lambda _json: [{"detections": [{"centroid": [1.0, 2.0]}]}],
    )
    monkeypatch.setattr(frame_processor_module, "process_detections", lambda frame, _camera_model: frame)

    frames_data = frame_processor_module.process_centroids(
        str(input_json),
        camera_nr=1,
        calibration_file="calibration.json",
        show_progress=False,
    )

    assert frames_data[0]["detections"][0]["projected_centroid"] == [1.0, 2.0, 100.0]
