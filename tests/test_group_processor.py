from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from cowbook.workflows import group_processor as package_group_processor
from cowbook.workflows.group_processor import _json_to_csv, process_video_group


def test_json_to_csv_converts_single_processed_json(fixtures_dir: Path, tmp_path):
    processed_json = tmp_path / "processed.json"
    shutil.copyfile(fixtures_dir / "processed_tracking_minimal.json", processed_json)

    csv_path = _json_to_csv(str(processed_json))

    assert csv_path is not None
    assert Path(csv_path).exists()
    rows = list(csv.DictReader(Path(csv_path).open()))
    assert len(rows) == 2
    assert rows[0]["proj_x"] == "100.0"
    assert rows[0]["proj_z"] == "100.0"


def test_process_video_group_accepts_existing_json_and_generates_processed_outputs(
    fixtures_dir: Path, tmp_path
):
    input_json = tmp_path / "input_tracking.json"
    shutil.copyfile(fixtures_dir / "smoke_tracking_ch1_short.json", input_json)

    output_json_folder = tmp_path / "output_json"
    output_image_folder = tmp_path / "output_frames"
    output_json_folder.mkdir()
    output_image_folder.mkdir()
    config = {
        "model_path": "models/yolov11_best.pt",
        "calibration_file": "assets/calibration/calibration_matrix.json",
        "save_tracking_video": False,
        "convert_to_csv": True,
        "num_plot_workers": 0,
        "output_image_format": "jpg",
    }

    output_json_paths, camera_nrs, merged_json_path = process_video_group(
        group_idx=1,
        video_group=[{"path": str(input_json), "camera_nr": 1}],
        model_ref=config["model_path"],
        config=config,
        output_json_folder=str(output_json_folder),
        output_image_folder=str(output_image_folder),
    )

    processed_json = tmp_path / "input_tracking_processed.json"

    assert output_json_paths == [str(input_json)]
    assert camera_nrs == [1]
    assert processed_json.exists()
    assert Path(merged_json_path).exists()
    assert (output_json_folder / "group_1_merged_processed.csv").exists()
    assert len(list(output_image_folder.glob("combined_projected_centroids_frame_*.jpg"))) == 3

    merged_doc = json.loads(Path(merged_json_path).read_text())
    assert [frame["frame_id"] for frame in merged_doc["frames"]] == [0, 1, 2]


def test_process_video_group_keeps_surviving_cameras_aligned_when_one_tracking_fails(
    tmp_path, monkeypatch
):
    output_json_folder = tmp_path / "output_json"
    output_image_folder = tmp_path / "output_frames"
    output_json_folder.mkdir()
    output_image_folder.mkdir()

    config = {
        "model_path": "models/yolov11_best.pt",
        "calibration_file": "assets/calibration/calibration_matrix.json",
        "save_tracking_video": False,
        "convert_to_csv": False,
        "num_plot_workers": 0,
        "output_image_format": "jpg",
        "tracking_concurrency": 2,
    }
    first_json = output_json_folder / "cam1_tracking.json"
    processed_first = str(first_json).replace(".json", "_processed.json")

    class FakePool:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starmap(self, func, tasks):
            assert len(tasks) == 2
            return [
                (str(first_json), None),
                (None, "Tracking failed for cam4.mp4: boom"),
            ]

    monkeypatch.setattr(
        package_group_processor,
        "mp",
        SimpleNamespace(
            get_context=lambda _name: SimpleNamespace(Pool=lambda processes: FakePool())
        ),
    )

    def fake_process_and_save_frames(json_paths, camera_nrs, *_args, **_kwargs):
        assert json_paths == [str(first_json)]
        assert camera_nrs == [1]
        return [processed_first]

    merge_calls = {}

    def fake_merge_json_files(input_files, output_file, *, camera_nrs=None):
        merge_calls["input_files"] = input_files
        merge_calls["output_file"] = output_file
        merge_calls["camera_nrs"] = camera_nrs

    monkeypatch.setattr(package_group_processor, "process_and_save_frames", fake_process_and_save_frames)
    monkeypatch.setattr(package_group_processor, "merge_json_files", fake_merge_json_files)

    output_json_paths, camera_nrs, merged_json_path = process_video_group(
        group_idx=1,
        video_group=[
            {"path": "cam1.mp4", "camera_nr": 1},
            {"path": "cam4.mp4", "camera_nr": 4},
        ],
        model_ref=config["model_path"],
        config=config,
        output_json_folder=str(output_json_folder),
        output_image_folder=str(output_image_folder),
    )

    assert output_json_paths == [str(first_json)]
    assert camera_nrs == [1]
    assert merged_json_path.endswith("group_1_merged_processed.json")
    assert merge_calls["input_files"] == [processed_first]
    assert merge_calls["camera_nrs"] == [1]


def test_process_video_group_reports_requested_and_effective_tracking_concurrency(
    tmp_path, monkeypatch
):
    output_json_folder = tmp_path / "output_json"
    output_image_folder = tmp_path / "output_frames"
    output_json_folder.mkdir()
    output_image_folder.mkdir()

    config = {
        "model_path": "models/yolov11_best.pt",
        "calibration_file": "assets/calibration/calibration_matrix.json",
        "save_tracking_video": False,
        "convert_to_csv": False,
        "num_plot_workers": 0,
        "output_image_format": "jpg",
        "tracking_concurrency": 4,
    }

    first_json = output_json_folder / "cam1_tracking.json"
    second_json = output_json_folder / "cam4_tracking.json"

    class FakePool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starmap(self, func, tasks):
            assert self.processes == 2
            return [
                (str(first_json), None),
                (str(second_json), None),
            ]

    monkeypatch.setattr(
        package_group_processor,
        "mp",
        SimpleNamespace(
            get_context=lambda _name: SimpleNamespace(Pool=lambda processes: FakePool(processes))
        ),
    )
    monkeypatch.setattr(
        package_group_processor,
        "process_and_save_frames",
        lambda json_paths, camera_nrs, *_args, **_kwargs: [],
    )
    monkeypatch.setattr(package_group_processor, "merge_json_files", lambda *args, **kwargs: None)

    events = []

    class Reporter:
        def emit(self, event_type, **kwargs):
            events.append((event_type, kwargs))

        def artifact(self, *_args, **_kwargs):
            return None

    process_video_group(
        group_idx=1,
        video_group=[
            {"path": "cam1.mp4", "camera_nr": 1},
            {"path": "cam4.mp4", "camera_nr": 4},
        ],
        model_ref=config["model_path"],
        config=config,
        output_json_folder=str(output_json_folder),
        output_image_folder=str(output_image_folder),
        reporter=Reporter(),
    )

    tracking_started = next(kwargs for event_type, kwargs in events if event_type == "tracking_started")
    assert tracking_started["payload"]["requested_tracking_concurrency"] == 4
    assert tracking_started["payload"]["effective_tracking_concurrency"] == 2
