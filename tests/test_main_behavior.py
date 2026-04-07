from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

import group_processor
from main import main


class _FakePool:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def starmap(self, func, tasks):
        return [func(*task) for task in tasks]


class _FakeContext:
    def Pool(self, processes):
        return _FakePool()


def _write_tiny_video(path: Path, frames: int = 3) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6, (64, 48))
    for i in range(frames):
        frame = np.full((48, 64, 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _build_json_smoke_config(tmp_path: Path, input_json: Path, clean_frames: bool) -> Path:
    config = {
        "model_path": "models/yolov11_best.pt",
        "calibration_file": "legacy/calibration_matrix.json",
        "mask_videos": False,
        "output_image_folder": str(tmp_path / "frames"),
        "output_video_folder": str(tmp_path / "videos"),
        "output_json_folder": str(tmp_path / "json"),
        "output_video_filename": "smoke_projection.mp4",
        "output_image_format": "jpg",
        "save_tracking_video": False,
        "create_projection_video": True,
        "clean_frames_after_video": clean_frames,
        "convert_to_csv": True,
        "fps": 6,
        "num_plot_workers": 0,
        "num_tracking_workers": 1,
        "video_groups": [[{"path": str(input_json), "camera_nr": 1}]],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_main_invalid_config_returns_without_creating_output_dirs(tmp_path):
    config_path = tmp_path / "bad_config.json"
    config_path.write_text(
        json.dumps(
            {
                "output_image_folder": str(tmp_path / "frames"),
                "output_video_folder": str(tmp_path / "videos"),
                "output_json_folder": str(tmp_path / "json"),
                "video_groups": [[
                    {"path": "videos/a.mp4", "camera_nr": 1},
                    {"path": "videos/b.mp4", "camera_nr": 1},
                ]],
            }
        )
    )

    main(str(config_path))

    assert not (tmp_path / "frames").exists()
    assert not (tmp_path / "videos").exists()
    assert not (tmp_path / "json").exists()


def test_main_clean_frames_after_video_removes_rendered_frames(fixtures_dir: Path, tmp_path):
    input_json = tmp_path / "input_tracking.json"
    shutil.copyfile(fixtures_dir / "smoke_tracking_ch1_short.json", input_json)
    config_path = _build_json_smoke_config(tmp_path, input_json, clean_frames=True)

    main(str(config_path))

    assert (tmp_path / "videos" / "smoke_projection.mp4").exists()
    assert list((tmp_path / "frames").glob("*.jpg")) == []


def test_main_video_input_pipeline_works_with_stubbed_tracking(fixtures_dir: Path, tmp_path, monkeypatch):
    source_fixture = fixtures_dir / "smoke_tracking_ch1_short.json"
    input_video = tmp_path / "clip.mp4"
    _write_tiny_video(input_video)

    def fake_track_video_with_yolo(video_path, output_json_path, model_path, save=False):
        shutil.copyfile(source_fixture, output_json_path)

    monkeypatch.setattr(group_processor, "track_video_with_yolo", fake_track_video_with_yolo)
    monkeypatch.setattr(group_processor.mp, "get_context", lambda _: _FakeContext())

    config = {
        "model_path": "models/yolov11_best.pt",
        "calibration_file": "legacy/calibration_matrix.json",
        "mask_videos": False,
        "output_image_folder": str(tmp_path / "frames"),
        "output_video_folder": str(tmp_path / "videos"),
        "output_json_folder": str(tmp_path / "json"),
        "output_video_filename": "smoke_projection.mp4",
        "output_image_format": "jpg",
        "save_tracking_video": False,
        "create_projection_video": True,
        "clean_frames_after_video": False,
        "convert_to_csv": True,
        "fps": 6,
        "num_plot_workers": 0,
        "num_tracking_workers": 1,
        "video_groups": [[{"path": str(input_video), "camera_nr": 1}]],
    }
    config_path = tmp_path / "video_config.json"
    config_path.write_text(json.dumps(config))

    main(str(config_path))

    assert (tmp_path / "json" / "clip_tracking_processed.json").exists()
    assert (tmp_path / "json" / "group_1_merged_processed.json").exists()
    assert (tmp_path / "videos" / "smoke_projection.mp4").exists()
    assert len(list((tmp_path / "frames").glob("combined_projected_centroids_frame_*.jpg"))) == 3


def test_main_rerun_is_structurally_deterministic_for_json_input(fixtures_dir: Path, tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()

    input_a = run_a / "input_tracking.json"
    input_b = run_b / "input_tracking.json"
    shutil.copyfile(fixtures_dir / "smoke_tracking_ch1_short.json", input_a)
    shutil.copyfile(fixtures_dir / "smoke_tracking_ch1_short.json", input_b)

    config_a = _build_json_smoke_config(run_a, input_a, clean_frames=False)
    config_b = _build_json_smoke_config(run_b, input_b, clean_frames=False)

    main(str(config_a))
    main(str(config_b))

    merged_a = json.loads((run_a / "json" / "group_1_merged_processed.json").read_text())
    merged_b = json.loads((run_b / "json" / "group_1_merged_processed.json").read_text())

    assert merged_a == merged_b
    assert sorted(p.name for p in (run_a / "frames").glob("*.jpg")) == sorted(
        p.name for p in (run_b / "frames").glob("*.jpg")
    )
