from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cowbook.app.pipeline import PipelineRunner


def _run_pipeline(config_path: Path | str) -> None:
    PipelineRunner().run(str(config_path))


@pytest.mark.smoke
def test_json_input_smoke_pipeline_generates_outputs(fixtures_dir: Path, tmp_path):
    source_fixture = fixtures_dir / "smoke_tracking_ch1_short.json"
    input_json = tmp_path / "input_tracking.json"
    shutil.copyfile(source_fixture, input_json)

    config_path = tmp_path / "config.json"
    output_frames = tmp_path / "frames"
    output_videos = tmp_path / "videos"
    output_json = tmp_path / "json"
    config = {
        "model_path": "models/yolov11_best.pt",
        "calibration_file": "legacy/calibration_matrix.json",
        "mask_videos": False,
        "output_image_folder": str(output_frames),
        "output_video_folder": str(output_videos),
        "output_json_folder": str(output_json),
        "output_video_filename": "smoke_projection.mp4",
        "output_image_format": "jpg",
        "save_tracking_video": False,
        "create_projection_video": True,
        "clean_frames_after_video": False,
        "convert_to_csv": True,
        "fps": 6,
        "num_plot_workers": 0,
        "num_tracking_workers": 1,
        "video_groups": [
            [
                {
                    "path": str(input_json),
                    "camera_nr": 1,
                }
            ]
        ],
    }
    config_path.write_text(json.dumps(config))

    _run_pipeline(config_path)

    processed_json = tmp_path / "input_tracking_processed.json"
    merged_json = output_json / "group_1_merged_processed.json"
    merged_csv = output_json / "group_1_merged_processed.csv"
    video = output_videos / "smoke_projection.mp4"
    rendered_frames = sorted(output_frames.glob("combined_projected_centroids_frame_*.jpg"))

    assert processed_json.exists()
    assert merged_json.exists()
    assert merged_csv.exists()
    assert video.exists()
    assert len(rendered_frames) == 3

    merged_doc = json.loads(merged_json.read_text())
    assert [frame["frame_id"] for frame in merged_doc["frames"]] == [0, 1, 2]
    assert "projected_centroids" in merged_doc["frames"][0]["detections"]
