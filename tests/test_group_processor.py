from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from group_processor import _json_to_csv, process_video_group


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
        "calibration_file": "legacy/calibration_matrix.json",
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
