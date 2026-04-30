from __future__ import annotations

import sys
from pathlib import Path

from cowbook.io.json_utils import load_path
from tools import project_sam3_tracking as module


def test_main_offsets_chunk_frame_ids_and_writes_outputs(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    first = input_dir / "Ch1_part_000_sam3_tracking.json"
    second = input_dir / "Ch1_part_001_sam3_tracking.json"
    first_summary = input_dir / "Ch1_part_000_sam3_export_summary.json"
    second_summary = input_dir / "Ch1_part_001_sam3_export_summary.json"
    first.write_text(
        '{"frames":[{"frame_id":0,"detections":{"xyxy":[[0,0,2,2]]},"labels":[{"class_id":0,"id":11}]}]}',
        encoding="utf-8",
    )
    second.write_text(
        '{"frames":['
        '{"frame_id":0,"detections":{"xyxy":[[1,1,3,3]]},"labels":[{"class_id":0,"id":21}]},'
        '{"frame_id":1,"detections":{"xyxy":[[2,2,4,4]]},"labels":[{"class_id":0,"id":22}]}'
        "]}",
        encoding="utf-8",
    )
    first_summary.write_text('{"width":4,"height":4}', encoding="utf-8")
    second_summary.write_text('{"width":4,"height":4}', encoding="utf-8")

    def fake_process_centroids(json_file, camera_nr, calibration_file, cancellation_token=None, show_progress=True):
        if Path(json_file).name.startswith("Ch1_part_000"):
            return [
                {
                    "frame_id": 0,
                    "detections": [{"bbox": [0.0, 0.0, 2.0, 2.0], "centroid": [1.0, 1.0], "projected_centroid": [10.0, 20.0]}],
                    "labels": [{"class_id": 0, "id": 11}],
                }
            ]
        return [
            {
                "frame_id": 0,
                "detections": [{"bbox": [1.0, 1.0, 3.0, 3.0], "centroid": [2.0, 2.0], "projected_centroid": [30.0, 40.0]}],
                "labels": [{"class_id": 0, "id": 21}],
            },
            {
                "frame_id": 1,
                "detections": [{"bbox": [2.0, 2.0, 4.0, 4.0], "centroid": [3.0, 3.0], "projected_centroid": [50.0, 60.0]}],
                "labels": [{"class_id": 0, "id": 22}],
            },
        ]

    rendered = {}

    def fake_plot_combined_projected_centroids(json_file_paths, base_filename, **kwargs):
        rendered["json_file_paths"] = json_file_paths
        rendered["base_filename"] = base_filename

    def fake_create_video_from_images(image_folder, output_video_path, fps=6, **kwargs):
        rendered["image_folder"] = image_folder
        rendered["output_video_path"] = output_video_path
        rendered["fps"] = fps

    monkeypatch.setattr(module, "process_centroids", fake_process_centroids)
    monkeypatch.setattr(module, "plot_combined_projected_centroids", fake_plot_combined_projected_centroids)
    monkeypatch.setattr(module, "create_video_from_images", fake_create_video_from_images)
    monkeypatch.setattr(
        module,
        "resolve_camera_spec",
        lambda camera_nr, calibration_file=None: type("Spec", (), {"image_size": (8, 8)})(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "project_sam3_tracking.py",
            "--tracking-json-dir",
            str(input_dir),
            "--output-root",
            str(tmp_path / "out"),
            "--fps",
            "12",
            "--no-log-progress",
        ],
    )

    assert module.main() == 0

    summary = load_path(tmp_path / "out" / "summary.json")
    assert summary["camera_nr"] == 1
    assert summary["chunks"] == [
        {
            "tracking_json_path": str(first),
            "projection_input_json_path": str(tmp_path / "out" / "json" / "Ch1_part_000_sam3_tracking_projection_input.json"),
            "processed_json_path": str(tmp_path / "out" / "json" / "Ch1_part_000_sam3_tracking_processed.json"),
            "frame_offset": 0,
            "frame_count": 1,
            "scale_info": {
                "source_frame_size": [4, 4],
                "calibration_image_size": [8, 8],
            },
        },
        {
            "tracking_json_path": str(second),
            "projection_input_json_path": str(tmp_path / "out" / "json" / "Ch1_part_001_sam3_tracking_projection_input.json"),
            "processed_json_path": str(tmp_path / "out" / "json" / "Ch1_part_001_sam3_tracking_processed.json"),
            "frame_offset": 1,
            "frame_count": 2,
            "scale_info": {
                "source_frame_size": [4, 4],
                "calibration_image_size": [8, 8],
            },
        },
    ]

    first_projection_input = load_path(tmp_path / "out" / "json" / "Ch1_part_000_sam3_tracking_projection_input.json")
    assert first_projection_input["frames"][0]["detections"]["xyxy"] == [[0.0, 0.0, 4.0, 4.0]]

    merged_tracking = load_path(tmp_path / "out" / "json" / "merged_sam3_tracking_global.json")
    assert [frame["frame_id"] for frame in merged_tracking["frames"]] == [0, 1, 2]

    merged_processed = load_path(tmp_path / "out" / "json" / "merged_sam3_tracking_global_processed.json")
    assert [frame["frame_id"] for frame in merged_processed["frames"]] == [0, 1, 2]
    assert merged_processed["frames"][2]["detections"]["projected_centroids"] == [[50.0, 60.0]]

    assert rendered["json_file_paths"] == [str(tmp_path / "out" / "json" / "merged_sam3_tracking_global_processed.json")]
    assert rendered["base_filename"] == str(tmp_path / "out" / "frames" / "combined_projected_centroids")
    assert rendered["image_folder"] == str(tmp_path / "out" / "frames")
    assert rendered["output_video_path"] == str(tmp_path / "out" / "videos" / "sam3_projection.mp4")
    assert rendered["fps"] == 12
