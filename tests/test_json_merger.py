from __future__ import annotations

import json

from cowbook.io.json_merger import merge_json_files


def test_merge_json_files_preserves_camera_local_identity_fields(tmp_path):
    first = tmp_path / "first_processed.json"
    second = tmp_path / "second_rawish.json"
    merged = tmp_path / "merged.json"

    first.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": {
                            "xyxy": [[0.0, 0.0, 10.0, 20.0]],
                            "centroids": [[5.0, 10.0]],
                            "projected_centroids": [[100.0, 200.0, 100.0]],
                        },
                        "labels": [{"class_id": 7, "id": 99}],
                    }
                ]
            }
        )
    )
    second.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": {"xyxy": [[10.0, 10.0, 20.0, 30.0]]},
                        "labels": [{"class_id": 8, "id": 42}],
                    }
                ]
            }
        )
    )

    merge_json_files([str(first), str(second)], str(merged), camera_nrs=[1, 4])
    doc = json.loads(merged.read_text())
    frame = doc["frames"][0]

    assert frame["frame_id"] == 0
    assert frame["detections"]["xyxy"] == [
        [0.0, 0.0, 10.0, 20.0],
        [10.0, 10.0, 20.0, 30.0],
    ]
    assert frame["detections"]["centroids"] == [
        [5.0, 10.0],
        [15.0, 20.0],
    ]
    assert frame["detections"]["projected_centroids"] == [[100.0, 200.0, 100.0]]
    assert frame["labels"] == [
        {"class_id": 7, "camera_nr": 1, "local_track_id": 99, "global_id": None},
        {"class_id": 8, "camera_nr": 4, "local_track_id": 42, "global_id": None},
    ]


def test_merge_single_processed_json_keeps_local_track_ids(fixtures_dir, tmp_path):
    source = tmp_path / "processed.json"
    merged = tmp_path / "merged.json"
    source.write_text((fixtures_dir / "processed_tracking_minimal.json").read_text())

    merge_json_files([str(source)], str(merged), camera_nrs=[1])
    doc = json.loads(merged.read_text())
    frame = doc["frames"][0]

    assert frame["frame_id"] == 0
    assert frame["detections"]["xyxy"] == [
        [10.0, 20.0, 30.0, 60.0],
        [40.0, 50.0, 70.0, 90.0],
    ]
    assert frame["detections"]["centroids"] == [
        [20.0, 40.0],
        [55.0, 70.0],
    ]
    assert frame["detections"]["projected_centroids"] == [
        [100.0, 200.0, 100.0],
        [110.0, 210.0, 100.0],
    ]
    assert frame["labels"] == [
        {"class_id": 0, "camera_nr": 1, "local_track_id": 11, "global_id": None},
        {"class_id": 0, "camera_nr": 1, "local_track_id": 12, "global_id": None},
    ]
