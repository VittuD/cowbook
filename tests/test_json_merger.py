from __future__ import annotations

import json

from json_merger import merge_json_files


def test_merge_json_files_reassigns_ids_and_falls_back_to_computed_centroids(tmp_path):
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

    merge_json_files([str(first), str(second)], str(merged))
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
        {"class_id": 7, "id": 1},
        {"class_id": 8, "id": 2},
    ]


def test_merge_single_processed_json_is_identity_like_except_for_label_ids(fixtures_dir, tmp_path):
    source = tmp_path / "processed.json"
    merged = tmp_path / "merged.json"
    source.write_text((fixtures_dir / "processed_tracking_minimal.json").read_text())

    merge_json_files([str(source)], str(merged))
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
        {"class_id": 0, "id": 1},
        {"class_id": 0, "id": 2},
    ]
