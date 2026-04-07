from __future__ import annotations

import numpy as np

from cowbook.vision.processing import convert_arrays_to_lists, extract_data, reconstruct_json


def test_extract_data_computes_centroids_and_preserves_labels(raw_tracking_doc):
    frames = extract_data(raw_tracking_doc)

    assert [frame["frame_id"] for frame in frames] == [0, 1]
    assert frames[0]["detections"][0]["bbox"] == [10.0, 20.0, 30.0, 60.0]
    assert frames[0]["detections"][0]["centroid"] == [20.0, 40.0]
    assert frames[0]["labels"] == [
        {"class_id": 0, "id": 11},
        {"class_id": 0, "id": 12},
    ]


def test_reconstruct_json_serializes_numpy_arrays_and_projected_centroids():
    frames_data = [
        {
            "frame_id": 3,
            "detections": [
                {
                    "bbox": np.array([1.0, 2.0, 11.0, 12.0]),
                    "centroid": np.array([6.0, 7.0]),
                    "projected_centroid": np.array([100.0, 200.0, 100.0]),
                }
            ],
            "labels": [{"class_id": 5, "id": 9}],
        }
    ]

    doc = reconstruct_json(frames_data)

    assert doc == {
        "frames": [
            {
                "frame_id": 3,
                "detections": {
                    "xyxy": [[1.0, 2.0, 11.0, 12.0]],
                    "centroids": [[6.0, 7.0]],
                    "projected_centroids": [[100.0, 200.0, 100.0]],
                },
                "labels": [{"class_id": 5, "id": 9}],
            }
        ]
    }
    assert convert_arrays_to_lists(np.array([1.0, 2.0])) == [1.0, 2.0]
