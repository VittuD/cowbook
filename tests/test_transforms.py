from __future__ import annotations

import numpy as np

from cowbook.transforms import (
    aggregate_projected_centroids,
    bbox_wh_area,
    centroid_from_xyxy,
    convert_arrays_to_lists,
    extract_frames_data,
    iter_csv_rows,
    merge_tracking_documents,
    reconstruct_tracking_document,
)


def test_extract_frames_data_is_pure_projection_free_transform(raw_tracking_doc):
    frames = extract_frames_data(raw_tracking_doc)

    assert frames == [
        {
            "frame_id": 0,
            "detections": [
                {"bbox": [10.0, 20.0, 30.0, 60.0], "centroid": [20.0, 40.0]},
                {"bbox": [40.0, 50.0, 70.0, 90.0], "centroid": [55.0, 70.0]},
            ],
            "labels": [{"class_id": 0, "id": 11}, {"class_id": 0, "id": 12}],
        },
        {
            "frame_id": 1,
            "detections": [
                {"bbox": [12.0, 18.0, 34.0, 58.0], "centroid": [23.0, 38.0]},
            ],
            "labels": [{"class_id": 1, "id": 21}],
        },
    ]


def test_reconstruct_tracking_document_preserves_current_wire_shape():
    frames_data = [
        {
            "frame_id": 5,
            "detections": [
                {
                    "bbox": np.array([1.0, 2.0, 3.0, 4.0]),
                    "centroid": np.array([2.0, 3.0]),
                    "projected_centroid": np.array([10.0, 11.0, 100.0]),
                }
            ],
            "labels": [{"class_id": 7, "id": 8}],
        }
    ]

    doc = reconstruct_tracking_document(frames_data)

    assert doc == {
        "frames": [
            {
                "frame_id": 5,
                "detections": {
                    "xyxy": [[1.0, 2.0, 3.0, 4.0]],
                    "centroids": [[2.0, 3.0]],
                    "projected_centroids": [[10.0, 11.0, 100.0]],
                },
                "labels": [{"class_id": 7, "id": 8}],
            }
        ]
    }


def test_merge_tracking_documents_is_pure_and_matches_current_merge_policy(processed_tracking_doc):
    merged = merge_tracking_documents([processed_tracking_doc, processed_tracking_doc])
    frame = merged["frames"][0]

    assert len(frame["detections"]["xyxy"]) == 4
    assert frame["labels"] == [
        {"class_id": 0, "id": 1},
        {"class_id": 0, "id": 2},
        {"class_id": 0, "id": 3},
        {"class_id": 0, "id": 4},
    ]


def test_aggregate_projected_centroids_combines_documents_by_frame():
    result = aggregate_projected_centroids(
        [
            {"frames": [{"frame_id": 0, "detections": {"projected_centroids": [[1, 2, 3]]}}]},
            {"frames": [{"frame_id": 0, "detections": {"projected_centroids": [[4, 5, 6]]}}]},
        ]
    )

    assert result == {0: [[1, 2, 3], [4, 5, 6]]}


def test_iter_csv_rows_is_pure_and_computes_fallback_fields(raw_tracking_doc):
    rows = list(iter_csv_rows(raw_tracking_doc, source_tag="fixture"))

    assert rows[0]["source"] == "fixture"
    assert rows[0]["centroid_x"] == 20.0
    assert rows[0]["proj_x"] is None
    assert rows[0]["w"] == 20.0
    assert rows[0]["area"] == 800.0


def test_basic_geometry_helpers_are_pure():
    assert centroid_from_xyxy([0.0, 0.0, 10.0, 20.0]) == (5.0, 10.0)
    assert bbox_wh_area([0.0, 0.0, 10.0, 20.0]) == (10.0, 20.0, 200.0)
    assert convert_arrays_to_lists(np.array([1.0, 2.0])) == [1.0, 2.0]
