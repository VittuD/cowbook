from __future__ import annotations

import json

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
