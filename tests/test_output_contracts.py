from __future__ import annotations

import json

from json_merger import merge_json_files


def _assert_raw_tracking_contract(doc: dict) -> None:
    assert isinstance(doc.get("frames"), list)
    for frame in doc["frames"]:
        assert isinstance(frame["frame_id"], int)
        assert isinstance(frame["detections"]["xyxy"], list)
        assert isinstance(frame["labels"], list)
        assert len(frame["labels"]) == len(frame["detections"]["xyxy"])
        for bbox in frame["detections"]["xyxy"]:
            assert len(bbox) == 4


def _assert_processed_tracking_contract(doc: dict) -> None:
    _assert_raw_tracking_contract(doc)
    for frame in doc["frames"]:
        dets = frame["detections"]
        xyxy = dets["xyxy"]
        assert "centroids" in dets
        assert "projected_centroids" in dets
        assert len(dets["centroids"]) == len(xyxy)
        assert len(dets["projected_centroids"]) == len(xyxy)
        for centroid in dets["centroids"]:
            assert len(centroid) == 2
        for projected in dets["projected_centroids"]:
            assert len(projected) >= 2


def test_raw_tracking_fixture_matches_current_contract(raw_tracking_doc):
    _assert_raw_tracking_contract(raw_tracking_doc)


def test_processed_tracking_fixture_matches_current_contract(processed_tracking_doc):
    _assert_processed_tracking_contract(processed_tracking_doc)


def test_merged_processed_output_matches_current_contract(fixtures_dir, tmp_path):
    first = tmp_path / "processed_a.json"
    second = tmp_path / "processed_b.json"
    merged = tmp_path / "merged.json"
    fixture_text = (fixtures_dir / "processed_tracking_minimal.json").read_text()
    first.write_text(fixture_text)
    second.write_text(fixture_text)

    merge_json_files([str(first), str(second)], str(merged), camera_nrs=[1, 4])
    doc = json.loads(merged.read_text())

    _assert_raw_tracking_contract(doc)
    for frame in doc["frames"]:
        dets = frame["detections"]
        assert "centroids" in dets
        assert "projected_centroids" in dets
        assert len(dets["xyxy"]) == 4
        assert len(frame["labels"]) == 4
        for label in frame["labels"]:
            assert "local_track_id" in label
            assert "camera_nr" in label
            assert "global_id" in label
            assert "id" not in label
