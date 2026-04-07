from __future__ import annotations

from csv_converter import _fieldnames, _iter_rows_from_json


def test_iter_rows_from_raw_json_computes_centroids_without_projection(raw_tracking_doc):
    rows = list(_iter_rows_from_json(raw_tracking_doc))

    assert len(rows) == 3
    assert rows[0]["frame_id"] == 0
    assert rows[0]["det_index"] == 1
    assert rows[0]["id"] == 11
    assert rows[0]["centroid_x"] == 20.0
    assert rows[0]["centroid_y"] == 40.0
    assert rows[0]["proj_x"] is None
    assert rows[0]["proj_y"] is None
    assert rows[0]["proj_z"] is None


def test_iter_rows_from_processed_json_preserves_projected_centroids(processed_tracking_doc):
    rows = list(_iter_rows_from_json(processed_tracking_doc, source_tag="fixture"))

    assert len(rows) == 2
    assert rows[0]["source"] == "fixture"
    assert rows[0]["proj_x"] == 100.0
    assert rows[0]["proj_y"] == 200.0
    assert rows[0]["proj_z"] == 100.0


def test_fieldnames_include_source_only_when_requested():
    assert "source" not in _fieldnames(include_source=False)
    assert _fieldnames(include_source=True)[0] == "source"
