from __future__ import annotations

import json
import runpy
import sys

from cowbook.io import csv_converter, json_merger


def test_csv_converter_helpers_cover_safe_access_and_fieldnames():
    assert csv_converter._safe_get([1, 2], 1) == 2
    assert csv_converter._safe_get([1, 2], 5) is None
    assert csv_converter._safe_get([], 0) is None
    assert csv_converter._fieldnames(None)[0] == "frame_id"
    assert csv_converter._fieldnames("source_name")[0] == "source_name"


def test_csv_converter_main_writes_source_column(tmp_path, monkeypatch):
    input_json = tmp_path / "input.json"
    input_json.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": {"xyxy": [[1, 2, 11, 12]]},
                        "labels": [{"class_id": 3, "id": 9}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_csv = tmp_path / "out.csv"

    monkeypatch.setattr(
        csv_converter,
        "_parse_args",
        lambda: type(
            "Args",
            (),
            {
                "input": str(input_json),
                "inputs": None,
                "output": str(output_csv),
                "source_col": "source_name",
            },
        )(),
    )

    csv_converter.main()

    content = output_csv.read_text(encoding="utf-8")
    assert "source_name" in content
    assert input_json.name in content
    assert "frame_id" in content


def test_json_merger_merge_json_files_writes_output(tmp_path, monkeypatch):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    output = tmp_path / "merged.json"
    first.write_text('{"frames": []}', encoding="utf-8")
    second.write_text('{"frames": []}', encoding="utf-8")

    monkeypatch.setattr(
        json_merger,
        "merge_tracking_documents",
        lambda documents, camera_nrs=None: {"frames": [{"frame_id": 0, "detections": {"xyxy": []}, "labels": []}]},
    )

    json_merger.merge_json_files([str(first), str(second)], str(output), camera_nrs=[1, 4])

    merged = json.loads(output.read_text(encoding="utf-8"))
    assert merged["frames"][0]["frame_id"] == 0


def test_json_merger_parse_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["json_merger.py", "--inputs", "a.json", "b.json", "--output", "out.json"])
    args = json_merger._parse_args()
    assert args.inputs == ["a.json", "b.json"]
    assert args.output == "out.json"


def test_module_main_invokes_entrypoint(monkeypatch):
    called: list[str] = []
    monkeypatch.setattr("cowbook.app.cli.entrypoint", lambda: called.append("entry"))

    runpy.run_module("cowbook", run_name="__main__")

    assert called == ["entry"]
