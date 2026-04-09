from __future__ import annotations

from cowbook.io.json_utils import dump_path_compact, dump_path_pretty, load_path


def test_dump_path_compact_round_trips_json(tmp_path):
    output_path = tmp_path / "compact.json"

    dump_path_compact(output_path, {"b": 2, "a": [1, True, None]})

    assert output_path.read_text(encoding="utf-8") == '{"b":2,"a":[1,true,null]}'
    assert load_path(output_path) == {"b": 2, "a": [1, True, None]}


def test_dump_path_pretty_writes_sorted_readable_json_with_newline(tmp_path):
    output_path = tmp_path / "pretty.json"

    dump_path_pretty(output_path, {"b": 2, "a": {"z": 1}}, trailing_newline=True)

    assert output_path.read_text(encoding="utf-8") == '{\n  "a": {\n    "z": 1\n  },\n  "b": 2\n}\n'
    assert load_path(output_path) == {"a": {"z": 1}, "b": 2}
