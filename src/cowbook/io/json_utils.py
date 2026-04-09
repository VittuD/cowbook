from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

_PRETTY_OPTIONS = orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS


def loads(data: str | bytes | bytearray | memoryview) -> Any:
    return orjson.loads(data)


def load_path(path: str | Path) -> Any:
    return loads(Path(path).read_bytes())


def dumps_compact(value: Any) -> bytes:
    return orjson.dumps(value)


def dumps_pretty(value: Any) -> bytes:
    return orjson.dumps(value, option=_PRETTY_OPTIONS)


def dump_path_compact(path: str | Path, value: Any) -> None:
    Path(path).write_bytes(dumps_compact(value))


def dump_path_pretty(path: str | Path, value: Any, *, trailing_newline: bool = False) -> None:
    payload = dumps_pretty(value)
    if trailing_newline:
        payload += b"\n"
    Path(path).write_bytes(payload)
