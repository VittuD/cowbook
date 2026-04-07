from __future__ import annotations

from cowbook.app import cli as _impl

for _exported_name, _value in vars(_impl).items():
    if not _exported_name.startswith("__"):
        globals()[_exported_name] = _value
