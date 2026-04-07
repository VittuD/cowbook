from __future__ import annotations

from _package_bootstrap import ensure_src_path

ensure_src_path()

from cowbook import preprocess_video as _impl

for _exported_name, _value in vars(_impl).items():
    if not _exported_name.startswith("__"):
        globals()[_exported_name] = _value
