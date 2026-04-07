from __future__ import annotations

from _package_bootstrap import ensure_src_path

ensure_src_path()

from cowbook import video_processor as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)
