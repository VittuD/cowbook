from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_path() -> Path:
    root = Path(__file__).resolve().parent
    src = root / "src"
    src_str = str(src)
    if src.exists() and src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src
