from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def assets_root() -> Path:
    return repo_root() / "assets"


def sample_data_root() -> Path:
    return repo_root() / "sample_data"


def ensure_repo_root_on_path() -> Path:
    root = repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
