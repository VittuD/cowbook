from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def raw_tracking_doc(fixtures_dir: Path) -> dict:
    return json.loads((fixtures_dir / "raw_tracking_minimal.json").read_text())


@pytest.fixture
def processed_tracking_doc(fixtures_dir: Path) -> dict:
    return json.loads((fixtures_dir / "processed_tracking_minimal.json").read_text())
