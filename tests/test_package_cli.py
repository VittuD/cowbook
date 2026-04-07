from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from cowbook import __version__
from cowbook import cli


def test_package_exposes_version():
    assert __version__ == "0.1.0"


def test_cli_prefers_flag_config_over_positional():
    args = cli.parse_args(["positional.json", "--config", "flag.json"])
    assert cli.resolve_config_path(args) == "flag.json"


def test_cli_defaults_to_config_json():
    args = cli.parse_args([])
    assert cli.resolve_config_path(args) == "config.json"


def test_cli_main_delegates_to_legacy_main(monkeypatch):
    called = {}

    class FakeLegacyMain:
        @staticmethod
        def main(config_path):
            called["config_path"] = config_path

    monkeypatch.setattr(cli, "load_legacy_main_module", lambda: FakeLegacyMain)

    exit_code = cli.main(["phase2.json"])

    assert exit_code == 0
    assert called["config_path"] == "phase2.json"


def test_python_m_cowbook_help_runs():
    repo_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [sys.executable, "-m", "cowbook", "--help"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Process videos and create projection video." in result.stdout
