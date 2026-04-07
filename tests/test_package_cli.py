from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from cowbook import __version__
from cowbook.app import cli


def test_package_exposes_version():
    assert __version__ == "0.1.0"


def test_cli_prefers_flag_config_over_positional():
    args = cli.parse_args(["positional.json", "--config", "flag.json"])
    assert cli.resolve_config_path(args) == "flag.json"


def test_cli_defaults_to_config_json():
    args = cli.parse_args([])
    assert cli.resolve_config_path(args) == "config.json"


def test_cli_resolves_supported_overrides():
    args = cli.parse_args(
        [
            "--fps",
            "12",
            "--output-video-filename",
            "demo.mp4",
            "--output-image-format",
            "png",
            "--num-plot-workers",
            "3",
            "--num-tracking-workers",
            "2",
            "--no-create-projection-video",
            "--no-clean-frames-after-video",
            "--mask-videos",
        ]
    )

    assert cli.resolve_overrides(args) == {
        "fps": 12,
        "output_video_filename": "demo.mp4",
        "output_image_format": "png",
        "num_plot_workers": 3,
        "num_tracking_workers": 2,
        "create_projection_video": False,
        "clean_frames_after_video": False,
        "mask_videos": True,
    }


def test_cli_main_delegates_to_legacy_main(monkeypatch):
    called = {}

    class FakeRunner:
        def run(self, config_path, overrides=None):
            called["config_path"] = config_path
            called["overrides"] = overrides

    monkeypatch.setattr(cli, "PipelineRunner", lambda: FakeRunner())

    exit_code = cli.main(["phase2.json", "--fps", "9", "--no-create-projection-video"])

    assert exit_code == 0
    assert called["config_path"] == "phase2.json"
    assert called["overrides"] == {"fps": 9, "create_projection_video": False}


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
