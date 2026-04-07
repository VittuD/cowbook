from __future__ import annotations

from pathlib import Path

from cowbook.io.directory_manager import (
    clear_output_directory,
    prepare_output_dirs,
    resolve_output_paths,
)


def test_resolve_output_paths_uses_single_runtime_root_by_default():
    paths = resolve_output_paths({})

    assert paths == (
        "var",
        "var/runs/default",
        "var/runs/default/frames",
        "var/runs/default/videos",
        "var/runs/default/json",
        "var/cache/masked_videos",
    )


def test_resolve_output_paths_respects_explicit_overrides():
    paths = resolve_output_paths(
        {
            "runtime_root": "runtime",
            "run_name": "job-42",
            "output_video_folder": "custom/videos",
            "masked_video_folder": "custom/masked",
        }
    )

    assert paths == (
        "runtime",
        "runtime/runs/job-42",
        "runtime/runs/job-42/frames",
        "custom/videos",
        "runtime/runs/job-42/json",
        "custom/masked",
    )


def test_prepare_output_dirs_creates_configured_directories(tmp_path):
    config = {
        "runtime_root": str(tmp_path / "runtime"),
        "run_name": "demo",
    }

    paths = prepare_output_dirs(config)

    assert paths == (
        str(tmp_path / "runtime" / "runs" / "demo" / "frames"),
        str(tmp_path / "runtime" / "runs" / "demo" / "videos"),
        str(tmp_path / "runtime" / "runs" / "demo" / "json"),
        str(tmp_path / "runtime" / "cache" / "masked_videos"),
    )
    for path in paths:
        assert Path(path).is_dir()


def test_clear_output_directory_removes_files_and_nested_directories(tmp_path):
    target = tmp_path / "output"
    target.mkdir()
    (target / "frame_0.jpg").write_text("x")
    nested = target / "nested"
    nested.mkdir()
    (nested / "artifact.txt").write_text("x")

    clear_output_directory(str(target))

    assert target.is_dir()
    assert list(target.iterdir()) == []
