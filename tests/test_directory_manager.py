from __future__ import annotations

from pathlib import Path

from directory_manager import clear_output_directory, prepare_output_dirs


def test_prepare_output_dirs_creates_configured_directories(tmp_path):
    config = {
        "output_image_folder": str(tmp_path / "frames"),
        "output_video_folder": str(tmp_path / "videos"),
        "output_json_folder": str(tmp_path / "json"),
        "masked_video_folder": str(tmp_path / "masked"),
    }

    paths = prepare_output_dirs(config)

    assert paths == (
        str(tmp_path / "frames"),
        str(tmp_path / "videos"),
        str(tmp_path / "json"),
        str(tmp_path / "masked"),
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
