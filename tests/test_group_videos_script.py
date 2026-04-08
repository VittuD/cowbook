from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "group_videos.sh"


def _run_group_videos(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_group_videos_script_dry_run_prints_planned_move(tmp_path: Path):
    src_dir = tmp_path / "raw_drop"
    dest_dir = tmp_path / "videos"
    src_dir.mkdir()
    (src_dir / "Ch1_batchA.mp4").write_text("video", encoding="utf-8")

    result = _run_group_videos("--src", str(src_dir), "--dest", str(dest_dir), "--dry-run", cwd=REPO_ROOT)

    assert result.returncode == 0
    assert f'mkdir -p "{dest_dir / "batchA"}"' in result.stdout
    assert f'mv "-n" "{src_dir / "Ch1_batchA.mp4"}" "{dest_dir / "batchA" / "Ch1.mp4"}"' in result.stdout
    assert result.stdout.strip().endswith("Done.")
    assert not dest_dir.exists()


def test_group_videos_script_copy_mode_preserves_source(tmp_path: Path):
    src_dir = tmp_path / "raw_drop"
    dest_dir = tmp_path / "videos"
    src_dir.mkdir()
    source_file = src_dir / "Ch4_group42.mp4"
    source_file.write_text("video-data", encoding="utf-8")

    result = _run_group_videos("--src", str(src_dir), "--dest", str(dest_dir), "--copy", cwd=REPO_ROOT)

    assert result.returncode == 0
    assert source_file.exists()
    assert (dest_dir / "group42" / "Ch4.mp4").read_text(encoding="utf-8") == "video-data"


def test_group_videos_script_fails_when_no_matching_files(tmp_path: Path):
    src_dir = tmp_path / "raw_drop"
    src_dir.mkdir()
    (src_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    result = _run_group_videos("--src", str(src_dir), cwd=REPO_ROOT)

    assert result.returncode == 1
    assert f"No matching files found in {src_dir}" in result.stderr
