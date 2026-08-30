from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tools import run_sam3_multi_gpu as module


def test_collect_channel_videos_matches_requested_channels_only(tmp_path: Path):
    for name in ["Ch1_a.mp4", "Ch1_b.mp4", "Ch4_a.mp4", "Ch2_a.mp4"]:
        (tmp_path / name).write_bytes(b"video")

    matched = module._collect_channel_videos(str(tmp_path), ["Ch1", "Ch4"])

    assert matched == [
        str(tmp_path / "Ch1_a.mp4"),
        str(tmp_path / "Ch1_b.mp4"),
        str(tmp_path / "Ch4_a.mp4"),
    ]


def test_collect_channel_videos_raises_on_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Missing input directory"):
        module._collect_channel_videos(str(tmp_path / "missing"), ["Ch1"])


def test_collect_channel_videos_raises_when_nothing_matches(tmp_path: Path):
    (tmp_path / "Ch2_a.mp4").write_bytes(b"video")

    with pytest.raises(FileNotFoundError, match="No videos matched channels"):
        module._collect_channel_videos(str(tmp_path), ["Ch1"])


def test_assign_longest_processing_time_first_balances_uneven_jobs():
    items = [
        module.VideoWorkItem(path="short_a", frame_count=100, fps=6.0, duration_s=16.7),
        module.VideoWorkItem(path="short_b", frame_count=100, fps=6.0, duration_s=16.7),
        module.VideoWorkItem(path="long_a", frame_count=1000, fps=6.0, duration_s=166.7),
        module.VideoWorkItem(path="long_b", frame_count=1000, fps=6.0, duration_s=166.7),
    ]

    assignments = module._assign_longest_processing_time_first(items, num_gpus=2)

    all_videos = sorted(video for assignment in assignments for video in assignment.videos)
    assert all_videos == ["long_a", "long_b", "short_a", "short_b"]
    # Each GPU should get exactly one long job plus one short job, not both
    # long jobs stacked on the same GPU.
    totals = sorted(assignment.total_frames for assignment in assignments)
    assert totals == [1100, 1100]


def test_assign_longest_processing_time_first_assigns_every_video_once():
    items = [
        module.VideoWorkItem(path=f"video_{index}", frame_count=index + 1, fps=6.0, duration_s=0.0)
        for index in range(7)
    ]

    assignments = module._assign_longest_processing_time_first(items, num_gpus=3)

    all_videos = [video for assignment in assignments for video in assignment.videos]
    assert sorted(all_videos) == sorted(item.path for item in items)
    assert len(all_videos) == len(items)


def test_detect_gpu_count_falls_back_to_one_without_nvidia_smi(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(module.subprocess, "run", _raise)

    assert module._detect_gpu_count() == 1


def test_detect_gpu_count_counts_nvidia_smi_lines(monkeypatch):
    def _fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="0\n1\n2\n3\n")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    assert module._detect_gpu_count() == 4
