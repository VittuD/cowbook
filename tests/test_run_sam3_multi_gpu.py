from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from tools import run_sam3_multi_gpu as module


def _write_real_video(path: Path, frame_count: int, fps: float, size: tuple[int, int] = (32, 24)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for index in range(frame_count):
        frame = np.full((size[1], size[0], 3), fill_value=index % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


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


def test_channel_for_video_path_matches_by_filename_prefix():
    assert module._channel_for_video_path("/videos/Ch1_60.mp4", ["Ch1", "Ch4"]) == "Ch1"
    assert module._channel_for_video_path("/videos/Ch4_60.mp4", ["Ch1", "Ch4"]) == "Ch4"
    assert module._channel_for_video_path("/videos/Ch2_60.mp4", ["Ch1", "Ch4"]) is None


def test_preprocess_videos_for_masking_crops_and_masks_per_channel(tmp_path: Path):
    width, height = 64, 64
    ch1_path = tmp_path / "Ch1_a.mp4"
    ch4_path = tmp_path / "Ch4_a.mp4"
    for path in (ch1_path, ch4_path):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
        writer.write(np.full((height, width, 3), 220, dtype=np.uint8))
        writer.release()

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[10:40, 15:55] = 255
    ch1_mask_path = tmp_path / "ch1_mask.png"
    ch4_mask_path = tmp_path / "ch4_mask.png"
    cv2.imwrite(str(ch1_mask_path), mask)
    cv2.imwrite(str(ch4_mask_path), mask)

    output_dir = tmp_path / "masked"
    processed = module._preprocess_videos_for_masking(
        [str(ch1_path), str(ch4_path)],
        ["Ch1", "Ch4"],
        output_dir=output_dir,
        crop_to_mask=True,
        channel_masks={"Ch1": str(ch1_mask_path), "Ch4": str(ch4_mask_path)},
        max_workers=1,
        log_progress=False,
    )

    assert sorted(Path(p).name for p in processed) == ["Ch1_a.mp4", "Ch4_a.mp4"]
    for path in processed:
        assert Path(path).exists()
        capture = cv2.VideoCapture(path)
        cropped_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        cropped_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        assert (cropped_width, cropped_height) == (40, 30)  # tight bbox around mask[10:40, 15:55]


def test_preprocess_videos_for_masking_raises_on_unmatched_channel(tmp_path: Path):
    video_path = tmp_path / "Ch2_a.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (16, 12))
    writer.write(np.zeros((12, 16, 3), dtype=np.uint8))
    writer.release()

    with pytest.raises(FileNotFoundError, match="No usable mask"):
        module._preprocess_videos_for_masking(
            [str(video_path)],
            ["Ch1", "Ch2"],
            output_dir=tmp_path / "masked",
            crop_to_mask=False,
            channel_masks={"Ch1": "/nonexistent/mask.png"},  # no entry for Ch2
            max_workers=1,
            log_progress=False,
        )


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


class _FakePopen:
    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.pid = 4242


def test_launch_assignment_dispatches_windowed_tool_by_default(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs.get("env")
        return _FakePopen(command, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", _fake_popen)
    assignment = module.GpuAssignment(gpu_index=3, videos=["a.mp4", "b.mp4"], total_frames=100)

    process = module._launch_assignment(
        assignment,
        output_root=tmp_path / "out",
        model_path="models/sam3.pt",
        prompts=["cow"],
        render_mode="none",
        log_dir=tmp_path / "logs",
        log_every_frames=200,
        windowed=True,
        window_seconds=600.0,
    )

    assert process.pid == 4242
    command = captured["command"]
    assert "tools.run_sam3_windowed" in command
    assert "tools.benchmark_sam3_semantic_tracking" not in command
    assert "--window-seconds" in command
    assert command[command.index("--window-seconds") + 1] == "600.0"
    assert command[command.index("--videos") + 1 : command.index("--videos") + 3] == ["a.mp4", "b.mp4"]
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "3"


def test_launch_assignment_dispatches_single_pass_tool_when_not_windowed(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        return _FakePopen(command, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", _fake_popen)
    assignment = module.GpuAssignment(gpu_index=0, videos=["a.mp4"], total_frames=50)

    module._launch_assignment(
        assignment,
        output_root=tmp_path / "out",
        model_path="models/sam3.pt",
        prompts=["cow"],
        render_mode="none",
        log_dir=tmp_path / "logs",
        log_every_frames=200,
        windowed=False,
        window_seconds=600.0,
    )

    command = captured["command"]
    assert "tools.benchmark_sam3_semantic_tracking" in command
    assert "tools.run_sam3_windowed" not in command
    assert "--window-seconds" not in command


def test_launch_assignment_forwards_target_fps_when_set(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        return _FakePopen(command, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", _fake_popen)
    assignment = module.GpuAssignment(gpu_index=0, videos=["a.mp4"], total_frames=50)

    module._launch_assignment(
        assignment,
        output_root=tmp_path / "out",
        model_path="models/sam3.pt",
        prompts=["cow"],
        render_mode="none",
        log_dir=tmp_path / "logs",
        log_every_frames=200,
        windowed=True,
        window_seconds=600.0,
        target_fps=1.0,
    )

    command = captured["command"]
    assert "--target-fps" in command
    assert command[command.index("--target-fps") + 1] == "1.0"


def test_launch_assignment_omits_target_fps_by_default(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        return _FakePopen(command, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", _fake_popen)
    assignment = module.GpuAssignment(gpu_index=0, videos=["a.mp4"], total_frames=50)

    module._launch_assignment(
        assignment,
        output_root=tmp_path / "out",
        model_path="models/sam3.pt",
        prompts=["cow"],
        render_mode="none",
        log_dir=tmp_path / "logs",
        log_every_frames=200,
        windowed=True,
        window_seconds=600.0,
    )

    assert "--target-fps" not in captured["command"]


def test_launch_assignment_chains_preview_after_tracking_when_set(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        return _FakePopen(command, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", _fake_popen)
    assignment = module.GpuAssignment(gpu_index=0, videos=["a.mp4", "b.mp4"], total_frames=100)

    module._launch_assignment(
        assignment,
        output_root=tmp_path / "out",
        model_path="models/sam3.pt",
        prompts=["cow"],
        render_mode="none",
        log_dir=tmp_path / "logs",
        log_every_frames=200,
        windowed=True,
        window_seconds=600.0,
        preview_sample_count=10,
    )

    command = captured["command"]
    # Chained via a shell, not launched concurrently: a second predictor
    # instance sharing the GPU with the still-running tracking pass would
    # compete for its memory instead of reusing it afterward.
    assert command[:3] == ["bash", "-lc", command[2]]
    shell_line = command[2]
    assert "tools.run_sam3_windowed" in shell_line
    assert "tools.preview_sam3_samples" in shell_line
    assert " && " in shell_line
    assert shell_line.index("tools.run_sam3_windowed") < shell_line.index("tools.preview_sam3_samples")
    assert "--sample-count 10" in shell_line
    assert str(tmp_path / "out" / "preview") in shell_line


def test_launch_assignment_runs_only_tracking_without_preview(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        return _FakePopen(command, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", _fake_popen)
    assignment = module.GpuAssignment(gpu_index=0, videos=["a.mp4"], total_frames=50)

    module._launch_assignment(
        assignment,
        output_root=tmp_path / "out",
        model_path="models/sam3.pt",
        prompts=["cow"],
        render_mode="none",
        log_dir=tmp_path / "logs",
        log_every_frames=200,
        windowed=True,
        window_seconds=600.0,
    )

    command = captured["command"]
    assert command[0] != "bash"
    assert "tools.preview_sam3_samples" not in command


def test_main_creates_plan_path_parent_directory_even_on_dry_run(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "videos"
    input_dir.mkdir()
    _write_real_video(input_dir / "Ch1_a.mp4", frame_count=10, fps=5.0)

    # A --plan-path nested under directories that don't exist yet, and no
    # --launch, so --output-root is never created either: writing the plan
    # must not depend on some other path having already made the directory.
    plan_path = tmp_path / "nested" / "does" / "not" / "exist" / "plan.json"
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "run_sam3_multi_gpu.py",
            "--input-dir",
            str(input_dir),
            "--channels",
            "Ch1",
            "--num-gpus",
            "1",
            "--plan-path",
            str(plan_path),
        ],
    )

    assert module.main() == 0
    assert plan_path.exists()
