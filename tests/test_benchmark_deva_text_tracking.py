from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools import benchmark_deva_text_tracking as module


def test_resolve_prompts_uses_default_and_trims_values():
    assert module._resolve_prompts(None) == ["cow"]
    assert module._resolve_prompts([" cow ", "", "dairy cow"]) == ["cow", "dairy cow"]


def test_resolve_prompts_rejects_empty_values():
    with pytest.raises(ValueError, match="At least one non-empty prompt"):
        module._resolve_prompts(["", "   "])


def test_build_deva_text_command_matches_expected_shape():
    command = module._build_deva_text_command(
        python_bin="/usr/bin/python3",
        frames_dir="/tmp/frames",
        output_dir="/tmp/out",
        prompts=["cow", "calf"],
        chunk_size=4,
        size=480,
        temporal_setting="semionline",
        amp=True,
        sam_variant="mobile",
    )

    assert command == [
        "/usr/bin/python3",
        "demo/demo_with_text.py",
        "--chunk_size",
        "4",
        "--img_path",
        "/tmp/frames",
        "--temporal_setting",
        "semionline",
        "--size",
        "480",
        "--output",
        "/tmp/out",
        "--prompt",
        "cow.calf",
        "--amp",
        "--sam_variant",
        "mobile",
    ]


def test_run_deva_text_tracking_for_video_writes_summary(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"video")
    deva_repo = tmp_path / "deva"
    gsa_repo = tmp_path / "gsa"
    deva_repo.mkdir()
    gsa_repo.mkdir()
    output_root = tmp_path / "out"
    frames_root = tmp_path / "frames"
    raw_dir = output_root / "deva_raw" / "input"
    raw_dir.mkdir(parents=True)
    rendered = raw_dir / "demo.mp4"
    rendered.write_bytes(b"mp4")
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "_probe_video_metadata",
        lambda _path: {"fps": 5.0, "width": 32, "height": 24, "frame_count": 2},
    )
    monkeypatch.setattr(
        module,
        "_extract_video_frames",
        lambda **_kwargs: {
            "frame_count": 2,
            "fps": 5.0,
            "width": 32,
            "height": 24,
            "frames_dir": str(frames_root / "input"),
        },
    )

    def fake_run(command, cwd, env, check):
        recorded["command"] = command
        recorded["cwd"] = cwd
        recorded["env"] = env
        recorded["check"] = check
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "_collect_rendered_artifacts", lambda _path: [str(rendered)])

    result = module._run_deva_text_tracking_for_video(
        video_path=str(video_path),
        output_root=output_root,
        prompts=["cow"],
        deva_repo=str(deva_repo),
        gsa_repo=str(gsa_repo),
        python_bin="/usr/bin/python3",
        sam_variant="original",
        size=480,
        chunk_size=4,
        temporal_setting="semionline",
        amp=True,
        max_frames=120,
        prepared_frame_dir=frames_root,
        log_progress=False,
    )

    assert recorded["cwd"] == str(deva_repo)
    assert recorded["check"] is True
    assert result.primary_rendered_artifact == str(rendered)
    assert result.frame_count == 2
    assert result.max_frames == 120
    assert Path(result.summary_json_path).exists()
