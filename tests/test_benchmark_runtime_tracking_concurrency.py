from __future__ import annotations

from pathlib import Path

import pytest

from tools import benchmark_runtime_tracking_concurrency as module


def test_resolve_camera_nrs_uses_explicit_values():
    assert module._resolve_camera_nrs(["a.mp4", "b.mp4"], [1, 4]) == [1, 4]


def test_resolve_camera_nrs_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        module._resolve_camera_nrs(["a.mp4", "b.mp4"], [1])


def test_infer_camera_nr_reads_channel_from_filename():
    assert module._infer_camera_nr("sample_data/videos/Ch8_60.mp4") == 8


def test_benchmark_backend_for_concurrency_reports_effective_workers(monkeypatch, tmp_path: Path):
    warmups = []

    monkeypatch.setattr(
        module,
        "_warmup_runtime_tracking",
        lambda **kwargs: warmups.append(
            (kwargs["backend_label"], kwargs["concurrency"], kwargs["warmup_runs"])
        ),
    )
    monkeypatch.setattr(
        module,
        "_repeat_mode",
        lambda mode_name, *, repeat_count, runner, extra=None: {
            "mode": mode_name,
            "repeat_count": repeat_count,
            "best_elapsed_s": 5.0,
            "mean_elapsed_s": 5.0,
            "runs": [],
            "best_run": {},
            **(extra or {}),
        },
    )

    result = module._benchmark_backend_for_concurrency(
        backend_label="engine",
        model_path="model.engine",
        concurrency=3,
        videos=["a.mp4", "b.mp4", "c.mp4", "d.mp4"],
        camera_nrs=[1, 2, 3, 4],
        repeat=1,
        warmup_runs=1,
        output_root=tmp_path,
        log_progress=False,
    )

    assert warmups == [("engine", 3, 1)]
    assert result["backend"] == "engine"
    assert result["requested_concurrency"] == 3
    assert result["effective_tracking_concurrency"] == 3
    assert result["process_workers"] == 3
