from __future__ import annotations

import pytest

from tools import benchmark_tensorrt_concurrency as module


def test_mode_for_concurrency_maps_one_to_sequential():
    assert module._mode_for_concurrency(1) == ("sequential_shared_model", 1)


def test_mode_for_concurrency_maps_parallel_values_to_process_workers():
    assert module._mode_for_concurrency(4) == ("process_parallel_models", 4)


def test_mode_for_concurrency_rejects_non_positive_values():
    for value in (0, -1):
        with pytest.raises(ValueError):
            module._mode_for_concurrency(value)


def test_benchmark_backend_for_concurrency_sets_expected_metadata(monkeypatch):
    warmups: list[tuple[str, str, int, int]] = []

    def fake_warmup_backend(**kwargs):
        warmups.append(
            (
                kwargs["backend_label"],
                kwargs["mode_name"],
                kwargs["process_workers"],
                kwargs["warmup_runs"],
            )
        )

    monkeypatch.setattr(module, "_warmup_backend", fake_warmup_backend)
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
        videos=["a.mp4", "b.mp4"],
        tracker_config="tracker.yaml",
        repeat=1,
        warmup_runs=2,
        log_progress=False,
    )

    assert warmups == [("engine", "process_parallel_models", 3, 2)]
    assert result["backend"] == "engine"
    assert result["requested_concurrency"] == 3
    assert result["effective_tracking_concurrency"] == 3
    assert result["process_workers"] == 3
