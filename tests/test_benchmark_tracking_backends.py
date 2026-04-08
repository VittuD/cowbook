from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import benchmark_tracking_backends as module


def test_normalize_imgsz_accepts_square_and_rectangular_shapes():
    assert module._normalize_imgsz([640]) == (640, 640)
    assert module._normalize_imgsz([448, 768]) == (448, 768)


def test_normalize_imgsz_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        module._normalize_imgsz([])
    with pytest.raises(ValueError):
        module._normalize_imgsz([0])
    with pytest.raises(ValueError):
        module._normalize_imgsz([1, 2, 3])


def test_artifact_name_encodes_backend_and_export_shape():
    artifact = module._artifact_name(
        "models/yolov11_best.pt",
        "onnx",
        (448, 768),
        half=False,
        dynamic=True,
    )
    assert artifact == "yolov11_best_onnx_448x768_fp32_dynamic.onnx"


def test_export_model_artifact_reuses_existing_file(tmp_path):
    artifact_dir = tmp_path / "exports"
    artifact_dir.mkdir()
    artifact_path = artifact_dir / "yolov11_best_engine_448x768_fp16_static.engine"
    artifact_path.write_text("ready", encoding="utf-8")

    result = module._export_model_artifact(
        source_model_path="models/yolov11_best.pt",
        backend="engine",
        export_dir=str(artifact_dir),
        imgsz=(448, 768),
        device="0",
        half=True,
        dynamic=False,
        simplify=False,
        workspace_gb=None,
        force_export=False,
        log_progress=False,
    )

    assert result.backend == "engine"
    assert result.artifact_path == str(artifact_path)
    assert result.reused_existing is True
    assert result.error is None


def test_run_sequential_shared_model_collects_frame_counts(monkeypatch):
    class FakeModel:
        def __init__(self, path: str):
            self.path = path

    counts = {
        "a.mp4": (12, 1.25),
        "b.mp4": (10, 1.00),
    }

    monkeypatch.setattr(module, "YOLO", FakeModel)
    monkeypatch.setattr(module, "_count_frames_for_source", lambda _model, video, _tracker: counts[video])

    result = module._run_sequential_shared_model(
        videos=["a.mp4", "b.mp4"],
        model_path="models/yolov11_best.pt",
        tracker_config="tracker.yaml",
    )

    assert result.per_source_frame_count == {"a.mp4": 12, "b.mp4": 10}
    assert result.per_source_elapsed_s == {"a.mp4": 1.25, "b.mp4": 1.0}
    assert result.elapsed_s >= 0.0


def test_prebuilt_artifact_path_reads_backend_specific_overrides():
    args = SimpleNamespace(
        onnx_artifact_path="exports/model.onnx",
        engine_artifact_path="exports/model.engine",
    )

    assert module._prebuilt_artifact_path(args, "onnx") == "exports/model.onnx"
    assert module._prebuilt_artifact_path(args, "engine") == "exports/model.engine"


def test_run_backend_mode_dispatches_to_process_parallel(monkeypatch):
    sentinel = module.BenchmarkModeResult(
        mode="process_parallel_models",
        elapsed_s=2.5,
        per_source_frame_count={"a.mp4": 12},
        per_source_elapsed_s={"a.mp4": 2.4},
    )
    recorded = {}

    def fake_run_process_parallel_models(*, videos, model_path, tracker_config, worker_count):
        recorded["args"] = (videos, model_path, tracker_config, worker_count)
        return sentinel

    monkeypatch.setattr(module, "_run_process_parallel_models", fake_run_process_parallel_models)

    result = module._run_backend_mode(
        mode_name="process_parallel_models",
        videos=["a.mp4", "b.mp4"],
        model_path="model.onnx",
        tracker_config="tracker.yaml",
        process_workers=2,
    )

    assert result is sentinel
    assert recorded["args"] == (["a.mp4", "b.mp4"], "model.onnx", "tracker.yaml", 2)
