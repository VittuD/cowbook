from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from tools.benchmark_tracking import (
    BenchmarkModeResult,
    _count_frames_for_source,
    _default_tracker_config,
    _prepare_benchmark_videos,
    _query_gpu_info,
    _repeat_mode,
    _run_process_parallel_models,
    _validate_videos,
)

_BACKEND_SUFFIXES = {
    "onnx": ".onnx",
    "engine": ".engine",
}


@dataclass(slots=True)
class ExportResult:
    backend: str
    artifact_path: str | None
    reused_existing: bool
    elapsed_s: float | None = None
    error: str | None = None


def _log_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _normalize_imgsz(values: list[int]) -> tuple[int, int]:
    if not values:
        raise ValueError("imgsz must contain one or two positive integers")
    if len(values) == 1:
        size = int(values[0])
        if size <= 0:
            raise ValueError("imgsz values must be positive")
        return size, size
    if len(values) == 2:
        height, width = int(values[0]), int(values[1])
        if height <= 0 or width <= 0:
            raise ValueError("imgsz values must be positive")
        return height, width
    raise ValueError("imgsz must contain one or two positive integers")


def _artifact_name(source_model_path: str, backend: str, imgsz: tuple[int, int], *, half: bool, dynamic: bool) -> str:
    stem = Path(source_model_path).stem
    height, width = imgsz
    precision = "fp16" if half else "fp32"
    shape_mode = "dynamic" if dynamic else "static"
    suffix = _BACKEND_SUFFIXES[backend]
    return f"{stem}_{backend}_{height}x{width}_{precision}_{shape_mode}{suffix}"


def _export_model_artifact(
    *,
    source_model_path: str,
    backend: str,
    export_dir: str,
    imgsz: tuple[int, int],
    device: str,
    half: bool,
    dynamic: bool,
    simplify: bool,
    workspace_gb: float | None,
    force_export: bool,
    log_progress: bool,
) -> ExportResult:
    export_root = Path(export_dir)
    export_root.mkdir(parents=True, exist_ok=True)
    target_path = export_root / _artifact_name(
        source_model_path,
        backend,
        imgsz,
        half=half,
        dynamic=dynamic,
    )
    if target_path.exists() and not force_export:
        _log_progress(log_progress, f"[export] reusing existing {backend} artifact: {target_path}")
        return ExportResult(
            backend=backend,
            artifact_path=str(target_path),
            reused_existing=True,
        )

    _log_progress(
        log_progress,
        f"[export] exporting {backend} artifact from {source_model_path} -> {target_path}",
    )
    start = time.perf_counter()
    model = YOLO(source_model_path, task="detect")
    try:
        export_kwargs: dict[str, Any] = {
            "format": backend,
            "imgsz": list(imgsz),
            "device": device,
            "half": half,
            "dynamic": dynamic,
            "simplify": simplify,
            "verbose": False,
        }
        if backend == "engine" and workspace_gb is not None:
            export_kwargs["workspace"] = workspace_gb
        exported_path = Path(str(model.export(**export_kwargs)))
        if exported_path.resolve() != target_path.resolve():
            if target_path.exists():
                target_path.unlink()
            shutil.move(str(exported_path), str(target_path))
        elapsed_s = time.perf_counter() - start
        _log_progress(log_progress, f"[export] exported {backend} artifact in {elapsed_s:.2f}s")
        return ExportResult(
            backend=backend,
            artifact_path=str(target_path),
            reused_existing=False,
            elapsed_s=elapsed_s,
        )
    except Exception as exc:
        return ExportResult(
            backend=backend,
            artifact_path=str(target_path),
            reused_existing=False,
            elapsed_s=time.perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        del model


def _run_sequential_shared_model(
    *,
    videos: list[str],
    model_path: str,
    tracker_config: str,
) -> BenchmarkModeResult:
    model = YOLO(model_path, task="detect")
    per_source_frame_count: dict[str, int] = {}
    per_source_elapsed_s: dict[str, float] = {}
    start = time.perf_counter()
    try:
        for video_path in videos:
            frame_count, elapsed_s = _count_frames_for_source(model, video_path, tracker_config)
            per_source_frame_count[video_path] = frame_count
            per_source_elapsed_s[video_path] = elapsed_s
    finally:
        del model

    return BenchmarkModeResult(
        mode="sequential_shared_model",
        elapsed_s=time.perf_counter() - start,
        per_source_frame_count=per_source_frame_count,
        per_source_elapsed_s=per_source_elapsed_s,
        notes=["One YOLO model instance is reused across videos, but only one video runs at a time."],
    )


def _warmup_backend(
    *,
    backend_label: str,
    mode_name: str,
    process_workers: int,
    warmup_runs: int,
    videos: list[str],
    model_path: str,
    tracker_config: str,
    log_progress: bool,
) -> None:
    for idx in range(warmup_runs):
        _log_progress(
            log_progress,
            f"[warmup] backend={backend_label} mode={mode_name} run={idx + 1}/{warmup_runs}",
        )
        _run_backend_mode(
            mode_name=mode_name,
            videos=videos,
            model_path=model_path,
            tracker_config=tracker_config,
            process_workers=process_workers,
        )


def _run_backend_mode(
    *,
    mode_name: str,
    videos: list[str],
    model_path: str,
    tracker_config: str,
    process_workers: int,
) -> BenchmarkModeResult:
    if mode_name == "sequential_shared_model":
        return _run_sequential_shared_model(
            videos=videos,
            model_path=model_path,
            tracker_config=tracker_config,
        )
    if mode_name == "process_parallel_models":
        return _run_process_parallel_models(
            videos=videos,
            model_path=model_path,
            tracker_config=tracker_config,
            worker_count=process_workers,
        )
    raise ValueError(f"Unsupported backend benchmark mode: {mode_name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a backend A/B benchmark for Cowbook tracking artifacts.",
    )
    parser.add_argument("--videos", nargs="+", required=True, help="Video paths to benchmark together.")
    parser.add_argument(
        "--mode",
        default="sequential_shared_model",
        choices=["sequential_shared_model", "process_parallel_models"],
        help="Benchmark mode to run for each backend.",
    )
    parser.add_argument(
        "--baseline-model-path",
        default="models/yolov11_best.pt",
        help="Baseline PyTorch model path.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["onnx", "engine"],
        choices=sorted(_BACKEND_SUFFIXES),
        help="Exported backends to compare against the PyTorch baseline.",
    )
    parser.add_argument(
        "--tracker-config",
        default=_default_tracker_config(),
        help="Path to the Ultralytics tracker YAML.",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=[448, 768],
        help="Export image size as one square size or two integers: height width.",
    )
    parser.add_argument("--device", default="0", help="Ultralytics export device argument.")
    parser.add_argument(
        "--process-workers",
        type=int,
        default=2,
        help="Worker count for process_parallel_models.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Measured runs per backend.")
    parser.add_argument("--warmup-runs", type=int, default=0, help="Unmeasured warmup runs per backend.")
    parser.add_argument(
        "--output",
        default="var/benchmarks/tracking_backend_benchmark.json",
        help="Path to write the JSON summary.",
    )
    parser.add_argument(
        "--export-dir",
        default="var/benchmarks/model_exports",
        help="Directory for exported backend artifacts.",
    )
    parser.add_argument(
        "--onnx-artifact-path",
        default=None,
        help="Optional prebuilt ONNX artifact path. When set, export is skipped for ONNX.",
    )
    parser.add_argument(
        "--engine-artifact-path",
        default=None,
        help="Optional prebuilt TensorRT engine path. When set, export is skipped for TensorRT.",
    )
    parser.add_argument(
        "--extend-seconds",
        type=int,
        default=0,
        help="If > 0, create longer benchmark copies of the input videos before running.",
    )
    parser.add_argument(
        "--prepared-video-dir",
        default="var/benchmarks/prepared_videos_backend",
        help="Directory for prepared benchmark videos when --extend-seconds is used.",
    )
    parser.add_argument("--half", action="store_true", help="Export candidate artifacts in fp16 when supported.")
    parser.add_argument("--dynamic", action="store_true", help="Export candidate artifacts with dynamic shapes.")
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Enable Ultralytics export graph simplification.",
    )
    parser.add_argument(
        "--workspace-gb",
        type=float,
        default=None,
        help="TensorRT workspace size in GiB for engine export.",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Rebuild exported backend artifacts even if they already exist.",
    )
    parser.add_argument(
        "--log-progress",
        action="store_true",
        help="Print coarse export and benchmark progress messages.",
    )
    return parser.parse_args()


def _prebuilt_artifact_path(args: argparse.Namespace, backend: str) -> str | None:
    if backend == "onnx":
        return args.onnx_artifact_path
    if backend == "engine":
        return args.engine_artifact_path
    return None


def main() -> int:
    args = _parse_args()
    videos = _validate_videos(args.videos)
    imgsz = _normalize_imgsz(list(args.imgsz))
    prepared_videos, prepared_video_metadata = _prepare_benchmark_videos(
        videos,
        target_duration_seconds=args.extend_seconds,
        prepared_video_dir=args.prepared_video_dir,
        log_progress=args.log_progress,
    )
    benchmark_videos = prepared_videos or videos
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "mode": args.mode,
        "source_videos": videos,
        "benchmark_videos": benchmark_videos,
        "video_count": len(benchmark_videos),
        "baseline_model_path": args.baseline_model_path,
        "tracker_config": args.tracker_config,
        "imgsz": list(imgsz),
        "device": args.device,
        "process_workers": int(args.process_workers),
        "half": bool(args.half),
        "dynamic": bool(args.dynamic),
        "simplify": bool(args.simplify),
        "workspace_gb": args.workspace_gb,
        "repeat": args.repeat,
        "warmup_runs": args.warmup_runs,
        "extend_seconds": args.extend_seconds,
        "prepared_video_dir": args.prepared_video_dir if args.extend_seconds > 0 else None,
        "prepared_videos": prepared_video_metadata,
        "gpu_info": _query_gpu_info(),
        "exports": [],
        "results": [],
    }

    _warmup_backend(
        backend_label="pt",
        mode_name=args.mode,
        process_workers=int(args.process_workers),
        warmup_runs=args.warmup_runs,
        videos=benchmark_videos,
        model_path=args.baseline_model_path,
        tracker_config=args.tracker_config,
        log_progress=args.log_progress,
    )
    pt_result = _repeat_mode(
        args.mode,
        repeat_count=args.repeat,
        runner=lambda: _run_backend_mode(
            mode_name=args.mode,
            videos=benchmark_videos,
            model_path=args.baseline_model_path,
            tracker_config=args.tracker_config,
            process_workers=int(args.process_workers),
        ),
        extra={
            "backend": "pt",
            "artifact_path": args.baseline_model_path,
            "process_workers": int(args.process_workers) if args.mode == "process_parallel_models" else None,
        },
    )
    summary["results"].append(pt_result)

    for backend in args.backends:
        prebuilt_artifact = _prebuilt_artifact_path(args, backend)
        if prebuilt_artifact:
            export_result = ExportResult(
                backend=backend,
                artifact_path=prebuilt_artifact,
                reused_existing=True,
            )
        else:
            export_result = _export_model_artifact(
                source_model_path=args.baseline_model_path,
                backend=backend,
                export_dir=args.export_dir,
                imgsz=imgsz,
                device=args.device,
                half=bool(args.half),
                dynamic=bool(args.dynamic),
                simplify=bool(args.simplify),
                workspace_gb=args.workspace_gb,
                force_export=bool(args.force_export),
                log_progress=bool(args.log_progress),
            )
        summary["exports"].append(asdict(export_result))
        if export_result.error:
            continue
        assert export_result.artifact_path is not None
        _warmup_backend(
            backend_label=backend,
            mode_name=args.mode,
            process_workers=int(args.process_workers),
            warmup_runs=args.warmup_runs,
            videos=benchmark_videos,
            model_path=export_result.artifact_path,
            tracker_config=args.tracker_config,
            log_progress=args.log_progress,
        )
        backend_result = _repeat_mode(
            args.mode,
            repeat_count=args.repeat,
            runner=lambda artifact_path=export_result.artifact_path: _run_backend_mode(
                mode_name=args.mode,
                videos=benchmark_videos,
                model_path=artifact_path,
                tracker_config=args.tracker_config,
                process_workers=int(args.process_workers),
            ),
            extra={
                "backend": backend,
                "artifact_path": export_result.artifact_path,
                "process_workers": int(args.process_workers) if args.mode == "process_parallel_models" else None,
            },
        )
        summary["results"].append(backend_result)

    pt_best = float(pt_result["best_elapsed_s"])
    for result in summary["results"]:
        result["speedup_vs_pt_best"] = pt_best / float(result["best_elapsed_s"]) if result["best_elapsed_s"] else None

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Tracking backend benchmark summary")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
