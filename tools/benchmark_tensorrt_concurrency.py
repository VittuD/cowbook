from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tools.benchmark_tracking import (
    _default_tracker_config,
    _prepare_benchmark_videos,
    _query_gpu_info,
    _repeat_mode,
    _validate_videos,
)
from tools.benchmark_tracking_backends import (
    ExportResult,
    _export_model_artifact,
    _normalize_imgsz,
    _run_backend_mode,
    _warmup_backend,
)


def _mode_for_concurrency(value: int) -> tuple[str, int]:
    concurrency = int(value)
    if concurrency <= 0:
        raise ValueError("concurrency values must be positive integers")
    if concurrency == 1:
        return "sequential_shared_model", 1
    return "process_parallel_models", concurrency


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch vs TensorRT across tracking concurrency values.",
    )
    parser.add_argument("--videos", nargs="+", required=True, help="Video paths to benchmark together.")
    parser.add_argument(
        "--baseline-model-path",
        default="models/yolov11_best.pt",
        help="Baseline PyTorch model path.",
    )
    parser.add_argument(
        "--tracker-config",
        default=_default_tracker_config(),
        help="Path to the Ultralytics tracker YAML.",
    )
    parser.add_argument(
        "--concurrency-values",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="Tracking concurrency values to benchmark. Concurrency 1 uses the single-model path.",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=[448, 768],
        help="TensorRT export image size as one square size or two integers: height width.",
    )
    parser.add_argument("--device", default="0", help="Ultralytics export device argument.")
    parser.add_argument("--repeat", type=int, default=1, help="Measured runs per backend and concurrency.")
    parser.add_argument("--warmup-runs", type=int, default=0, help="Unmeasured warmup runs per backend and concurrency.")
    parser.add_argument(
        "--output",
        default="var/benchmarks/tensorrt_concurrency_benchmark.json",
        help="Path to write the JSON summary.",
    )
    parser.add_argument(
        "--export-dir",
        default="var/benchmarks/model_exports",
        help="Directory for the TensorRT engine artifact.",
    )
    parser.add_argument(
        "--engine-artifact-path",
        default=None,
        help="Optional prebuilt TensorRT engine path. When set, export is skipped.",
    )
    parser.add_argument(
        "--extend-seconds",
        type=int,
        default=0,
        help="If > 0, create longer benchmark copies of the input videos before running.",
    )
    parser.add_argument(
        "--prepared-video-dir",
        default="var/benchmarks/prepared_videos_tensorrt",
        help="Directory for prepared benchmark videos when --extend-seconds is used.",
    )
    parser.add_argument("--half", action="store_true", help="Export the TensorRT engine in fp16 when supported.")
    parser.add_argument("--dynamic", action="store_true", help="Export the TensorRT engine with dynamic shapes.")
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
        help="Rebuild the TensorRT engine even if it already exists.",
    )
    parser.add_argument(
        "--log-progress",
        action="store_true",
        help="Print coarse export and benchmark progress messages.",
    )
    return parser.parse_args()


def _benchmark_backend_for_concurrency(
    *,
    backend_label: str,
    model_path: str,
    concurrency: int,
    videos: list[str],
    tracker_config: str,
    repeat: int,
    warmup_runs: int,
    log_progress: bool,
) -> dict[str, Any]:
    mode_name, process_workers = _mode_for_concurrency(concurrency)
    _warmup_backend(
        backend_label=backend_label,
        mode_name=mode_name,
        process_workers=process_workers,
        warmup_runs=warmup_runs,
        videos=videos,
        model_path=model_path,
        tracker_config=tracker_config,
        log_progress=log_progress,
    )
    result = _repeat_mode(
        mode_name,
        repeat_count=repeat,
        runner=lambda: _run_backend_mode(
            mode_name=mode_name,
            videos=videos,
            model_path=model_path,
            tracker_config=tracker_config,
            process_workers=process_workers,
        ),
        extra={
            "backend": backend_label,
            "artifact_path": model_path,
            "requested_concurrency": concurrency,
            "effective_tracking_concurrency": process_workers,
            "process_workers": process_workers if mode_name == "process_parallel_models" else None,
        },
    )
    return result


def main() -> int:
    args = _parse_args()
    videos = _validate_videos(args.videos)
    prepared_videos, prepared_video_metadata = _prepare_benchmark_videos(
        videos,
        target_duration_seconds=args.extend_seconds,
        prepared_video_dir=args.prepared_video_dir,
        log_progress=args.log_progress,
    )
    benchmark_videos = prepared_videos or videos
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imgsz = _normalize_imgsz(list(args.imgsz))
    concurrency_values = [int(value) for value in args.concurrency_values]

    if args.engine_artifact_path:
        export_result = ExportResult(
            backend="engine",
            artifact_path=args.engine_artifact_path,
            reused_existing=True,
        )
    else:
        export_result = _export_model_artifact(
            source_model_path=args.baseline_model_path,
            backend="engine",
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

    summary: dict[str, Any] = {
        "source_videos": videos,
        "benchmark_videos": benchmark_videos,
        "video_count": len(benchmark_videos),
        "baseline_model_path": args.baseline_model_path,
        "tracker_config": args.tracker_config,
        "imgsz": list(imgsz),
        "device": args.device,
        "concurrency_values": concurrency_values,
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
        "exports": [asdict(export_result)],
        "results": [],
    }

    if export_result.error:
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("TensorRT concurrency benchmark summary")
        print(json.dumps(summary, indent=2))
        return 0

    assert export_result.artifact_path is not None
    for concurrency in concurrency_values:
        summary["results"].append(
            _benchmark_backend_for_concurrency(
                backend_label="pt",
                model_path=args.baseline_model_path,
                concurrency=concurrency,
                videos=benchmark_videos,
                tracker_config=args.tracker_config,
                repeat=args.repeat,
                warmup_runs=args.warmup_runs,
                log_progress=bool(args.log_progress),
            )
        )
        summary["results"].append(
            _benchmark_backend_for_concurrency(
                backend_label="engine",
                model_path=export_result.artifact_path,
                concurrency=concurrency,
                videos=benchmark_videos,
                tracker_config=args.tracker_config,
                repeat=args.repeat,
                warmup_runs=args.warmup_runs,
                log_progress=bool(args.log_progress),
            )
        )

    pt_best_by_concurrency = {
        int(result["requested_concurrency"]): float(result["best_elapsed_s"])
        for result in summary["results"]
        if result["backend"] == "pt"
    }
    for result in summary["results"]:
        baseline = pt_best_by_concurrency.get(int(result["requested_concurrency"]))
        result["speedup_vs_pt_same_concurrency"] = (
            baseline / float(result["best_elapsed_s"]) if baseline and result["best_elapsed_s"] else None
        )

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("TensorRT concurrency benchmark summary")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
