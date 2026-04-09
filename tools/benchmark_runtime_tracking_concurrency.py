from __future__ import annotations

import argparse
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cowbook.io.json_utils import dump_path_compact, dumps_pretty, load_path
from cowbook.workflows import group_processor as group_processor_module
from tools.benchmark_tracking import _prepare_benchmark_videos, _query_gpu_info, _repeat_mode
from tools.benchmark_tracking_backends import ExportResult, _export_model_artifact, _normalize_imgsz


@dataclass(slots=True)
class RuntimeTrackingRunResult:
    mode: str
    elapsed_s: float
    per_source_frame_count: dict[str, int]
    tracking_error_count: int
    tracking_errors: list[str]
    tracked_output_jsons: list[str]
    notes: list[str] | None = None


def _default_videos() -> list[str]:
    return [
        "sample_data/videos/Ch1_60.mp4",
        "sample_data/videos/Ch4_60.mp4",
        "sample_data/videos/Ch6_60.mp4",
        "sample_data/videos/Ch8_60.mp4",
    ]


def _validate_videos(video_paths: list[str]) -> list[str]:
    normalized = [str(Path(path)) for path in video_paths]
    missing = [path for path in normalized if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing runtime benchmark video(s): {missing}")
    return normalized


def _infer_camera_nr(video_path: str) -> int:
    match = re.search(r"(?i)ch(\d+)", Path(video_path).name)
    if match is None:
        raise ValueError(
            f"Could not infer camera number from video path: {video_path}. "
            "Provide --camera-nrs explicitly."
        )
    return int(match.group(1))


def _resolve_camera_nrs(video_paths: list[str], requested_camera_nrs: list[int] | None) -> list[int]:
    if requested_camera_nrs is not None:
        if len(requested_camera_nrs) != len(video_paths):
            raise ValueError("--camera-nrs must match the number of benchmark videos.")
        return [int(camera_nr) for camera_nr in requested_camera_nrs]
    return [_infer_camera_nr(video_path) for video_path in video_paths]


def _count_frames_from_tracking_json(output_json_path: str) -> int:
    document = load_path(output_json_path)
    return len(document.get("frames", []))


def _video_group(video_paths: list[str], camera_nrs: list[int]) -> list[dict[str, Any]]:
    return [
        {"path": video_path, "camera_nr": camera_nr}
        for video_path, camera_nr in zip(video_paths, camera_nrs)
    ]


def _tracking_run(
    *,
    videos: list[str],
    camera_nrs: list[int],
    model_path: str,
    concurrency: int,
    output_json_folder: str,
    log_progress: bool,
) -> RuntimeTrackingRunResult:
    config = {
        "model_path": model_path,
        "tracking_concurrency": int(concurrency),
        "save_tracking_video": False,
        "log_progress": bool(log_progress),
    }
    source_entries, tasks, precomputed_json_count = group_processor_module._collect_source_entries_and_tracking_tasks(
        _video_group(videos, camera_nrs),
        model_ref=model_path,
        config=config,
        output_json_folder=output_json_folder,
        reporter=None,
        group_idx=1,
    )
    if source_entries or precomputed_json_count:
        raise ValueError("Runtime tracking benchmark expects video inputs only.")

    start = time.perf_counter()
    tracked_source_entries, tracking_errors = group_processor_module._run_tracking_tasks(
        tasks,
        config=config,
        reporter=None,
        group_idx=1,
        precomputed_json_count=precomputed_json_count,
        cancellation_token=None,
    )
    elapsed_s = time.perf_counter() - start

    per_source_frame_count = {
        output_json_path: _count_frames_from_tracking_json(output_json_path)
        for output_json_path, _camera_nr in tracked_source_entries
    }
    return RuntimeTrackingRunResult(
        mode="group_processor_tracking",
        elapsed_s=elapsed_s,
        per_source_frame_count=per_source_frame_count,
        tracking_error_count=len(tracking_errors),
        tracking_errors=tracking_errors,
        tracked_output_jsons=[path for path, _camera_nr in tracked_source_entries],
        notes=[
            "Exercises cowbook.workflows.group_processor._run_tracking_tasks.",
            "Concurrency 1 uses the inline path; higher values use the multiprocessing pool path.",
        ],
    )


def _warmup_runtime_tracking(
    *,
    backend_label: str,
    concurrency: int,
    warmup_runs: int,
    videos: list[str],
    camera_nrs: list[int],
    model_path: str,
    warmup_root: Path,
    log_progress: bool,
) -> None:
    for idx in range(warmup_runs):
        run_dir = warmup_root / backend_label / f"concurrency_{concurrency}" / f"warmup_{idx + 1}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if log_progress:
            print(
                f"[warmup] backend={backend_label} concurrency={concurrency} run={idx + 1}/{warmup_runs}",
                flush=True,
            )
        _tracking_run(
            videos=videos,
            camera_nrs=camera_nrs,
            model_path=model_path,
            concurrency=concurrency,
            output_json_folder=str(run_dir),
            log_progress=log_progress,
        )


def _benchmark_backend_for_concurrency(
    *,
    backend_label: str,
    model_path: str,
    concurrency: int,
    videos: list[str],
    camera_nrs: list[int],
    repeat: int,
    warmup_runs: int,
    output_root: Path,
    log_progress: bool,
) -> dict[str, Any]:
    _warmup_runtime_tracking(
        backend_label=backend_label,
        concurrency=concurrency,
        warmup_runs=warmup_runs,
        videos=videos,
        camera_nrs=camera_nrs,
        model_path=model_path,
        warmup_root=output_root / "warmups",
        log_progress=log_progress,
    )

    run_counter = {"value": 0}

    def runner() -> dict[str, Any]:
        run_counter["value"] += 1
        run_dir = output_root / "runs" / backend_label / f"concurrency_{concurrency}" / f"run_{run_counter['value']}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return _tracking_run(
            videos=videos,
            camera_nrs=camera_nrs,
            model_path=model_path,
            concurrency=concurrency,
            output_json_folder=str(run_dir),
            log_progress=log_progress,
        )

    return _repeat_mode(
        "group_processor_tracking",
        repeat_count=repeat,
        runner=runner,
        extra={
            "backend": backend_label,
            "artifact_path": model_path,
            "requested_concurrency": int(concurrency),
            "effective_tracking_concurrency": min(int(concurrency), len(videos)),
            "process_workers": min(int(concurrency), len(videos)) if int(concurrency) > 1 else None,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Cowbook's runtime tracking path across TensorRT concurrency values.",
    )
    parser.add_argument("--videos", nargs="+", default=_default_videos(), help="Video paths to benchmark together.")
    parser.add_argument(
        "--camera-nrs",
        nargs="+",
        type=int,
        default=None,
        help="Optional camera numbers matching --videos order after preparation.",
    )
    parser.add_argument(
        "--baseline-model-path",
        default="models/yolov11_best.pt",
        help="Baseline PyTorch model path.",
    )
    parser.add_argument(
        "--concurrency-values",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="Tracking concurrency values to benchmark.",
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
        default="var/benchmarks/runtime_tracking_concurrency.json",
        help="Path to write the JSON summary.",
    )
    parser.add_argument(
        "--output-root",
        default="var/benchmarks/runtime_tracking_concurrency",
        help="Directory where per-run tracking JSON outputs are written.",
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
        default=300,
        help="If > 0, create longer benchmark copies of the input videos before running.",
    )
    parser.add_argument(
        "--prepared-video-dir",
        default="var/benchmarks/prepared_videos_runtime_tracking",
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
        help="Print coarse benchmark progress messages.",
    )
    return parser.parse_args()


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
    camera_nrs = _resolve_camera_nrs(benchmark_videos, args.camera_nrs)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
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
        "output_root": str(output_root),
        "gpu_info": _query_gpu_info(),
        "exports": [asdict(export_result)],
        "results": [],
    }

    if export_result.error:
        dump_path_compact(output_path, summary)
        print("Runtime tracking concurrency benchmark summary")
        print(dumps_pretty(summary).decode("utf-8"))
        return 0

    assert export_result.artifact_path is not None
    for concurrency in concurrency_values:
        summary["results"].append(
            _benchmark_backend_for_concurrency(
                backend_label="pt",
                model_path=args.baseline_model_path,
                concurrency=concurrency,
                videos=benchmark_videos,
                camera_nrs=camera_nrs,
                repeat=args.repeat,
                warmup_runs=args.warmup_runs,
                output_root=output_root,
                log_progress=bool(args.log_progress),
            )
        )
        summary["results"].append(
            _benchmark_backend_for_concurrency(
                backend_label="engine",
                model_path=export_result.artifact_path,
                concurrency=concurrency,
                videos=benchmark_videos,
                camera_nrs=camera_nrs,
                repeat=args.repeat,
                warmup_runs=args.warmup_runs,
                output_root=output_root,
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

    dump_path_compact(output_path, summary)
    print("Runtime tracking concurrency benchmark summary")
    print(dumps_pretty(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
