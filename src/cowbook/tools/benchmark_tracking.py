from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
from ultralytics import YOLO
from ultralytics.utils.ops import clean_str

from cowbook.core.runtime import assets_root


@dataclass(slots=True)
class BenchmarkModeResult:
    mode: str
    elapsed_s: float
    per_source_frame_count: dict[str, int]
    per_source_elapsed_s: dict[str, float] | None = None
    notes: list[str] | None = None


def _default_tracker_config() -> str:
    return str(assets_root() / "trackers" / "cows_botsort.yaml")


def _model_track_kwargs(tracker_config: str) -> dict[str, Any]:
    return {
        "save": False,
        "conf": 0.45,
        "iou": 0.5,
        "tracker": tracker_config,
        "verbose": False,
    }


def _count_frames_for_source(model: YOLO, video_path: str, tracker_config: str) -> tuple[int, float]:
    count = 0
    t0 = time.perf_counter()
    for _ in model.track(source=video_path, stream=True, **_model_track_kwargs(tracker_config)):
        count += 1
    return count, time.perf_counter() - t0


def _run_sequential_shared_model(
    *,
    videos: list[str],
    model_path: str,
    tracker_config: str,
) -> BenchmarkModeResult:
    model = YOLO(model_path)
    per_source_frame_count: dict[str, int] = {}
    per_source_elapsed_s: dict[str, float] = {}
    start = time.perf_counter()
    for video_path in videos:
        frame_count, elapsed_s = _count_frames_for_source(model, video_path, tracker_config)
        per_source_frame_count[video_path] = frame_count
        per_source_elapsed_s[video_path] = elapsed_s
    elapsed_s = time.perf_counter() - start
    return BenchmarkModeResult(
        mode="sequential_shared_model",
        elapsed_s=elapsed_s,
        per_source_frame_count=per_source_frame_count,
        per_source_elapsed_s=per_source_elapsed_s,
        notes=["One YOLO model instance is reused across videos, but only one video runs at a time."],
    )


def _run_multistream_shared_model(
    *,
    videos: list[str],
    model_path: str,
    tracker_config: str,
) -> BenchmarkModeResult:
    model = YOLO(model_path)
    sanitized_to_source = {clean_str(path).replace(os.sep, "_"): path for path in videos}
    per_source_frame_count = {path: 0 for path in videos}
    with tempfile.TemporaryDirectory(prefix="cowbook_streams_") as tmpdir:
        streams_file = Path(tmpdir) / "videos.streams"
        streams_file.write_text("\n".join(videos) + "\n", encoding="utf-8")
        start = time.perf_counter()
        results = model.track(
            source=str(streams_file),
            stream=True,
            stream_buffer=True,
            **_model_track_kwargs(tracker_config),
        )
        for result in results:
            source_key = sanitized_to_source.get(str(result.path), str(result.path))
            per_source_frame_count[source_key] = per_source_frame_count.get(source_key, 0) + 1
        elapsed_s = time.perf_counter() - start
    return BenchmarkModeResult(
        mode="shared_model_multistream",
        elapsed_s=elapsed_s,
        per_source_frame_count=per_source_frame_count,
        notes=[
            "One YOLO model instance tracks all sources in a single multi-stream call.",
            "Uses stream_buffer=True because offline file streams otherwise collapse to a single yielded frame.",
        ],
    )


def _parallel_worker(video_path: str, model_path: str, tracker_config: str) -> tuple[str, int, float]:
    try:
        import torch  # type: ignore

        torch.set_num_threads(1)
    except Exception:
        pass

    model = YOLO(model_path)
    frame_count = 0
    start = time.perf_counter()
    for _ in model.track(source=video_path, stream=True, **_model_track_kwargs(tracker_config)):
        frame_count += 1
    elapsed_s = time.perf_counter() - start
    return video_path, frame_count, elapsed_s


def _run_process_parallel_models(
    *,
    videos: list[str],
    model_path: str,
    tracker_config: str,
    worker_count: int,
) -> BenchmarkModeResult:
    worker_count = max(1, min(worker_count, len(videos)))
    start = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=worker_count) as pool:
        results = pool.starmap(
            _parallel_worker,
            [(video_path, model_path, tracker_config) for video_path in videos],
        )
    elapsed_s = time.perf_counter() - start
    per_source_frame_count = {video_path: frame_count for video_path, frame_count, _ in results}
    per_source_elapsed_s = {video_path: per_elapsed_s for video_path, _, per_elapsed_s in results}
    return BenchmarkModeResult(
        mode="process_parallel_models",
        elapsed_s=elapsed_s,
        per_source_frame_count=per_source_frame_count,
        per_source_elapsed_s=per_source_elapsed_s,
        notes=[
            f"Spawns {worker_count} worker process(es).",
            "Each worker loads its own YOLO model and CUDA context.",
        ],
    )


def _repeat_mode(
    mode_name: str,
    *,
    repeat_count: int,
    runner,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runs = [asdict(runner()) for _ in range(repeat_count)]
    elapsed_values = [float(run["elapsed_s"]) for run in runs]
    best_run = min(runs, key=lambda run: float(run["elapsed_s"]))
    result = {
        "mode": mode_name,
        "repeat_count": repeat_count,
        "best_elapsed_s": min(elapsed_values),
        "mean_elapsed_s": sum(elapsed_values) / len(elapsed_values),
        "runs": runs,
        "best_run": best_run,
    }
    if extra:
        result.update(extra)
    return result


def _query_gpu_info() -> list[str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _probe_video_metadata(video_path: str) -> dict[str, float | int]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()

    if fps <= 0:
        fps = 30.0

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0,
    }


def _load_video_frames(video_path: str) -> tuple[list[Any], float, tuple[int, int]]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    frames: list[Any] = []
    frame_size = (0, 0)
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_size == (0, 0):
                frame_size = (int(frame.shape[1]), int(frame.shape[0]))
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames read from video: {video_path}")

    return frames, fps, frame_size


def _build_extended_video(
    source_video: str,
    *,
    target_duration_seconds: int,
    output_dir: Path,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_video)
    output_path = output_dir / f"{source_path.stem}_{target_duration_seconds}s{source_path.suffix}"

    if output_path.exists():
        existing_duration = float(_probe_video_metadata(str(output_path))["duration_s"])
        if existing_duration >= float(target_duration_seconds) - 0.5:
            return str(output_path)

    frames, fps, frame_size = _load_video_frames(str(source_path))
    target_frame_count = max(1, int(round(float(target_duration_seconds) * fps)))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise ValueError(f"Failed to open output video for writing: {output_path}")

    try:
        for frame_idx in range(target_frame_count):
            writer.write(frames[frame_idx % len(frames)])
    finally:
        writer.release()

    return str(output_path)


def _prepare_benchmark_videos(
    videos: list[str],
    *,
    target_duration_seconds: int,
    prepared_video_dir: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    if target_duration_seconds <= 0:
        return videos, []

    output_dir = Path(prepared_video_dir)
    prepared_videos: list[str] = []
    prepared_metadata: list[dict[str, Any]] = []
    for video_path in videos:
        prepared_path = _build_extended_video(
            video_path,
            target_duration_seconds=target_duration_seconds,
            output_dir=output_dir,
        )
        prepared_videos.append(prepared_path)
        prepared_metadata.append(
            {
                "source_video": video_path,
                "prepared_video": prepared_path,
                "prepared_duration_s": _probe_video_metadata(prepared_path)["duration_s"],
            }
        )
    return prepared_videos, prepared_metadata


def _print_summary(summary: dict[str, Any]) -> None:
    print("Tracking benchmark summary")
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Cowbook tracking modes for multi-video GPU scheduling.",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Video paths to benchmark together.",
    )
    parser.add_argument(
        "--model-path",
        default="models/yolov11_best.pt",
        help="Path to Ultralytics model weights.",
    )
    parser.add_argument(
        "--tracker-config",
        default=_default_tracker_config(),
        help="Path to the Ultralytics tracker YAML.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "sequential_shared_model",
            "shared_model_multistream",
            "process_parallel_models",
        ],
        choices=[
            "sequential_shared_model",
            "shared_model_multistream",
            "process_parallel_models",
        ],
        help="Benchmark modes to run.",
    )
    parser.add_argument(
        "--process-workers",
        type=int,
        default=0,
        help="Worker count for process_parallel_models. Defaults to the number of videos.",
    )
    parser.add_argument(
        "--process-worker-sweep",
        nargs="+",
        type=int,
        default=None,
        help="If provided, run process_parallel_models once for each listed worker count.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to run each mode.",
    )
    parser.add_argument(
        "--output",
        default="var/benchmarks/tracking_benchmark.json",
        help="Path to write the JSON benchmark summary.",
    )
    parser.add_argument(
        "--extend-seconds",
        type=int,
        default=0,
        help="If > 0, create longer benchmark copies of the input videos with ffmpeg before running.",
    )
    parser.add_argument(
        "--prepared-video-dir",
        default="var/benchmarks/prepared_videos",
        help="Directory for ffmpeg-generated benchmark videos when --extend-seconds is used.",
    )
    return parser.parse_args()


def _validate_videos(video_paths: Iterable[str]) -> list[str]:
    normalized = [str(Path(path)) for path in video_paths]
    missing = [path for path in normalized if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing benchmark video(s): {missing}")
    return normalized


def main() -> int:
    args = _parse_args()
    videos = _validate_videos(args.videos)
    prepared_videos, prepared_video_metadata = _prepare_benchmark_videos(
        videos,
        target_duration_seconds=args.extend_seconds,
        prepared_video_dir=args.prepared_video_dir,
    )
    benchmark_videos = prepared_videos or videos
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process_workers = args.process_workers or len(benchmark_videos)
    process_worker_sweep = [
        worker_count
        for worker_count in (args.process_worker_sweep or [])
        if worker_count > 0
    ]

    summary: dict[str, Any] = {
        "source_videos": videos,
        "videos": benchmark_videos,
        "video_count": len(benchmark_videos),
        "model_path": args.model_path,
        "tracker_config": args.tracker_config,
        "extend_seconds": args.extend_seconds,
        "prepared_video_dir": args.prepared_video_dir if args.extend_seconds > 0 else None,
        "prepared_videos": prepared_video_metadata,
        "process_worker_sweep": process_worker_sweep or None,
        "gpu_info": _query_gpu_info(),
        "results": [],
    }

    for mode_name in args.modes:
        if mode_name == "sequential_shared_model":
            mode_result = _repeat_mode(
                mode_name,
                repeat_count=args.repeat,
                runner=lambda: _run_sequential_shared_model(
                    videos=benchmark_videos,
                    model_path=args.model_path,
                    tracker_config=args.tracker_config,
                ),
            )
        elif mode_name == "shared_model_multistream":
            mode_result = _repeat_mode(
                mode_name,
                repeat_count=args.repeat,
                runner=lambda: _run_multistream_shared_model(
                    videos=benchmark_videos,
                    model_path=args.model_path,
                    tracker_config=args.tracker_config,
                ),
            )
        elif mode_name == "process_parallel_models":
            worker_counts = process_worker_sweep or [process_workers]
            for worker_count in worker_counts:
                effective_worker_count = max(1, min(worker_count, len(benchmark_videos)))
                mode_result = _repeat_mode(
                    mode_name,
                    repeat_count=args.repeat,
                    runner=lambda worker_count=effective_worker_count: _run_process_parallel_models(
                        videos=benchmark_videos,
                        model_path=args.model_path,
                        tracker_config=args.tracker_config,
                        worker_count=worker_count,
                    ),
                    extra={
                        "requested_worker_count": worker_count,
                        "effective_worker_count": effective_worker_count,
                    },
                )
                summary["results"].append(mode_result)
            continue
        else:  # pragma: no cover
            raise ValueError(f"Unsupported benchmark mode: {mode_name}")
        summary["results"].append(mode_result)

    sequential = next(
        (result for result in summary["results"] if result["mode"] == "sequential_shared_model"),
        None,
    )
    if sequential is not None:
        baseline = float(sequential["best_elapsed_s"])
        for result in summary["results"]:
            result["speedup_vs_sequential_best"] = (
                baseline / float(result["best_elapsed_s"]) if result["best_elapsed_s"] else None
            )
    elif summary["results"]:
        best_overall = min(summary["results"], key=lambda result: float(result["best_elapsed_s"]))
        baseline = float(best_overall["best_elapsed_s"])
        summary["best_mode"] = {
            "mode": best_overall["mode"],
            "best_elapsed_s": best_overall["best_elapsed_s"],
            "requested_worker_count": best_overall.get("requested_worker_count"),
            "effective_worker_count": best_overall.get("effective_worker_count"),
        }
        for result in summary["results"]:
            result["speedup_vs_best_mode"] = (
                baseline / float(result["best_elapsed_s"]) if result["best_elapsed_s"] else None
            )

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
