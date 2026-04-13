from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import ultralytics
from ultralytics.models.sam import SAM3VideoSemanticPredictor

from cowbook.io.json_utils import dump_path_compact, dumps_pretty


@dataclass(slots=True)
class Sam3RawVideoRunResult:
    video_path: str
    summary_json_path: str
    elapsed_s: float
    frame_count: int
    fps: float
    width: int
    height: int
    prompts: list[str]
    model_path: str
    imgsz: int
    mean_instances_per_frame: float
    max_instances_per_frame: int
    tracked_object_ids: list[int]
    max_frames: int


def _default_videos() -> list[str]:
    return [
        "sample_data/videos/Ch1_60.mp4",
        "sample_data/videos/Ch4_60.mp4",
        "sample_data/videos/Ch6_60.mp4",
        "sample_data/videos/Ch8_60.mp4",
    ]


def _default_prompts() -> list[str]:
    return ["cow"]


def _log_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


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
    log_progress: bool = False,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_video)
    output_path = output_dir / f"{source_path.stem}_{target_duration_seconds}s{source_path.suffix}"

    if output_path.exists():
        existing_duration = float(_probe_video_metadata(str(output_path))["duration_s"])
        if existing_duration >= float(target_duration_seconds) - 0.5:
            _log_progress(
                log_progress,
                f"[prepare] reusing existing extended video: {output_path} ({existing_duration:.1f}s)",
            )
            return str(output_path)

    _log_progress(
        log_progress,
        f"[prepare] building extended video for {source_path} -> {output_path} ({target_duration_seconds}s)",
    )
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

    _log_progress(
        log_progress,
        f"[prepare] wrote extended video: {output_path} ({target_frame_count} frames at {fps:.2f} FPS)",
    )
    return str(output_path)


def _prepare_benchmark_videos(
    videos: list[str],
    *,
    target_duration_seconds: int,
    prepared_video_dir: str,
    log_progress: bool = False,
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
            log_progress=log_progress,
        )
        prepared_videos.append(prepared_path)
        prepared_metadata.append(
            {
                "source_video": video_path,
                "prepared_video": prepared_path,
                "prepared_duration_s": _probe_video_metadata(prepared_path)["duration_s"],
                "preparation_method": "opencv_repeat",
            }
        )
    return prepared_videos, prepared_metadata


def _validate_videos(video_paths: Iterable[str]) -> list[str]:
    normalized = [str(Path(path)) for path in video_paths]
    missing = [path for path in normalized if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing SAM3 benchmark video(s): {missing}")
    return normalized


def _resolve_prompts(prompts: Iterable[str] | None) -> list[str]:
    if prompts is None:
        return _default_prompts()
    resolved = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
    if not resolved:
        raise ValueError("At least one non-empty prompt is required.")
    return resolved


def _validate_model_path(model_path: str) -> str:
    candidate = Path(model_path)
    if candidate.suffix == ".pt" or candidate.parent != Path("."):
        if not candidate.exists():
            raise FileNotFoundError(
                "SAM3 weights are not auto-downloaded by Ultralytics. "
                f"Expected local weights at: {candidate}. Download `sam3.pt` first or pass a valid full path."
            )
    return model_path


def _artifact_stem(video_path: str) -> str:
    return Path(video_path).stem


def _extract_object_ids(boxes) -> list[int]:
    if boxes is None:
        return []
    ids = getattr(boxes, "id", None)
    if ids is None:
        return []
    if hasattr(ids, "tolist"):
        values = ids.tolist()
    else:
        values = list(ids)
    return [int(value) for value in values]


def _run_raw_tracking_for_video(
    *,
    video_path: str,
    output_root: Path,
    prompts: list[str],
    model_path: str,
    conf_threshold: float,
    imgsz: int,
    device: str | None,
    half: bool,
    max_frames: int,
    log_progress: bool,
    log_every_frames: int,
) -> Sam3RawVideoRunResult:
    metadata = _probe_video_metadata(video_path)
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    expected_frame_count = int(metadata["frame_count"])
    expected_logged_frame_count = min(expected_frame_count, max_frames) if max_frames > 0 else expected_frame_count

    json_dir = output_root / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(video_path)
    summary_json_path = json_dir / f"{stem}_sam3_raw_summary.json"

    predictor = SAM3VideoSemanticPredictor(
        overrides={
            "conf": conf_threshold,
            "imgsz": imgsz,
            "task": "segment",
            "mode": "predict",
            "model": model_path,
            "save": False,
            "verbose": False,
            "device": device,
            "half": half,
            "show_boxes": False,
            "show_labels": False,
            "show_conf": False,
        }
    )

    tracked_ids: set[int] = set()
    frame_count = 0
    total_instances = 0
    max_instances = 0
    _log_progress(log_progress, f"[sam3-raw] start: {video_path} prompts={prompts}")
    start = time.perf_counter()
    for frame_index, result in enumerate(predictor(source=video_path, text=prompts, stream=True)):
        object_ids = _extract_object_ids(getattr(result, "boxes", None))
        instance_count = len(object_ids) if object_ids else len(getattr(result, "boxes", []) or [])
        tracked_ids.update(object_ids)
        total_instances += int(instance_count)
        max_instances = max(max_instances, int(instance_count))
        frame_count += 1
        if log_progress and log_every_frames > 0:
            if frame_index == 0 or ((frame_index + 1) % log_every_frames == 0) or ((frame_index + 1) == expected_logged_frame_count):
                print(
                    f"[sam3-raw] progress: {video_path} frame={frame_index + 1}/{expected_logged_frame_count} instances={instance_count}",
                    flush=True,
                )
        if max_frames > 0 and frame_count >= max_frames:
            break
    elapsed_s = time.perf_counter() - start
    mean_instances = (total_instances / frame_count) if frame_count else 0.0

    summary_payload = {
        "video_path": video_path,
        "elapsed_s": elapsed_s,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "prompts": prompts,
        "model_path": model_path,
        "imgsz": imgsz,
        "max_frames": max_frames,
        "mean_instances_per_frame": mean_instances,
        "max_instances_per_frame": max_instances,
        "tracked_object_ids": sorted(tracked_ids),
        "effective_fps": (frame_count / elapsed_s) if elapsed_s > 0 else 0.0,
    }
    dump_path_compact(summary_json_path, summary_payload)
    _log_progress(log_progress, f"[sam3-raw] done: {video_path} in {elapsed_s:.2f}s -> {summary_json_path}")
    return Sam3RawVideoRunResult(
        video_path=video_path,
        summary_json_path=str(summary_json_path),
        elapsed_s=elapsed_s,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        prompts=prompts,
        model_path=model_path,
        imgsz=imgsz,
        mean_instances_per_frame=mean_instances,
        max_instances_per_frame=max_instances,
        tracked_object_ids=sorted(tracked_ids),
        max_frames=max_frames,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw Ultralytics SAM3 semantic video tracking for timing benchmarks.",
    )
    parser.add_argument("--videos", nargs="+", default=_default_videos(), help="Video paths to run.")
    parser.add_argument(
        "--model-path",
        default="sam3.pt",
        help="Local SAM3 model weights path. Ultralytics docs require downloading `sam3.pt` manually first.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=_default_prompts(),
        help="Text prompts forwarded to SAM3 semantic video mode.",
    )
    parser.add_argument(
        "--output-root",
        default="var/benchmarks/sam3_raw",
        help="Directory where JSON summaries are written.",
    )
    parser.add_argument(
        "--summary-name",
        default="summary.json",
        help="Top-level summary filename under --output-root.",
    )
    parser.add_argument(
        "--extend-seconds",
        type=int,
        default=0,
        help="If > 0, create longer benchmark copies of the input videos before running.",
    )
    parser.add_argument(
        "--prepared-video-dir",
        default="var/benchmarks/prepared_videos_sam3_raw",
        help="Directory for prepared benchmark videos when --extend-seconds is used.",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold forwarded to SAM3.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size forwarded to SAM3.")
    parser.add_argument("--device", default=None, help="Optional device string passed to Ultralytics.")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable fp16 inference where supported.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If > 0, process at most this many frames from each input video.",
    )
    parser.add_argument(
        "--log-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable coarse progress logs for console and swarm logs.",
    )
    parser.add_argument(
        "--log-every-frames",
        type=int,
        default=25,
        help="Log every N frames when --log-progress is enabled.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if int(args.imgsz) < 1:
        raise ValueError("--imgsz must be >= 1.")
    if int(args.max_frames) < 0:
        raise ValueError("--max-frames must be >= 0.")
    if int(args.log_every_frames) < 1:
        raise ValueError("--log-every-frames must be >= 1.")

    videos = _validate_videos(args.videos)
    prompts = _resolve_prompts(args.prompts)
    model_path = _validate_model_path(args.model_path)

    prepared_videos, prepared_video_metadata = _prepare_benchmark_videos(
        videos,
        target_duration_seconds=args.extend_seconds,
        prepared_video_dir=args.prepared_video_dir,
        log_progress=bool(args.log_progress),
    )
    benchmark_videos = prepared_videos or videos

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    runtime_info = {
        "ultralytics_version": ultralytics.__version__,
        "gpu_info": _query_gpu_info(),
        "model_path": model_path,
        "device": args.device,
        "sam3_semantic_predictor": SAM3VideoSemanticPredictor.__name__,
    }
    runs = [
        _run_raw_tracking_for_video(
            video_path=video_path,
            output_root=output_root,
            prompts=prompts,
            model_path=model_path,
            conf_threshold=float(args.conf_threshold),
            imgsz=int(args.imgsz),
            device=args.device,
            half=bool(args.half),
            max_frames=int(args.max_frames),
            log_progress=bool(args.log_progress),
            log_every_frames=int(args.log_every_frames),
        )
        for video_path in benchmark_videos
    ]

    summary_payload = {
        "tool": "benchmark_sam3_raw",
        "runtime": runtime_info,
        "prepared_videos": prepared_video_metadata,
        "runs": [asdict(run) for run in runs],
    }
    summary_path = output_root / str(args.summary_name)
    dump_path_compact(summary_path, summary_payload)
    print("SAM3 raw benchmark summary")
    print(dumps_pretty(summary_payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
