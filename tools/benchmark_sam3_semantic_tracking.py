from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import ultralytics
from ultralytics.models.sam import SAM3VideoSemanticPredictor

from cowbook.io.json_utils import dump_path_compact, dumps_pretty
from tools.benchmark_tracking import _prepare_benchmark_videos, _probe_video_metadata, _query_gpu_info


@dataclass(slots=True)
class Sam3VideoRunResult:
    video_path: str
    annotated_video_path: str
    summary_json_path: str
    elapsed_s: float
    frame_count: int
    fps: float
    width: int
    height: int
    prompts: list[str]
    model_path: str
    mean_instances_per_frame: float
    max_instances_per_frame: int
    tracked_object_ids: list[int]
    dump_frame_metadata: bool


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


def _open_video_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(frame_size[0]), int(frame_size[1])),
    )
    if not writer.isOpened():
        raise ValueError(f"Failed to open output video for writing: {path}")
    return writer


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


def _extract_confidences(boxes) -> list[float]:
    if boxes is None:
        return []
    confidences = getattr(boxes, "conf", None)
    if confidences is None:
        return []
    if hasattr(confidences, "tolist"):
        values = confidences.tolist()
    else:
        values = list(confidences)
    return [float(value) for value in values]


def _frame_summary(frame_index: int, result) -> dict[str, Any]:
    boxes = getattr(result, "boxes", None)
    object_ids = _extract_object_ids(boxes)
    confidences = _extract_confidences(boxes)
    names = getattr(result, "names", {}) or {}

    class_ids: list[int] = []
    if boxes is not None and getattr(boxes, "cls", None) is not None:
        cls_values = boxes.cls.tolist() if hasattr(boxes.cls, "tolist") else list(boxes.cls)
        class_ids = [int(value) for value in cls_values]

    class_names = [str(names.get(class_id, class_id)) for class_id in class_ids]
    return {
        "frame_index": frame_index,
        "instance_count": len(object_ids) if object_ids else (len(boxes) if boxes is not None else 0),
        "object_ids": object_ids,
        "confidences": confidences,
        "class_ids": class_ids,
        "class_names": class_names,
    }


def _overlay_prompt_text(image: np.ndarray, prompts: list[str]) -> np.ndarray:
    annotated = image.copy()
    label = f"SAM3 prompts: {', '.join(prompts)}"
    cv2.rectangle(annotated, (8, 8), (min(annotated.shape[1] - 8, 12 + 9 * len(label)), 40), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        label,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _collect_runtime_info(model_path: str, device: str | None) -> dict[str, Any]:
    return {
        "ultralytics_version": ultralytics.__version__,
        "gpu_info": _query_gpu_info(),
        "model_path": model_path,
        "device": device,
        "sam3_semantic_predictor": SAM3VideoSemanticPredictor.__name__,
    }


def _run_semantic_tracking_for_video(
    *,
    video_path: str,
    output_root: Path,
    prompts: list[str],
    model_path: str,
    conf_threshold: float,
    device: str | None,
    half: bool,
    dump_frame_metadata: bool,
    log_progress: bool,
) -> Sam3VideoRunResult:
    metadata = _probe_video_metadata(video_path)
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])

    json_dir = output_root / "json"
    video_dir = output_root / "videos"
    json_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    stem = _artifact_stem(video_path)
    annotated_video_path = video_dir / f"{stem}_sam3_annotated.mp4"
    summary_json_path = json_dir / f"{stem}_sam3_summary.json"

    predictor = SAM3VideoSemanticPredictor(
        overrides={
            "conf": conf_threshold,
            "task": "segment",
            "mode": "predict",
            "model": model_path,
            "save": False,
            "verbose": False,
            "device": device,
            "half": half,
            "show_boxes": True,
            "show_labels": True,
            "show_conf": True,
        }
    )

    writer = _open_video_writer(annotated_video_path, fps=fps, frame_size=(width, height))
    frame_summaries: list[dict[str, Any]] = []
    tracked_ids: set[int] = set()
    frame_count = 0
    total_instances = 0
    max_instances = 0
    start = time.perf_counter()
    _log_progress(log_progress, f"[sam3] start: {video_path} prompts={prompts}")

    try:
        for frame_index, result in enumerate(
            predictor(source=video_path, model=model_path, text=prompts, stream=True)
        ):
            summary = _frame_summary(frame_index, result)
            tracked_ids.update(summary["object_ids"])
            total_instances += int(summary["instance_count"])
            max_instances = max(max_instances, int(summary["instance_count"]))
            if dump_frame_metadata:
                frame_summaries.append(summary)

            plotted = result.plot()
            annotated = _overlay_prompt_text(plotted, prompts)
            writer.write(annotated)
            frame_count += 1
    finally:
        writer.release()

    elapsed_s = time.perf_counter() - start
    mean_instances = (total_instances / frame_count) if frame_count else 0.0

    summary_payload: dict[str, Any] = {
        "video_path": video_path,
        "annotated_video_path": str(annotated_video_path),
        "elapsed_s": elapsed_s,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "prompts": prompts,
        "model_path": model_path,
        "mean_instances_per_frame": mean_instances,
        "max_instances_per_frame": max_instances,
        "tracked_object_ids": sorted(tracked_ids),
        "dump_frame_metadata": dump_frame_metadata,
    }
    if dump_frame_metadata:
        summary_payload["frames"] = frame_summaries

    dump_path_compact(summary_json_path, summary_payload)
    _log_progress(log_progress, f"[sam3] done: {video_path} in {elapsed_s:.2f}s -> {annotated_video_path}")
    return Sam3VideoRunResult(
        video_path=video_path,
        annotated_video_path=str(annotated_video_path),
        summary_json_path=str(summary_json_path),
        elapsed_s=elapsed_s,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        prompts=prompts,
        model_path=model_path,
        mean_instances_per_frame=mean_instances,
        max_instances_per_frame=max_instances,
        tracked_object_ids=sorted(tracked_ids),
        dump_frame_metadata=dump_frame_metadata,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ultralytics SAM3 semantic video tracking for all-instance concept prompts.",
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
        help="Text prompts for all-instance semantic tracking.",
    )
    parser.add_argument(
        "--output-root",
        default="var/benchmarks/sam3_semantic_tracking",
        help="Directory where annotated videos and JSON summaries are written.",
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
        default="var/benchmarks/prepared_videos_sam3",
        help="Directory for prepared benchmark videos when --extend-seconds is used.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold passed to the semantic predictor.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override forwarded to Ultralytics.",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable half-precision inference.",
    )
    parser.add_argument(
        "--dump-frame-metadata",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include per-frame object summaries in the per-video JSON output.",
    )
    parser.add_argument(
        "--log-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable coarse progress logs for console and swarm logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    videos = _validate_videos(args.videos)
    prompts = _resolve_prompts(args.prompts)
    model_path = _validate_model_path(args.model_path)
    prepared_videos, prepared_metadata = _prepare_benchmark_videos(
        videos,
        target_duration_seconds=args.extend_seconds,
        prepared_video_dir=args.prepared_video_dir,
        log_progress=bool(args.log_progress),
    )
    benchmark_videos = prepared_videos or videos

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    runtime_info = _collect_runtime_info(model_path, args.device)
    runs = [
        _run_semantic_tracking_for_video(
            video_path=video_path,
            output_root=output_root,
            prompts=prompts,
            model_path=model_path,
            conf_threshold=float(args.conf_threshold),
            device=args.device,
            half=bool(args.half),
            dump_frame_metadata=bool(args.dump_frame_metadata),
            log_progress=bool(args.log_progress),
        )
        for video_path in benchmark_videos
    ]

    summary = {
        "tool": "benchmark_sam3_semantic_tracking",
        "runtime": runtime_info,
        "videos": benchmark_videos,
        "prompts": prompts,
        "prepared_video_metadata": prepared_metadata,
        "runs": [asdict(run) for run in runs],
    }
    summary_path = output_root / args.summary_name
    dump_path_compact(summary_path, summary)
    if args.log_progress:
        print("SAM3 semantic tracking summary")
        print(dumps_pretty(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
