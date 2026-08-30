from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import ultralytics
from ultralytics.models.sam import SAM3SemanticPredictor

from cowbook.io.json_utils import dump_path_compact, dumps_pretty
from tools.benchmark_sam3_semantic_tracking import (
    Sam3FrameArtifacts,
    TrackingCleanupConfig,
    _build_tracking_document,
    _color_from_seed,
    _default_cleanup_config,
    _draw_processed_frames,
    _extract_frame_artifacts,
    _log_progress,
    _resolve_prompts,
    _select_cleanup_keep_indices,
    _subset_frame,
    _validate_model_path,
    _validate_videos,
)
from tools.benchmark_tracking import _probe_video_metadata, _query_gpu_info


def _default_videos() -> list[str]:
    return [
        "sample_data/videos/Ch1_60.mp4",
        "sample_data/videos/Ch4_60.mp4",
        "sample_data/videos/Ch6_60.mp4",
        "sample_data/videos/Ch8_60.mp4",
    ]


def _default_prompts() -> list[str]:
    return ["cow"]


@dataclass(slots=True)
class Sam3SparsePreviewRunResult:
    video_path: str
    image_dir: str
    tracking_json_path: str
    summary_json_path: str
    requested_sample_count: int
    sample_count: int
    frame_count: int
    fps: float
    width: int
    height: int
    prompts: list[str]
    model_path: str
    imgsz: int
    elapsed_s: float
    mean_instances_per_sample: float
    max_instances_per_sample: int
    sample_image_paths: list[str]


def _select_equally_spaced_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    """Pick `sample_count` frame indices evenly spaced across [0, frame_count - 1].

    Endpoints are inclusive, so a preview always includes the first and
    last frame of the clip. Positions are rounded to the nearest integer
    frame and deduplicated, so the result can have fewer than
    `sample_count` entries for a very short video or a requested count
    larger than the frame count -- that's expected, not an error.
    """
    if frame_count <= 0:
        return []
    if sample_count <= 1:
        return [0]
    positions = np.linspace(0, frame_count - 1, num=sample_count)
    return sorted({int(round(position)) for position in positions})


def _resolve_sample_count(*, frame_count: int, fps: float, sample_count: int, interval_seconds: float) -> int:
    if sample_count > 0:
        return sample_count
    duration_s = (frame_count / fps) if fps > 0 else 0.0
    return max(1, int(round(duration_s / interval_seconds)) + 1)


def _open_capture(video_path: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    return capture


def _seek_frame(capture: cv2.VideoCapture, frame_index: int) -> tuple[np.ndarray, int]:
    """Seek to `frame_index` and read it, reporting the frame actually landed on.

    OpenCV's frame-index seek isn't guaranteed exact on every codec/container
    -- it can snap to the nearest keyframe -- so this hands the landed index
    back to the caller instead of silently trusting the request.
    """
    capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ok, frame = capture.read()
    if not ok or frame is None:
        raise ValueError(f"Failed to read frame {frame_index}")
    landed_index = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
    return frame, landed_index


def _sparse_sample_color(_track_id: int | None, _class_id: int | None, index: int) -> tuple[int, int, int]:
    # Independent, untracked samples have no track identity to color by.
    # Seed directly off each detection's index within its own frame so
    # multiple animals in the same sample still render in distinct colors.
    return _color_from_seed(index)


def _run_sparse_preview_for_video(
    *,
    video_path: str,
    output_root: Path,
    predictor: SAM3SemanticPredictor,
    model_path: str,
    prompts: list[str],
    sample_count: int,
    interval_seconds: float,
    imgsz: int,
    cleanup_config: TrackingCleanupConfig,
    log_progress: bool,
) -> Sam3SparsePreviewRunResult:
    metadata = _probe_video_metadata(video_path)
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    frame_count = int(metadata["frame_count"])

    requested_sample_count = _resolve_sample_count(
        frame_count=frame_count, fps=fps, sample_count=sample_count, interval_seconds=interval_seconds
    )
    frame_indices = _select_equally_spaced_frame_indices(frame_count, requested_sample_count)

    stem = Path(video_path).stem
    image_dir = output_root / "images" / stem
    json_dir = output_root / "json"
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    tracking_json_path = json_dir / f"{stem}_sam3_sparse_tracking.json"
    summary_json_path = json_dir / f"{stem}_sam3_sparse_summary.json"

    capture = _open_capture(video_path)

    sample_frames: list[Sam3FrameArtifacts] = []
    sample_image_paths: list[str] = []
    total_instances = 0
    max_instances = 0
    start = time.perf_counter()
    _log_progress(
        log_progress,
        f"[sam3-sparse] start: {video_path} samples={len(frame_indices)} prompts={prompts}",
    )
    try:
        for sample_index, frame_index in enumerate(frame_indices):
            frame, landed_index = _seek_frame(capture, frame_index)
            if landed_index != frame_index:
                _log_progress(
                    log_progress,
                    f"[sam3-sparse] seek drift: requested={frame_index} landed={landed_index}",
                )

            results = predictor(source=frame, text=prompts)
            artifacts = _extract_frame_artifacts(landed_index, results[0])
            keep_indices, _removed_by_mask_fill = _select_cleanup_keep_indices(artifacts, cleanup_config)
            cleaned = _subset_frame(artifacts, keep_indices)
            sample_frames.append(cleaned)

            instance_count = int(cleaned.xyxy.shape[0])
            total_instances += instance_count
            max_instances = max(max_instances, instance_count)

            timestamp_s = (landed_index / fps) if fps > 0 else 0.0
            detailed_frame, _clean_frame = _draw_processed_frames(
                cleaned, prompts=prompts, color_for_detection=_sparse_sample_color
            )
            image_path = (
                image_dir / f"{stem}_sample{sample_index:04d}_frame{landed_index:07d}_t{timestamp_s:07.1f}s.jpg"
            )
            cv2.imwrite(str(image_path), detailed_frame)
            sample_image_paths.append(str(image_path))

            _log_progress(
                log_progress,
                f"[sam3-sparse] sample {sample_index + 1}/{len(frame_indices)} frame={landed_index} "
                f"t={timestamp_s:.1f}s instances={instance_count}",
            )
    finally:
        capture.release()

    elapsed_s = time.perf_counter() - start
    mean_instances = (total_instances / len(sample_frames)) if sample_frames else 0.0

    tracking_document = _build_tracking_document(sample_frames)
    dump_path_compact(tracking_json_path, tracking_document.to_dict())

    summary_payload: dict[str, Any] = {
        "video_path": video_path,
        "image_dir": str(image_dir),
        "tracking_json_path": str(tracking_json_path),
        "requested_sample_count": requested_sample_count,
        "sample_count": len(sample_frames),
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "prompts": prompts,
        "model_path": model_path,
        "imgsz": imgsz,
        "elapsed_s": elapsed_s,
        "mean_instances_per_sample": mean_instances,
        "max_instances_per_sample": max_instances,
        "sample_image_paths": sample_image_paths,
    }
    dump_path_compact(summary_json_path, summary_payload)
    _log_progress(log_progress, f"[sam3-sparse] done: {video_path} in {elapsed_s:.2f}s -> {image_dir}")

    return Sam3SparsePreviewRunResult(
        video_path=video_path,
        image_dir=str(image_dir),
        tracking_json_path=str(tracking_json_path),
        summary_json_path=str(summary_json_path),
        requested_sample_count=requested_sample_count,
        sample_count=len(sample_frames),
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        prompts=prompts,
        model_path=model_path,
        imgsz=imgsz,
        elapsed_s=elapsed_s,
        mean_instances_per_sample=mean_instances,
        max_instances_per_sample=max_instances,
        sample_image_paths=sample_image_paths,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cheap SAM3 sanity-check preview: run semantic segmentation "
            "independently on N equally spaced frames per video instead of "
            "tracking and rendering the whole thing."
        )
    )
    parser.add_argument("--videos", nargs="+", default=_default_videos(), help="Video paths to preview.")
    parser.add_argument(
        "--model-path",
        default="sam3.pt",
        help="Local SAM3 model weights path. Ultralytics docs require downloading `sam3.pt` manually first.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=_default_prompts(),
        help="Text prompts for semantic segmentation.",
    )
    parser.add_argument(
        "--output-root",
        default="var/previews/sam3_sparse",
        help="Directory where sample images and JSON summaries are written.",
    )
    sampling = parser.add_mutually_exclusive_group(required=True)
    sampling.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Exact number of equally spaced frames to sample per video.",
    )
    sampling.add_argument(
        "--interval-seconds",
        type=float,
        default=0.0,
        help=(
            "Approximate seconds between samples (e.g. 60 for one frame per "
            "minute); converted to an equally spaced sample count per video."
        ),
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold passed to the semantic predictor.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size forwarded to Ultralytics.")
    parser.add_argument("--device", default=None, help="Optional device override forwarded to Ultralytics.")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable half-precision inference.",
    )
    parser.add_argument(
        "--summary-name",
        default="summary.json",
        help="Top-level summary filename under --output-root.",
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
    if int(args.sample_count) < 0:
        raise ValueError("--sample-count must be >= 0.")
    if float(args.interval_seconds) < 0:
        raise ValueError("--interval-seconds must be >= 0.")
    if int(args.imgsz) < 1:
        raise ValueError("--imgsz must be >= 1.")

    videos = _validate_videos(args.videos)
    prompts = _resolve_prompts(args.prompts)
    model_path = _validate_model_path(args.model_path)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # One predictor, reused across every video and every sample: SAM3 weight
    # loading costs seconds, while a single-frame forward pass costs tens of
    # milliseconds, so reloading per video would dominate the runtime this
    # tool exists to avoid.
    predictor = SAM3SemanticPredictor(
        overrides={
            "conf": float(args.conf_threshold),
            "imgsz": int(args.imgsz),
            "task": "segment",
            "mode": "predict",
            "model": model_path,
            "save": False,
            "verbose": False,
            "device": args.device,
            "half": bool(args.half),
        }
    )
    cleanup_config = _default_cleanup_config()

    runtime_info = {
        "ultralytics_version": ultralytics.__version__,
        "gpu_info": _query_gpu_info(),
        "model_path": model_path,
        "device": args.device,
        "sam3_semantic_predictor": SAM3SemanticPredictor.__name__,
    }
    runs = [
        _run_sparse_preview_for_video(
            video_path=video_path,
            output_root=output_root,
            predictor=predictor,
            model_path=model_path,
            prompts=prompts,
            sample_count=int(args.sample_count),
            interval_seconds=float(args.interval_seconds),
            imgsz=int(args.imgsz),
            cleanup_config=cleanup_config,
            log_progress=bool(args.log_progress),
        )
        for video_path in videos
    ]

    summary_payload = {
        "tool": "preview_sam3_samples",
        "runtime": runtime_info,
        "runs": [asdict(run) for run in runs],
    }
    summary_path = output_root / str(args.summary_name)
    dump_path_compact(summary_path, summary_payload)
    print("SAM3 sparse preview summary")
    print(dumps_pretty(summary_payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
