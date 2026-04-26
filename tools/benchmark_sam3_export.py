from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import ultralytics
from ultralytics.models.sam import SAM3VideoSemanticPredictor

from cowbook.core.contracts import Detections, TrackingDocument, TrackingFrame, TrackingLabel
from cowbook.core.transforms import centroid_from_xyxy
from cowbook.io.json_utils import dump_path_compact, dumps_pretty
from tools.benchmark_sam3_raw import (
    _artifact_stem,
    _default_prompts,
    _default_videos,
    _extract_object_ids,
    _log_progress,
    _prepare_benchmark_videos,
    _probe_video_metadata,
    _query_gpu_info,
    _resolve_prompts,
    _validate_model_path,
    _validate_videos,
)


@dataclass(slots=True)
class Sam3ExportVideoRunResult:
    video_path: str
    tracking_json_path: str
    masks_dir: str
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


@dataclass(slots=True)
class Sam3ExportFrameArtifacts:
    frame_index: int
    path: str
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray
    object_ids: np.ndarray
    masks: np.ndarray


def _extract_frame_artifacts(frame_index: int, result) -> Sam3ExportFrameArtifacts:
    boxes = getattr(result, "boxes", None)
    xyxy = np.zeros((0, 4), dtype=np.float32)
    conf = np.zeros((0,), dtype=np.float32)
    cls = np.zeros((0,), dtype=np.int32)
    object_ids = np.zeros((0,), dtype=np.int32)
    if boxes is not None:
        xyxy = np.asarray(boxes.xyxy.cpu().numpy(), dtype=np.float32)
        if getattr(boxes, "conf", None) is not None:
            conf = np.asarray(boxes.conf.cpu().numpy(), dtype=np.float32)
        if getattr(boxes, "cls", None) is not None:
            cls = np.asarray(boxes.cls.cpu().numpy(), dtype=np.int32)
        if getattr(boxes, "id", None) is not None:
            object_ids = np.asarray(boxes.id.cpu().numpy(), dtype=np.int32)
        elif xyxy.shape[0]:
            object_ids = np.full((xyxy.shape[0],), -1, dtype=np.int32)

    masks_data = np.zeros((0, result.orig_img.shape[0], result.orig_img.shape[1]), dtype=np.uint8)
    masks = getattr(result, "masks", None)
    if masks is not None and getattr(masks, "data", None) is not None:
        masks_data = np.asarray(masks.data.cpu().numpy(), dtype=np.uint8)

    return Sam3ExportFrameArtifacts(
        frame_index=frame_index,
        path=str(result.path),
        xyxy=xyxy,
        conf=conf,
        cls=cls,
        object_ids=object_ids,
        masks=masks_data,
    )


def _save_frame_masks(frame: Sam3ExportFrameArtifacts, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{frame.frame_index:07d}.npz"
    np.savez_compressed(
        output_path,
        frame_index=np.int32(frame.frame_index),
        path=np.asarray(frame.path),
        xyxy=frame.xyxy.astype(np.float32),
        conf=frame.conf.astype(np.float32),
        cls=frame.cls.astype(np.int32),
        object_ids=frame.object_ids.astype(np.int32),
        masks=frame.masks.astype(np.uint8),
    )
    return str(output_path)


def _build_tracking_document(frames: list[Sam3ExportFrameArtifacts]) -> TrackingDocument:
    tracking_frames: list[TrackingFrame] = []
    for frame in frames:
        centroids = [list(centroid_from_xyxy(box.tolist())) for box in frame.xyxy]
        labels = [
            TrackingLabel(
                class_id=int(frame.cls[index]) if frame.cls.size else None,
                id=(int(frame.object_ids[index]) if frame.object_ids.size and int(frame.object_ids[index]) >= 0 else None),
                det_idx=index,
                real=1,
                src="sam3_export",
            )
            for index in range(frame.xyxy.shape[0])
        ]
        tracking_frames.append(
            TrackingFrame(
                frame_id=frame.frame_index,
                detections=Detections(
                    xyxy=frame.xyxy.tolist(),
                    centroids=centroids,
                ),
                labels=labels,
            )
        )
    return TrackingDocument(frames=tracking_frames)


def _run_export_for_video(
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
) -> Sam3ExportVideoRunResult:
    metadata = _probe_video_metadata(video_path)
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    expected_frame_count = int(metadata["frame_count"])
    expected_logged_frame_count = min(expected_frame_count, max_frames) if max_frames > 0 else expected_frame_count

    json_dir = output_root / "json"
    masks_root = output_root / "masks"
    json_dir.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)

    stem = _artifact_stem(video_path)
    tracking_json_path = json_dir / f"{stem}_sam3_tracking.json"
    summary_json_path = json_dir / f"{stem}_sam3_export_summary.json"
    masks_dir = masks_root / stem
    if masks_dir.exists():
        for existing in masks_dir.glob("*.npz"):
            existing.unlink()
    masks_dir.mkdir(parents=True, exist_ok=True)

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
    exported_frames: list[Sam3ExportFrameArtifacts] = []
    frame_count = 0
    total_instances = 0
    max_instances = 0
    _log_progress(log_progress, f"[sam3-export] start: {video_path} prompts={prompts}")
    start = time.perf_counter()
    for frame_index, result in enumerate(predictor(source=video_path, text=prompts, stream=True)):
        frame = _extract_frame_artifacts(frame_index, result)
        object_ids = _extract_object_ids(getattr(result, "boxes", None))
        instance_count = len(object_ids) if object_ids else len(getattr(result, "boxes", []) or [])
        tracked_ids.update(object_ids)
        total_instances += int(instance_count)
        max_instances = max(max_instances, int(instance_count))
        _save_frame_masks(frame, masks_dir)
        exported_frames.append(frame)
        frame_count += 1
        if log_progress and log_every_frames > 0:
            if frame_index == 0 or ((frame_index + 1) % log_every_frames == 0) or ((frame_index + 1) == expected_logged_frame_count):
                print(
                    f"[sam3-export] progress: {video_path} frame={frame_index + 1}/{expected_logged_frame_count} instances={instance_count}",
                    flush=True,
                )
        if max_frames > 0 and frame_count >= max_frames:
            break

    elapsed_s = time.perf_counter() - start
    mean_instances = (total_instances / frame_count) if frame_count else 0.0

    tracking_document = _build_tracking_document(exported_frames)
    dump_path_compact(tracking_json_path, tracking_document.to_dict())

    summary_payload = {
        "video_path": video_path,
        "tracking_json_path": str(tracking_json_path),
        "masks_dir": str(masks_dir),
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
    _log_progress(log_progress, f"[sam3-export] done: {video_path} in {elapsed_s:.2f}s -> {tracking_json_path}")
    return Sam3ExportVideoRunResult(
        video_path=video_path,
        tracking_json_path=str(tracking_json_path),
        masks_dir=str(masks_dir),
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
        description="Run SAM3 semantic tracking and export reusable per-frame masks plus tracking JSON.",
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
        default="var/benchmarks/sam3_export",
        help="Directory where export JSONs and mask bundles are written.",
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
        default="var/benchmarks/prepared_videos_sam3_export",
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
        _run_export_for_video(
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
        "tool": "benchmark_sam3_export",
        "runtime": runtime_info,
        "prepared_videos": prepared_video_metadata,
        "runs": [asdict(run) for run in runs],
    }
    summary_path = output_root / str(args.summary_name)
    dump_path_compact(summary_path, summary_payload)
    print("SAM3 export benchmark summary")
    print(dumps_pretty(summary_payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
