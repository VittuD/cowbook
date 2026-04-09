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

from cowbook.core.contracts import (
    Detections,
    TrackingCleanupConfig,
    TrackingDocument,
    TrackingFrame,
    TrackingLabel,
)
from cowbook.io.json_utils import dump_path_compact, dumps_pretty
from cowbook.vision.cleanup import (
    clip_boxes,
    find_prunable_track_ids,
    footprint_nms_xyxy,
    hybrid_nms_xyxy,
    iou_nms_xyxy,
    point_in_poly,
)
from tools.benchmark_tracking import _prepare_benchmark_videos, _probe_video_metadata, _query_gpu_info


@dataclass(slots=True)
class Sam3VideoRunResult:
    video_path: str
    annotated_video_path: str
    clean_annotated_video_path: str
    processed_annotated_video_path: str
    processed_clean_annotated_video_path: str
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
    processed_mean_instances_per_frame: float
    processed_max_instances_per_frame: int
    processed_tracked_object_ids: list[int]
    dump_frame_metadata: bool


@dataclass(slots=True)
class Sam3FrameArtifacts:
    frame_index: int
    orig_img: np.ndarray
    path: str
    names: dict[int, str]
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray
    object_ids: np.ndarray
    masks: np.ndarray


@dataclass(slots=True)
class Sam3CleanupStats:
    raw_detection_count: int
    prefiltered_detection_count: int
    cleaned_detection_count: int
    removed_by_prefilter: int
    removed_by_short_tracks: int
    short_track_count: int


def _default_videos() -> list[str]:
    return [
        "sample_data/videos/Ch1_60.mp4",
        "sample_data/videos/Ch4_60.mp4",
        "sample_data/videos/Ch6_60.mp4",
        "sample_data/videos/Ch8_60.mp4",
    ]


def _default_prompts() -> list[str]:
    return ["cow"]


def _default_cleanup_config() -> TrackingCleanupConfig:
    return TrackingCleanupConfig(
        enabled=True,
        conf_threshold=0.25,
        nms_mode="hybrid_nms",
        hybrid_iou_hard=0.65,
        hybrid_iou_guard=0.02,
        hybrid_footpoint_dist_k=0.40,
        hybrid_footpoint_dist_min_px=36.0,
        drop_edge_boxes=True,
        edge_margin_px=28,
        two_pass_prune_short_tracks=False,
        min_track_length=24,
        min_track_total_observations=45,
        short_track_gap_tolerance=6,
        postprocess_smoothing=False,
    )


def _log_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _maybe_log_frame_progress(
    *,
    enabled: bool,
    video_path: str,
    frame_index: int,
    frame_count: int,
    instance_count: int,
    every_n_frames: int,
) -> None:
    if not enabled or every_n_frames <= 0:
        return
    if frame_index == 0 or ((frame_index + 1) % every_n_frames == 0) or ((frame_index + 1) == frame_count):
        print(
            f"[sam3] progress: {video_path} frame={frame_index + 1}/{frame_count} instances={instance_count}",
            flush=True,
        )


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


def _normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def _frame_summary(frame_index: int, result) -> dict[str, Any]:
    boxes = getattr(result, "boxes", None)
    object_ids = _extract_object_ids(boxes)
    confidences = _extract_confidences(boxes)
    names = _normalize_names(getattr(result, "names", {}) or {})

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


def _extract_frame_artifacts(frame_index: int, result) -> Sam3FrameArtifacts:
    boxes = getattr(result, "boxes", None)
    names = _normalize_names(getattr(result, "names", {}) or {})

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

    return Sam3FrameArtifacts(
        frame_index=frame_index,
        orig_img=result.orig_img.copy(),
        path=str(result.path),
        names=names,
        xyxy=xyxy,
        conf=conf,
        cls=cls,
        object_ids=object_ids,
        masks=masks_data,
    )


def _overlay_prompt_text(image: np.ndarray, prompts: list[str]) -> np.ndarray:
    label = f"SAM3 prompts: {', '.join(prompts)}"
    cv2.rectangle(image, (8, 8), (min(image.shape[1] - 8, 12 + 9 * len(label)), 40), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _collect_runtime_info(model_path: str, device: str | None) -> dict[str, Any]:
    return {
        "ultralytics_version": ultralytics.__version__,
        "gpu_info": _query_gpu_info(),
        "model_path": model_path,
        "device": device,
        "sam3_semantic_predictor": SAM3VideoSemanticPredictor.__name__,
    }


def _color_for_detection(track_id: int | None, class_id: int | None, index: int) -> tuple[int, int, int]:
    seed = track_id if track_id is not None and track_id >= 0 else (class_id if class_id is not None else index)
    value = int(seed) * 2654435761 % (1 << 32)
    return (
        48 + int((value >> 16) & 0x7F),
        48 + int((value >> 8) & 0x7F),
        48 + int(value & 0x7F),
    )


def _draw_mask_overlay(image: np.ndarray, frame: Sam3FrameArtifacts, alpha: float = 0.45) -> np.ndarray:
    rendered = image.copy()
    if frame.masks.shape[0] == 0:
        return rendered
    overlay = rendered.copy()
    for index, mask in enumerate(frame.masks):
        track_id = int(frame.object_ids[index]) if frame.object_ids.size else None
        class_id = int(frame.cls[index]) if frame.cls.size else None
        overlay[mask > 0] = _color_for_detection(track_id, class_id, index)
    return cv2.addWeighted(rendered, 1.0 - alpha, overlay, alpha, 0.0)


def _draw_box_label(
    image: np.ndarray,
    xyxy: np.ndarray,
    label: str | None,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in xyxy.tolist()]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    if not label:
        return
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_y = max(text_h + baseline + 4, y1)
    text_box_y1 = max(0, text_y - text_h - baseline - 4)
    text_box_y2 = min(image.shape[0] - 1, text_y + 2)
    text_box_x2 = min(image.shape[1] - 1, x1 + text_w + 8)
    cv2.rectangle(image, (x1, text_box_y1), (text_box_x2, text_box_y2), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 4, text_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_processed_frames(
    frame: Sam3FrameArtifacts,
    *,
    prompts: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    clean_frame = _overlay_prompt_text(_draw_mask_overlay(frame.orig_img, frame), prompts)
    detailed_frame = clean_frame.copy()
    for index, xyxy in enumerate(frame.xyxy):
        track_id = int(frame.object_ids[index]) if frame.object_ids.size and int(frame.object_ids[index]) >= 0 else None
        class_id = int(frame.cls[index]) if frame.cls.size else None
        confidence = float(frame.conf[index]) if frame.conf.size else None
        class_name = frame.names.get(class_id, str(class_id)) if class_id is not None else "obj"
        label_parts = []
        if track_id is not None:
            label_parts.append(f"id:{track_id}")
        label_parts.append(class_name)
        if confidence is not None:
            label_parts.append(f"{confidence:.2f}")
        _draw_box_label(
            detailed_frame,
            xyxy,
            " ".join(label_parts),
            _color_for_detection(track_id, class_id, index),
        )
    return detailed_frame, clean_frame


def _select_cleanup_keep_indices(
    frame: Sam3FrameArtifacts,
    cleanup_config: TrackingCleanupConfig,
) -> np.ndarray:
    xyxy = clip_boxes(frame.xyxy, frame.orig_img.shape[1], frame.orig_img.shape[0])
    conf = frame.conf.copy()

    if xyxy.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    keep = np.arange(xyxy.shape[0], dtype=np.int64)
    widths = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0])
    heights = np.maximum(0.0, xyxy[:, 3] - xyxy[:, 1])

    valid = (widths > 1.0) & (heights > 1.0)
    keep = keep[valid]
    xyxy = xyxy[valid]
    conf = conf[valid]
    widths = widths[valid]
    heights = heights[valid]

    mask = conf >= float(cleanup_config.conf_threshold)
    keep = keep[mask]
    xyxy = xyxy[mask]
    conf = conf[mask]
    widths = widths[mask]
    heights = heights[mask]

    if conf.size == 0:
        return np.zeros((0,), dtype=np.int64)

    if cleanup_config.roi is not None:
        centers_x = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
        centers_y = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
        roi_keep = np.array(
            [
                point_in_poly(float(centers_x[i]), float(centers_y[i]), cleanup_config.roi)
                for i in range(xyxy.shape[0])
            ],
            dtype=bool,
        )
        keep = keep[roi_keep]
        xyxy = xyxy[roi_keep]
        conf = conf[roi_keep]
        widths = widths[roi_keep]
        heights = heights[roi_keep]

    if conf.size == 0:
        return np.zeros((0,), dtype=np.int64)

    if cleanup_config.drop_edge_boxes:
        margin = int(cleanup_config.edge_margin_px)
        edge_keep = (
            (xyxy[:, 0] >= margin)
            & (xyxy[:, 1] >= margin)
            & (xyxy[:, 2] <= (frame.orig_img.shape[1] - 1 - margin))
            & (xyxy[:, 3] <= (frame.orig_img.shape[0] - 1 - margin))
        )
        keep = keep[edge_keep]
        xyxy = xyxy[edge_keep]
        conf = conf[edge_keep]
        widths = widths[edge_keep]
        heights = heights[edge_keep]

    if conf.size == 0:
        return np.zeros((0,), dtype=np.int64)

    area = widths * heights
    aspect_ratio = widths / np.maximum(heights, 1e-9)

    if cleanup_config.min_area_px is not None:
        area_keep = area >= float(cleanup_config.min_area_px)
        keep, xyxy, conf, area, aspect_ratio = (
            keep[area_keep],
            xyxy[area_keep],
            conf[area_keep],
            area[area_keep],
            aspect_ratio[area_keep],
        )
    if cleanup_config.max_area_px is not None and conf.size:
        area_keep = area <= float(cleanup_config.max_area_px)
        keep, xyxy, conf, area, aspect_ratio = (
            keep[area_keep],
            xyxy[area_keep],
            conf[area_keep],
            area[area_keep],
            aspect_ratio[area_keep],
        )
    if cleanup_config.min_aspect_ratio is not None and conf.size:
        aspect_keep = aspect_ratio >= float(cleanup_config.min_aspect_ratio)
        keep, xyxy, conf, area, aspect_ratio = (
            keep[aspect_keep],
            xyxy[aspect_keep],
            conf[aspect_keep],
            area[aspect_keep],
            aspect_ratio[aspect_keep],
        )
    if cleanup_config.max_aspect_ratio is not None and conf.size:
        aspect_keep = aspect_ratio <= float(cleanup_config.max_aspect_ratio)
        keep = keep[aspect_keep]
        xyxy = xyxy[aspect_keep]
        conf = conf[aspect_keep]

    if conf.size == 0:
        return np.zeros((0,), dtype=np.int64)

    if cleanup_config.nms_mode == "iou_nms":
        nms_keep = iou_nms_xyxy(xyxy, conf, cleanup_config.nms_iou)
    elif cleanup_config.nms_mode == "footpoint_nms":
        nms_keep = footprint_nms_xyxy(
            xyxy,
            conf,
            dist_k=cleanup_config.footpoint_dist_k,
            dist_min_px=cleanup_config.footpoint_dist_min_px,
            iou_guard=cleanup_config.footpoint_iou_guard,
        )
    else:
        nms_keep = hybrid_nms_xyxy(
            xyxy,
            conf,
            iou_hard=cleanup_config.hybrid_iou_hard,
            iou_guard=cleanup_config.hybrid_iou_guard,
            dist_k=cleanup_config.hybrid_footpoint_dist_k,
            dist_min_px=cleanup_config.hybrid_footpoint_dist_min_px,
        )
    return keep[nms_keep]


def _subset_frame(frame: Sam3FrameArtifacts, keep_indices: np.ndarray) -> Sam3FrameArtifacts:
    if keep_indices.size == 0:
        return Sam3FrameArtifacts(
            frame_index=frame.frame_index,
            orig_img=frame.orig_img,
            path=frame.path,
            names=frame.names,
            xyxy=np.zeros((0, 4), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.int32),
            object_ids=np.zeros((0,), dtype=np.int32),
            masks=np.zeros((0, frame.orig_img.shape[0], frame.orig_img.shape[1]), dtype=np.uint8),
        )
    return Sam3FrameArtifacts(
        frame_index=frame.frame_index,
        orig_img=frame.orig_img,
        path=frame.path,
        names=frame.names,
        xyxy=frame.xyxy[keep_indices],
        conf=frame.conf[keep_indices],
        cls=frame.cls[keep_indices],
        object_ids=frame.object_ids[keep_indices] if frame.object_ids.size else np.zeros((0,), dtype=np.int32),
        masks=frame.masks[keep_indices] if frame.masks.shape[0] else np.zeros((0, frame.orig_img.shape[0], frame.orig_img.shape[1]), dtype=np.uint8),
    )


def _build_tracking_document(frames: list[Sam3FrameArtifacts]) -> TrackingDocument:
    return TrackingDocument(
        frames=[
            TrackingFrame(
                frame_id=frame.frame_index,
                detections=Detections(xyxy=frame.xyxy.tolist()),
                labels=[
                    TrackingLabel(
                        class_id=int(frame.cls[index]) if frame.cls.size else None,
                        id=(int(frame.object_ids[index]) if frame.object_ids.size and int(frame.object_ids[index]) >= 0 else None),
                        det_idx=index,
                        real=1,
                        src="sam3",
                    )
                    for index in range(frame.xyxy.shape[0])
                ],
            )
            for frame in frames
        ]
    )


def _apply_cowbook_box_cleanup(
    frames: list[Sam3FrameArtifacts],
    cleanup_config: TrackingCleanupConfig,
) -> tuple[list[Sam3FrameArtifacts], set[int], Sam3CleanupStats]:
    raw_detection_count = int(sum(frame.xyxy.shape[0] for frame in frames))
    prefiltered = [_subset_frame(frame, _select_cleanup_keep_indices(frame, cleanup_config)) for frame in frames]
    prefiltered_detection_count = int(sum(frame.xyxy.shape[0] for frame in prefiltered))
    document = _build_tracking_document(prefiltered)
    short_track_ids = find_prunable_track_ids(
        document,
        cleanup_config.min_track_length,
        min_total_observations=cleanup_config.min_track_total_observations,
        gap_tolerance=cleanup_config.short_track_gap_tolerance,
    )

    cleaned_frames: list[Sam3FrameArtifacts] = []
    for frame in prefiltered:
        if not short_track_ids or frame.object_ids.size == 0:
            cleaned_frames.append(frame)
            continue
        keep = np.array(
            [int(object_id) not in short_track_ids for object_id in frame.object_ids],
            dtype=bool,
        )
        cleaned_frames.append(_subset_frame(frame, np.flatnonzero(keep)))
    cleaned_detection_count = int(sum(frame.xyxy.shape[0] for frame in cleaned_frames))
    stats = Sam3CleanupStats(
        raw_detection_count=raw_detection_count,
        prefiltered_detection_count=prefiltered_detection_count,
        cleaned_detection_count=cleaned_detection_count,
        removed_by_prefilter=max(0, raw_detection_count - prefiltered_detection_count),
        removed_by_short_tracks=max(0, prefiltered_detection_count - cleaned_detection_count),
        short_track_count=len(short_track_ids),
    )
    return cleaned_frames, short_track_ids, stats


def _write_overlay_videos(
    frames: list[Sam3FrameArtifacts],
    *,
    detailed_path: Path,
    clean_path: Path,
    fps: float,
    prompts: list[str],
    log_progress: bool = False,
    log_every_frames: int = 25,
    stage_label: str = "processed_overlays",
) -> tuple[float, int, list[int]]:
    if not frames:
        return 0.0, 0, []

    writer = _open_video_writer(detailed_path, fps=fps, frame_size=(frames[0].orig_img.shape[1], frames[0].orig_img.shape[0]))
    clean_writer = _open_video_writer(clean_path, fps=fps, frame_size=(frames[0].orig_img.shape[1], frames[0].orig_img.shape[0]))
    tracked_ids: set[int] = set()
    total_instances = 0
    max_instances = 0
    try:
        for frame in frames:
            tracked_ids.update(int(object_id) for object_id in frame.object_ids.tolist() if int(object_id) >= 0)
            total_instances += int(frame.xyxy.shape[0])
            max_instances = max(max_instances, int(frame.xyxy.shape[0]))
            annotated, clean_annotated = _draw_processed_frames(frame, prompts=prompts)
            writer.write(annotated)
            clean_writer.write(clean_annotated)
            _maybe_log_frame_progress(
                enabled=log_progress,
                video_path=stage_label,
                frame_index=frame.frame_index,
                frame_count=len(frames),
                instance_count=int(frame.xyxy.shape[0]),
                every_n_frames=log_every_frames,
            )
    finally:
        writer.release()
        clean_writer.release()

    mean_instances = (total_instances / len(frames)) if frames else 0.0
    return mean_instances, max_instances, sorted(tracked_ids)


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
    log_every_frames: int,
) -> Sam3VideoRunResult:
    metadata = _probe_video_metadata(video_path)
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    expected_frame_count = int(metadata["frame_count"])

    json_dir = output_root / "json"
    video_dir = output_root / "videos"
    json_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    stem = _artifact_stem(video_path)
    annotated_video_path = video_dir / f"{stem}_sam3_annotated.mp4"
    clean_annotated_video_path = video_dir / f"{stem}_sam3_annotated_clean.mp4"
    processed_annotated_video_path = video_dir / f"{stem}_sam3_annotated_processed.mp4"
    processed_clean_annotated_video_path = video_dir / f"{stem}_sam3_annotated_processed_clean.mp4"
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
    clean_writer = _open_video_writer(clean_annotated_video_path, fps=fps, frame_size=(width, height))
    frame_summaries: list[dict[str, Any]] = []
    raw_frames: list[Sam3FrameArtifacts] = []
    tracked_ids: set[int] = set()
    frame_count = 0
    total_instances = 0
    max_instances = 0
    start = time.perf_counter()
    _log_progress(log_progress, f"[sam3] start: {video_path} prompts={prompts}")

    try:
        for frame_index, result in enumerate(
            predictor(source=video_path, text=prompts, stream=True)
        ):
            summary = _frame_summary(frame_index, result)
            tracked_ids.update(summary["object_ids"])
            total_instances += int(summary["instance_count"])
            max_instances = max(max_instances, int(summary["instance_count"]))
            if dump_frame_metadata:
                frame_summaries.append(summary)
            raw_frames.append(_extract_frame_artifacts(frame_index, result))

            plotted = result.plot()
            clean_plotted = result.plot(labels=False, conf=False, boxes=False, masks=True)
            annotated = _overlay_prompt_text(plotted, prompts)
            clean_annotated = _overlay_prompt_text(clean_plotted, prompts)
            writer.write(annotated)
            clean_writer.write(clean_annotated)
            frame_count += 1
            _maybe_log_frame_progress(
                enabled=log_progress,
                video_path=video_path,
                frame_index=frame_index,
                frame_count=expected_frame_count,
                instance_count=int(summary["instance_count"]),
                every_n_frames=log_every_frames,
            )
    finally:
        writer.release()
        clean_writer.release()

    elapsed_s = time.perf_counter() - start
    mean_instances = (total_instances / frame_count) if frame_count else 0.0
    _log_progress(log_progress, f"[sam3] cleanup: {video_path} frames={frame_count}")
    cleanup_start = time.perf_counter()
    processed_frames, short_track_ids, cleanup_stats = _apply_cowbook_box_cleanup(
        raw_frames,
        _default_cleanup_config(),
    )
    cleanup_elapsed_s = time.perf_counter() - cleanup_start
    _log_progress(
        log_progress,
        (
            f"[sam3] cleanup done: {video_path} raw={cleanup_stats.raw_detection_count} "
            f"prefiltered={cleanup_stats.prefiltered_detection_count} "
            f"cleaned={cleanup_stats.cleaned_detection_count} "
            f"removed_prefilter={cleanup_stats.removed_by_prefilter} "
            f"removed_short_tracks={cleanup_stats.removed_by_short_tracks} "
            f"short_tracks_removed={cleanup_stats.short_track_count} "
            f"elapsed_s={cleanup_elapsed_s:.2f}"
        ),
    )
    processed_overlay_start = time.perf_counter()
    processed_mean_instances, processed_max_instances, processed_tracked_ids = _write_overlay_videos(
        processed_frames,
        detailed_path=processed_annotated_video_path,
        clean_path=processed_clean_annotated_video_path,
        fps=fps,
        prompts=prompts,
        log_progress=log_progress,
        log_every_frames=log_every_frames,
        stage_label=f"{video_path} processed_overlays",
    )
    processed_overlay_elapsed_s = time.perf_counter() - processed_overlay_start
    _log_progress(
        log_progress,
        f"[sam3] processed overlays written: {video_path} elapsed_s={processed_overlay_elapsed_s:.2f}",
    )

    summary_payload: dict[str, Any] = {
        "video_path": video_path,
        "annotated_video_path": str(annotated_video_path),
        "clean_annotated_video_path": str(clean_annotated_video_path),
        "processed_annotated_video_path": str(processed_annotated_video_path),
        "processed_clean_annotated_video_path": str(processed_clean_annotated_video_path),
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
        "processed_mean_instances_per_frame": processed_mean_instances,
        "processed_max_instances_per_frame": processed_max_instances,
        "processed_tracked_object_ids": processed_tracked_ids,
        "short_track_ids_removed": sorted(short_track_ids),
        "cleanup_stats": asdict(cleanup_stats),
        "cleanup_elapsed_s": cleanup_elapsed_s,
        "processed_overlay_elapsed_s": processed_overlay_elapsed_s,
        "dump_frame_metadata": dump_frame_metadata,
    }
    if dump_frame_metadata:
        summary_payload["frames"] = frame_summaries

    dump_path_compact(summary_json_path, summary_payload)
    _log_progress(log_progress, f"[sam3] done: {video_path} in {elapsed_s:.2f}s -> {annotated_video_path}")
    return Sam3VideoRunResult(
        video_path=video_path,
        annotated_video_path=str(annotated_video_path),
        clean_annotated_video_path=str(clean_annotated_video_path),
        processed_annotated_video_path=str(processed_annotated_video_path),
        processed_clean_annotated_video_path=str(processed_clean_annotated_video_path),
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
        processed_mean_instances_per_frame=processed_mean_instances,
        processed_max_instances_per_frame=processed_max_instances,
        processed_tracked_object_ids=processed_tracked_ids,
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
    parser.add_argument(
        "--log-every-frames",
        type=int,
        default=25,
        help="When --log-progress is enabled, log every N processed frames.",
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
            log_every_frames=int(args.log_every_frames),
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
