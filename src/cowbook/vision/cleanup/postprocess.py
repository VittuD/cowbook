from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cowbook.core.contracts import (
    TrackingCleanupConfig,
    TrackingDocument,
    TrackingLabel,
)
from cowbook.vision.cleanup.preprocess import DetectionFrame

_SMOOTH_EPS = 1e-3


@dataclass(slots=True)
class _TrackObs:
    frame_idx: int
    box: np.ndarray
    class_id: int
    label_index: int
    real: int


def _xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    w = max(0.0, float(x2 - x1))
    h = max(0.0, float(y2 - y1))
    cx = float(x1 + x2) * 0.5
    cy = float(y1 + y2) * 0.5
    return np.array([cx, cy, w, h], dtype=np.float32)


def _cxcywh_to_xyxy(values: np.ndarray) -> np.ndarray:
    cx, cy, w, h = [float(value) for value in values]
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


def compute_short_track_ids(document: TrackingDocument, min_track_length: int) -> set[int]:
    counts: dict[int, int] = {}
    for frame in document.frames:
        for label in frame.labels:
            if label.id is None or label.det_idx is None or label.det_idx < 0:
                continue
            counts[label.id] = counts.get(label.id, 0) + 1
    return {track_id for track_id, count in counts.items() if count < min_track_length}


def prune_detection_frames_by_track_ids(
    detection_frames: list[DetectionFrame],
    document: TrackingDocument,
    short_track_ids: set[int],
) -> list[DetectionFrame]:
    pruned_frames: list[DetectionFrame] = []
    for det_frame, track_frame in zip(detection_frames, document.frames):
        drop: set[int] = set()
        for label in track_frame.labels:
            if label.id is None or label.id not in short_track_ids:
                continue
            if label.det_idx is None or label.det_idx < 0 or label.det_idx >= det_frame.conf.shape[0]:
                continue
            drop.add(label.det_idx)
        if not drop:
            pruned_frames.append(det_frame)
            continue
        keep_mask = np.ones((det_frame.conf.shape[0],), dtype=bool)
        for index in drop:
            keep_mask[index] = False
        pruned_frames.append(
            DetectionFrame(
                frame_idx=det_frame.frame_idx,
                shape=det_frame.shape,
                xyxy=det_frame.xyxy[keep_mask],
                conf=det_frame.conf[keep_mask],
                cls=det_frame.cls[keep_mask],
            )
        )
    return pruned_frames


def _build_track_observations(document: TrackingDocument) -> dict[int, list[_TrackObs]]:
    observations: dict[int, list[_TrackObs]] = {}
    for frame in document.frames:
        boxes = frame.detections.xyxy
        for index, label in enumerate(frame.labels):
            if label.id is None or index >= len(boxes):
                continue
            observations.setdefault(label.id, []).append(
                _TrackObs(
                    frame_idx=frame.frame_id,
                    box=np.asarray(boxes[index], dtype=np.float32),
                    class_id=label.class_id or 0,
                    label_index=index,
                    real=label.real if label.real is not None else 1,
                )
            )
    for obs_list in observations.values():
        obs_list.sort(key=lambda obs: obs.frame_idx)
    return observations


def _relative_change(a: float, b: float) -> float:
    return float(abs(b - a) / max(1e-9, abs(a)))


def _center_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    center_a = _xyxy_to_cxcywh(box_a)
    center_b = _xyxy_to_cxcywh(box_b)
    dx = float(center_a[0] - center_b[0])
    dy = float(center_a[1] - center_b[1])
    return float((dx * dx + dy * dy) ** 0.5)


def _can_fill_gap(
    box_a: np.ndarray,
    box_b: np.ndarray,
    gap: int,
    cleanup_config: TrackingCleanupConfig,
) -> bool:
    speed = _center_distance(box_a, box_b) / max(1, gap + 1)
    if speed > cleanup_config.max_center_speed_px_per_frame:
        return False
    vec_a = _xyxy_to_cxcywh(box_a)
    vec_b = _xyxy_to_cxcywh(box_b)
    area_a = float(vec_a[2] * vec_a[3])
    area_b = float(vec_b[2] * vec_b[3])
    aspect_a = float(vec_a[2] / max(1e-9, vec_a[3]))
    aspect_b = float(vec_b[2] / max(1e-9, vec_b[3]))
    if _relative_change(area_a, area_b) > cleanup_config.max_relative_area_change:
        return False
    if _relative_change(aspect_a, aspect_b) > cleanup_config.max_relative_aspect_change:
        return False
    return True


def _interpolate_boxes(box_a: np.ndarray, box_b: np.ndarray, t: float) -> np.ndarray:
    vec_a = _xyxy_to_cxcywh(box_a)
    vec_b = _xyxy_to_cxcywh(box_b)
    return _cxcywh_to_xyxy((1.0 - t) * vec_a + t * vec_b)


def _apply_gap_fill(document: TrackingDocument, cleanup_config: TrackingCleanupConfig) -> None:
    observations = _build_track_observations(document)
    frame_index = {frame.frame_id: frame for frame in document.frames}
    for track_id, obs_list in observations.items():
        if len(obs_list) < 2:
            continue
        for current, nxt in zip(obs_list, obs_list[1:]):
            gap = nxt.frame_idx - current.frame_idx - 1
            if gap <= 0 or gap > cleanup_config.gap_fill_max_frames:
                continue
            if not _can_fill_gap(current.box, nxt.box, gap, cleanup_config):
                continue
            for offset in range(1, gap + 1):
                frame_id = current.frame_idx + offset
                target_frame = frame_index.get(frame_id)
                if target_frame is None:
                    continue
                target_frame.detections.xyxy.append(
                    _interpolate_boxes(current.box, nxt.box, offset / float(gap + 1)).tolist()
                )
                target_frame.labels.append(
                    TrackingLabel(
                        class_id=current.class_id,
                        id=track_id,
                        det_idx=-1,
                        real=0,
                        src="gap_fill",
                    )
                )


def _apply_ema_smoothing(document: TrackingDocument, cleanup_config: TrackingCleanupConfig) -> None:
    observations = _build_track_observations(document)
    frame_index = {frame.frame_id: frame for frame in document.frames}
    alpha = cleanup_config.smoothing_alpha
    for obs_list in observations.values():
        if len(obs_list) < 2:
            continue
        smoothed = _xyxy_to_cxcywh(obs_list[0].box)
        for obs in obs_list[1:]:
            current = _xyxy_to_cxcywh(obs.box)
            smoothed = alpha * smoothed + (1.0 - alpha) * current
            smoothed_box = _cxcywh_to_xyxy(smoothed)
            if np.max(np.abs(smoothed_box - obs.box)) > _SMOOTH_EPS:
                frame = frame_index.get(obs.frame_idx)
                if frame is None or obs.label_index >= len(frame.detections.xyxy):
                    continue
                frame.detections.xyxy[obs.label_index] = smoothed_box.tolist()
                label = frame.labels[obs.label_index]
                if label.real == 1:
                    label.src = "smooth"


def postprocess_tracking_document(
    document: TrackingDocument,
    cleanup_config: TrackingCleanupConfig,
) -> TrackingDocument:
    postprocessed = TrackingDocument.from_mapping(document.to_dict())
    _apply_gap_fill(postprocessed, cleanup_config)
    _apply_ema_smoothing(postprocessed, cleanup_config)
    return postprocessed
