from __future__ import annotations

import numpy as np

from cowbook.core.contracts import TrackingDocument
from cowbook.vision.cleanup.preprocess import DetectionFrame


def _compute_max_track_streak(frame_ids: list[int], gap_tolerance: int) -> int:
    if not frame_ids:
        return 0
    best = 1
    current = 1
    for previous, current_frame in zip(frame_ids, frame_ids[1:]):
        gap = current_frame - previous - 1
        if gap <= gap_tolerance:
            current += 1
        else:
            best = max(best, current)
            current = 1
    return max(best, current)


def find_prunable_track_ids(
    document: TrackingDocument,
    min_track_streak: int,
    *,
    min_total_observations: int | None = None,
    gap_tolerance: int = 6,
) -> set[int]:
    frame_ids_by_track: dict[int, set[int]] = {}
    for frame in document.frames:
        for label in frame.labels:
            if label.id is None or label.det_idx is None or label.det_idx < 0:
                continue
            frame_ids_by_track.setdefault(label.id, set()).add(frame.frame_id)
    return {
        track_id
        for track_id, frame_ids in frame_ids_by_track.items()
        if (
            _compute_max_track_streak(sorted(frame_ids), gap_tolerance) < min_track_streak
            or (
                min_total_observations is not None
                and len(frame_ids) < min_total_observations
            )
        )
    }


def drop_pruned_tracks_from_detection_frames(
    detection_frames: list[DetectionFrame],
    document: TrackingDocument,
    prunable_track_ids: set[int],
) -> list[DetectionFrame]:
    if len(detection_frames) != len(document.frames):
        raise ValueError(
            "Detection-frame pruning requires detection_frames and document.frames to have the same length."
        )

    pruned_frames: list[DetectionFrame] = []
    for det_frame, track_frame in zip(detection_frames, document.frames, strict=True):
        if int(det_frame.frame_idx) != int(track_frame.frame_id):
            raise ValueError(
                "Detection-frame pruning requires aligned frame indices between detection_frames and document.frames."
            )
        drop: set[int] = set()
        for label in track_frame.labels:
            if label.id is None or label.id not in prunable_track_ids:
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
