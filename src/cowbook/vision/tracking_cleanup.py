from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker

from cowbook.core.contracts import (
    Detections,
    TrackingCleanupConfig,
    TrackingDocument,
    TrackingFrame,
    TrackingLabel,
)
from cowbook.vision.cleanup import DetectionFrame

_CACHE_CONF = 0.001
_CACHE_NMS_IOU = 0.90
_CACHE_MAX_DET = 100
ProgressCallback = Callable[[str, int, int | None], None]


def _read_video_meta(video_path: str) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return float(fps), width, height, frame_count


def _open_writer(path: Path, fps: float, width: int, height: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {path}")
    return writer


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    value = (track_id * 2654435761) & 0xFFFFFFFF
    b = 64 + (value & 191)
    g = 64 + ((value >> 8) & 191)
    r = 64 + ((value >> 16) & 191)
    return int(b), int(g), int(r)


def detect_video_to_frames(
    video_path: str,
    model_path: str,
    cleanup_config: TrackingCleanupConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[DetectionFrame]:
    model = YOLO(model_path)
    _, width, height, frame_count = _read_video_meta(video_path)
    results = model.predict(
        source=video_path,
        stream=True,
        conf=min(_CACHE_CONF, cleanup_config.conf_threshold),
        iou=_CACHE_NMS_IOU,
        max_det=_CACHE_MAX_DET,
        verbose=False,
    )

    frames: list[DetectionFrame] = []
    iterable = (
        results
        if progress_callback is not None
        else tqdm(results, desc="Detecting video for cleanup", total=frame_count or None)
    )
    for frame_idx, result in enumerate(iterable):
        if progress_callback is not None:
            progress_callback("detect", frame_idx + 1, frame_count or None)
        shape = tuple(result.orig_shape) if hasattr(result, "orig_shape") else (height, width)
        if result.boxes is None or len(result.boxes) == 0:
            frames.append(
                DetectionFrame(
                    frame_idx=frame_idx,
                    shape=(int(shape[0]), int(shape[1])),
                    xyxy=np.zeros((0, 4), dtype=np.float32),
                    conf=np.zeros((0,), dtype=np.float32),
                    cls=np.zeros((0,), dtype=np.int32),
                )
            )
            continue
        boxes = result.boxes
        frames.append(
            DetectionFrame(
                frame_idx=frame_idx,
                shape=(int(shape[0]), int(shape[1])),
                xyxy=boxes.xyxy.cpu().numpy().astype(np.float32),
                conf=boxes.conf.cpu().numpy().astype(np.float32),
                cls=boxes.cls.cpu().numpy().astype(np.int32),
            )
        )
    del model
    return frames


def build_tracker(tracker_yaml_path: Path, fps: float):
    cfg = yaml.safe_load(tracker_yaml_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(**cfg)
    tracker_type = str(cfg.get("tracker_type", "botsort")).lower()
    if tracker_type == "bytetrack":
        return BYTETracker(args, frame_rate=int(round(fps)) if fps else 30)
    return BOTSORT(args, frame_rate=int(round(fps)) if fps else 30)


def make_ultralytics_boxes(
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    shape_hw: tuple[int, int],
) -> Boxes:
    import torch

    h, w = shape_hw
    if xyxy.shape[0] == 0:
        data = torch.zeros((0, 6), dtype=torch.float32)
    else:
        data_np = np.concatenate(
            [
                xyxy.astype(np.float32),
                conf.reshape(-1, 1).astype(np.float32),
                cls.reshape(-1, 1).astype(np.float32),
            ],
            axis=1,
        )
        data = torch.from_numpy(data_np)
    return Boxes(data, orig_shape=(int(h), int(w)))


def _render_tracked_frame(
    frame: np.ndarray,
    cur_xyxy: np.ndarray,
    cur_ids: np.ndarray,
    cur_scores: np.ndarray,
) -> np.ndarray:
    for index in range(cur_ids.shape[0]):
        x1, y1, x2, y2 = cur_xyxy[index]
        track_id = int(cur_ids[index])
        color = _color_for_id(track_id)
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(frame, p1, p2, color, thickness=2)
        label = f"id={track_id}"
        if cur_scores.size:
            label += f" {cur_scores[index]:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx, ty = p1[0], max(0, p1[1] - 5)
        cv2.rectangle(frame, (tx, max(0, ty - text_h - 4)), (tx + text_w + 4, ty + 2), color, -1)
        cv2.putText(frame, label, (tx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return frame


def track_from_detection_frames(
    video_path: str,
    detection_frames: list[DetectionFrame],
    tracker_yaml_path: Path,
    *,
    save_video_path: str | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_stage: str = "track",
) -> TrackingDocument:
    fps, width, height, _ = _read_video_meta(video_path)
    tracker = build_tracker(tracker_yaml_path, fps=fps)
    if hasattr(tracker, "reset"):
        tracker.reset()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    writer = None
    if save_video_path is not None:
        writer = _open_writer(Path(save_video_path), fps=fps, width=width, height=height)

    frames: list[TrackingFrame] = []
    iterable = (
        detection_frames
        if progress_callback is not None
        else tqdm(detection_frames, desc="Tracking cleaned detections", total=len(detection_frames))
    )
    total_frames = len(detection_frames)
    for index, det_frame in enumerate(iterable):
        if progress_callback is not None:
            progress_callback(progress_stage, index + 1, total_frames)
        ok, frame = cap.read()
        if not ok:
            break
        boxes = make_ultralytics_boxes(det_frame.xyxy, det_frame.conf, det_frame.cls, det_frame.shape)
        tracks = tracker.update(boxes, img=frame)
        if tracks is None or len(tracks) == 0:
            tracks_arr = np.zeros((0, 8), dtype=np.float32)
        else:
            tracks_arr = np.asarray(tracks)

        cur_xyxy = (
            tracks_arr[:, :4].astype(np.float32)
            if tracks_arr.shape[0]
            else np.zeros((0, 4), dtype=np.float32)
        )
        cur_ids = (
            tracks_arr[:, 4].astype(np.int64)
            if tracks_arr.shape[0] and tracks_arr.shape[1] >= 5
            else np.zeros((0,), dtype=np.int64)
        )
        cur_scores = (
            tracks_arr[:, 5].astype(np.float32)
            if tracks_arr.shape[0] and tracks_arr.shape[1] >= 6
            else np.zeros((0,), dtype=np.float32)
        )
        cur_cls = (
            tracks_arr[:, 6].astype(np.int64)
            if tracks_arr.shape[0] and tracks_arr.shape[1] >= 7
            else np.zeros((0,), dtype=np.int64)
        )
        cur_det_idx = (
            tracks_arr[:, 7].astype(np.int64)
            if tracks_arr.shape[0] and tracks_arr.shape[1] >= 8
            else np.full((cur_ids.shape[0],), -1, dtype=np.int64)
        )

        labels = [
            TrackingLabel(
                class_id=int(cur_cls[i]) if cur_cls.size else 0,
                id=int(cur_ids[i]),
                det_idx=int(cur_det_idx[i]),
                real=1,
                src="tracker",
            )
            for i in range(cur_ids.shape[0])
        ]
        frames.append(
            TrackingFrame(
                frame_id=int(det_frame.frame_idx),
                detections=Detections(xyxy=cur_xyxy.tolist()),
                labels=labels,
            )
        )

        if writer is not None:
            writer.write(_render_tracked_frame(frame, cur_xyxy, cur_ids, cur_scores))

    cap.release()
    if writer is not None:
        writer.release()
    return TrackingDocument(frames=frames)
