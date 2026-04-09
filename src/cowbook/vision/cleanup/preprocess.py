from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cowbook.core.contracts import TrackingCleanupConfig


@dataclass(slots=True)
class DetectionFrame:
    frame_idx: int
    shape: tuple[int, int]
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray


def _box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _iou_pair(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(1e-9, _box_area(box_a) + _box_area(box_b) - inter)
    return float(inter / union)


def _iou_xyxy_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = np.maximum(1e-9, area_a + area_b - inter)
    return inter / union


def iou_nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int64)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _iou_xyxy_one_to_many(boxes[idx], boxes[rest])
        order = rest[ious <= float(iou_thr)]
    return np.asarray(keep, dtype=np.int64)


def footprint_nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    dist_k: float,
    dist_min_px: float,
    iou_guard: float | None,
) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int64)

    order = scores.argsort()[::-1]
    keep: list[int] = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    foot_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
    foot_y = boxes[:, 3]
    heights = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])

    for ii in order:
        if suppressed[ii]:
            continue
        keep.append(int(ii))
        for jj in order:
            if jj == ii or suppressed[jj]:
                continue
            threshold = max(float(dist_min_px), float(dist_k) * float(min(heights[ii], heights[jj])))
            dx = float(foot_x[jj] - foot_x[ii])
            dy = float(foot_y[jj] - foot_y[ii])
            distance = (dx * dx + dy * dy) ** 0.5
            if distance <= threshold:
                if iou_guard is None or _iou_pair(boxes[ii], boxes[jj]) >= float(iou_guard):
                    suppressed[jj] = True
    return np.asarray(keep, dtype=np.int64)


def hybrid_nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    iou_hard: float,
    iou_guard: float,
    dist_k: float,
    dist_min_px: float,
) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int64)

    order = scores.argsort()[::-1]
    keep: list[int] = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    foot_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
    foot_y = boxes[:, 3]
    heights = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])

    for ii in order:
        if suppressed[ii]:
            continue
        keep.append(int(ii))
        for jj in order:
            if jj == ii or suppressed[jj]:
                continue
            iou = _iou_pair(boxes[ii], boxes[jj])
            if iou >= float(iou_hard):
                suppressed[jj] = True
                continue
            if iou >= float(iou_guard):
                threshold = max(float(dist_min_px), float(dist_k) * float(min(heights[ii], heights[jj])))
                dx = float(foot_x[jj] - foot_x[ii])
                dy = float(foot_y[jj] - foot_y[ii])
                distance = (dx * dx + dy * dy) ** 0.5
                if distance <= threshold:
                    suppressed[jj] = True
    return np.asarray(keep, dtype=np.int64)


def point_in_poly(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_intersection = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x_intersection > x:
                inside = not inside
    return inside


def clip_boxes(xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    if xyxy.size == 0:
        return xyxy
    clipped = xyxy.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, max(0, width - 1))
    clipped[:, 2] = np.clip(clipped[:, 2], 0, max(0, width - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, max(0, height - 1))
    clipped[:, 3] = np.clip(clipped[:, 3], 0, max(0, height - 1))
    return clipped


def filter_detection_frame(frame: DetectionFrame, cleanup_config: TrackingCleanupConfig) -> DetectionFrame:
    xyxy = clip_boxes(frame.xyxy, frame.shape[1], frame.shape[0])
    conf = frame.conf.copy()
    cls = frame.cls.copy()

    if xyxy.shape[0] == 0:
        return DetectionFrame(frame_idx=frame.frame_idx, shape=frame.shape, xyxy=xyxy, conf=conf, cls=cls)

    widths = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0])
    heights = np.maximum(0.0, xyxy[:, 3] - xyxy[:, 1])
    valid = (widths > 1.0) & (heights > 1.0)
    mask = valid & (conf >= float(cleanup_config.conf_threshold))
    xyxy = xyxy[mask]
    conf = conf[mask]
    cls = cls[mask]
    widths = widths[mask]
    heights = heights[mask]

    if conf.size == 0:
        return DetectionFrame(frame_idx=frame.frame_idx, shape=frame.shape, xyxy=xyxy, conf=conf, cls=cls)

    if cleanup_config.roi is not None:
        centers_x = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
        centers_y = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
        keep_mask = np.array(
            [
                point_in_poly(float(centers_x[i]), float(centers_y[i]), cleanup_config.roi)
                for i in range(xyxy.shape[0])
            ],
            dtype=bool,
        )
        xyxy = xyxy[keep_mask]
        conf = conf[keep_mask]
        cls = cls[keep_mask]
        widths = widths[keep_mask]
        heights = heights[keep_mask]

    if conf.size == 0:
        return DetectionFrame(frame_idx=frame.frame_idx, shape=frame.shape, xyxy=xyxy, conf=conf, cls=cls)

    if cleanup_config.drop_edge_boxes:
        margin = int(cleanup_config.edge_margin_px)
        keep_mask = (
            (xyxy[:, 0] >= margin)
            & (xyxy[:, 1] >= margin)
            & (xyxy[:, 2] <= (frame.shape[1] - 1 - margin))
            & (xyxy[:, 3] <= (frame.shape[0] - 1 - margin))
        )
        xyxy = xyxy[keep_mask]
        conf = conf[keep_mask]
        cls = cls[keep_mask]
        widths = widths[keep_mask]
        heights = heights[keep_mask]

    if conf.size == 0:
        return DetectionFrame(frame_idx=frame.frame_idx, shape=frame.shape, xyxy=xyxy, conf=conf, cls=cls)

    area = widths * heights
    frame_area = max(1.0, float(frame.shape[0] * frame.shape[1]))
    area_ratio = area / frame_area
    aspect_ratio = widths / np.maximum(heights, 1e-9)

    if cleanup_config.min_area_px is not None:
        keep_mask = area >= float(cleanup_config.min_area_px)
        xyxy, conf, cls, area, area_ratio, aspect_ratio = (
            xyxy[keep_mask],
            conf[keep_mask],
            cls[keep_mask],
            area[keep_mask],
            area_ratio[keep_mask],
            aspect_ratio[keep_mask],
        )
    if cleanup_config.max_area_px is not None and conf.size:
        keep_mask = area <= float(cleanup_config.max_area_px)
        xyxy, conf, cls, area, area_ratio, aspect_ratio = (
            xyxy[keep_mask],
            conf[keep_mask],
            cls[keep_mask],
            area[keep_mask],
            area_ratio[keep_mask],
            aspect_ratio[keep_mask],
        )
    if cleanup_config.min_area_ratio is not None and conf.size:
        keep_mask = area_ratio >= float(cleanup_config.min_area_ratio)
        xyxy, conf, cls, area, area_ratio, aspect_ratio = (
            xyxy[keep_mask],
            conf[keep_mask],
            cls[keep_mask],
            area[keep_mask],
            area_ratio[keep_mask],
            aspect_ratio[keep_mask],
        )
    if cleanup_config.max_area_ratio is not None and conf.size:
        keep_mask = area_ratio <= float(cleanup_config.max_area_ratio)
        xyxy, conf, cls, area, area_ratio, aspect_ratio = (
            xyxy[keep_mask],
            conf[keep_mask],
            cls[keep_mask],
            area[keep_mask],
            area_ratio[keep_mask],
            aspect_ratio[keep_mask],
        )
    if cleanup_config.min_aspect_ratio is not None and conf.size:
        keep_mask = aspect_ratio >= float(cleanup_config.min_aspect_ratio)
        xyxy, conf, cls, area, area_ratio, aspect_ratio = (
            xyxy[keep_mask],
            conf[keep_mask],
            cls[keep_mask],
            area[keep_mask],
            area_ratio[keep_mask],
            aspect_ratio[keep_mask],
        )
    if cleanup_config.max_aspect_ratio is not None and conf.size:
        keep_mask = aspect_ratio <= float(cleanup_config.max_aspect_ratio)
        xyxy = xyxy[keep_mask]
        conf = conf[keep_mask]
        cls = cls[keep_mask]

    if conf.size == 0:
        return DetectionFrame(frame_idx=frame.frame_idx, shape=frame.shape, xyxy=xyxy, conf=conf, cls=cls)

    if cleanup_config.nms_mode == "iou_nms":
        keep = iou_nms_xyxy(xyxy, conf, cleanup_config.nms_iou)
    elif cleanup_config.nms_mode == "footpoint_nms":
        keep = footprint_nms_xyxy(
            xyxy,
            conf,
            dist_k=cleanup_config.footpoint_dist_k,
            dist_min_px=cleanup_config.footpoint_dist_min_px,
            iou_guard=cleanup_config.footpoint_iou_guard,
        )
    else:
        keep = hybrid_nms_xyxy(
            xyxy,
            conf,
            iou_hard=cleanup_config.hybrid_iou_hard,
            iou_guard=cleanup_config.hybrid_iou_guard,
            dist_k=cleanup_config.hybrid_footpoint_dist_k,
            dist_min_px=cleanup_config.hybrid_footpoint_dist_min_px,
        )

    return DetectionFrame(
        frame_idx=frame.frame_idx,
        shape=frame.shape,
        xyxy=xyxy[keep],
        conf=conf[keep],
        cls=cls[keep],
    )


def preprocess_detection_frames(
    frames: list[DetectionFrame],
    cleanup_config: TrackingCleanupConfig,
) -> list[DetectionFrame]:
    return [filter_detection_frame(frame, cleanup_config) for frame in frames]
