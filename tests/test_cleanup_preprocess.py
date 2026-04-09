from __future__ import annotations

import numpy as np
import pytest

from cowbook.core.contracts import TrackingCleanupConfig
from cowbook.vision.cleanup import (
    DetectionFrame,
    filter_detection_frame,
    preprocess_detection_frames,
)
from cowbook.vision.cleanup.preprocess import (
    clip_boxes,
    footprint_nms_xyxy,
    hybrid_nms_xyxy,
    iou_nms_xyxy,
    point_in_poly,
)


def test_clip_boxes_and_point_in_poly_cover_boundary_helpers():
    boxes = np.asarray([[-5, -5, 15, 25], [90, 95, 120, 130]], dtype=np.float32)

    clipped = clip_boxes(boxes, width=100, height=100)

    assert clipped.tolist() == [[0.0, 0.0, 15.0, 25.0], [90.0, 95.0, 99.0, 99.0]]
    polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_poly(5, 5, polygon) is True
    assert point_in_poly(15, 5, polygon) is False


def test_nms_variants_suppress_near_duplicates():
    boxes = np.asarray(
        [
            [10, 10, 30, 30],
            [11, 11, 31, 31],
            [70, 70, 90, 90],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.95, 0.80, 0.70], dtype=np.float32)

    iou_keep = iou_nms_xyxy(boxes, scores, iou_thr=0.5)
    footprint_keep = footprint_nms_xyxy(
        boxes,
        scores,
        dist_k=0.5,
        dist_min_px=5.0,
        iou_guard=0.1,
    )
    hybrid_keep = hybrid_nms_xyxy(
        boxes,
        scores,
        iou_hard=0.8,
        iou_guard=0.1,
        dist_k=0.5,
        dist_min_px=5.0,
    )

    assert iou_keep.tolist() == [0, 2]
    assert footprint_keep.tolist() == [0, 2]
    assert hybrid_keep.tolist() == [0, 2]


def test_filter_detection_frame_applies_edge_area_aspect_and_iou_filters():
    cleanup = TrackingCleanupConfig.from_mapping(
        {
            "enabled": True,
            "nms_mode": "iou_nms",
            "nms_iou": 0.5,
            "conf_threshold": 0.3,
            "drop_edge_boxes": True,
            "edge_margin_px": 5,
            "min_area_px": 100,
            "max_area_px": 900,
            "min_aspect_ratio": 0.5,
            "max_aspect_ratio": 2.0,
        }
    )
    frame = DetectionFrame(
        frame_idx=0,
        shape=(100, 100),
        xyxy=np.asarray(
            [
                [0, 0, 12, 12],      # edge -> dropped
                [10, 10, 15, 15],    # too small -> dropped
                [20, 20, 70, 30],    # aspect too wide -> dropped
                [20, 20, 40, 40],    # kept
                [21, 21, 41, 41],    # IoU-suppressed
            ],
            dtype=np.float32,
        ),
        conf=np.asarray([0.9, 0.9, 0.9, 0.95, 0.8], dtype=np.float32),
        cls=np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
    )

    filtered = filter_detection_frame(frame, cleanup)

    assert filtered.xyxy.tolist() == [[20.0, 20.0, 40.0, 40.0]]
    assert filtered.conf.tolist() == pytest.approx([0.95])
    assert filtered.cls.tolist() == [1]


def test_filter_detection_frame_applies_area_ratio_filters():
    cleanup = TrackingCleanupConfig.from_mapping(
        {
            "enabled": True,
            "nms_mode": "iou_nms",
            "nms_iou": 0.5,
            "min_area_ratio": 0.03,
            "max_area_ratio": 0.20,
        }
    )
    frame = DetectionFrame(
        frame_idx=0,
        shape=(100, 100),
        xyxy=np.asarray(
            [
                [10, 10, 20, 20],   # 100 px -> 0.01, too small
                [10, 10, 70, 70],   # 3600 px -> 0.36, too large
                [20, 20, 60, 50],   # 1200 px -> 0.12, kept
            ],
            dtype=np.float32,
        ),
        conf=np.asarray([0.9, 0.9, 0.95], dtype=np.float32),
        cls=np.asarray([0, 0, 1], dtype=np.int32),
    )

    filtered = filter_detection_frame(frame, cleanup)

    assert filtered.xyxy.tolist() == [[20.0, 20.0, 60.0, 50.0]]
    assert filtered.conf.tolist() == pytest.approx([0.95])
    assert filtered.cls.tolist() == [1]


def test_preprocess_detection_frames_runs_framewise_with_footpoint_mode():
    cleanup = TrackingCleanupConfig.from_mapping(
        {
            "enabled": True,
            "nms_mode": "footpoint_nms",
            "footpoint_dist_k": 0.5,
            "footpoint_dist_min_px": 5.0,
            "footpoint_iou_guard": 0.1,
            "roi": [[0, 0], [50, 0], [50, 50], [0, 50]],
        }
    )
    frames = [
        DetectionFrame(
            frame_idx=0,
            shape=(60, 60),
            xyxy=np.asarray([[10, 10, 30, 30], [11, 11, 31, 31]], dtype=np.float32),
            conf=np.asarray([0.9, 0.8], dtype=np.float32),
            cls=np.asarray([0, 0], dtype=np.int32),
        ),
        DetectionFrame(
            frame_idx=1,
            shape=(60, 60),
            xyxy=np.asarray([[70, 70, 90, 90]], dtype=np.float32),
            conf=np.asarray([0.9], dtype=np.float32),
            cls=np.asarray([0], dtype=np.int32),
        ),
    ]

    processed = preprocess_detection_frames(frames, cleanup)

    assert len(processed) == 2
    assert processed[0].xyxy.tolist() == [[10.0, 10.0, 30.0, 30.0]]
    assert processed[1].xyxy.tolist() == []
