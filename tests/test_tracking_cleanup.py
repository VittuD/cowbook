from __future__ import annotations

import json

import numpy as np

from cowbook.core.contracts import (
    Detections,
    TrackingCleanupConfig,
    TrackingDocument,
    TrackingFrame,
    TrackingLabel,
)
from cowbook.vision import tracking as tracking_module
from cowbook.vision.cleanup import (
    DetectionFrame,
    compute_short_track_ids,
    filter_detection_frame,
    postprocess_tracking_document,
    prune_detection_frames_by_track_ids,
)


def test_filter_detection_frame_applies_roi_and_hybrid_nms():
    cleanup = TrackingCleanupConfig.from_mapping(
        {
            "enabled": True,
            "roi": [[0, 0], [60, 0], [60, 60], [0, 60]],
            "nms_mode": "hybrid_nms",
            "conf_threshold": 0.15,
            "hybrid_iou_hard": 0.8,
            "hybrid_iou_guard": 0.1,
            "hybrid_footpoint_dist_k": 0.5,
            "hybrid_footpoint_dist_min_px": 5.0,
        }
    )
    frame = DetectionFrame(
        frame_idx=0,
        shape=(100, 100),
        xyxy=np.asarray(
            [
                [10, 10, 30, 30],
                [11, 11, 31, 31],
                [70, 70, 90, 90],
            ],
            dtype=np.float32,
        ),
        conf=np.asarray([0.9, 0.8, 0.95], dtype=np.float32),
        cls=np.asarray([0, 0, 0], dtype=np.int32),
    )

    filtered = filter_detection_frame(frame, cleanup)

    assert filtered.xyxy.shape[0] == 1
    assert filtered.xyxy.tolist() == [[10.0, 10.0, 30.0, 30.0]]


def test_compute_short_track_ids_and_prune_detection_frames():
    detection_frames = [
        DetectionFrame(
            frame_idx=0,
            shape=(48, 64),
            xyxy=np.asarray([[1, 1, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
            conf=np.asarray([0.9, 0.8], dtype=np.float32),
            cls=np.asarray([0, 0], dtype=np.int32),
        ),
        DetectionFrame(
            frame_idx=1,
            shape=(48, 64),
            xyxy=np.asarray([[2, 2, 11, 11]], dtype=np.float32),
            conf=np.asarray([0.9], dtype=np.float32),
            cls=np.asarray([0], dtype=np.int32),
        ),
        DetectionFrame(
            frame_idx=3,
            shape=(48, 64),
            xyxy=np.asarray([[3, 3, 12, 12]], dtype=np.float32),
            conf=np.asarray([0.9], dtype=np.float32),
            cls=np.asarray([0], dtype=np.int32),
        ),
    ]
    document = TrackingDocument(
        frames=[
            TrackingFrame(
                frame_id=0,
                detections=Detections(xyxy=[[1, 1, 10, 10], [20, 20, 30, 30]]),
                labels=[
                    TrackingLabel(class_id=0, id=100, det_idx=0, real=1, src="tracker"),
                    TrackingLabel(class_id=0, id=200, det_idx=1, real=1, src="tracker"),
                ],
            ),
            TrackingFrame(
                frame_id=1,
                detections=Detections(xyxy=[[2, 2, 11, 11]]),
                labels=[TrackingLabel(class_id=0, id=100, det_idx=0, real=1, src="tracker")],
            ),
            TrackingFrame(
                frame_id=3,
                detections=Detections(xyxy=[[3, 3, 12, 12]]),
                labels=[TrackingLabel(class_id=0, id=100, det_idx=0, real=1, src="tracker")],
            ),
        ]
    )

    short_ids = compute_short_track_ids(document, min_track_length=3, gap_tolerance=1)
    pruned = prune_detection_frames_by_track_ids(detection_frames, document, short_ids)

    assert short_ids == {200}
    assert pruned[0].xyxy.tolist() == [[1.0, 1.0, 10.0, 10.0]]
    assert pruned[1].xyxy.tolist() == [[2.0, 2.0, 11.0, 11.0]]
    assert pruned[2].xyxy.tolist() == [[3.0, 3.0, 12.0, 12.0]]


def test_compute_short_track_ids_uses_gap_tolerant_streaks():
    document = TrackingDocument(
        frames=[
            TrackingFrame(
                frame_id=0,
                detections=Detections(xyxy=[[1, 1, 10, 10]]),
                labels=[TrackingLabel(class_id=0, id=100, det_idx=0, real=1, src="tracker")],
            ),
            TrackingFrame(
                frame_id=2,
                detections=Detections(xyxy=[[2, 2, 11, 11]]),
                labels=[TrackingLabel(class_id=0, id=100, det_idx=0, real=1, src="tracker")],
            ),
            TrackingFrame(
                frame_id=4,
                detections=Detections(xyxy=[[4, 4, 13, 13]]),
                labels=[TrackingLabel(class_id=0, id=100, det_idx=0, real=1, src="tracker")],
            ),
        ]
    )

    assert compute_short_track_ids(document, min_track_length=3, gap_tolerance=1) == set()
    assert compute_short_track_ids(document, min_track_length=3, gap_tolerance=0) == {100}


def test_postprocess_tracking_document_gap_fills_and_marks_synthetic_frames():
    cleanup = TrackingCleanupConfig.from_mapping(
        {
            "enabled": True,
            "postprocess_smoothing": True,
            "gap_fill_max_frames": 1,
            "smoothing_alpha": 0.65,
        }
    )
    document = TrackingDocument(
        frames=[
            TrackingFrame(
                frame_id=0,
                detections=Detections(xyxy=[[10, 10, 20, 20]]),
                labels=[TrackingLabel(class_id=0, id=7, det_idx=0, real=1, src="tracker")],
            ),
            TrackingFrame(frame_id=1, detections=Detections(xyxy=[]), labels=[]),
            TrackingFrame(
                frame_id=2,
                detections=Detections(xyxy=[[14, 14, 24, 24]]),
                labels=[TrackingLabel(class_id=0, id=7, det_idx=0, real=1, src="tracker")],
            ),
        ]
    )

    postprocessed = postprocess_tracking_document(document, cleanup)

    mid_frame = postprocessed.frames[1]
    assert len(mid_frame.detections.xyxy) == 1
    assert mid_frame.labels[0].real == 0
    assert mid_frame.labels[0].src == "gap_fill"

    final_label = postprocessed.frames[2].labels[0]
    assert final_label.src in {"tracker", "smooth"}


def test_track_video_with_yolo_uses_cleanup_path_when_enabled(tmp_path, monkeypatch):
    output_json = tmp_path / "tracking.json"
    calls: list[str] = []

    fake_frames = [
        DetectionFrame(
            frame_idx=0,
            shape=(48, 64),
            xyxy=np.asarray([[1, 1, 10, 10]], dtype=np.float32),
            conf=np.asarray([0.9], dtype=np.float32),
            cls=np.asarray([0], dtype=np.int32),
        )
    ]
    fake_doc = TrackingDocument(
        frames=[
            TrackingFrame(
                frame_id=0,
                detections=Detections(xyxy=[[1, 1, 10, 10]]),
                labels=[TrackingLabel(class_id=0, id=1, det_idx=0, real=1, src="tracker")],
            )
        ]
    )

    monkeypatch.setattr(
        tracking_module,
        "detect_video_to_frames",
        lambda *args, **kwargs: calls.append("detect") or fake_frames,
    )
    monkeypatch.setattr(
        tracking_module,
        "preprocess_detection_frames",
        lambda frames, *_args, **_kwargs: calls.append("preprocess") or frames,
    )
    monkeypatch.setattr(
        tracking_module,
        "track_from_detection_frames",
        lambda *args, **kwargs: calls.append("track") or fake_doc,
    )
    monkeypatch.setattr(
        tracking_module,
        "postprocess_tracking_document",
        lambda doc, *_args, **_kwargs: calls.append("smooth") or doc,
    )

    tracking_module.track_video_with_yolo(
        "video.mp4",
        str(output_json),
        "model.pt",
        tracking_cleanup={"enabled": True, "postprocess_smoothing": True},
    )

    saved = json.loads(output_json.read_text())
    assert calls == ["detect", "preprocess", "track", "smooth"]
    assert saved["frames"][0]["labels"][0]["det_idx"] == 0
