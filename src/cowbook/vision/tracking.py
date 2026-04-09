# tracking.py
import logging
from pathlib import Path

import cv2
from tqdm import tqdm  # Used for a progress bar during tracking
from ultralytics import YOLO

from cowbook.core.contracts import (
    Detections,
    TrackingCleanupConfig,
    TrackingDocument,
    TrackingFrame,
    TrackingLabel,
)
from cowbook.core.runtime import assets_root
from cowbook.execution.observers import JobReporter
from cowbook.execution.progress import TrackingProgressReporter
from cowbook.io.json_utils import dump_path_compact
from cowbook.vision.cleanup import (
    compute_short_track_ids,
    postprocess_tracking_document,
    preprocess_detection_frames,
    prune_detection_frames_by_track_ids,
)
from cowbook.vision.tracking_cleanup import (
    detect_video_to_frames,
    track_from_detection_frames,
)

logger = logging.getLogger(__name__)

def _track_video_direct(
    video_path,
    output_json_path,
    model_path,
    save=False,
    *,
    model: YOLO | None = None,
    log_progress: bool = False,
    reporter: JobReporter | None = None,
    group_idx: int | None = None,
    camera_nr: int | None = None,
    progress_event_sink=None,
):
    """
    Track objects in a video using YOLOv8 and save results as a JSON file.

    Parameters:
        video_path (str): Path to the input video file.
        output_json_path (str): Path to save the output JSON file containing tracking data.
        model_path (str): Path to the YOLO model weights file.
        save (bool): If True, saves video output with tracking visualization. Default is False.

    This function loads the YOLO model, processes each frame in the video to detect and track objects,
    extracts bounding boxes and class labels, then saves the results in JSON format. The model is unloaded
    at the end to free resources.
    """
    # Reuse a caller-provided model when available. Ultralytics resets tracker
    # state per call when persist=False, so this preserves tracking semantics.
    owns_model = model is None
    model = model or load_yolo_model(model_path)

    # Open video to get total frame count
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Run YOLO tracking on the video
    results = model.track(
        source=video_path,
        stream=True,
        save=save,
        conf=0.45,      # ↑ fewer false positives
        iou=0.5,        # NMS
        tracker=str(assets_root() / "trackers" / "cows_botsort.yaml"),
        verbose=False,
    )
    frames: list[TrackingFrame] = []
    progress_reporter = TrackingProgressReporter(
        tracking_mode="direct",
        stage_name="direct",
        video_path=video_path,
        camera_nr=camera_nr,
        frame_total=total_frames or None,
        log_progress=log_progress,
        reporter=reporter,
        group_idx=group_idx,
        event_sink=progress_event_sink,
    )
    if log_progress or reporter is not None or progress_event_sink is not None:
        progress_reporter.stage_started()

    # Process each frame and collect tracking data
    iterable = (
        results
        if (log_progress or reporter is not None or progress_event_sink is not None)
        else tqdm(results, desc="Tracking video", total=total_frames)
    )
    for frame_num, result in enumerate(iterable):
        if log_progress or reporter is not None or progress_event_sink is not None:
            progress_reporter.frame_progress(frame_num + 1, total_frames or None)
        boxes: list[list[float]] = []
        labels: list[TrackingLabel] = []
        for box in result.boxes:
            # Extract bounding box coordinates and class labels
            bbox = [float(coord) for coord in box.xyxy[0]]
            boxes.append(bbox)

            # Safely convert tensors or numbers to Python ints
            cls_val = box.cls.item() if hasattr(box.cls, "item") else box.cls
            id_raw = box.id if (box.id is not None) else None
            id_val = id_raw.item() if (id_raw is not None and hasattr(id_raw, "item")) else id_raw

            labels.append(
                TrackingLabel(
                    class_id=int(cls_val),
                    id=int(id_val) if id_val is not None else None,
                )
            )
        frames.append(
            TrackingFrame(
                frame_id=frame_num,
                detections=Detections(xyxy=boxes),
                labels=labels,
            )
        )

    # Save tracking data to a JSON file
    json_data = TrackingDocument(frames=frames).to_dict()
    dump_path_compact(output_json_path, json_data)
    logger.info("Tracking data saved to %s", output_json_path)
    if log_progress or reporter is not None or progress_event_sink is not None:
        progress_reporter.stage_completed()

    # Unload the model to free resources
    if owns_model:
        del model


def _track_video_with_cleanup(
    video_path: str,
    output_json_path: str,
    model_path: str,
    *,
    save: bool,
    cleanup_config: TrackingCleanupConfig,
    model: YOLO | None = None,
    log_progress: bool = False,
    reporter: JobReporter | None = None,
    group_idx: int | None = None,
    camera_nr: int | None = None,
    progress_event_sink=None,
) -> None:
    tracker_yaml_path = assets_root() / "trackers" / "cows_botsort.yaml"
    detect_progress = TrackingProgressReporter(
        tracking_mode="cleanup",
        stage_name="detect",
        video_path=video_path,
        camera_nr=camera_nr,
        log_progress=log_progress,
        reporter=reporter,
        group_idx=group_idx,
        event_sink=progress_event_sink,
    )
    detect_progress.stage_started()
    detection_frames = detect_video_to_frames(
        video_path,
        model_path,
        cleanup_config,
        model=model,
        progress_reporter=detect_progress,
    )
    detect_progress.stage_completed()

    preprocess_progress = TrackingProgressReporter(
        tracking_mode="cleanup",
        stage_name="preprocess",
        video_path=video_path,
        camera_nr=camera_nr,
        log_progress=log_progress,
        reporter=reporter,
        group_idx=group_idx,
        event_sink=progress_event_sink,
    )
    preprocess_progress.stage_started()
    preprocessed_frames = preprocess_detection_frames(detection_frames, cleanup_config)
    preprocess_progress.stage_completed()
    video_output_path = None
    if save:
        video_output_path = str(Path(output_json_path).with_suffix(".mp4"))

    pass1_progress = TrackingProgressReporter(
        tracking_mode="cleanup",
        stage_name="cleanup_pass1",
        video_path=video_path,
        camera_nr=camera_nr,
        frame_total=len(preprocessed_frames),
        log_progress=log_progress,
        reporter=reporter,
        group_idx=group_idx,
        event_sink=progress_event_sink,
    )
    pass1_progress.stage_started()
    tracked = track_from_detection_frames(
        video_path,
        preprocessed_frames,
        tracker_yaml_path,
        save_video_path=video_output_path,
        progress_reporter=pass1_progress,
    )
    pass1_progress.stage_completed()

    if cleanup_config.two_pass_prune_short_tracks:
        prune_progress = TrackingProgressReporter(
            tracking_mode="cleanup",
            stage_name="prune",
            video_path=video_path,
            camera_nr=camera_nr,
            log_progress=log_progress,
            reporter=reporter,
            group_idx=group_idx,
            event_sink=progress_event_sink,
        )
        prune_progress.stage_started()
        short_track_ids = compute_short_track_ids(tracked, cleanup_config.min_track_length)
        pruned_frames = prune_detection_frames_by_track_ids(preprocessed_frames, tracked, short_track_ids)
        prune_progress.stage_completed()
        pass2_progress = TrackingProgressReporter(
            tracking_mode="cleanup",
            stage_name="cleanup_pass2",
            video_path=video_path,
            camera_nr=camera_nr,
            frame_total=len(pruned_frames),
            log_progress=log_progress,
            reporter=reporter,
            group_idx=group_idx,
            event_sink=progress_event_sink,
        )
        pass2_progress.stage_started()
        tracked = track_from_detection_frames(
            video_path,
            pruned_frames,
            tracker_yaml_path,
            save_video_path=video_output_path,
            progress_reporter=pass2_progress,
        )
        pass2_progress.stage_completed()

    if cleanup_config.postprocess_smoothing:
        postprocess_progress = TrackingProgressReporter(
            tracking_mode="cleanup",
            stage_name="postprocess",
            video_path=video_path,
            camera_nr=camera_nr,
            log_progress=log_progress,
            reporter=reporter,
            group_idx=group_idx,
            event_sink=progress_event_sink,
        )
        postprocess_progress.stage_started()
        tracked = postprocess_tracking_document(tracked, cleanup_config)
        postprocess_progress.stage_completed()

    dump_path_compact(output_json_path, tracked.to_dict())
    logger.info("Tracking data saved to %s", output_json_path)


def track_video_with_yolo(
    video_path,
    output_json_path,
    model_path,
    save=False,
    tracking_cleanup: dict | None = None,
    log_progress: bool = False,
    reporter: JobReporter | None = None,
    group_idx: int | None = None,
    camera_nr: int | None = None,
    progress_event_sink=None,
    model: YOLO | None = None,
):
    cleanup_config = TrackingCleanupConfig.from_mapping(tracking_cleanup)
    if cleanup_config.enabled:
        _track_video_with_cleanup(
            video_path,
            output_json_path,
            model_path,
            save=save,
            cleanup_config=cleanup_config,
            model=model,
            log_progress=log_progress,
            reporter=reporter,
            group_idx=group_idx,
            camera_nr=camera_nr,
            progress_event_sink=progress_event_sink,
        )
        return
    _track_video_direct(
        video_path,
        output_json_path,
        model_path,
        save=save,
        model=model,
        log_progress=log_progress,
        reporter=reporter,
        group_idx=group_idx,
        camera_nr=camera_nr,
        progress_event_sink=progress_event_sink,
    )

def load_yolo_model(model_path):
    """
    Load a YOLO model with specified weights.

    Parameters:
        model_path (str): Path to the YOLO model weights file.

    Returns:
        YOLO: Loaded YOLO model instance.
    """
    return YOLO(model_path, task="detect")
