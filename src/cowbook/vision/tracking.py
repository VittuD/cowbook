# tracking.py

import json
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

def _track_video_direct(video_path, output_json_path, model_path, save=False):
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
    # Load the YOLO model from the provided model path
    model = load_yolo_model(model_path)

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
    )
    frames: list[TrackingFrame] = []

    # Process each frame and collect tracking data
    for frame_num, result in enumerate(tqdm(results, desc="Tracking video", total=total_frames)):
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
    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    logger.info("Tracking data saved to %s", output_json_path)

    # Unload the model to free resources
    del model


def _track_video_with_cleanup(
    video_path: str,
    output_json_path: str,
    model_path: str,
    *,
    save: bool,
    cleanup_config: TrackingCleanupConfig,
) -> None:
    tracker_yaml_path = assets_root() / "trackers" / "cows_botsort.yaml"
    detection_frames = detect_video_to_frames(video_path, model_path, cleanup_config)
    preprocessed_frames = preprocess_detection_frames(detection_frames, cleanup_config)
    video_output_path = None
    if save:
        video_output_path = str(Path(output_json_path).with_suffix(".mp4"))

    tracked = track_from_detection_frames(
        video_path,
        preprocessed_frames,
        tracker_yaml_path,
        save_video_path=video_output_path,
    )

    if cleanup_config.two_pass_prune_short_tracks:
        short_track_ids = compute_short_track_ids(tracked, cleanup_config.min_track_length)
        pruned_frames = prune_detection_frames_by_track_ids(preprocessed_frames, tracked, short_track_ids)
        tracked = track_from_detection_frames(
            video_path,
            pruned_frames,
            tracker_yaml_path,
            save_video_path=video_output_path,
        )

    if cleanup_config.postprocess_smoothing:
        tracked = postprocess_tracking_document(tracked, cleanup_config)

    with open(output_json_path, "w") as f:
        json.dump(tracked.to_dict(), f, indent=4)
    logger.info("Tracking data saved to %s", output_json_path)


def track_video_with_yolo(
    video_path,
    output_json_path,
    model_path,
    save=False,
    tracking_cleanup: dict | None = None,
):
    cleanup_config = TrackingCleanupConfig.from_mapping(tracking_cleanup)
    if cleanup_config.enabled:
        _track_video_with_cleanup(
            video_path,
            output_json_path,
            model_path,
            save=save,
            cleanup_config=cleanup_config,
        )
        return
    _track_video_direct(video_path, output_json_path, model_path, save=save)

def load_yolo_model(model_path):
    """
    Load a YOLO model with specified weights.

    Parameters:
        model_path (str): Path to the YOLO model weights file.

    Returns:
        YOLO: Loaded YOLO model instance.
    """
    return YOLO(model_path)
