# tracking.py

import json
import logging

import cv2
from tqdm import tqdm  # Used for a progress bar during tracking
from ultralytics import YOLO

from cowbook.core.contracts import Detections, TrackingDocument, TrackingFrame, TrackingLabel
from cowbook.core.runtime import assets_root

logger = logging.getLogger(__name__)

def track_video_with_yolo(video_path, output_json_path, model_path, save=False):
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

def load_yolo_model(model_path):
    """
    Load a YOLO model with specified weights.

    Parameters:
        model_path (str): Path to the YOLO model weights file.

    Returns:
        YOLO: Loaded YOLO model instance.
    """
    return YOLO(model_path)
