# tracking.py

import cv2
import json
from ultralytics import YOLO
from tqdm import tqdm  # Used for a progress bar during tracking

def track_video_with_yolov8(video_path, output_json_path, model, save=False):
    """
    Track objects in a video using YOLOv8 and save results as a JSON file.

    Parameters:
        video_path (str): Path to the input video file.
        output_json_path (str): Path to save the output JSON file containing tracking data.
        model (YOLO): YOLO model loaded with pre-trained weights for object detection and tracking.
        save (bool): If True, saves video output with tracking visualization. Default is False.

    This function processes each frame in the video to detect and track objects. 
    It extracts bounding boxes and class labels, then saves the results in JSON format.
    """
    # Open video to get total frame count
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Run YOLO tracking on the video
    results = model.track(source=video_path, stream=True, save=save)
    json_data = {"frames": []}  # Initialize the JSON data structure

    # Process each frame and collect tracking data
    for frame_num, result in enumerate(tqdm(results, desc="Tracking video", total=total_frames)):
        frame_data = {
            "frame_id": frame_num,
            "detections": {"xyxy": []},  # List of bounding boxes for detections
            "labels": []  # List of labels associated with each detection
        }
        for box in result.boxes:
            # Extract bounding box coordinates and class labels
            bbox = [float(coord) for coord in box.xyxy[0]]
            frame_data["detections"]["xyxy"].append(bbox)

            # Safely convert tensors or numbers to Python ints
            cls_val = box.cls.item() if hasattr(box.cls, "item") else box.cls
            id_raw = box.id if (box.id is not None) else None
            id_val = id_raw.item() if (id_raw is not None and hasattr(id_raw, "item")) else id_raw

            frame_data["labels"].append({
                "class_id": int(cls_val),
                "id": int(id_val) if id_val is not None else None
            })
        json_data["frames"].append(frame_data)  # Append data for each frame

    # Save tracking data to a JSON file
    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"Tracking data saved to {output_json_path}")

def load_yolo_model(model_path):
    """
    Load a YOLOv8 model with specified weights.

    Parameters:
        model_path (str): Path to the YOLO model weights file.

    Returns:
        YOLO: Loaded YOLO model instance.
    """
    return YOLO(model_path)
