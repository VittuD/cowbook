# tracking.py

import cv2
import json
from ultralytics import YOLO
from tqdm import tqdm

def track_video_with_yolov8(video_path, output_json_path, model, save=False):
    """
    Track multiple object classes in a video using YOLOv8, saving results as JSON with refined tracking data.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    results = model.track(source=video_path, stream=True, save=save)
    tracking_data = {"frames": []}

    for frame_num, result in enumerate(tqdm(results, desc="Tracking video", total=total_frames)):
        frame_data = {
            "frame_id": frame_num,
            "detections": []
        }
        for box in result.boxes:
            bbox = [float(coord) for coord in box.xyxy[0]]
            detection_info = {
                "bbox": bbox,
                "class_id": int(box.cls),
                "track_id": int(box.id) if box.id is not None else None
            }
            frame_data["detections"].append(detection_info)
        tracking_data["frames"].append(frame_data)

    with open(output_json_path, 'w') as f:
        json.dump(tracking_data, f, indent=4)
    print(f"Tracking data saved to {output_json_path}")
