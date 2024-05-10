# processing.py

import json

def parse_json(json_file_path):
    """
    Load and parse JSON tracking data.

    Parameters:
        json_file_path (str): Path to JSON file containing tracking data.

    Returns:
        dict: Parsed JSON data as a Python dictionary.
    """
    with open(json_file_path) as file:
        return json.load(file)

def extract_data(json_data):
    """
    Extract detection data from parsed JSON and compute centroids.

    Parameters:
        json_data (dict): Parsed JSON data with frames and detections.

    Returns:
        list: A list of dictionaries, each containing frame ID, bounding boxes,
              centroids, and labels for each detection in that frame.
    """
    frames_data = []
    for frame in json_data['frames']:
        frame_data = {
            'frame_id': frame['frame_id'],
            'detections': [{'bbox': bbox, 'centroid': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]}
                           for bbox in frame['detections']['xyxy']],
            'labels': frame.get('labels', [])
        }
        frames_data.append(frame_data)
    return frames_data

def process_detections(frame_data):
    """
    Process each detection by normalizing and filtering.

    Parameters:
        frame_data (dict): Data for each frame's detections.

    Returns:
        list: Filtered and processed detection data.
    """
    processed_data = []
    for detection in frame_data['detections']:
        # Normalize or process bounding boxes here if needed
        # Placeholder for possible future processing steps
        processed_data.append(detection)
    return processed_data
