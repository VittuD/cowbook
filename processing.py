# processing.py

import json
import legacy.image_utils as utils  # Legacy utilities for image processing

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
    
    Logic:
        Iterates over each frame in the JSON data, calculates centroids for each detection,
        and appends relevant data to a list of frames.
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

def process_detections(frame_data, mtx, dist):
    """
    Undistort bounding boxes and centroids for a given frame.

    Parameters:
        frame_data (dict): A dictionary containing detections data for a frame.
        mtx (ndarray): Camera matrix from calibration.
        dist (ndarray): Distortion coefficients from calibration.

    Returns:
        dict: Updated frame data with undistorted bounding boxes and centroids.
    
    Logic:
        Each bounding box and centroid is undistorted using the provided camera matrix 
        and distortion coefficients.
    """
    for detection in frame_data['detections']:
        detection['bbox'] = utils.undistort_points_given(detection['bbox'], mtx, dist)
        detection['centroid'] = utils.undistort_points_given(detection['centroid'], mtx, dist)[0]
    return frame_data

def project_to_ground(centroids, mtx, dist, channel):
    """
    Project centroids onto the ground plane based on camera calibration.

    Parameters:
        centroids (list): List of centroids to project.
        mtx (ndarray): Camera matrix.
        dist (ndarray): Distortion coefficients.
        channel (int): Camera channel identifier for projection.

    Returns:
        list: Ground-projected coordinates of the centroids.
    
    Logic:
        Utilizes camera parameters and a specific channel identifier to project each centroid
        onto the ground plane for spatial analysis.
    """
    return utils.groundProjectPoint(channel, mtx, dist, centroids)
