# processing.py

import json
import numpy as np
import cv2
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
            'detections': [
                {
                    'bbox': bbox,
                    'centroid': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                }
                for bbox in frame['detections']['xyxy']
            ],
            # Updated to store class_id and id for each label
            'labels': [{"class_id": label["class_id"], "id": label["id"]} for label in frame.get('labels', [])]
        }
        frames_data.append(frame_data)
    return frames_data

def extract_projected_centroids_from_files(json_file_paths):
    """
    Extract projected centroids from each frame in multiple JSON files.

    Parameters:
        json_file_paths (list): List of paths to JSON files containing frame data.

    Returns:
        dict: A dictionary where keys are frame IDs and values are lists of projected centroids.
    """
    all_projected_centroids = {}

    for json_file_path in json_file_paths:
        print(f"Extracting projected centroids from {json_file_path}")
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        for frame in data.get('frames', []):
            frame_id = frame['frame_id']
            projected_centroids = frame['detections'].get('projected_centroids', [])
            if frame_id not in all_projected_centroids:
                all_projected_centroids[frame_id] = []
            all_projected_centroids[frame_id].extend(projected_centroids)
    
    return all_projected_centroids

def convert_arrays_to_lists(data):
    """
    Recursively convert numpy arrays within the data structure to lists for JSON serialization.
    
    Parameters:
        data (any): The data to process, typically a dict or list.
    
    Returns:
        any: A new data structure where all numpy arrays are replaced by lists.
    """
    if isinstance(data, dict):
        return {key: convert_arrays_to_lists(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_arrays_to_lists(element) for element in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy array to a list
    else:
        return data

def reconstruct_json(frames_data):
    """
    Reconstruct JSON structure from extracted detection data and convert all arrays to lists.
    
    Parameters:
        frames_data (list): A list of dictionaries, each containing frame ID,
                            bounding boxes, centroids, and labels for each detection.
    
    Returns:
        dict: A JSON-like dictionary with the original structure,
              containing frames and detections with bounding boxes.
    """
    json_data = {'frames': []}
    for frame_data in frames_data:
        frame = {
            'frame_id': frame_data['frame_id'],
            'detections': {
                'xyxy': [convert_arrays_to_lists(detection['bbox']) for detection in frame_data['detections']],
                'centroids': [convert_arrays_to_lists(detection['centroid']) for detection in frame_data['detections']],
                # Convert 'projected_centroid' if present in each detection
                'projected_centroids': [convert_arrays_to_lists(detection.get('projected_centroid')) for detection in frame_data['detections']]
            },
            'labels': [{"class_id": label["class_id"], "id": label["id"]} for label in frame_data.get('labels', [])]
        }
        json_data['frames'].append(frame)
    
    # Final conversion to ensure all arrays are converted
    json_data = convert_arrays_to_lists(json_data)

    return json_data

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
    dets = frame_data['detections']
    if not dets:
        frame_data['centroids'] = []
        return frame_data

    # Build arrays: two corners per bbox, and one centroid per detection
    bps = []
    cps = []
    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        bps.extend([[x1, y1], [x2, y2]])
        cx, cy = d['centroid']
        cps.append([cx, cy])

    # Batch undistort
    bps_u = utils.undistort_points_given(bps, mtx, dist)  # shape (2N, 2)
    cps_u = utils.undistort_points_given(cps, mtx, dist)  # shape (N, 2)

    # Write back undistorted bbox corners and centroids
    centroids_out = []
    for i, d in enumerate(dets):
        (x1u, y1u) = bps_u[2*i + 0]
        (x2u, y2u) = bps_u[2*i + 1]
        d['bbox'] = [float(x1u), float(y1u), float(x2u), float(y2u)]

        (cxu, cyu) = cps_u[i]
        d['centroid'] = [float(cxu), float(cyu)]
        centroids_out.append(d['centroid'])

    frame_data['centroids'] = centroids_out
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

# Example function that might be used to save each processed frame as an image (if needed)
def save_frame_image(projected_points, frame_num, base_filename):
    """
    Save processed frame with projected points to an image file.
    
    Args:
        projected_points (list): List of points projected on the ground plane.
        frame_num (int): Frame number for naming.
        base_filename (str): Base name for the output image.
    """
    # Convert projected points to an image format if necessary
    barn_image = utils.points_to_barn(projected_points)
    output_filename = f"{base_filename}_frame{frame_num}.png"
    cv2.imwrite(output_filename, barn_image[1])
