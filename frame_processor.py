import json
import os
import math
from tqdm import tqdm
from processing import parse_json, extract_data, reconstruct_json, process_detections, project_to_ground, extract_projected_centroids_from_files
import legacy.image_utils as utils  # Legacy utilities for image processing

def process_and_save_frames(json_file_paths, camera_nrs, output_image_folder, calibration_file):
    """
    Process detections from each of the JSON files, project centroids,
    later saves each frame as an image by comining the projected points of each JSON.
    Inputs: json_file_paths: list of paths to JSON files containing tracking data
            camera_nr: list of camera numbers, one for each JSON file
            output_image_folder: path to the output folder where images will be saved
            calibration_file: path to the calibration file for the cameras
    """
    # Process centroids for each JSON file
    updated_json_file_paths = []
    for json_file_path, camera_nr in zip(json_file_paths, camera_nrs):
        frames_data = process_centroids(json_file_path, camera_nr, calibration_file)
        
        # Save updated frame data with projected centroids to a new JSON file, replace .json with _processed.json
        updated_json_file_path = os.path.join(json_file_path.replace(".json", "_processed.json"))
        save_frame_data_json(frames_data, updated_json_file_path)
        updated_json_file_paths.append(updated_json_file_path)
        print(f"Processed frame data saved to {updated_json_file_path}")
    
    # Plot combined projected centroids from multiple JSON files for each frame and save as images
    base_filename = os.path.join(output_image_folder, "combined_projected_centroids")
    plot_combined_projected_centroids(updated_json_file_paths, base_filename)

def process_centroids(json_file, camera_nr, calibration_file):
    """
    Process detections from JSON, project centroids, and return the updated data.
    """
    # Load calibration data
    mtx, dist = utils.get_calibrated_camera_model(calibration_file)

    # Load and extract data from the JSON tracking file
    json_data = parse_json(json_file)
    frames_data = extract_data(json_data)
    
    # Process each frame
    for frame in tqdm(frames_data, desc=f"Processing frames for camera {camera_nr}", unit="frame"):
        # Step 1: Undistort detections and centroids
        frame = process_detections(frame, mtx, dist)
        
        # Step 2: Project centroids to the ground plane
        projected_centroids = project_to_ground(
            [detection["centroid"] for detection in frame["detections"]],
            mtx, dist, camera_nr
        )
        
        # Step 3: Add projected centroids to each detection and prepare for JSON output
        for detection, projected_centroid in zip(frame["detections"], projected_centroids):
            detection["projected_centroid"] = projected_centroid

    return frames_data

def save_frame_data_json(frames_data, output_json_path):
    """
    Save updated frame data with projected centroids to a new JSON file.
    """
    frames_data_json = reconstruct_json(frames_data)
    with open(output_json_path, 'w') as output_file:
        json.dump(frames_data_json, output_file, indent=4)

def plot_combined_projected_centroids(json_file_paths, base_filename):
    """
    Plot combined projected centroids from multiple JSON files for each frame and save as images.

    Parameters:
        json_file_paths (list): List of paths to JSON files containing frame data.
        base_filename (str): Base name for the output images.

    Returns:
        None
    """
    # Extract projected centroids from each JSON file
    all_projected_centroids = extract_projected_centroids_from_files(json_file_paths)

    # Determine the number of digits for zero-padded filenames based on total frames
    total_frames = len(all_projected_centroids)
    num_digits = math.ceil(math.log10(total_frames)) if total_frames > 1 else 1

    # Iterate over each frame and plot combined centroids
    for frame_id in sorted(all_projected_centroids.keys()):
        projected_centroids = all_projected_centroids[frame_id]
        
        # Zero-pad the frame number for filename consistency
        frame_num_str = str(frame_id).zfill(num_digits)
        frame_output_path = f"{base_filename}_frame_{frame_num_str}.png"

        # Save the frame image with the projected centroids
        utils.save_frame_image(projected_centroids, frame_id, frame_output_path)

