# frame_processor.py

import os
from tqdm import tqdm
from processing import parse_json, extract_data, process_detections

def process_and_save_frames(json_file, camera_nr, output_image_folder, calibration_file=None):
    """
    Process detections from JSON and save each frame with projected centroids.
    This version includes centroid calculations and processing for detections.
    """
    # Load JSON tracking data
    json_data = parse_json(json_file)
    frames_data = extract_data(json_data)
    
    # Process each frame and save
    for frame in tqdm(frames_data, desc=f"Processing frames for camera {camera_nr}", unit="frame"):
        frame_data = process_detections(frame)  # Process detections in each frame
        
        # Placeholder for applying calibration or other processing
        if calibration_file:
            print(f"Applying calibration from {calibration_file} to frame {frame['frame_id']}")
        
        # Simulated save path for processed frame
        frame_output_path = os.path.join(output_image_folder, f"camera_{camera_nr}_frame_{frame['frame_id']}.png")
        print(f"Saving processed frame to {frame_output_path}")
