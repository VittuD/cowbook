# frame_processor.py

import os
import math
from tqdm import tqdm
from processing import parse_json, extract_data, process_detections, project_to_ground
import legacy.image_utils as utils

def process_and_save_frames(json_file, camera_nr, output_image_folder, calibration_file):
    """
    Process detections from JSON and save each frame with projected centroids.
    """
    # Load calibration data
    mtx, dist = utils.get_calibrated_camera_model(calibration_file)

    # Load JSON tracking data
    json_data = parse_json(json_file)
    frames_data = extract_data(json_data)
    
    # Determine the number of digits for zero-padded filenames
    total_frames = len(frames_data)
    num_digits = math.ceil(math.log10(total_frames)) if total_frames > 1 else 1

    # Process each frame and save
    for frame in tqdm(frames_data, desc=f"Processing frames for camera {camera_nr}", unit="frame"):
        process_detections(frame, mtx, dist)
        projected_centroids = project_to_ground(
            [det["centroid"] for det in frame["detections"]],
            mtx, dist, camera_nr
        )
        
        # Save the processed frame with projected centroids
        frame_num_str = str(frame['frame_id']).zfill(num_digits)
        frame_output_path = os.path.join(output_image_folder, f"camera_{camera_nr}_frame_{frame_num_str}.png")
        utils.save_frame_image(projected_centroids, frame['frame_id'], frame_output_path)
