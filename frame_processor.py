# frame_processor.py

import os
from processing import parse_json, extract_data

def process_and_save_frames(json_file, camera_nr, output_image_folder):
    """Process detections from JSON and save each frame."""
    json_data = parse_json(json_file)
    frames_data = extract_data(json_data)
    
    for frame in frames_data:
        # Saving frames without projection or calibration handling
        frame_output_path = os.path.join(output_image_folder, f"camera_{camera_nr}_frame_{frame['frame_id']}.png")
        # Dummy save function (in real usage, it would save actual images)
        print(f"Saving frame to {frame_output_path}")
