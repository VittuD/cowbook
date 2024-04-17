# main.py

import os
import argparse
from config_loader import load_config
from directory_manager import clear_output_directory
from frame_processor import process_and_save_frames  # Updated import for new processing function

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    
    # Prepare the output directory
    output_image_folder = "output_frames"
    clear_output_directory(output_image_folder)
    
    # Process each video and camera pair with calibration
    for video_path, camera_nr in config["video_paths"]:
        print(f"Processing video {video_path} for camera {camera_nr}")
        output_json = f"{os.path.splitext(video_path)[0]}_tracking.json"
        
        # Simulate tracking step (placeholder for actual tracking functionality)
        print(f"Tracking and saving data to {output_json}")
        
        # Step 2: Process frames with optional calibration file
        process_and_save_frames(
            output_json,
            camera_nr,
            output_image_folder,
            calibration_file=config.get("calibration_file", None)  # Pass calibration file if available
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and save frames with optional calibration.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    args = parser.parse_args()

    main(args.config)
