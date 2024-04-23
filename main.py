# main.py

import os
import argparse
from config_loader import load_config
from directory_manager import clear_output_directory
from tracking import track_video_with_yolov8, load_yolo_model
from frame_processor import process_and_save_frames
from video_generator import create_video_from_images  # New import

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    
    # Load YOLO model
    model = load_yolo_model(config["model_path"])
    
    # Prepare the output directory
    output_image_folder = "output_frames"
    clear_output_directory(output_image_folder)
    
    # Process each video and camera pair
    for video_path, camera_nr in config["video_paths"]:
        print(f"Processing video {video_path} for camera {camera_nr}")
        output_json = f"{os.path.splitext(video_path)[0]}_tracking.json"
        
        # Step 1: Track and save detections to JSON
        track_video_with_yolov8(
            video_path,
            output_json,
            model,
            save=config.get("save_tracking_video", False)
        )
        
        # Step 2: Process frames with projected centroids
        process_and_save_frames(
            output_json, 
            camera_nr, 
            output_image_folder, 
            config["calibration_file"]
        )
    
    # Step 3: Create projection video if specified in config
    if config.get("create_projection_video", False):
        output_video_path = "output_video.mp4"
        fps = config.get("fps", 6)
        print("Generating projection video...")
        create_video_from_images(output_image_folder, output_video_path, fps)
        print("Projection video generated successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and save frames with optional calibration.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    args = parser.parse_args()

    main(args.config)
