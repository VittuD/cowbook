# main.py

import os
import argparse
from config_loader import load_config
from directory_manager import clear_output_directory
from tracking import track_video_with_yolov8, load_yolo_model
from frame_processor import process_and_save_frames
from video_processor import create_video_from_images

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    if not config:
        return
    
    # Load the YOLO model
    model = load_yolo_model(config["model_path"])
    
    # Prepare output directories
    output_image_folder = config.get("output_image_folder", "output_frames")
    output_video_folder = config.get("output_video_folder", "output_videos")
    output_json_folder = config.get("output_json_folder", "output_json")
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_video_folder, exist_ok=True)
    clear_output_directory(output_image_folder)

    # Process each video group
    for video_group in config["video_groups"]:
        output_json_paths = []
        camera_nrs = []
        for video_info in video_group:
            video_path = video_info["path"]
            camera_nr = video_info["camera_nr"]
            print(f"Processing video {video_path} for camera {camera_nr}")
            
            # Define JSON output path for tracking data
            if video_path.endswith('.json'):
                output_json = video_path
            else:
                output_json = os.path.join(output_json_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_tracking.json")
                track_video_with_yolov8(video_path, output_json, model, save=config["save_tracking_video"])
            
            output_json_paths.append(output_json)
            camera_nrs.append(camera_nr)

        # Process frames and save projections, handling multiple JSON files per group
        process_and_save_frames(output_json_paths, camera_nrs, output_image_folder, config["calibration_file"])

    # Generate a single combined projection video if specified
    if config["create_projection_video"]:
        output_video_path = os.path.join(output_video_folder, "combined_projection.mp4")
        fps = config["fps"]
        print("Generating combined projection video...")
        create_video_from_images(output_image_folder, output_video_path, fps)
        print("Combined projection video generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and create projection video.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    args = parser.parse_args()

    main(args.config)