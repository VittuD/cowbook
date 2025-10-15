# config_loader.py

import json

def load_config(config_path):
    """
    Load configuration settings from a JSON file with error handling and default values.
    """
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        # Set default values for missing config options
        config.setdefault("model_path", "models/best.pt")
        config.setdefault("fps", 6)
        config.setdefault("save_tracking_video", False)
        config.setdefault("create_projection_video", True)
        config.setdefault("video_groups", [])
        config.setdefault("calibration_file", "legacy/calibration_matrix.json")
        config.setdefault("save_tracking_video", False)
        # Parallel rendering & image format
        config.setdefault("num_plot_workers", 0)           # 0 = sequential; >0 uses ProcessPoolExecutor
        config.setdefault("output_image_format", "jpg")    # "png" or "jpg"

        # Validate each video group in the config
        for group in config["video_groups"]:
            num_videos = len(group)
            if num_videos == 0 or num_videos > 4:
                raise ValueError("Each video group must contain between 1 and 4 videos.")
            
            # Check that each video in a group has a unique camera number
            camera_ids = [video["camera_nr"] for video in group]
            if len(set(camera_ids)) < num_videos:
                raise ValueError("Each video in a group must be associated with a unique camera.")

        return config
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading config: {e}")
        return {}
