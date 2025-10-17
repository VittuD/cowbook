# config_loader.py

import json

def load_config(config_path, overrides=None):
    """
    Load configuration settings from a JSON file with error handling, defaults,
    light normalization, and optional overrides (e.g., from CLI).
    """
    try:
        with open(config_path) as f:
            config = json.load(f)

        # ---- Defaults (centralized) ----
        config.setdefault("model_path", "models/best.pt")
        config.setdefault("fps", 6)
        config.setdefault("save_tracking_video", False)
        config.setdefault("create_projection_video", True)
        config.setdefault("video_groups", [])
        config.setdefault("calibration_file", "legacy/calibration_matrix.json")
        # Parallel rendering & image format
        config.setdefault("num_plot_workers", 0)            # 0 = sequential; >0 uses ProcessPoolExecutor
        config.setdefault("output_image_format", "jpg")     # "png" or "jpg"
        # Output directories & filename
        config.setdefault("output_image_folder", "output_frames")
        config.setdefault("output_video_folder", "output_videos")
        config.setdefault("output_json_folder", "output_json")
        config.setdefault("output_video_filename", "combined_projection.mp4")
        # CSV conversion
        config.setdefault("convert_to_csv", True)

        # ---- Apply optional overrides (from CLI or caller) ----
        if overrides:
            for k, v in overrides.items():
                if v is not None:
                    config[k] = v

        # ---- Normalization ----
        # fps as int
        try:
            config["fps"] = int(config["fps"])
        except Exception as e:
            raise ValueError(f"'fps' must be an integer (got {config.get('fps')!r}).") from e

        # image format normalized
        fmt = str(config.get("output_image_format", "jpg")).lower()
        if fmt not in {"png", "jpg", "jpeg"}:
            raise ValueError(f"Invalid output_image_format: {fmt}. Use 'png' or 'jpg'.")
        # normalize "jpeg" -> "jpg"
        config["output_image_format"] = "jpg" if fmt in {"jpg", "jpeg"} else "png"

        # ---- Validate video groups ----
        # Expected: List[List[dict(path:str, camera_nr:int)]]
        groups = config["video_groups"]
        if not isinstance(groups, list):
            raise ValueError("'video_groups' must be a list of groups.")

        for gi, group in enumerate(groups, start=1):
            if not isinstance(group, list):
                raise ValueError(f"Group {gi} must be a list.")
            num_videos = len(group)
            if num_videos == 0 or num_videos > 4:
                raise ValueError("Each video group must contain between 1 and 4 videos.")

            camera_ids = []
            for vi, video in enumerate(group, start=1):
                if not isinstance(video, dict):
                    raise ValueError(f"Group {gi}, item {vi} must be an object with 'path' and 'camera_nr'.")
                if "path" not in video or "camera_nr" not in video:
                    raise ValueError(f"Group {gi}, item {vi} missing 'path' or 'camera_nr'.")
                # normalize camera_nr to int
                try:
                    video["camera_nr"] = int(video["camera_nr"])
                except Exception as e:
                    raise ValueError(f"Group {gi}, item {vi} has non-integer camera_nr.") from e
                camera_ids.append(video["camera_nr"])

            if len(set(camera_ids)) < num_videos:
                raise ValueError(f"Each video in group {gi} must be associated with a unique camera.")

        return config

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading config: {e}")
        return {}
