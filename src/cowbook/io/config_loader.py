# config_loader.py

import json
import logging
import os

from cowbook.core.contracts import PipelineConfig
from cowbook.io.directory_manager import resolve_output_paths
from cowbook.vision.calibration import default_calibration_file

logger = logging.getLogger(__name__)

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
        config.setdefault("calibration_file", default_calibration_file())
        config.setdefault("runtime_root", "var")
        config.setdefault("run_name", "default")
        # Parallel rendering & image format
        config.setdefault("num_plot_workers", max(1, os.cpu_count() - 1) if hasattr(os, 'cpu_count') else 0)            # 0 = sequential; >0 uses ProcessPoolExecutor
        config.setdefault("output_image_format", "jpg")     # "png" or "jpg"
        # Output directories & filename
        config.setdefault("output_video_filename", "combined_projection.mp4")
        # CSV conversion
        config.setdefault("convert_to_csv", True)
        # Clean-up frames after assembling the video (default ON)
        config.setdefault("clean_frames_after_video", True)
        # Tracking workers (default to 1 to avoid GPU OOM)
        config.setdefault("num_tracking_workers", 1)
        # ---- Masking at inference ----
        config.setdefault("mask_videos", False)
        config.setdefault("num_mask_workers", max(1, os.cpu_count() - 1) if hasattr(os, 'cpu_count') else 0)
        config.setdefault("mask_strict_half_rule", True)
        # Per-channel mask image paths (override in your JSON if you store masks elsewhere)
        config.setdefault("masks", {
            "Ch1": "assets/masks/combined_mask_ch1.png",
            "Ch4": "assets/masks/combined_mask_ch4.png",
            "Ch6": "assets/masks/combined_mask_ch6.png",
            "Ch8": "assets/masks/combined_mask_ch8.png",
        })

        # Optional: explicit camera->channel map (JSON keys are strings)
        # Example: {"1":"Ch1","4":"Ch4","6":"Ch6","8":"Ch8"}
        config.setdefault("camera_to_mask_map", {})

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

        (
            runtime_root,
            output_root,
            output_image_folder,
            output_video_folder,
            output_json_folder,
            output_masked_folder,
        ) = resolve_output_paths(config)
        config["runtime_root"] = runtime_root
        config["output_root"] = output_root
        config["output_image_folder"] = output_image_folder
        config["output_video_folder"] = output_video_folder
        config["output_json_folder"] = output_json_folder
        config["masked_video_folder"] = output_masked_folder

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

        return PipelineConfig.from_mapping(config).to_dict()

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error("Error loading config: %s", e)
        return {}
