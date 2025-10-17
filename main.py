# main.py

import os
import argparse
import logging

from config_loader import load_config
from directory_manager import (
    clear_output_directory,
    prepare_output_dirs,
)
from tracking import load_yolo_model
from video_processor import create_video_from_images
from group_processor import process_video_group


logger = logging.getLogger(__name__)


def main(config_path: str, save_tracking_video_flag: bool | None = None) -> None:
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load configuration (with optional CLI overrides applied inside the loader)
    overrides = (
        {"save_tracking_video": bool(save_tracking_video_flag)}
        if save_tracking_video_flag is not None
        else None
    )
    config = load_config(config_path, overrides=overrides)
    if not config:
        logger.error("Failed to load config from %s", config_path)
        return

    # Prepare and verify output directories (creation + writability checks)
    output_image_folder, output_video_folder, output_json_folder = prepare_output_dirs(config)

    # Clear previous per-frame renders (will create the folder if missing)
    clear_output_directory(output_image_folder)

    # Load the YOLO model
    # try:
    #     model = load_yolo_model(config["model_path"])
    # except Exception as e:
    #     logger.exception("Failed to load YOLO model from %s: %s", config.get("model_path"), e)
    #     return

    # Process each video group
    groups = config.get("video_groups", [])
    if not groups:
        logger.warning("No video groups specified in config.")
    else:
        for idx, video_group in enumerate(groups, start=1):
            logger.info("=== Group %d/%d ===", idx, len(groups))
            try:
                process_video_group(
                    idx, video_group, config["model_path"], config, output_json_folder, output_image_folder
                )
            except Exception as e:
                logger.exception("Group %d failed: %s", idx, e)

    # Generate a single combined projection video if specified
    if config.get("create_projection_video", True):
        output_video_path = os.path.join(
            output_video_folder, config.get("output_video_filename", "combined_projection.mp4")
        )
        fps = config["fps"]  # normalized in config_loader
        logger.info("Generating combined projection video at %d FPS -> %s", fps, output_video_path)
        try:
            create_video_from_images(output_image_folder, output_video_path, fps)
            logger.info("Combined projection video generated successfully.")
            # Optional: clean up intermediate frames after video creation
            logger.info("Cleaning up intermediate frames in %s ...", output_image_folder)
            clear_output_directory(output_image_folder)
        except Exception as e:
            logger.exception("Failed to create video: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and create projection video.")
    parser.add_argument(
        "config", nargs="?", help="Path to the configuration file (positional, optional)"
    )
    parser.add_argument("--config", dest="config_opt", type=str, help="Path to the configuration file")

    # Mutually exclusive override for saving YOLO tracking videos
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--save-tracking-video",
        dest="save_tracking_video_flag",
        action="store_true",
        help="Force saving YOLO annotated tracking videos",
    )
    group.add_argument(
        "--no-save-tracking-video",
        dest="save_tracking_video_flag",
        action="store_false",
        help="Disable saving YOLO annotated tracking videos",
    )
    parser.set_defaults(save_tracking_video_flag=None)

    args = parser.parse_args()
    config_path = args.config_opt or args.config or "config.json"
    main(config_path, save_tracking_video_flag=args.save_tracking_video_flag)
