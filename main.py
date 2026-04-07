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
from preprocess_video import preprocess_videos

logger = logging.getLogger(__name__)


def main(
    config_path: str,
) -> None:
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load configuration
    config = load_config(config_path)
    if not config:
        logger.error("Failed to load config from %s", config_path)
        return

    # Prepare and verify output directories (creation + writability checks)
    output_image_folder, output_video_folder, output_json_folder, output_masked_folder = prepare_output_dirs(config)

    # Clear previous per-frame renders (will create the folder if missing)
    clear_output_directory(output_image_folder)

    # Process each video group
    groups = config.get("video_groups", [])
    if config.get("mask_videos", False):
        logger.info("mask_videos=true -> generating masked copies before inference...")
        try:
            groups = preprocess_videos(config)
            logger.info("Masked video groups prepared.")
        except Exception as e:
            logger.exception("Video masking failed. Falling back to original videos: %s", e)
            
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
            # Clean up intermediate frames if enabled
            if config.get("clean_frames_after_video", True):
                logger.info("Cleaning up intermediate frames in %s ...", output_image_folder)
                clear_output_directory(output_image_folder)
            else:
                logger.info("Keeping intermediate frames (clean_frames_after_video=false).")
        except Exception as e:
            logger.exception("Failed to create video: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and create projection video.")
    parser.add_argument(
        "config", nargs="?", help="Path to the configuration file (positional, optional)"
    )
    parser.add_argument("--config", dest="config_opt", type=str, help="Path to the configuration file")

    args = parser.parse_args()
    config_path = args.config_opt or args.config or "config.json"
    main(
        config_path,
    )
