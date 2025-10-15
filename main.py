# main.py

import os
import argparse
import logging
from typing import List, Tuple

from config_loader import load_config
from directory_manager import clear_output_directory
from tracking import track_video_with_yolov8, load_yolo_model
from frame_processor import process_and_save_frames
from video_processor import create_video_from_images
from json_merger import merge_json_files


logger = logging.getLogger(__name__)


def _ensure_output_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _process_video_group(
    group_idx: int,
    video_group: List[dict],
    model,
    config: dict,
    output_json_folder: str,
    output_image_folder: str,
) -> Tuple[List[str], List[int], str]:
    """
    Process one group of videos:
      - run tracking when needed (or accept precomputed JSONs),
      - collect JSON paths and camera numbers,
      - merge per-group JSONs into a single file,
      - render projected frames (parallel if configured).

    Returns:
      (output_json_paths, camera_nrs, merged_json_path)
    """
    output_json_paths: List[str] = []
    camera_nrs: List[int] = []

    # 1) Track (or accept .json) for each entry in the group
    for video_info in video_group:
        video_path = video_info["path"]
        camera_nr = int(video_info["camera_nr"])
        logger.info("Processing %s (camera %d)", video_path, camera_nr)

        if str(video_path).lower().endswith(".json"):
            output_json = video_path
            logger.debug("Using existing tracking JSON: %s", output_json)
        else:
            base = os.path.splitext(os.path.basename(video_path))[0]
            output_json = os.path.join(output_json_folder, f"{base}_tracking.json")
            try:
                track_video_with_yolov8(
                    video_path, output_json, model, save=config.get("save_tracking_video", False)
                )
            except Exception as e:
                logger.exception("Tracking failed for %s: %s", video_path, e)
                # Continue to next video; skip adding this JSON
                continue

        output_json_paths.append(output_json)
        camera_nrs.append(camera_nr)

    if not output_json_paths:
        raise RuntimeError(f"No JSONs produced for group {group_idx}; aborting group.")

    # 2) Render + produce processed JSONs (also draws frames)
    try:
        processed_json_paths = process_and_save_frames(
            output_json_paths,
            camera_nrs,
            output_image_folder,
            config["calibration_file"],
            num_plot_workers=config.get("num_plot_workers", 0),
            output_image_format=config.get("output_image_format", "png"),
        )
    except Exception as e:
        logger.exception("Rendering frames failed for group %d: %s", group_idx, e)
        processed_json_paths = []

    # 3) Merge the PROCESSED JSONs so merged output includes centroids & projected_centroids
    merged_json_path = os.path.join(
        output_json_folder, f"group_{group_idx}_merged_processed.json"
    )
    try:
        if processed_json_paths:
            logger.info("Merging %d processed JSON(s) -> %s", len(processed_json_paths), merged_json_path)
            merge_json_files(processed_json_paths, merged_json_path)
        else:
            logger.warning("No processed JSONs to merge for group %d.", group_idx)
    except Exception as e:
        logger.exception("Merging processed JSONs failed for group %d: %s", group_idx, e)

    return output_json_paths, camera_nrs, merged_json_path


def main(config_path: str, save_tracking_video_flag: bool | None = None) -> None:
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

    # CLI override for saving YOLO tracking videos (if provided)
    if save_tracking_video_flag is not None:
        config["save_tracking_video"] = bool(save_tracking_video_flag)

    # Prepare output directories
    output_image_folder = config.get("output_image_folder", "output_frames")
    output_video_folder = config.get("output_video_folder", "output_videos")
    output_json_folder = config.get("output_json_folder", "output_json")
    _ensure_output_dirs(output_image_folder, output_video_folder, output_json_folder)

    # Clear previous per-frame renders
    clear_output_directory(output_image_folder)

    # Load the YOLO model
    try:
        model = load_yolo_model(config["model_path"])
    except Exception as e:
        logger.exception("Failed to load YOLO model from %s: %s", config.get("model_path"), e)
        return

    # Process each video group
    groups = config.get("video_groups", [])
    if not groups:
        logger.warning("No video groups specified in config.")
    else:
        for idx, video_group in enumerate(groups, start=1):
            logger.info("=== Group %d/%d ===", idx, len(groups))
            try:
                _process_video_group(
                    idx, video_group, model, config, output_json_folder, output_image_folder
                )
            except Exception as e:
                logger.exception("Group %d failed: %s", idx, e)

    # Generate a single combined projection video if specified
    if config.get("create_projection_video", True):
        output_video_path = os.path.join(output_video_folder, "combined_projection.mp4")
        fps = int(config.get("fps", 6))
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
