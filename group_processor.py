# group_processor.py

import os
import logging
from typing import List, Tuple

from tracking import track_video_with_yolov8
from frame_processor import process_and_save_frames
from json_merger import merge_json_files

logger = logging.getLogger(__name__)


def process_video_group(
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

    # Delete unprocessed JSONs to avoid confusion
    for json_path in output_json_paths:
        try:
            if os.path.exists(json_path) and json_path not in processed_json_paths:
                os.remove(json_path)
                logger.debug("Deleted unprocessed JSON: %s", json_path)
        except Exception as e:
            logger.warning("Failed to delete unprocessed JSON %s: %s", json_path, e)

    # 3) Merge the PROCESSED JSONs so merged output includes centroids & projected_centroids
    merged_json_path = os.path.join(
        output_json_folder, f"group_{group_idx}_merged_processed.json"
    )
    try:
        if processed_json_paths:
            logger.info(
                "Merging %d processed JSON(s) -> %s",
                len(processed_json_paths),
                merged_json_path,
            )
            merge_json_files(processed_json_paths, merged_json_path)
        else:
            logger.warning("No processed JSONs to merge for group %d.", group_idx)
    except Exception as e:
        logger.exception("Merging processed JSONs failed for group %d: %s", group_idx, e)

    return output_json_paths, camera_nrs, merged_json_path
