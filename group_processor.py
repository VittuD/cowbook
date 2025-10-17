# group_processor.py

import os
import json
import logging
from typing import List, Tuple

import multiprocessing as mp

from tracking import track_video_with_yolo
from frame_processor import process_and_save_frames
from json_merger import merge_json_files

logger = logging.getLogger(__name__)

# Try to reuse the csv_converter implementation to avoid duplication.
# If it's not importable, we'll log a warning and skip conversion.
try:
    import csv_converter as _csv
except Exception:  # pragma: no cover
    _csv = None



def _tracking_worker(video_path: str,
                     output_json: str,
                     model_ref: str,
                     save: bool) -> Tuple[str | None, str | None]:
    """
    Run tracking for a single video in a separate process.

    Returns:
        (output_json_or_none, error_message_or_none)
    """
    # Optional: keep CPU threads per worker low to avoid oversubscription.
    try:
        import torch  # type: ignore
        torch.set_num_threads(1)
    except Exception:
        pass

    try:
        # IMPORTANT: track_video_with_yolo should *load its own model* from model_ref
        # (e.g., a weights path or config). Do *not* pass in-memory model objects here.
        track_video_with_yolo(
            video_path,
            output_json,
            model_ref,
            save=save,
        )
        return output_json, None
    except Exception as e:
        return None, f"Tracking failed for {video_path}: {e}"

def _json_to_csv(json_path: str) -> str | None:
    """
    Convert a single processed JSON to CSV using csv_converter's internals.
    Returns the CSV path on success, or None if conversion couldn't run.
    """
    if _csv is None:
        logger.warning(
            "csv_converter not available; skipping CSV conversion for %s", json_path
        )
        return None

    csv_path = os.path.splitext(json_path)[0] + ".csv"

    try:
        doc = _csv._load_json(json_path)  # type: ignore[attr-defined]
        # Generate rows and write CSV (no source column for single-file conversion)
        def _rows():
            yield from _csv._iter_rows_from_json(doc, source_tag=None)  # type: ignore[attr-defined]

        _csv._write_csv(_rows(), csv_path, include_source=False)  # type: ignore[attr-defined]
        logger.info("Wrote CSV: %s", csv_path)
        return csv_path
    except Exception as e:
        logger.exception("Failed converting %s to CSV: %s", json_path, e)
        return None


def process_video_group(
    group_idx: int,
    video_group: List[dict],
    model_ref: str,
    config: dict,
    output_json_folder: str,
    output_image_folder: str,
) -> Tuple[List[str], List[int], str]:
    """
    Process one group of videos:
      - run tracking when needed (or accept precomputed JSONs),
      - collect JSON paths and camera numbers,
      - merge per-group JSONs into a single file,
      - render projected frames (parallel if configured),
      - optionally convert all *_processed.json (and merged) to CSV.

    Returns:
      (output_json_paths, camera_nrs, merged_json_path)
    """
    output_json_paths: List[str] = []
    camera_nrs: List[int] = []

# group_processor.py (replace your current "1) Track (or accept .json)..." loop inside process_video_group)

    # 1) Track (or accept .json) for each entry in the group
    tasks: List[Tuple[str, str, object, bool]] = []
    for video_info in video_group:
        video_path = video_info["path"]
        camera_nr = int(video_info["camera_nr"])
        logger.info("Processing %s (camera %d)", video_path, camera_nr)

        if str(video_path).lower().endswith(".json"):
            output_json = video_path
            logger.debug("Using existing tracking JSON: %s", output_json)
            output_json_paths.append(output_json)
            camera_nrs.append(camera_nr)
            continue

        base = os.path.splitext(os.path.basename(video_path))[0]
        output_json = os.path.join(output_json_folder, f"{base}_tracking.json")

        # Collect a task for the pool. DO NOT create/load the model here.
        tasks.append((
            video_path,
            output_json,
            # Pass a *model reference* (e.g., a weights path or config),
            # NOT an in-memory model object.
            config.get("model_path", None) if "model_path" in config else model_ref,
            bool(config.get("save_tracking_video", False)),
        ))
        # We still record the camera numbers aligned with videos
        camera_nrs.append(camera_nr)

    # If there are tasks to run, execute them in parallel (1 worker per video by default)
    if tasks:
        # Default: N workers = number of videos being tracked now.
        # You can cap it via config["num_tracking_workers"] if VRAM or GPU-usage is tight.
        max_workers = int(config.get("num_tracking_workers", len(tasks)))
        if max_workers < 1:
            max_workers = 1

        logger.info("Launching %d tracking worker(s) for %d video(s)", max_workers, len(tasks))

        # Use spawn to be CUDA-safe.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=max_workers) as pool:
            results = pool.starmap(_tracking_worker, tasks)

        # Collect successful outputs; log failures
        for (output_json, err), (video_path, _, _, _) in zip(results, tasks):
            if err:
                logger.exception(err)
                # Skip adding this JSON
                continue
            if output_json:
                output_json_paths.append(output_json)

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

    # 4) Optional: convert all *_processed.json (and merged) to CSV
    if config.get("convert_to_csv", True):
        # Convert each processed JSON
        for pjson in processed_json_paths:
            if pjson and pjson.endswith("_processed.json") and os.path.exists(pjson):
                _json_to_csv(pjson)

        # Convert merged JSON (if exists)
        if merged_json_path and os.path.exists(merged_json_path):
            _json_to_csv(merged_json_path)

    return output_json_paths, camera_nrs, merged_json_path
