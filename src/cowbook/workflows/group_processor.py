# group_processor.py

import logging
import multiprocessing as mp
import os
from typing import List, Tuple

from cowbook.execution import CancellationToken, JobCancelledError, JobReporter
from cowbook.io.json_merger import merge_json_files
from cowbook.vision.frame_processor import process_and_save_frames
from cowbook.vision.tracking import track_video_with_yolo

logger = logging.getLogger(__name__)

# Try to reuse the csv_converter implementation to avoid duplication.
# If it's not importable, we'll log a warning and skip conversion.
try:
    from cowbook.io import csv_converter as _csv
except Exception:  # pragma: no cover
    _csv = None


def _tracking_worker(
    video_path: str,
    output_json: str,
    model_ref: str,
    save: bool,
    tracking_cleanup: dict | None = None,
) -> Tuple[str | None, str | None]:
    """
    Run tracking for a single video in a separate process.

    Returns:
        (output_json_or_none, error_message_or_none)
    """
    try:
        import torch  # type: ignore

        torch.set_num_threads(1)
    except Exception:
        pass

    try:
        kwargs = {"save": save}
        if tracking_cleanup and tracking_cleanup.get("enabled"):
            kwargs["tracking_cleanup"] = tracking_cleanup
        track_video_with_yolo(video_path, output_json, model_ref, **kwargs)
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
    reporter: JobReporter | None = None,
    cancellation_token: CancellationToken | None = None,
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
    source_entries: List[Tuple[str, int]] = []
    precomputed_json_count = 0
    _raise_if_cancelled(cancellation_token)

    if reporter is not None:
        reporter.emit(
            "group_started",
            stage="group",
            group_idx=group_idx,
            payload={"input_count": len(video_group)},
        )

    tasks: List[Tuple[str, str, object, bool, int]] = []
    for video_info in video_group:
        video_path = video_info["path"]
        camera_nr = int(video_info["camera_nr"])
        logger.info("Processing %s (camera %d)", video_path, camera_nr)

        if str(video_path).lower().endswith(".json"):
            output_json = video_path
            logger.debug("Using existing tracking JSON: %s", output_json)
            source_entries.append((output_json, camera_nr))
            precomputed_json_count += 1
            if reporter is not None:
                reporter.artifact(
                    "tracking_json",
                    output_json,
                    group_idx=group_idx,
                    metadata={"camera_nr": camera_nr, "source": "precomputed"},
                )
            continue

        base = os.path.splitext(os.path.basename(video_path))[0]
        output_json = os.path.join(output_json_folder, f"{base}_tracking.json")
        tasks.append(
            (
                video_path,
                output_json,
                config.get("model_path", None) if "model_path" in config else model_ref,
                bool(config.get("save_tracking_video", False)),
                config.get("tracking_cleanup"),
                camera_nr,
            )
        )

    tracking_errors: List[str] = []
    if tasks:
        requested_tracking_concurrency = int(config["tracking_concurrency"])
        effective_tracking_concurrency = min(requested_tracking_concurrency, len(tasks))

        logger.info(
            "Launching tracking with requested concurrency %d and effective concurrency %d for %d video(s)",
            requested_tracking_concurrency,
            effective_tracking_concurrency,
            len(tasks),
        )
        if reporter is not None:
            reporter.emit(
                "tracking_started",
                stage="tracking",
                group_idx=group_idx,
                payload={
                    "video_count": len(tasks),
                    "requested_tracking_concurrency": requested_tracking_concurrency,
                    "effective_tracking_concurrency": effective_tracking_concurrency,
                },
            )

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=effective_tracking_concurrency) as pool:
            results = pool.starmap(_tracking_worker, [task[:5] for task in tasks])
        _raise_if_cancelled(cancellation_token)

        for (output_json, err), (_, _, _, _, _, camera_nr) in zip(results, tasks):
            if err:
                logger.error("%s", err)
                tracking_errors.append(err)
                continue
            if output_json:
                source_entries.append((output_json, camera_nr))
                if reporter is not None:
                    reporter.artifact(
                        "tracking_json",
                        output_json,
                        group_idx=group_idx,
                        metadata={"camera_nr": camera_nr, "source": "inference"},
                    )

        if reporter is not None:
            reporter.emit(
                "tracking_completed",
                stage="tracking",
                group_idx=group_idx,
                payload={
                    "success_count": len(source_entries) - precomputed_json_count,
                    "failure_count": len(tracking_errors),
                    "precomputed_json_count": precomputed_json_count,
                },
            )
            if tracking_errors:
                reporter.emit(
                    "tracking_failed",
                    stage="tracking",
                    group_idx=group_idx,
                    message="One or more tracking jobs failed.",
                    payload={"error": "; ".join(tracking_errors), "failure_count": len(tracking_errors)},
                )
    elif reporter is not None:
        reporter.emit(
            "tracking_skipped",
            stage="tracking",
            group_idx=group_idx,
            payload={"precomputed_json_count": precomputed_json_count},
        )

    output_json_paths = [path for path, _camera_nr in source_entries]
    camera_nrs = [camera_nr for _path, camera_nr in source_entries]

    if not output_json_paths:
        raise RuntimeError(f"No JSONs produced for group {group_idx}; aborting group.")

    _raise_if_cancelled(cancellation_token)
    if reporter is not None:
        reporter.emit(
            "processing_started",
            stage="processing",
            group_idx=group_idx,
            payload={"input_json_count": len(output_json_paths)},
        )
    try:
        processed_json_paths = process_and_save_frames(
            output_json_paths,
            camera_nrs,
            output_image_folder,
            config["calibration_file"],
            num_plot_workers=config.get("num_plot_workers", 0),
            output_image_format=config.get("output_image_format", "png"),
            cancellation_token=cancellation_token,
        )
    except Exception as e:
        if isinstance(e, JobCancelledError):
            raise
        logger.exception("Rendering frames failed for group %d: %s", group_idx, e)
        processed_json_paths = []
        if reporter is not None:
            reporter.emit(
                "processing_failed",
                stage="processing",
                group_idx=group_idx,
                message=f"Frame processing failed for group {group_idx}.",
                payload={"error": str(e)},
            )

    processed_camera_nrs = [
        camera_nr
        for input_json_path, camera_nr in source_entries
        if input_json_path.replace(".json", "_processed.json") in processed_json_paths
    ]

    if reporter is not None:
        for processed_json_path, camera_nr in zip(processed_json_paths, processed_camera_nrs):
            reporter.artifact(
                "processed_json",
                processed_json_path,
                group_idx=group_idx,
                metadata={"camera_nr": camera_nr},
            )
        reporter.emit(
            "processing_completed",
            stage="processing",
            group_idx=group_idx,
            payload={"processed_json_count": len(processed_json_paths)},
        )

    for json_path in output_json_paths:
        try:
            if os.path.exists(json_path) and json_path not in processed_json_paths:
                os.remove(json_path)
                logger.debug("Deleted unprocessed JSON: %s", json_path)
        except Exception as e:
            logger.warning("Failed to delete unprocessed JSON %s: %s", json_path, e)

    merged_json_path = os.path.join(
        output_json_folder, f"group_{group_idx}_merged_processed.json"
    )
    _raise_if_cancelled(cancellation_token)
    try:
        if processed_json_paths:
            if reporter is not None:
                reporter.emit(
                    "merge_started",
                    stage="merge",
                    group_idx=group_idx,
                    payload={"processed_json_count": len(processed_json_paths)},
                )
            logger.info(
                "Merging %d processed JSON(s) -> %s",
                len(processed_json_paths),
                merged_json_path,
            )
            merge_json_files(
                processed_json_paths,
                merged_json_path,
                camera_nrs=processed_camera_nrs,
            )
            if reporter is not None:
                reporter.artifact("merged_json", merged_json_path, group_idx=group_idx)
                reporter.emit(
                    "merge_completed",
                    stage="merge",
                    group_idx=group_idx,
                    payload={"path": merged_json_path},
                )
        else:
            logger.warning("No processed JSONs to merge for group %d.", group_idx)
    except Exception as e:
        logger.exception("Merging processed JSONs failed for group %d: %s", group_idx, e)
        if reporter is not None:
            reporter.emit(
                "merge_failed",
                stage="merge",
                group_idx=group_idx,
                message=f"Merging processed JSONs failed for group {group_idx}.",
                payload={"error": str(e), "path": merged_json_path},
            )

    if config.get("convert_to_csv", True):
        csv_paths: List[str] = []
        for pjson in processed_json_paths:
            _raise_if_cancelled(cancellation_token)
            if pjson and pjson.endswith("_processed.json") and os.path.exists(pjson):
                csv_path = _json_to_csv(pjson)
                if csv_path:
                    csv_paths.append(csv_path)
                    if reporter is not None:
                        reporter.artifact("csv", csv_path, group_idx=group_idx)

        if merged_json_path and os.path.exists(merged_json_path):
            _raise_if_cancelled(cancellation_token)
            csv_path = _json_to_csv(merged_json_path)
            if csv_path:
                csv_paths.append(csv_path)
                if reporter is not None:
                    reporter.artifact("csv", csv_path, group_idx=group_idx)
        if reporter is not None:
            reporter.emit(
                "csv_export_completed",
                stage="export",
                group_idx=group_idx,
                payload={"csv_count": len(csv_paths)},
            )

    if reporter is not None:
        reporter.emit(
            "group_completed",
            stage="group",
            group_idx=group_idx,
            payload={
                "tracking_json_count": len(output_json_paths),
                "processed_json_count": len(processed_json_paths),
                "merged_json_path": merged_json_path,
            },
        )

    return output_json_paths, camera_nrs, merged_json_path


def _raise_if_cancelled(cancellation_token: CancellationToken | None) -> None:
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()
