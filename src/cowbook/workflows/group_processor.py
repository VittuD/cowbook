from dataclasses import dataclass
import logging
import multiprocessing as _mp_std
import multiprocessing as mp
import os
import time
from queue import Empty
from typing import List, Tuple

from cowbook.execution import (
    MERGE_FAILED,
    PROCESSING_FAILED,
    TRACKING_FAILED,
    CancellationToken,
    JobCancelledError,
    JobReporter,
)
from cowbook.io.csv_converter import json_to_csv
from cowbook.io.json_merger import merge_json_files
from cowbook.vision.frame_processor import process_and_save_frames
from cowbook.vision.tracking import load_yolo_model, track_video_with_yolo

logger = logging.getLogger(__name__)
_WORKER_MODEL_CACHE: dict[tuple[str, str], object] = {}


@dataclass(slots=True)
class _TrackingTask:
    video_path: str
    output_json: str
    model_ref: str
    save: bool
    tracking_cleanup: dict | None
    camera_nr: int


def _tracking_mode_from_cleanup(tracking_cleanup: dict | None) -> str:
    if tracking_cleanup and tracking_cleanup.get("enabled"):
        return "cleanup"
    return "track"


def _cached_tracking_model(
    model_cache: dict[tuple[str, str], object],
    *,
    model_ref: str,
    tracking_cleanup: dict | None,
):
    cache_key = (model_ref, _tracking_mode_from_cleanup(tracking_cleanup))
    model = model_cache.get(cache_key)
    if model is None:
        model = load_yolo_model(model_ref)
        model_cache[cache_key] = model
    return model


def _tracking_kwargs(
    *,
    save: bool,
    tracking_cleanup: dict | None,
    camera_nr: int | None,
    log_progress: bool,
    reporter: JobReporter | None = None,
    group_idx: int | None = None,
    progress_event_sink=None,
    model=None,
) -> dict:
    kwargs = {
        "save": save,
        "log_progress": log_progress,
        "camera_nr": camera_nr,
    }
    if reporter is not None:
        kwargs["reporter"] = reporter
    if group_idx is not None:
        kwargs["group_idx"] = group_idx
    if progress_event_sink is not None:
        kwargs["progress_event_sink"] = progress_event_sink
    if tracking_cleanup and tracking_cleanup.get("enabled"):
        kwargs["tracking_cleanup"] = tracking_cleanup
    if model is not None:
        kwargs["model"] = model
    return kwargs


def _tracking_worker(
    video_path: str,
    output_json: str,
    model_ref: str,
    save: bool,
    tracking_cleanup: dict | None = None,
    camera_nr: int | None = None,
    log_progress: bool = False,
    group_idx: int | None = None,
    progress_queue=None,
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
        progress_event_sink = None
        if progress_queue is not None:
            def progress_event_sink(event_type: str, payload: dict) -> None:
                progress_queue.put(
                    {
                        "event_type": event_type,
                        "group_idx": group_idx,
                        "payload": payload,
                    }
                )
        model = _cached_tracking_model(
            _WORKER_MODEL_CACHE,
            model_ref=model_ref,
            tracking_cleanup=tracking_cleanup,
        )
        kwargs = _tracking_kwargs(
            save=save,
            tracking_cleanup=tracking_cleanup,
            camera_nr=camera_nr,
            log_progress=log_progress,
            group_idx=group_idx,
            progress_event_sink=progress_event_sink,
            model=model,
        )
        track_video_with_yolo(video_path, output_json, model_ref, **kwargs)
        return output_json, None
    except Exception as e:
        return None, f"Tracking failed for {video_path}: {e}"


def _tracking_mode(task: _TrackingTask) -> str:
    """Return the cache partition for a tracking task.

    Direct tracking uses Ultralytics' built-in tracker callbacks, while cleanup
    tracking performs a plain detection pass before running the custom tracker.
    Those modes must not share the same YOLO instance because Ultralytics stores
    tracking callbacks on the model object.
    """
    return _tracking_mode_from_cleanup(task.tracking_cleanup)


def _run_tracking_inline(
    tasks: list[_TrackingTask],
    *,
    reporter: JobReporter | None,
    group_idx: int,
    log_progress: bool,
    cancellation_token: CancellationToken | None,
) -> list[Tuple[str | None, str | None]]:
    """Run tracking synchronously in the current process.

    This mirrors the pooled execution contract: same return shape, same error
    formatting, and the same progress/reporter behavior. It is intentionally
    used only for the effective single-worker case where multiprocessing adds
    process startup and teardown cost without adding throughput.
    """
    model_cache: dict[tuple[str, str], object] = {}
    results: list[Tuple[str | None, str | None]] = []
    try:
        for task in tasks:
            _raise_if_cancelled(cancellation_token)
            try:
                model = _cached_tracking_model(
                    model_cache,
                    model_ref=task.model_ref,
                    tracking_cleanup=task.tracking_cleanup,
                )
                kwargs = _tracking_kwargs(
                    save=task.save,
                    tracking_cleanup=task.tracking_cleanup,
                    camera_nr=task.camera_nr,
                    log_progress=log_progress,
                    reporter=reporter,
                    group_idx=group_idx,
                    model=model,
                )
                track_video_with_yolo(
                    task.video_path,
                    task.output_json,
                    task.model_ref,
                    **kwargs,
                )
                results.append((task.output_json, None))
            except Exception as e:
                results.append((None, f"Tracking failed for {task.video_path}: {e}"))
        return results
    finally:
        model_cache.clear()


def _drain_tracking_progress_queue(progress_queue, reporter: JobReporter | None) -> None:
    if reporter is None or progress_queue is None:
        return
    while True:
        try:
            item = progress_queue.get_nowait()
        except Empty:
            break
        reporter.emit(
            item["event_type"],
            stage="tracking",
            group_idx=item.get("group_idx"),
            payload=item.get("payload", {}),
        )


def _build_tracking_worker_args(
    task: _TrackingTask,
    *,
    log_progress: bool,
    group_idx: int,
    progress_queue,
) -> tuple[object, ...]:
    return (
        task.video_path,
        task.output_json,
        task.model_ref,
        task.save,
        task.tracking_cleanup,
        task.camera_nr,
        log_progress,
        group_idx,
        progress_queue,
    )


def _collect_source_entries_and_tracking_tasks(
    video_group: List[dict],
    *,
    model_ref: str,
    config: dict,
    output_json_folder: str,
    reporter: JobReporter | None,
    group_idx: int,
) -> tuple[list[tuple[str, int]], list[_TrackingTask], int]:
    source_entries: list[tuple[str, int]] = []
    tasks: list[_TrackingTask] = []
    precomputed_json_count = 0

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
        tasks.append(
            _TrackingTask(
                video_path=video_path,
                output_json=os.path.join(output_json_folder, f"{base}_tracking.json"),
                model_ref=config["model_path"] if "model_path" in config else model_ref,
                save=bool(config.get("save_tracking_video", False)),
                tracking_cleanup=config.get("tracking_cleanup"),
                camera_nr=camera_nr,
            )
        )

    return source_entries, tasks, precomputed_json_count


def _run_tracking_pool(
    tasks: list[_TrackingTask],
    *,
    effective_tracking_concurrency: int,
    reporter: JobReporter | None,
    group_idx: int,
    log_progress: bool,
) -> list[Tuple[str | None, str | None]]:
    """Run tracking in worker processes using the supported pooled path.

    Each worker maintains its own process-local model cache, so repeated tasks
    for the same `(model_ref, tracking_mode)` reuse one YOLO instance instead
    of paying load cost for every video. This is the supported high-throughput
    path for effective tracking concurrency greater than one.
    """
    ctx = mp.get_context("spawn")
    if reporter is None:
        with ctx.Pool(processes=effective_tracking_concurrency) as pool:
            return pool.starmap(
                _tracking_worker,
                [
                    _build_tracking_worker_args(
                        task,
                        log_progress=log_progress,
                        group_idx=group_idx,
                        progress_queue=None,
                    )
                    for task in tasks
                ],
            )

    with _mp_std.Manager() as manager:
        progress_queue = manager.Queue()
        with ctx.Pool(processes=effective_tracking_concurrency) as pool:
            async_results = [
                pool.apply_async(
                    _tracking_worker,
                    _build_tracking_worker_args(
                        task,
                        log_progress=log_progress,
                        group_idx=group_idx,
                        progress_queue=progress_queue,
                    ),
                )
                for task in tasks
            ]
            while not all(result.ready() for result in async_results):
                _drain_tracking_progress_queue(progress_queue, reporter)
                time.sleep(0.1)
            _drain_tracking_progress_queue(progress_queue, reporter)
            return [result.get() for result in async_results]


def _run_tracking_tasks(
    tasks: list[_TrackingTask],
    *,
    config: dict,
    reporter: JobReporter | None,
    group_idx: int,
    precomputed_json_count: int,
    cancellation_token: CancellationToken | None,
) -> tuple[list[tuple[str, int]], list[str]]:
    """Execute group tracking with one public concurrency knob.

    The runtime contract is intentionally simple: effective concurrency one uses
    the inline path, while higher values use the pooled worker path. Both keep
    the same result shape, event flow, and tracking-mode isolation guarantees.
    """
    if not tasks:
        if reporter is not None:
            reporter.emit(
                "tracking_skipped",
                stage="tracking",
                group_idx=group_idx,
                payload={"precomputed_json_count": precomputed_json_count},
            )
        return [], []

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

    log_progress = bool(config.get("log_progress", False))
    if effective_tracking_concurrency == 1:
        results = _run_tracking_inline(
            tasks,
            reporter=reporter,
            group_idx=group_idx,
            log_progress=log_progress,
            cancellation_token=cancellation_token,
        )
    else:
        results = _run_tracking_pool(
            tasks,
            effective_tracking_concurrency=effective_tracking_concurrency,
            reporter=reporter,
            group_idx=group_idx,
            log_progress=log_progress,
        )
    _raise_if_cancelled(cancellation_token)

    tracked_source_entries: list[tuple[str, int]] = []
    tracking_errors: list[str] = []
    for (output_json, err), task in zip(results, tasks):
        if err:
            logger.error("%s", err)
            tracking_errors.append(err)
            continue
        if output_json:
            tracked_source_entries.append((output_json, task.camera_nr))
            if reporter is not None:
                reporter.artifact(
                    "tracking_json",
                    output_json,
                    group_idx=group_idx,
                    metadata={"camera_nr": task.camera_nr, "source": "inference"},
                )

    if reporter is not None:
        reporter.emit(
            "tracking_completed",
            stage="tracking",
            group_idx=group_idx,
            payload={
                "success_count": len(tracked_source_entries),
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
                payload={
                    "error_code": TRACKING_FAILED,
                    "error_detail": "; ".join(tracking_errors),
                    "failure_count": len(tracking_errors),
                },
            )

    return tracked_source_entries, tracking_errors


def _processed_camera_nrs(
    source_entries: list[tuple[str, int]],
    processed_json_paths: list[str],
) -> list[int]:
    return [
        camera_nr
        for input_json_path, camera_nr in source_entries
        if input_json_path.replace(".json", "_processed.json") in processed_json_paths
    ]


def _emit_processed_json_artifacts(
    processed_json_paths: list[str],
    processed_camera_nrs: list[int],
    *,
    reporter: JobReporter | None,
    group_idx: int,
) -> None:
    if reporter is None:
        return

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


def _remove_unprocessed_jsons(output_json_paths: list[str], processed_json_paths: list[str]) -> None:
    for json_path in output_json_paths:
        try:
            if os.path.exists(json_path) and json_path not in processed_json_paths:
                os.remove(json_path)
                logger.debug("Deleted unprocessed JSON: %s", json_path)
        except Exception as e:
            logger.warning("Failed to delete unprocessed JSON %s: %s", json_path, e)


def _merge_processed_jsons(
    *,
    group_idx: int,
    output_json_folder: str,
    processed_json_paths: list[str],
    processed_camera_nrs: list[int],
    reporter: JobReporter | None,
    cancellation_token: CancellationToken | None,
) -> str:
    merged_json_path = os.path.join(output_json_folder, f"group_{group_idx}_merged_processed.json")
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
                payload={
                    "error_code": MERGE_FAILED,
                    "error_detail": str(e),
                    "path": merged_json_path,
                },
            )
    return merged_json_path


def _export_csv_artifacts(
    *,
    config: dict,
    processed_json_paths: list[str],
    merged_json_path: str,
    reporter: JobReporter | None,
    group_idx: int,
    cancellation_token: CancellationToken | None,
) -> list[str]:
    if not config.get("convert_to_csv", True):
        return []

    csv_paths: list[str] = []
    for json_path in processed_json_paths:
        _raise_if_cancelled(cancellation_token)
        if json_path and json_path.endswith("_processed.json") and os.path.exists(json_path):
            csv_path = json_to_csv(json_path)
            if csv_path:
                csv_paths.append(csv_path)
                if reporter is not None:
                    reporter.artifact("csv", csv_path, group_idx=group_idx)

    if merged_json_path and os.path.exists(merged_json_path):
        _raise_if_cancelled(cancellation_token)
        csv_path = json_to_csv(merged_json_path)
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

    return csv_paths


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
    _raise_if_cancelled(cancellation_token)

    if reporter is not None:
        reporter.emit(
            "group_started",
            stage="group",
            group_idx=group_idx,
            payload={"input_count": len(video_group)},
        )

    source_entries, tasks, precomputed_json_count = _collect_source_entries_and_tracking_tasks(
        video_group,
        model_ref=model_ref,
        config=config,
        output_json_folder=output_json_folder,
        reporter=reporter,
        group_idx=group_idx,
    )
    tracked_source_entries, _ = _run_tracking_tasks(
        tasks,
        config=config,
        reporter=reporter,
        group_idx=group_idx,
        precomputed_json_count=precomputed_json_count,
        cancellation_token=cancellation_token,
    )
    source_entries.extend(tracked_source_entries)

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
            reporter=reporter,
            group_idx=group_idx,
            log_progress=bool(config.get("log_progress", False)),
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
                payload={"error_code": PROCESSING_FAILED, "error_detail": str(e)},
            )

    processed_camera_nrs = _processed_camera_nrs(source_entries, processed_json_paths)
    _emit_processed_json_artifacts(
        processed_json_paths,
        processed_camera_nrs,
        reporter=reporter,
        group_idx=group_idx,
    )
    _remove_unprocessed_jsons(output_json_paths, processed_json_paths)

    merged_json_path = _merge_processed_jsons(
        group_idx=group_idx,
        output_json_folder=output_json_folder,
        processed_json_paths=processed_json_paths,
        processed_camera_nrs=processed_camera_nrs,
        reporter=reporter,
        cancellation_token=cancellation_token,
    )

    _export_csv_artifacts(
        config=config,
        processed_json_paths=processed_json_paths,
        merged_json_path=merged_json_path,
        reporter=reporter,
        group_idx=group_idx,
        cancellation_token=cancellation_token,
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
