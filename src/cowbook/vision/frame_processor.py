from dataclasses import dataclass
import concurrent.futures as _fut  # parallel rendering
import logging
import math
import os

from tqdm import tqdm

from cowbook.execution import CancellationToken, JobReporter, StageProgressReporter
from cowbook.io.json_utils import dump_path_compact
from cowbook.vision.calibration import (
    load_camera_setup,
    load_projection_context,
)
from cowbook.vision.processing import (
    extract_data,
    extract_projected_centroids_from_files,
    parse_json,
    process_detections,
    reconstruct_json,
)
from cowbook.vision.rendering import (
    default_barn_image_path,
    load_barn_image,
    render_projection_frame,
)

logger = logging.getLogger(__name__)

# Global cache used within each worker process to avoid reloading the barn image on every frame
_BARN_IMG = None


@dataclass(slots=True)
class ProcessedJsonArtifact:
    input_json_path: str
    processed_json_path: str
    camera_nr: int

def _render_frame_worker(args):
    """
    Worker that renders a single frame image.
    Loads barn.png once per process and reuses it across tasks.
    """
    global _BARN_IMG
    frame_id, projected_centroids, frame_output_path, barn_image_path = args

    if _BARN_IMG is None:
        if os.path.exists(barn_image_path):
            import cv2 as _cv2
            _BARN_IMG = _cv2.imread(barn_image_path)
        else:
            _BARN_IMG = None

    render_projection_frame(
        projected_centroids,
        frame_id,
        frame_output_path,
        barn_image_path=barn_image_path,
        barn_image=_BARN_IMG
    )
    return frame_output_path


def _process_centroids_worker(args):
    json_file_path, camera_nr, calibration_file = args
    frames_data = process_centroids(
        json_file_path,
        camera_nr,
        calibration_file,
        cancellation_token=None,
        show_progress=False,
    )
    updated_json_file_path = os.path.join(json_file_path.replace(".json", "_processed.json"))
    save_frame_data_json(frames_data, updated_json_file_path)
    return updated_json_file_path, camera_nr


def _processed_json_artifact(
    *,
    input_json_path: str,
    processed_json_path: str,
    camera_nr: int,
) -> ProcessedJsonArtifact:
    return ProcessedJsonArtifact(
        input_json_path=input_json_path,
        processed_json_path=processed_json_path,
        camera_nr=camera_nr,
    )


def process_and_save_frames_with_metadata(
    json_file_paths,
    camera_nrs,
    output_image_folder,
    calibration_file,
    num_plot_workers=0,
    output_image_format="jpg",
    cancellation_token: CancellationToken | None = None,
    reporter: JobReporter | None = None,
    group_idx: int | None = None,
    log_progress: bool = False,
):
    """Process detections and return explicit per-camera processed-artifact metadata."""
    processed_artifacts: list[ProcessedJsonArtifact] = []
    processing_tasks = list(zip(json_file_paths, camera_nrs))
    process_workers = min(len(processing_tasks), os.cpu_count() or 1)
    process_progress = StageProgressReporter(
        event_prefix="processing",
        reporter_stage="processing",
        stage_name="process_centroids",
        total=len(processing_tasks),
        log_progress=log_progress,
        reporter=reporter,
        group_idx=group_idx,
    )
    process_progress.stage_started()

    if process_workers > 1:
        processed_by_input: dict[str, ProcessedJsonArtifact] = {}
        with _fut.ProcessPoolExecutor(max_workers=process_workers) as ex:
            future_map = {
                ex.submit(
                    _process_centroids_worker,
                    (json_file_path, camera_nr, calibration_file),
                ): (json_file_path, camera_nr)
                for json_file_path, camera_nr in processing_tasks
            }
            for future in _fut.as_completed(future_map):
                json_file_path, camera_nr = future_map[future]
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
                try:
                    processed_json_path, _ = future.result()
                    processed_by_input[json_file_path] = _processed_json_artifact(
                        input_json_path=json_file_path,
                        processed_json_path=processed_json_path,
                        camera_nr=camera_nr,
                    )
                    process_progress.step_progress(len(processed_by_input), len(processing_tasks))
                    logger.info("Processed frame data saved to %s", processed_json_path)
                except Exception as exc:
                    logger.exception(
                        "Failed processing %s for camera %d: %s",
                        json_file_path,
                        camera_nr,
                        exc,
                    )
        processed_artifacts = [
            processed_by_input[json_file_path]
            for json_file_path, _camera_nr in processing_tasks
            if json_file_path in processed_by_input
        ]
    else:
        for json_file_path, camera_nr in processing_tasks:
            try:
                frames_data = process_centroids(
                    json_file_path,
                    camera_nr,
                    calibration_file,
                    cancellation_token=cancellation_token,
                    show_progress=True,
                )

                processed_json_path = os.path.join(
                    json_file_path.replace(".json", "_processed.json")
                )
                save_frame_data_json(frames_data, processed_json_path)
                processed_artifacts.append(
                    _processed_json_artifact(
                        input_json_path=json_file_path,
                        processed_json_path=processed_json_path,
                        camera_nr=camera_nr,
                    )
                )
                process_progress.step_progress(len(processed_artifacts), len(processing_tasks))
                logger.info("Processed frame data saved to %s", processed_json_path)
            except Exception as exc:
                logger.exception(
                    "Failed processing %s for camera %d: %s",
                    json_file_path,
                    camera_nr,
                    exc,
                )
    process_progress.stage_completed()

    processed_json_paths = [artifact.processed_json_path for artifact in processed_artifacts]

    if processed_json_paths:
        base_filename = os.path.join(output_image_folder, "combined_projected_centroids")
        plot_combined_projected_centroids(
            processed_json_paths,
            base_filename,
            num_workers=num_plot_workers,
            image_format=output_image_format,
            cancellation_token=cancellation_token,
            reporter=reporter,
            group_idx=group_idx,
            log_progress=log_progress,
        )

    return processed_artifacts

def process_and_save_frames(
    json_file_paths,
    camera_nrs,
    output_image_folder,
    calibration_file,
    num_plot_workers=0,
    output_image_format="jpg",
    cancellation_token: CancellationToken | None = None,
    reporter: JobReporter | None = None,
    group_idx: int | None = None,
    log_progress: bool = False,
    return_artifacts: bool = False,
):
    """
    Process detections from each of the JSON files, project centroids,
    later saves each frame as an image by combining the projected points of each JSON.
    Inputs: json_file_paths: list of paths to JSON files containing tracking data
            camera_nr: list of camera numbers, one for each JSON file
            output_image_folder: path to the output folder where images will be saved
            calibration_file: path to the calibration file for the cameras
            num_plot_workers: number of parallel workers for rendering images (0 = sequential)
            output_image_format: "png" or "jpg" for the output images (jpg is smaller file size and faster to write)
    Returns: None
    """
    processed_artifacts = process_and_save_frames_with_metadata(
        json_file_paths,
        camera_nrs,
        output_image_folder,
        calibration_file,
        num_plot_workers=num_plot_workers,
        output_image_format=output_image_format,
        cancellation_token=cancellation_token,
        reporter=reporter,
        group_idx=group_idx,
        log_progress=log_progress,
    )
    if return_artifacts:
        return processed_artifacts
    return [artifact.processed_json_path for artifact in processed_artifacts]

def process_centroids(
    json_file,
    camera_nr,
    calibration_file,
    cancellation_token: CancellationToken | None = None,
    show_progress: bool = True,
):
    """
    Process detections from JSON, project centroids, and return the updated data.
    """
    camera_model = load_camera_setup(camera_nr, calibration_file=calibration_file)
    projection_context = load_projection_context(camera_nr, calibration_file=calibration_file)

    # Load and extract data from the JSON tracking file
    json_data = parse_json(json_file)
    frames_data = extract_data(json_data)
    
    # Process each frame
    iterable = (
        tqdm(frames_data, desc=f"Processing frames for camera {camera_nr}", unit="frame")
        if show_progress
        else frames_data
    )
    for frame in iterable:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        # Step 1: Undistort detections and centroids
        frame = process_detections(frame, camera_model)
        
        # Step 2: Project centroids to the ground plane
        projected_centroids = projection_context.project_points_to_ground(
            [detection["centroid"] for detection in frame["detections"]],
        )
        
        # Step 3: Add projected centroids to each detection and prepare for JSON output
        for detection, projected_centroid in zip(frame["detections"], projected_centroids):
            detection["projected_centroid"] = projected_centroid

    return frames_data

def save_frame_data_json(frames_data, output_json_path):
    """
    Save updated frame data with projected centroids to a new JSON file.
    """
    frames_data_json = reconstruct_json(frames_data)
    dump_path_compact(output_json_path, frames_data_json)

def plot_combined_projected_centroids(
    json_file_paths,
    base_filename,
    num_workers=0,
    image_format="png",
    cancellation_token: CancellationToken | None = None,
    reporter: JobReporter | None = None,
    group_idx: int | None = None,
    log_progress: bool = False,
):
    """
    Plot combined projected centroids from multiple JSON files for each frame and save as images.

    Parameters:
        json_file_paths (list): List of paths to JSON files containing frame data.
        base_filename (str): Base name for the output images (without suffix).
        num_workers (int): 0 => sequential; >0 => ProcessPoolExecutor workers.
        image_format (str): "png" or "jpg".
    """
    # Extract projected centroids from each JSON file
    all_projected_centroids = extract_projected_centroids_from_files(json_file_paths)

    # Determine the number of digits for zero-padded filenames based on total frames
    total_frames = len(all_projected_centroids)
    num_digits = math.ceil(math.log10(total_frames)) if total_frames > 1 else 1

    # Normalize format & set file extension
    image_format = (image_format or "png").lower()
    ext = ".jpg" if image_format in ("jpg", "jpeg") else ".png"

    # Build work items
    barn_image_path = default_barn_image_path()
    items = []
    for frame_id in sorted(all_projected_centroids.keys()):
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        projected_centroids = all_projected_centroids[frame_id]
        frame_num_str = str(frame_id).zfill(num_digits)
        frame_output_path = f"{base_filename}_frame_{frame_num_str}{ext}"
        items.append((frame_id, projected_centroids, frame_output_path, barn_image_path))

    render_progress = StageProgressReporter(
        event_prefix="processing",
        reporter_stage="processing",
        stage_name="render_frames",
        path_value=base_filename,
        path_key="base_filename",
        total=len(items),
        log_progress=log_progress,
        reporter=reporter,
        group_idx=group_idx,
    )
    render_progress.stage_started()

    # Execute work
    if num_workers and num_workers > 0:
        # Use processes (safe for OpenCV; each proc caches barn image once)
        with _fut.ProcessPoolExecutor(max_workers=num_workers) as ex:
            future_map = {ex.submit(_render_frame_worker, item): idx for idx, item in enumerate(items, start=1)}
            completed = 0
            for future in _fut.as_completed(future_map):
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
                future.result()
                completed += 1
                render_progress.step_progress(completed, len(items))
    else:
        # Sequential fallback; still avoids reloading barn per frame
        barn_img = load_barn_image(barn_image_path)
        for idx, (frame_id, projected_centroids, frame_output_path, _) in enumerate(items, start=1):
            render_projection_frame(
                projected_centroids,
                frame_id,
                frame_output_path,
                barn_image_path=barn_image_path,
                barn_image=barn_img,
            )
            render_progress.step_progress(idx, len(items))
    render_progress.stage_completed()
