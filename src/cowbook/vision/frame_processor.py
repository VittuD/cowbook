import concurrent.futures as _fut  # parallel rendering
import json
import logging
import math
import os

from tqdm import tqdm

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

def process_and_save_frames(json_file_paths, camera_nrs, output_image_folder, calibration_file,
                            num_plot_workers=0, output_image_format="jpg"):
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
    # Process centroids for each JSON file
    updated_json_file_paths = []
    for json_file_path, camera_nr in zip(json_file_paths, camera_nrs):
        try:
            frames_data = process_centroids(json_file_path, camera_nr, calibration_file)

            updated_json_file_path = os.path.join(
                json_file_path.replace(".json", "_processed.json")
            )
            save_frame_data_json(frames_data, updated_json_file_path)
            updated_json_file_paths.append(updated_json_file_path)
            logger.info("Processed frame data saved to %s", updated_json_file_path)
        except Exception as exc:
            logger.exception(
                "Failed processing %s for camera %d: %s",
                json_file_path,
                camera_nr,
                exc,
            )

    # Plot combined projected centroids from multiple JSON files for each frame and save as images
    if updated_json_file_paths:
        base_filename = os.path.join(output_image_folder, "combined_projected_centroids")
        plot_combined_projected_centroids(
            updated_json_file_paths,
            base_filename,
            num_workers=num_plot_workers,
            image_format=output_image_format,
        )

    # Return processed JSON paths so caller can merge them
    return updated_json_file_paths

def process_centroids(json_file, camera_nr, calibration_file):
    """
    Process detections from JSON, project centroids, and return the updated data.
    """
    camera_model = load_camera_setup(calibration_file)
    projection_context = load_projection_context(camera_nr, calibration_file)

    # Load and extract data from the JSON tracking file
    json_data = parse_json(json_file)
    frames_data = extract_data(json_data)
    
    # Process each frame
    for frame in tqdm(frames_data, desc=f"Processing frames for camera {camera_nr}", unit="frame"):
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
    with open(output_json_path, 'w') as output_file:
        json.dump(frames_data_json, output_file, indent=4)

def plot_combined_projected_centroids(json_file_paths, base_filename, num_workers=0, image_format="png"):
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
        projected_centroids = all_projected_centroids[frame_id]
        frame_num_str = str(frame_id).zfill(num_digits)
        frame_output_path = f"{base_filename}_frame_{frame_num_str}{ext}"
        items.append((frame_id, projected_centroids, frame_output_path, barn_image_path))

    # Execute work
    if num_workers and num_workers > 0:
        # Use processes (safe for OpenCV; each proc caches barn image once)
        with _fut.ProcessPoolExecutor(max_workers=num_workers) as ex:
            list(ex.map(_render_frame_worker, items, chunksize=16))
    else:
        # Sequential fallback; still avoids reloading barn per frame
        barn_img = load_barn_image(barn_image_path)
        for frame_id, projected_centroids, frame_output_path, _ in items:
            render_projection_frame(
                projected_centroids,
                frame_id,
                frame_output_path,
                barn_image_path=barn_image_path,
                barn_image=barn_img,
            )
