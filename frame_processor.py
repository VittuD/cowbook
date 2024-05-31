# frame_processor.py

from processing import parse_json, extract_data, process_detections, project_to_ground

def process_and_save_frames(json_file, camera_nr, output_image_folder, calibration_file=None):
    """
    Process detections with calibration data and save frames with projected centroids.
    """
    if calibration_file:
        mtx, dist = load_calibration(calibration_file)
        print(f"Loaded calibration data from {calibration_file}")
    
    json_data = parse_json(json_file)
    frames_data = extract_data(json_data)
    
    for frame in frames_data:
        frame_data = process_detections(frame)
        if calibration_file:
            projected_centroids = project_to_ground([det['centroid'] for det in frame_data], mtx, dist, camera_nr)
        print(f"Processed frame {frame['frame_id']} for camera {camera_nr}")
