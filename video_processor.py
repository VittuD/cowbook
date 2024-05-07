# video_processor.py

import cv2
import os
import re

def extract_frame_number(filename):
    """
    Extract numeric frame number from filename for sorting.
    """
    match = re.search(r'frame_(\d+)', filename)
    return int(match.group(1)) if match else -1

def create_video_from_images(image_folder, output_video_path, fps=6):
    """
    Generate a video from a sequence of images in the specified folder.

    Parameters:
        image_folder (str): Path to the folder containing projected images.
        output_video_path (str): Path where the output video will be saved.
        fps (int): Frames per second for the output video. Default is 6.
    """
    # Gather image filenames and sort by frame number
    images = sorted(
        [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))],
        key=extract_frame_number
    )
    
    if not images:
        raise ValueError("No images found in the specified folder.")

    # Read the first image to set the video dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # Define codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Add each image frame to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video_path}")
