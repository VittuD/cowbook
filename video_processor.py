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
    Generate a video from a sequence of images.
    """
    # Collect image filenames and sort them by frame number
    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))],
                    key=extract_frame_number)
    
    if not images:
        raise ValueError("No images found in the specified folder.")
    
    # Load the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # Define codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Font settings for overlaying frame number text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 255, 0)
    thickness = 1

    # Iterate through each image, add frame number, and write to video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        
        # Extract frame number and overlay it
        frame_number = extract_frame_number(image)
        text = f"Frame: {frame_number}"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = width - text_size[0] - 10
        text_y = text_size[1] + 10
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness)
        
        # Write the frame to the video
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video saved as {output_video_path}")
