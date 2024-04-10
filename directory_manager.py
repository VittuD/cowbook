# directory_manager.py
# TODO Add simlink management and feedback to the user

import os

def clear_output_directory(directory_path):
    """Clear all files in the specified directory."""
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    else:
        os.makedirs(directory_path)
