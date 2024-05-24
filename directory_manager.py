# directory_manager.py

import os
import shutil
import datetime

def clear_output_directory(directory_path, archive=False):
    """
    Clear all files in the specified directory.
    
    Parameters:
        directory_path (str): Path to the directory to clear.
        archive (bool): If True, archive directory contents before clearing.
    """
    if os.path.exists(directory_path):
        if archive:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = f"{directory_path}_backup_{timestamp}"
            shutil.copytree(directory_path, archive_path)
            print(f"Directory contents archived at {archive_path}")

        # Proceed with clearing the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(directory_path)
        print(f"Created new directory at {directory_path}")
