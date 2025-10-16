# directory_manager.py

import os
import shutil
import tempfile
import logging
from typing import Iterable, Tuple

logger = logging.getLogger(__name__)


def _verify_writable(directory_path: str) -> None:
    """
    Verify that the given directory is writable by attempting to create a temp file.
    Raises PermissionError if not writable.
    """
    try:
        with tempfile.NamedTemporaryFile(dir=directory_path):
            pass
    except Exception as e:
        raise PermissionError(f"Directory is not writable: {directory_path}. Reason: {e}") from e


def ensure_directory(directory_path: str, verify_writable: bool = True) -> None:
    """
    Ensure a directory exists and (optionally) is writable.

    - Creates the directory (and parents) if missing.
    - Raises NotADirectoryError if the path exists but is not a directory.
    - Optionally verifies write access by creating a temp file.
    """
    if os.path.exists(directory_path) and not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Path exists and is not a directory: {directory_path}")

    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        logger.debug("Created directory: %s", directory_path)

    if verify_writable:
        _verify_writable(directory_path)


def ensure_directories(dirs: Iterable[str], verify_writable: bool = True) -> None:
    """
    Ensure multiple directories exist (and are writable if requested).
    """
    for d in dirs:
        ensure_directory(d, verify_writable=verify_writable)


def ensure_parent_dir(file_path: str, verify_writable: bool = True) -> None:
    """
    Ensure the parent directory of a file exists (and is writable if requested).
    """
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent:
        ensure_directory(parent, verify_writable=verify_writable)


def prepare_output_dirs(config: dict) -> Tuple[str, str, str]:
    """
    Read output directory settings from config, ensure they exist and are writable,
    and return (output_image_folder, output_video_folder, output_json_folder).
    """
    output_image_folder = config.get("output_image_folder", "output_frames")
    output_video_folder = config.get("output_video_folder", "output_videos")
    output_json_folder = config.get("output_json_folder", "output_json")

    ensure_directories(
        [output_image_folder, output_video_folder, output_json_folder],
        verify_writable=True,
    )
    logger.info(
        "Output directories ready: images='%s', videos='%s', json='%s'",
        output_image_folder,
        output_video_folder,
        output_json_folder,
    )
    return output_image_folder, output_video_folder, output_json_folder


def clear_output_directory(directory_path: str) -> None:
    """
    Clear all files and subdirectories in the specified directory.
    If the directory does not exist, it will be created.
    """
    ensure_directory(directory_path, verify_writable=True)

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.warning("Failed to delete %s. Reason: %s", file_path, e)
