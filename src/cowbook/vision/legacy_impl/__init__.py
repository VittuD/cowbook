"""
The legacy package contains utilities for image processing, calibration, and point projection.

Modules:
    utils: Provides functions for camera calibration, image undistortion, point rotation, and bird-view projection.

"""

# If you want to make certain functions accessible directly from legacy,
# import them here from image_utils. For example:
from .image_utils import (
    get_calibrated_camera_model,
    undistort_image,
    undistort_points_given,
    groundProjectPoint,
)

# Import points_data since it needs to be accessible directly from legacy
from . import points_data

# Optional versioning information
__version__ = "0.0.1"
