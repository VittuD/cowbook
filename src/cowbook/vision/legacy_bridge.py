from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from numpy.typing import NDArray

from cowbook.core.runtime import assets_root
from cowbook.vision.legacy_impl import image_utils as legacy_utils


def default_calibration_file() -> str:
    return str(assets_root() / "calibration" / "calibration_matrix.json")


def default_barn_image_path() -> str:
    return str(assets_root() / "images" / "barn.png")


def load_camera_model(calibration_file: str | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    path = calibration_file or default_calibration_file()
    return legacy_utils.get_calibrated_camera_model(path)


def undistort_points(points: Iterable[Iterable[float]], mtx, dist):
    return legacy_utils.undistort_points_given(points, mtx, dist)


def project_points_to_ground(channel: int, mtx, dist, points):
    return legacy_utils.groundProjectPoint(channel, mtx, dist, points)


def load_barn_image(barn_image_path: str | None = None):
    path = barn_image_path or default_barn_image_path()
    if not Path(path).exists():
        return None
    return cv2.imread(path)


def render_projection_frame(
    projected_points,
    frame_num: int,
    output_path: str,
    *,
    barn_image_path: str | None = None,
    barn_image=None,
) -> None:
    legacy_utils.save_frame_image(
        projected_points,
        frame_num,
        output_path,
        barn_image_path=barn_image_path or default_barn_image_path(),
        barn_image=barn_image,
    )
