from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2 as cv
import numpy as np

from cowbook.core.runtime import assets_root
from cowbook.vision.calibration import BARN_HEIGHT_CM, BARN_WIDTH_CM


def default_barn_image_path() -> str:
    return str(assets_root() / "images" / "barn.png")


def load_barn_image(barn_image_path: str | None = None):
    path = barn_image_path or default_barn_image_path()
    if not Path(path).exists():
        return None
    return cv.imread(path)


def points_to_barn(points: Iterable[Iterable[float]], barn_img, show: bool = False):
    img_height, img_width = barn_img.shape[:2]
    pixel_to_cm_height = BARN_HEIGHT_CM / img_height
    pixel_to_cm_width = BARN_WIDTH_CM / img_width

    barn_points = []
    for point in points:
        img_px_x = point[0] / pixel_to_cm_width
        img_px_y = img_height - (point[1] / pixel_to_cm_height)
        barn_points.append((int(img_px_x), int(img_px_y)))
        cv.circle(barn_img, (int(img_px_x), int(img_px_y)), 2, (162, 81, 250), 4)

    if show:
        cv.imshow("barn", barn_img)

    return barn_points, barn_img


def render_projection_frame(
    projected_points,
    frame_num: int,
    output_path: str,
    *,
    barn_image_path: str | None = None,
    barn_image=None,
) -> None:
    if barn_image is not None:
        image = barn_image.copy()
    else:
        image = load_barn_image(barn_image_path)

    if image is None:
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    _, drawn_image = points_to_barn(projected_points, image)
    cv.putText(
        drawn_image,
        f"Frame: {frame_num}",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    cv.imwrite(output_path, drawn_image)
