from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from cowbook.core.runtime import assets_root

CAMERA_WIDTH = 2688
CAMERA_HEIGHT = 1520
UNDISTORTION_COEFFICENT = 0
GROUND_PLANE_Z_CM = 100.0
BARN_WIDTH_CM = 4200
BARN_HEIGHT_CM = 2950


@dataclass(frozen=True)
class CameraModel:
    mtx: NDArray[np.float64]
    dist: NDArray[np.float64]
    optimal_mtx: NDArray[np.float64]


@dataclass(frozen=True)
class CameraCorrespondences:
    image_points: NDArray[np.float32]
    object_points: NDArray[np.float32]


@dataclass(frozen=True)
class ProjectionContext:
    camera_nr: int
    camera_model: CameraModel
    rotation_matrix: NDArray[np.float64]
    inverse_rotation_matrix: NDArray[np.float64]
    inverse_camera_matrix: NDArray[np.float64]
    translation_vector: NDArray[np.float64]
    rotated_translation_vector: NDArray[np.float64]

    def project_points_to_ground(
        self,
        points: Iterable[Iterable[float]],
        *,
        z_cm: float = GROUND_PLANE_Z_CM,
    ) -> list[NDArray[np.int32]]:
        points_array = np.asarray(list(points), dtype=np.float64).reshape(-1, 2)
        if points_array.size == 0:
            return []

        real_points: list[NDArray[np.int32]] = []
        for point in points_array:
            uv_point = np.ones((3, 1), dtype=np.float64)
            uv_point[0, 0] = point[0]
            uv_point[1, 0] = point[1]

            rotated_inverse_projected = self.inverse_rotation_matrix @ (
                self.inverse_camera_matrix @ uv_point
            )
            scale = (z_cm + self.rotated_translation_vector[2, 0]) / rotated_inverse_projected[2, 0]
            world_point = self.inverse_rotation_matrix @ (
                (scale * (self.inverse_camera_matrix @ uv_point)) - self.translation_vector
            )

            world_point[2, 0] = z_cm
            real_point = world_point.reshape(-1).astype(np.int32)
            real_point[0] = np.clip(real_point[0], 0, BARN_WIDTH_CM)
            real_point[1] = np.clip(real_point[1], 0, BARN_HEIGHT_CM)
            real_points.append(real_point)

        return real_points


def default_calibration_file() -> str:
    return str(assets_root() / "calibration" / "calibration_matrix.json")


def default_correspondences_file() -> str:
    return str(assets_root() / "calibration" / "camera_correspondences.json")


def _normalize_path(path: str | Path | None, *, default: str) -> str:
    resolved = Path(path or default).resolve()
    return str(resolved)


def build_camera_model(mtx, dist) -> CameraModel:
    mtx_arr = np.asarray(mtx, dtype=np.float64)
    dist_arr = np.asarray(dist, dtype=np.float64)
    optimal_mtx, _ = cv.getOptimalNewCameraMatrix(
        mtx_arr,
        dist_arr,
        (CAMERA_WIDTH, CAMERA_HEIGHT),
        UNDISTORTION_COEFFICENT,
        (CAMERA_WIDTH, CAMERA_HEIGHT),
    )
    return CameraModel(mtx=mtx_arr, dist=dist_arr, optimal_mtx=np.asarray(optimal_mtx, dtype=np.float64))


@lru_cache(maxsize=None)
def load_camera_setup(calibration_file: str | None = None) -> CameraModel:
    calibration_path = _normalize_path(calibration_file, default=default_calibration_file())
    with open(calibration_path) as file:
        data = json.load(file)
    return build_camera_model(data["camera_matrix"], data["dist_coeff"])


def load_camera_model(calibration_file: str | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    camera_model = load_camera_setup(calibration_file)
    return camera_model.mtx, camera_model.dist


@lru_cache(maxsize=None)
def load_camera_correspondences(
    correspondences_file: str | None = None,
) -> dict[int, CameraCorrespondences]:
    correspondences_path = _normalize_path(
        correspondences_file,
        default=default_correspondences_file(),
    )
    with open(correspondences_path) as file:
        payload = json.load(file)

    correspondences: dict[int, CameraCorrespondences] = {}
    for camera_nr_str, values in payload.items():
        correspondences[int(camera_nr_str)] = CameraCorrespondences(
            image_points=np.asarray(values["image_points"], dtype=np.float32),
            object_points=np.asarray(values["object_points"], dtype=np.float32),
        )
    return correspondences


def undistort_points_with_model(
    points: Iterable[Iterable[float]],
    camera_model: CameraModel,
) -> NDArray[np.float64]:
    points_array = np.asarray(list(points), dtype=np.float32)
    if points_array.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    reshaped_points = points_array.reshape(-1, 1, 2)
    undistorted_points = cv.undistortPointsIter(
        reshaped_points,
        camera_model.mtx,
        camera_model.dist,
        None,
        camera_model.optimal_mtx,
        (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 40, 0.03),
    )
    return undistorted_points.reshape(-1, 2)


def undistort_points(points: Iterable[Iterable[float]], mtx, dist) -> NDArray[np.float64]:
    return undistort_points_with_model(points, build_camera_model(mtx, dist))


def build_projection_context(
    camera_nr: int,
    camera_model: CameraModel,
    correspondences: CameraCorrespondences,
) -> ProjectionContext:
    undistorted_reference_points = undistort_points_with_model(
        correspondences.image_points,
        camera_model,
    ).astype(np.int32).astype(np.float32)
    success, rotation_vector, translation_vector = cv.solvePnP(
        correspondences.object_points,
        undistorted_reference_points,
        camera_model.optimal_mtx,
        0,
    )
    if not success:
        raise RuntimeError(f"solvePnP failed for camera {camera_nr}")

    rotation_matrix, _ = cv.Rodrigues(rotation_vector)
    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    inverse_camera_matrix = np.linalg.inv(camera_model.optimal_mtx)
    rotated_translation_vector = inverse_rotation_matrix @ translation_vector

    return ProjectionContext(
        camera_nr=camera_nr,
        camera_model=camera_model,
        rotation_matrix=rotation_matrix,
        inverse_rotation_matrix=inverse_rotation_matrix,
        inverse_camera_matrix=inverse_camera_matrix,
        translation_vector=translation_vector,
        rotated_translation_vector=rotated_translation_vector,
    )


@lru_cache(maxsize=None)
def load_projection_context(
    camera_nr: int,
    calibration_file: str | None = None,
    correspondences_file: str | None = None,
) -> ProjectionContext:
    camera_model = load_camera_setup(calibration_file)
    correspondences = load_camera_correspondences(correspondences_file)
    try:
        camera_correspondences = correspondences[camera_nr]
    except KeyError as exc:
        raise KeyError(f"Camera {camera_nr} has no calibration correspondences.") from exc
    return build_projection_context(camera_nr, camera_model, camera_correspondences)


def project_points_to_ground(
    camera_nr: int,
    mtx,
    dist,
    points: Iterable[Iterable[float]],
) -> list[NDArray[np.int32]]:
    correspondences = load_camera_correspondences()
    try:
        camera_correspondences = correspondences[camera_nr]
    except KeyError as exc:
        raise KeyError(f"Camera {camera_nr} has no calibration correspondences.") from exc
    projection_context = build_projection_context(
        camera_nr,
        build_camera_model(mtx, dist),
        camera_correspondences,
    )
    return projection_context.project_points_to_ground(points)
