from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from cowbook.core.runtime import assets_root
from cowbook.io.json_utils import load_path

CalibrationModelType = Literal["pinhole", "fisheye"]
SUPPORTED_MODEL_TYPES: tuple[CalibrationModelType, ...] = ("pinhole", "fisheye")
DEFAULT_IMAGE_SIZE = (2688, 1520)
UNDISTORTION_COEFFICENT = 0
BARN_WIDTH_CM = 4200
BARN_HEIGHT_CM = 2950
GROUND_PLANE_Z_CM = 100.0


@dataclass(frozen=True)
class WorldGeometry:
    ground_plane_z_cm: float = GROUND_PLANE_Z_CM
    barn_width_cm: int = BARN_WIDTH_CM
    barn_height_cm: int = BARN_HEIGHT_CM


@dataclass(frozen=True)
class CameraCorrespondences:
    image_points: NDArray[np.float32]
    object_points: NDArray[np.float32]


@dataclass(frozen=True)
class CameraCalibrationSpec:
    camera_nr: int | None
    model_type: CalibrationModelType
    image_size: tuple[int, int]
    camera_matrix: NDArray[np.float64]
    dist_coeff: NDArray[np.float64]
    reference_points: CameraCorrespondences | None = None


@dataclass(frozen=True)
class CalibrationBundle:
    world: WorldGeometry
    default_spec: CameraCalibrationSpec
    cameras: dict[int, CameraCalibrationSpec]


@dataclass(frozen=True)
class CameraModel:
    spec: CameraCalibrationSpec
    optimal_mtx: NDArray[np.float64]

    @property
    def mtx(self) -> NDArray[np.float64]:
        return self.spec.camera_matrix

    @property
    def dist(self) -> NDArray[np.float64]:
        return self.spec.dist_coeff

    @property
    def image_size(self) -> tuple[int, int]:
        return self.spec.image_size

    @property
    def model_type(self) -> CalibrationModelType:
        return self.spec.model_type


@dataclass(frozen=True)
class ProjectionContext:
    camera_nr: int
    world: WorldGeometry
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
        z_cm: float | None = None,
    ) -> list[NDArray[np.int32]]:
        ground_z_cm = self.world.ground_plane_z_cm if z_cm is None else z_cm
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
            scale = (ground_z_cm + self.rotated_translation_vector[2, 0]) / rotated_inverse_projected[2, 0]
            world_point = self.inverse_rotation_matrix @ (
                (scale * (self.inverse_camera_matrix @ uv_point)) - self.translation_vector
            )

            world_point[2, 0] = ground_z_cm
            real_point = world_point.reshape(-1).astype(np.int32)
            real_point[0] = np.clip(real_point[0], 0, self.world.barn_width_cm)
            real_point[1] = np.clip(real_point[1], 0, self.world.barn_height_cm)
            real_points.append(real_point)

        return real_points


def default_calibration_file() -> str:
    return str(assets_root() / "calibration" / "camera_system.json")


def default_correspondences_file() -> str:
    return str(assets_root() / "calibration" / "camera_correspondences.json")


def _normalize_path(path: str | Path | None, *, default: str) -> str:
    return str(Path(path or default).resolve())


def _as_model_type(value: str | None, *, default: CalibrationModelType = "pinhole") -> CalibrationModelType:
    model_type = str(value or default).lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported calibration model_type: {model_type}")
    return model_type  # type: ignore[return-value]


def _as_image_size(value, *, default: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> tuple[int, int]:
    if value is None:
        return default
    if len(value) != 2:
        raise ValueError(f"image_size must contain [width, height], got {value!r}")
    return int(value[0]), int(value[1])


def _as_float_array(value) -> NDArray[np.float64]:
    return np.asarray(value, dtype=np.float64)


def _parse_reference_points(value) -> CameraCorrespondences | None:
    if not value:
        return None
    image_points = value.get("image_points")
    object_points = value.get("object_points")
    if image_points is None or object_points is None:
        return None
    return CameraCorrespondences(
        image_points=np.asarray(image_points, dtype=np.float32),
        object_points=np.asarray(object_points, dtype=np.float32),
    )


def _build_spec_from_payload(
    payload: dict,
    *,
    camera_nr: int | None,
    fallback: CameraCalibrationSpec | None = None,
) -> CameraCalibrationSpec:
    if fallback is None:
        if "camera_matrix" not in payload or "dist_coeff" not in payload:
            raise ValueError("Calibration payload requires camera_matrix and dist_coeff.")
        model_type = _as_model_type(payload.get("model_type"))
        image_size = _as_image_size(payload.get("image_size"))
        camera_matrix = _as_float_array(payload["camera_matrix"])
        dist_coeff = _as_float_array(payload["dist_coeff"])
    else:
        model_type = _as_model_type(payload.get("model_type"), default=fallback.model_type)
        image_size = _as_image_size(payload.get("image_size"), default=fallback.image_size)
        camera_matrix = _as_float_array(payload.get("camera_matrix", fallback.camera_matrix))
        dist_coeff = _as_float_array(payload.get("dist_coeff", fallback.dist_coeff))

    reference_points = _parse_reference_points(payload.get("reference_points"))
    return CameraCalibrationSpec(
        camera_nr=camera_nr,
        model_type=model_type,
        image_size=image_size,
        camera_matrix=camera_matrix,
        dist_coeff=dist_coeff,
        reference_points=reference_points,
    )


def _is_structured_bundle(payload: dict) -> bool:
    return any(key in payload for key in ("version", "world", "defaults", "cameras"))


def _parse_world(payload: dict | None) -> WorldGeometry:
    payload = payload or {}
    barn_bounds = payload.get("barn_bounds_cm", {}) or {}
    return WorldGeometry(
        ground_plane_z_cm=float(payload.get("ground_plane_z_cm", 100.0)),
        barn_width_cm=int(barn_bounds.get("width", 4200)),
        barn_height_cm=int(barn_bounds.get("height", 2950)),
    )


@lru_cache(maxsize=None)
def load_calibration_bundle(calibration_file: str | None = None) -> CalibrationBundle:
    calibration_path = _normalize_path(calibration_file, default=default_calibration_file())
    payload = load_path(calibration_path)

    if _is_structured_bundle(payload):
        world = _parse_world(payload.get("world"))
        default_spec = _build_spec_from_payload(payload.get("defaults", {}), camera_nr=None)
        cameras = {
            int(camera_nr): _build_spec_from_payload(camera_payload, camera_nr=int(camera_nr), fallback=default_spec)
            for camera_nr, camera_payload in (payload.get("cameras", {}) or {}).items()
        }
        return CalibrationBundle(world=world, default_spec=default_spec, cameras=cameras)

    default_spec = _build_spec_from_payload(payload, camera_nr=None)
    return CalibrationBundle(world=WorldGeometry(), default_spec=default_spec, cameras={})


@lru_cache(maxsize=None)
def load_camera_correspondences(
    calibration_file: str | None = None,
    correspondences_file: str | None = None,
) -> dict[int, CameraCorrespondences]:
    bundle = load_calibration_bundle(calibration_file)
    bundled = {
        camera_nr: spec.reference_points
        for camera_nr, spec in bundle.cameras.items()
        if spec.reference_points is not None
    }
    if bundled:
        return bundled  # type: ignore[return-value]

    correspondences_path = _normalize_path(
        correspondences_file,
        default=default_correspondences_file(),
    )
    payload = load_path(correspondences_path)

    correspondences: dict[int, CameraCorrespondences] = {}
    for camera_nr_str, values in payload.items():
        correspondences[int(camera_nr_str)] = CameraCorrespondences(
            image_points=np.asarray(values["image_points"], dtype=np.float32),
            object_points=np.asarray(values["object_points"], dtype=np.float32),
        )
    return correspondences


def resolve_camera_spec(
    camera_nr: int | None = None,
    *,
    calibration_file: str | None = None,
    correspondences_file: str | None = None,
) -> CameraCalibrationSpec:
    bundle = load_calibration_bundle(calibration_file)
    if camera_nr is None:
        return bundle.default_spec

    spec = bundle.cameras.get(camera_nr)
    if spec is None:
        spec = replace(bundle.default_spec, camera_nr=camera_nr)

    if spec.reference_points is not None:
        return spec

    correspondences = load_camera_correspondences(calibration_file, correspondences_file)
    if camera_nr not in correspondences:
        return spec
    return replace(spec, reference_points=correspondences[camera_nr])


def build_camera_model(
    spec_or_mtx,
    dist=None,
    *,
    model_type: CalibrationModelType = "pinhole",
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> CameraModel:
    if isinstance(spec_or_mtx, CameraCalibrationSpec):
        spec = spec_or_mtx
    else:
        spec = CameraCalibrationSpec(
            camera_nr=None,
            model_type=model_type,
            image_size=image_size,
            camera_matrix=_as_float_array(spec_or_mtx),
            dist_coeff=_as_float_array(dist),
        )

    width, height = spec.image_size
    if spec.model_type == "pinhole":
        optimal_mtx, _ = cv.getOptimalNewCameraMatrix(
            spec.camera_matrix,
            spec.dist_coeff,
            (width, height),
            UNDISTORTION_COEFFICENT,
            (width, height),
        )
    else:
        optimal_mtx = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
            spec.camera_matrix,
            spec.dist_coeff,
            (width, height),
            np.eye(3),
            balance=0.0,
            new_size=(width, height),
        )
    return CameraModel(spec=spec, optimal_mtx=np.asarray(optimal_mtx, dtype=np.float64))


@lru_cache(maxsize=None)
def load_camera_setup(
    camera_nr: int | None = None,
    calibration_file: str | None = None,
    correspondences_file: str | None = None,
) -> CameraModel:
    return build_camera_model(
        resolve_camera_spec(
            camera_nr,
            calibration_file=calibration_file,
            correspondences_file=correspondences_file,
        )
    )


def load_camera_model(
    calibration_file: str | None = None,
    camera_nr: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    camera_model = load_camera_setup(camera_nr, calibration_file)
    return camera_model.mtx, camera_model.dist


def undistort_points_with_model(
    points: Iterable[Iterable[float]],
    camera_model: CameraModel,
) -> NDArray[np.float64]:
    points_array = np.asarray(list(points), dtype=np.float32)
    if points_array.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    reshaped_points = points_array.reshape(-1, 1, 2)
    if camera_model.model_type == "pinhole":
        undistorted_points = cv.undistortPointsIter(
            reshaped_points,
            camera_model.mtx,
            camera_model.dist,
            None,
            camera_model.optimal_mtx,
            (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 40, 0.03),
        )
    else:
        undistorted_points = cv.fisheye.undistortPoints(
            reshaped_points,
            camera_model.mtx,
            camera_model.dist,
            R=np.eye(3),
            P=camera_model.optimal_mtx,
        )
    return undistorted_points.reshape(-1, 2)


def undistort_points(
    points: Iterable[Iterable[float]],
    mtx,
    dist,
    *,
    model_type: CalibrationModelType = "pinhole",
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> NDArray[np.float64]:
    return undistort_points_with_model(
        points,
        build_camera_model(mtx, dist, model_type=model_type, image_size=image_size),
    )


def build_projection_context(
    camera_nr: int,
    camera_model: CameraModel,
    correspondences: CameraCorrespondences | None = None,
    *,
    world: WorldGeometry | None = None,
) -> ProjectionContext:
    reference_points = correspondences or camera_model.spec.reference_points
    if reference_points is None:
        raise KeyError(f"Camera {camera_nr} has no calibration correspondences.")

    undistorted_reference_points = undistort_points_with_model(
        reference_points.image_points,
        camera_model,
    ).astype(np.int32).astype(np.float32)
    success, rotation_vector, translation_vector = cv.solvePnP(
        reference_points.object_points,
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
        world=world or WorldGeometry(),
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
    bundle = load_calibration_bundle(calibration_file)
    spec = resolve_camera_spec(
        camera_nr,
        calibration_file=calibration_file,
        correspondences_file=correspondences_file,
    )
    return build_projection_context(
        camera_nr,
        build_camera_model(spec),
        spec.reference_points,
        world=bundle.world,
    )


def project_points_to_ground(
    camera_nr: int,
    mtx,
    dist,
    points: Iterable[Iterable[float]],
    *,
    calibration_file: str | None = None,
    correspondences_file: str | None = None,
) -> list[NDArray[np.int32]]:
    bundle = load_calibration_bundle(calibration_file)
    base_spec = resolve_camera_spec(
        camera_nr,
        calibration_file=calibration_file,
        correspondences_file=correspondences_file,
    )
    spec = replace(
        base_spec,
        camera_matrix=_as_float_array(mtx),
        dist_coeff=_as_float_array(dist),
    )
    projection_context = build_projection_context(
        camera_nr,
        build_camera_model(spec),
        spec.reference_points,
        world=bundle.world,
    )
    return projection_context.project_points_to_ground(points)
