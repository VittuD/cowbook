from __future__ import annotations

import json

import numpy as np

from cowbook.vision import calibration


def test_default_bundle_contains_world_and_supported_cameras():
    bundle = calibration.load_calibration_bundle()

    assert bundle.world.ground_plane_z_cm == 100.0
    assert bundle.world.barn_width_cm == 4200
    assert bundle.world.barn_height_cm == 2950
    assert bundle.default_spec.model_type == "pinhole"
    assert bundle.default_spec.image_size == (2688, 1520)
    assert sorted(bundle.cameras) == [1, 4, 6, 8]


def test_legacy_calibration_file_remains_supported():
    spec = calibration.resolve_camera_spec(
        1,
        calibration_file="assets/calibration/calibration_matrix.json",
    )

    assert spec.model_type == "pinhole"
    assert spec.image_size == (2688, 1520)
    assert spec.reference_points is not None
    assert spec.reference_points.image_points.shape == (91, 2)


def test_undistort_points_regression_values():
    mtx, dist = calibration.load_camera_model()
    points = [[1200.0, 700.0], [1500.0, 850.0]]

    actual = calibration.undistort_points(points, mtx, dist)

    np.testing.assert_allclose(
        actual,
        np.array(
            [
                [1290.7413, 710.2025],
                [1411.9877, 832.5745],
            ]
        ),
        rtol=1e-6,
        atol=1e-4,
    )


def test_projection_context_regression_values():
    mtx, dist = calibration.load_camera_model()
    image_points = [[1200.0, 700.0], [1500.0, 850.0]]
    undistorted_points = calibration.undistort_points(image_points, mtx, dist)

    actual = calibration.load_projection_context(1).project_points_to_ground(undistorted_points)

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.array(
            [
                [3307, 268, 100],
                [3478, 143, 100],
            ],
            dtype=np.int32,
        ),
    )


def test_projection_context_is_cached(monkeypatch):
    calibration.load_projection_context.cache_clear()

    calls = {"count": 0}
    original_solve_pnp = calibration.cv.solvePnP

    def counting_solve_pnp(*args, **kwargs):
        calls["count"] += 1
        return original_solve_pnp(*args, **kwargs)

    monkeypatch.setattr(calibration.cv, "solvePnP", counting_solve_pnp)

    first = calibration.load_projection_context(1)
    second = calibration.load_projection_context(1)

    assert first is second
    assert calls["count"] == 1


def test_camera_correspondences_are_available_for_supported_cameras():
    correspondences = calibration.load_camera_correspondences()

    assert sorted(correspondences) == [1, 4, 6, 8]
    assert correspondences[1].image_points.shape == (91, 2)
    assert correspondences[1].object_points.shape == (91, 3)


def test_structured_bundle_supports_per_camera_override(tmp_path):
    bundle_path = tmp_path / "camera_bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "version": 1,
                "defaults": {
                    "model_type": "pinhole",
                    "image_size": [100, 80],
                    "camera_matrix": [[50.0, 0.0, 50.0], [0.0, 50.0, 40.0], [0.0, 0.0, 1.0]],
                    "dist_coeff": [[0.0, 0.0, 0.0, 0.0]],
                },
                "cameras": {
                    "9": {
                        "model_type": "fisheye",
                        "image_size": [64, 48],
                        "dist_coeff": [[0.0], [0.0], [0.0], [0.0]],
                        "reference_points": {
                            "image_points": [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]],
                            "object_points": [
                                [0.0, 0.0, 100.0],
                                [100.0, 0.0, 100.0],
                                [100.0, 100.0, 100.0],
                                [0.0, 100.0, 100.0],
                            ],
                        },
                    }
                },
            }
        )
    )

    spec = calibration.resolve_camera_spec(9, calibration_file=str(bundle_path))

    assert spec.model_type == "fisheye"
    assert spec.image_size == (64, 48)
    assert spec.reference_points is not None
    assert spec.reference_points.image_points.shape == (4, 2)


def test_fisheye_camera_model_and_undistortion_path():
    spec = calibration.CameraCalibrationSpec(
        camera_nr=9,
        model_type="fisheye",
        image_size=(64, 48),
        camera_matrix=np.array(
            [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        dist_coeff=np.zeros((4, 1), dtype=np.float64),
    )

    camera_model = calibration.build_camera_model(spec)
    undistorted = calibration.undistort_points_with_model([[12.0, 13.0], [20.0, 22.0]], camera_model)

    assert camera_model.model_type == "fisheye"
    assert undistorted.shape == (2, 2)
    assert np.isfinite(undistorted).all()


def test_scale_camera_spec_scales_intrinsics_and_reference_points():
    spec = calibration.CameraCalibrationSpec(
        camera_nr=1,
        model_type="pinhole",
        image_size=(100, 50),
        camera_matrix=np.array(
            [[40.0, 0.0, 30.0], [0.0, 20.0, 10.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        dist_coeff=np.zeros((1, 5), dtype=np.float64),
        reference_points=calibration.CameraCorrespondences(
            image_points=np.array([[10.0, 5.0], [20.0, 10.0]], dtype=np.float32),
            object_points=np.array([[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]], dtype=np.float32),
        ),
    )

    scaled = calibration.scale_camera_spec(spec, (200, 100))

    np.testing.assert_allclose(
        scaled.camera_matrix,
        np.array(
            [[80.0, 0.0, 60.0], [0.0, 40.0, 20.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )
    assert scaled.image_size == (200, 100)
    assert scaled.reference_points is not None
    np.testing.assert_allclose(
        scaled.reference_points.image_points,
        np.array([[20.0, 10.0], [40.0, 20.0]], dtype=np.float32),
    )
