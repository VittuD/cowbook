from __future__ import annotations

import numpy as np

from cowbook.vision import calibration


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
