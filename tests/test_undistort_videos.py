from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cowbook.io.json_utils import load_path
from cowbook.vision import calibration
from tools import undistort_videos as module


def test_main_undistorts_folder_and_scales_calibration(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "chunks"
    input_dir.mkdir()
    video_path = input_dir / "Ch1_part_000.mp4"
    video_path.write_bytes(b"video")

    class FakeCapture:
        def __init__(self):
            self._frames = [
                np.zeros((720, 1280, 3), dtype=np.uint8),
                np.ones((720, 1280, 3), dtype=np.uint8),
            ]

        def get(self, prop):
            if prop == module.cv2.CAP_PROP_FRAME_WIDTH:
                return 1280
            if prop == module.cv2.CAP_PROP_FRAME_HEIGHT:
                return 720
            if prop == module.cv2.CAP_PROP_FPS:
                return 6.0
            return 0.0

        def read(self):
            if not self._frames:
                return False, None
            return True, self._frames.pop(0)

        def release(self):
            return None

    written_frames: list[np.ndarray] = []

    class FakeWriter:
        def __init__(self, path, fourcc, fps, frame_size):
            self.path = path
            self.fps = fps
            self.frame_size = frame_size
            self.opened = True

        def isOpened(self):
            return self.opened

        def write(self, frame):
            written_frames.append(frame.copy())

        def release(self):
            return None

    calls: dict[str, object] = {}

    monkeypatch.setattr(module, "_open_capture", lambda path: FakeCapture())
    monkeypatch.setattr(module.cv2, "VideoWriter", FakeWriter)
    monkeypatch.setattr(module.cv2, "VideoWriter_fourcc", lambda *args: 1234)
    monkeypatch.setattr(
        module,
        "resolve_camera_spec",
        lambda camera_nr, calibration_file=None: calibration.CameraCalibrationSpec(
            camera_nr=camera_nr,
            model_type="pinhole",
            image_size=(2688, 1520),
            camera_matrix=np.eye(3, dtype=np.float64),
            dist_coeff=np.zeros((1, 5), dtype=np.float64),
        ),
    )

    original_scale = module.scale_camera_spec

    def tracking_scale(spec, image_size, scale_reference_points=False):
        calls["scaled_to"] = image_size
        return original_scale(spec, image_size, scale_reference_points=scale_reference_points)

    monkeypatch.setattr(module, "scale_camera_spec", tracking_scale)
    monkeypatch.setattr(module, "build_camera_model", lambda spec: {"image_size": spec.image_size})
    monkeypatch.setattr(module, "build_undistort_maps", lambda camera_model: ("map1", "map2"))

    def fake_undistort_image(frame, camera_model, maps=None):
        calls["maps"] = maps
        calls["camera_model"] = camera_model
        return frame + 2

    monkeypatch.setattr(module, "undistort_image_with_model", fake_undistort_image)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "undistort_videos.py",
            "--input-dir",
            str(input_dir),
            "--camera-nr",
            "1",
            "--output-dir",
            str(tmp_path / "out"),
            "--suffix",
            "_rectified",
            "--no-log-progress",
        ],
    )

    assert module.main() == 0

    assert calls["scaled_to"] == (1280, 720)
    assert calls["maps"] == ("map1", "map2")
    assert calls["camera_model"] == {"image_size": (1280, 720)}
    assert len(written_frames) == 2
    assert int(written_frames[0][0, 0, 0]) == 2

    summary = load_path(tmp_path / "out" / "summary.json")
    assert summary["camera_nr"] == 1
    assert summary["runs"][0]["input_video_path"] == str(video_path)
    assert summary["runs"][0]["output_video_path"].endswith("Ch1_part_000_rectified.mp4")
    assert summary["runs"][0]["frame_count"] == 2
