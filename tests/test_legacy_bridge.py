from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cowbook.vision import legacy_bridge


def test_legacy_bridge_defaults_resolve_inside_repo():
    calibration_path = Path(legacy_bridge.default_calibration_file())
    barn_path = Path(legacy_bridge.default_barn_image_path())

    assert calibration_path.name == "calibration_matrix.json"
    assert barn_path.name == "barn.png"
    assert calibration_path.exists()


def test_legacy_bridge_loads_camera_model_from_default_path():
    mtx, dist = legacy_bridge.load_camera_model()

    assert mtx.shape == (3, 3)
    assert dist.ndim == 2


def test_legacy_bridge_renders_projection_frame_without_direct_legacy_imports(tmp_path):
    output_path = tmp_path / "frame.png"
    barn_image = np.zeros((64, 64, 3), dtype=np.uint8)

    legacy_bridge.render_projection_frame(
        [[10.0, 20.0, 100.0]],
        3,
        str(output_path),
        barn_image=barn_image,
    )

    written = cv2.imread(str(output_path))
    assert written is not None
    assert written.shape == (64, 64, 3)


def test_package_modules_only_import_legacy_through_bridge():
    package_root = Path(__file__).resolve().parent.parent / "src" / "cowbook"

    for module_path in package_root.glob("*.py"):
        if module_path.name == "legacy_bridge.py":
            continue

        source = module_path.read_text()
        assert "import legacy." not in source
        assert "from legacy " not in source
