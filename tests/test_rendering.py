from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np

from cowbook.vision import calibration, rendering


def test_calibration_and_rendering_defaults_resolve_inside_repo():
    calibration_path = Path(calibration.default_calibration_file())
    correspondences_path = Path(calibration.default_correspondences_file())
    barn_path = Path(rendering.default_barn_image_path())

    assert calibration_path.name == "calibration_matrix.json"
    assert correspondences_path.name == "camera_correspondences.json"
    assert barn_path.name == "barn.png"
    assert calibration_path.exists()
    assert correspondences_path.exists()


def test_render_projection_frame_writes_image(tmp_path):
    output_path = tmp_path / "frame.png"
    barn_image = np.zeros((64, 64, 3), dtype=np.uint8)

    rendering.render_projection_frame(
        [[10.0, 20.0, 100.0]],
        3,
        str(output_path),
        barn_image=barn_image,
    )

    written = cv2.imread(str(output_path))
    assert written is not None
    assert written.shape == (64, 64, 3)


def test_package_modules_do_not_import_removed_legacy_modules():
    package_root = Path(__file__).resolve().parent.parent / "src" / "cowbook"

    for module_path in package_root.rglob("*.py"):
        tree = ast.parse(module_path.read_text(), filename=str(module_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "legacy_bridge" not in alias.name
                    assert "legacy_impl" not in alias.name
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                assert "legacy_bridge" not in module_name
                assert "legacy_impl" not in module_name
