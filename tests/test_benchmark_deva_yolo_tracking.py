from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import benchmark_deva_yolo_tracking as module


def test_validate_model_path_requires_existing_weights(tmp_path: Path):
    weights = tmp_path / "yolo.pt"
    weights.write_bytes(b"weights")
    assert module._validate_model_path(str(weights)) == str(weights)

    with pytest.raises(FileNotFoundError, match="YOLO segmentation weights"):
        module._validate_model_path(str(tmp_path / "missing.pt"))


def test_build_deva_detection_command_matches_expected_shape():
    command = module._build_deva_detection_command(
        python_bin="/usr/bin/python3",
        frames_root="/tmp/frames",
        masks_root="/tmp/masks",
        output_root="/tmp/out",
        deva_model_path="/opt/deva/saves/DEVA-propagation.pth",
        chunk_size=4,
        size=480,
        temporal_setting="semionline",
        amp=True,
    )

    assert command == [
        "/usr/bin/python3",
        "evaluation/eval_with_detections.py",
        "--model",
        "/opt/deva/saves/DEVA-propagation.pth",
        "--img_path",
        "/tmp/frames",
        "--mask_path",
        "/tmp/masks",
        "--output",
        "/tmp/out",
        "--dataset",
        "demo",
        "--detection_every",
        "1",
        "--num_voting_frames",
        "1",
        "--chunk_size",
        "4",
        "--size",
        "480",
        "--temporal_setting",
        "semionline",
        "--amp",
    ]


def test_write_detection_artifacts_for_result_exports_png_and_json(tmp_path: Path):
    class FakeTensor:
        def __init__(self, values):
            self._values = np.asarray(values)

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    class FakeMasks:
        def __init__(self, values):
            self.data = FakeTensor(values)

    class FakeBoxes:
        def __init__(self, conf):
            self.conf = FakeTensor(conf)

    class FakeResult:
        def __init__(self):
            self.path = str(tmp_path / "0000000.jpg")
            self.orig_shape = (4, 4)
            self.masks = FakeMasks(
                [
                    [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]],
                ]
            )
            self.boxes = FakeBoxes([0.7, 0.9])

    count = module._write_detection_artifacts_for_result(result=FakeResult(), masks_dir=tmp_path)

    assert count == 2
    encoded = cv2.imread(str(tmp_path / "0000000.png"), cv2.IMREAD_GRAYSCALE)
    assert encoded is not None
    nonzero = encoded[encoded != 0]
    assert np.unique(nonzero).tolist() == [1, 2]
    segments = json.loads((tmp_path / "0000000.json").read_text(encoding="utf-8"))
    assert segments == [
        {"id": 1, "category_id": 1, "score": 0.7},
        {"id": 2, "category_id": 1, "score": 0.9},
    ]


def test_run_deva_yolo_tracking_for_video_writes_summary(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"video")
    deva_repo = tmp_path / "deva"
    deva_repo.mkdir()
    output_root = tmp_path / "out"
    frames_root = tmp_path / "frames"
    masks_root = tmp_path / "masks"
    raw_dir = output_root / "deva_raw" / "input"
    raw_dir.mkdir(parents=True)
    rendered = raw_dir / "demo.mp4"
    rendered.write_bytes(b"mp4")
    deva_model = tmp_path / "DEVA-propagation.pth"
    deva_model.write_bytes(b"weights")
    yolo_model = tmp_path / "yolo-seg.pt"
    yolo_model.write_bytes(b"weights")
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "_probe_video_metadata",
        lambda _path: {"fps": 5.0, "width": 32, "height": 24, "frame_count": 2},
    )

    def fake_extract_video_frames(*, video_path: str, output_dir: Path, max_frames: int):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "0000000.jpg").write_bytes(b"jpg")
        (output_dir / "0000001.jpg").write_bytes(b"jpg")
        return {
            "frame_count": 2,
            "fps": 5.0,
            "width": 32,
            "height": 24,
            "frames_dir": str(output_dir),
        }

    monkeypatch.setattr(module, "_extract_video_frames", fake_extract_video_frames)

    def fake_yolo_export(**kwargs):
        recorded["yolo_export"] = kwargs
        masks_dir = kwargs["masks_dir"]
        masks_dir.mkdir(parents=True, exist_ok=True)
        (masks_dir / "0000000.png").write_bytes(b"png")
        (masks_dir / "0000000.json").write_text("[]", encoding="utf-8")
        (masks_dir / "0000001.png").write_bytes(b"png")
        (masks_dir / "0000001.json").write_text("[]", encoding="utf-8")
        return 1.5, 0

    monkeypatch.setattr(module, "_run_yolo_segmentation_export", fake_yolo_export)

    def fake_run(command, cwd, env, check):
        recorded["command"] = command
        recorded["cwd"] = cwd
        recorded["env"] = env
        recorded["check"] = check
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "_collect_rendered_artifacts", lambda _path: [str(rendered)])

    result = module._run_deva_yolo_tracking_for_video(
        video_path=str(video_path),
        output_root=output_root,
        deva_repo=str(deva_repo),
        python_bin="/usr/bin/python3",
        deva_model_path=str(deva_model),
        yolo_model_path=str(yolo_model),
        yolo_conf=0.25,
        size=480,
        chunk_size=4,
        temporal_setting="semionline",
        amp=True,
        max_frames=120,
        prepared_frame_dir=frames_root,
        prepared_mask_dir=masks_root,
        log_progress=False,
        log_every_frames=25,
    )

    assert recorded["cwd"] == str(deva_repo)
    assert recorded["check"] is True
    assert result.primary_rendered_artifact == str(rendered)
    assert result.frame_count == 2
    assert result.max_frames == 120
    assert Path(result.summary_json_path).exists()
