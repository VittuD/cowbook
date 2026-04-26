from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools import benchmark_sam3_export as module


def test_build_tracking_document_preserves_ids_and_centroids():
    frame = module.Sam3ExportFrameArtifacts(
        frame_index=3,
        path="frame003.jpg",
        xyxy=np.asarray([[0.0, 2.0, 4.0, 6.0]], dtype=np.float32),
        conf=np.asarray([0.9], dtype=np.float32),
        cls=np.asarray([0], dtype=np.int32),
        object_ids=np.asarray([7], dtype=np.int32),
        masks=np.zeros((1, 8, 8), dtype=np.uint8),
    )

    document = module._build_tracking_document([frame]).to_dict()

    assert document == {
        "frames": [
            {
                "frame_id": 3,
                "detections": {
                    "xyxy": [[0.0, 2.0, 4.0, 6.0]],
                    "centroids": [[2.0, 4.0]],
                },
                "labels": [
                    {
                        "class_id": 0,
                        "id": 7,
                        "det_idx": 0,
                        "real": 1,
                        "src": "sam3_export",
                    }
                ],
            }
        ]
    }


def test_save_frame_masks_writes_npz(tmp_path: Path):
    frame = module.Sam3ExportFrameArtifacts(
        frame_index=1,
        path="frame001.jpg",
        xyxy=np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        conf=np.asarray([0.8], dtype=np.float32),
        cls=np.asarray([0], dtype=np.int32),
        object_ids=np.asarray([9], dtype=np.int32),
        masks=np.ones((1, 4, 5), dtype=np.uint8),
    )

    output_path = module._save_frame_masks(frame, tmp_path)
    saved = np.load(output_path)

    assert saved["frame_index"] == 1
    assert saved["xyxy"].shape == (1, 4)
    assert saved["masks"].shape == (1, 4, 5)
    assert saved["object_ids"].tolist() == [9]


def test_run_export_for_video_writes_summary_and_tracking(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"video")
    recorded = {}

    class FakeTensor:
        def __init__(self, values):
            self._values = np.asarray(values)

        def cpu(self):
            return self

        def numpy(self):
            return self._values

        def tolist(self):
            return self._values.tolist()

    class FakeBoxes:
        def __init__(self, object_ids):
            self.id = FakeTensor(object_ids)
            self.xyxy = FakeTensor([[0.0, 0.0, 2.0, 2.0] for _ in object_ids])
            self.conf = FakeTensor([0.9 for _ in object_ids])
            self.cls = FakeTensor([0 for _ in object_ids])

        def __len__(self):
            return len(self.id.tolist())

    class FakeMasks:
        def __init__(self, count):
            self.data = FakeTensor(np.ones((count, 4, 4), dtype=np.uint8))

    class FakeResult:
        def __init__(self, frame_index, object_ids):
            self.boxes = FakeBoxes(object_ids)
            self.masks = FakeMasks(len(object_ids))
            self.orig_img = np.zeros((4, 4, 3), dtype=np.uint8)
            self.path = f"frame{frame_index:03d}.jpg"

    class FakePredictor:
        def __init__(self, overrides):
            recorded["overrides"] = overrides

        def __call__(self, *, source, text, stream):
            recorded["call"] = {"source": source, "text": text, "stream": stream}
            return iter([FakeResult(0, [1, 2]), FakeResult(1, [2])])

    monkeypatch.setattr(module, "SAM3VideoSemanticPredictor", FakePredictor)
    monkeypatch.setattr(
        module,
        "_probe_video_metadata",
        lambda _path: {"fps": 5.0, "width": 32, "height": 24, "frame_count": 2},
    )

    result = module._run_export_for_video(
        video_path=str(video_path),
        output_root=tmp_path / "out",
        prompts=["cow"],
        model_path="sam3.pt",
        conf_threshold=0.25,
        imgsz=512,
        device="0",
        half=True,
        max_frames=0,
        log_progress=False,
        log_every_frames=25,
    )

    assert recorded["overrides"]["conf"] == 0.25
    assert recorded["call"] == {"source": str(video_path), "text": ["cow"], "stream": True}
    assert result.frame_count == 2
    assert result.tracked_object_ids == [1, 2]
    assert Path(result.summary_json_path).exists()
    assert Path(result.tracking_json_path).exists()
    tracking_doc = json.loads(Path(result.tracking_json_path).read_text(encoding="utf-8"))
    assert len(tracking_doc["frames"]) == 2
    assert sorted((Path(result.masks_dir)).glob("*.npz"))
