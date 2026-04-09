from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools import benchmark_sam3_semantic_tracking as module


def test_resolve_prompts_uses_default_and_trims_values():
    assert module._resolve_prompts(None) == ["cow"]
    assert module._resolve_prompts([" cow ", "", "dairy cow"]) == ["cow", "dairy cow"]


def test_resolve_prompts_rejects_empty_values():
    with pytest.raises(ValueError, match="At least one non-empty prompt"):
        module._resolve_prompts(["", "   "])


def test_validate_model_path_requires_downloaded_weights(tmp_path: Path):
    model_path = tmp_path / "sam3.pt"
    model_path.write_bytes(b"weights")
    assert module._validate_model_path(str(model_path)) == str(model_path)

    with pytest.raises(FileNotFoundError, match="not auto-downloaded"):
        module._validate_model_path(str(tmp_path / "missing.pt"))


def test_frame_summary_reads_object_ids_and_class_names():
    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

    class FakeBoxes:
        def __init__(self):
            self.id = FakeTensor([7, 8])
            self.conf = FakeTensor([0.8, 0.9])
            self.cls = FakeTensor([0, 0])

        def __len__(self):
            return 2

    class FakeResult:
        boxes = FakeBoxes()
        names = {0: "cow"}

    summary = module._frame_summary(3, FakeResult())
    assert summary["frame_index"] == 3
    assert summary["instance_count"] == 2
    assert summary["object_ids"] == [7, 8]
    assert summary["confidences"] == [0.8, 0.9]
    assert summary["class_names"] == ["cow", "cow"]


def test_frame_summary_accepts_list_names():
    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

    class FakeBoxes:
        id = FakeTensor([3])
        conf = FakeTensor([0.7])
        cls = FakeTensor([1])

        def __len__(self):
            return 1

    class FakeResult:
        boxes = FakeBoxes()
        names = ["background", "cow"]

    summary = module._frame_summary(0, FakeResult())
    assert summary["class_names"] == ["cow"]


def test_run_semantic_tracking_for_video_writes_summary_and_video(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"placeholder")
    recorded = {}

    class FakeBoxes:
        def __init__(self, object_ids):
            self.id = FakeTensor(object_ids)
            self.conf = FakeTensor([0.85 for _ in object_ids])
            self.cls = FakeTensor([0 for _ in object_ids])

        def __len__(self):
            return len(self.id.tolist())

    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

    class FakeResult:
        def __init__(self, object_ids):
            self.boxes = FakeBoxes(object_ids)
            self.names = {0: "cow"}

        def plot(self):
            return np.zeros((24, 32, 3), dtype=np.uint8)

    class FakePredictor:
        def __init__(self, overrides):
            recorded["overrides"] = overrides

        def __call__(self, *, source, text, stream):
            recorded["call"] = {
                "source": source,
                "text": text,
                "stream": stream,
            }
            return iter([FakeResult([1, 2]), FakeResult([2])])

    monkeypatch.setattr(module, "SAM3VideoSemanticPredictor", FakePredictor)
    monkeypatch.setattr(
        module,
        "_probe_video_metadata",
        lambda _path: {"fps": 5.0, "width": 32, "height": 24, "frame_count": 2},
    )

    result = module._run_semantic_tracking_for_video(
        video_path=str(video_path),
        output_root=tmp_path / "out",
        prompts=["cow"],
        model_path="sam3.pt",
        conf_threshold=0.25,
        device="0",
        half=True,
        dump_frame_metadata=True,
        log_progress=False,
    )

    assert recorded["overrides"]["conf"] == 0.25
    assert recorded["call"] == {
        "source": str(video_path),
        "text": ["cow"],
        "stream": True,
    }
    assert result.frame_count == 2
    assert result.tracked_object_ids == [1, 2]
    assert result.mean_instances_per_frame == 1.5
    assert result.max_instances_per_frame == 2
    assert Path(result.annotated_video_path).exists()
    assert Path(result.summary_json_path).exists()
