from __future__ import annotations

from pathlib import Path

import pytest

from tools import benchmark_sam3_raw as module


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


def test_run_raw_tracking_for_video_writes_summary(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"placeholder")
    recorded = {}

    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

    class FakeBoxes:
        def __init__(self, object_ids):
            self.id = FakeTensor(object_ids)

        def __len__(self):
            return len(self.id.tolist())

    class FakeResult:
        def __init__(self, object_ids):
            self.boxes = FakeBoxes(object_ids)

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

    result = module._run_raw_tracking_for_video(
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
    assert recorded["overrides"]["imgsz"] == 512
    assert recorded["call"] == {
        "source": str(video_path),
        "text": ["cow"],
        "stream": True,
    }
    assert result.frame_count == 2
    assert result.tracked_object_ids == [1, 2]
    assert result.mean_instances_per_frame == 1.5
    assert result.max_instances_per_frame == 2
    assert Path(result.summary_json_path).exists()


def test_run_raw_tracking_for_video_respects_max_frames(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"placeholder")

    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

    class FakeBoxes:
        def __init__(self, object_ids):
            self.id = FakeTensor(object_ids)

        def __len__(self):
            return len(self.id.tolist())

    class FakeResult:
        def __init__(self, object_ids):
            self.boxes = FakeBoxes(object_ids)

    class FakePredictor:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, **_kwargs):
            return iter([FakeResult([1]), FakeResult([2]), FakeResult([3])])

    monkeypatch.setattr(module, "SAM3VideoSemanticPredictor", FakePredictor)
    monkeypatch.setattr(
        module,
        "_probe_video_metadata",
        lambda _path: {"fps": 5.0, "width": 32, "height": 24, "frame_count": 3},
    )

    result = module._run_raw_tracking_for_video(
        video_path=str(video_path),
        output_root=tmp_path / "out",
        prompts=["cow"],
        model_path="sam3.pt",
        conf_threshold=0.25,
        imgsz=512,
        device=None,
        half=True,
        max_frames=2,
        log_progress=False,
        log_every_frames=25,
    )

    assert result.frame_count == 2
    assert result.tracked_object_ids == [1, 2]
