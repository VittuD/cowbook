from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from tools import benchmark_sam3_semantic_tracking as module


def _write_real_video(path: Path, frame_count: int, fps: float, size: tuple[int, int] = (32, 24)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for index in range(frame_count):
        frame = np.full((size[1], size[0], 3), fill_value=index % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_resolve_frame_stride_returns_one_when_target_fps_is_none():
    assert module._resolve_frame_stride(30.0, None) == 1


def test_resolve_frame_stride_computes_nearest_integer_stride():
    assert module._resolve_frame_stride(6.0, 1.0) == 6
    assert module._resolve_frame_stride(30.0, 7.0) == 4
    # A target fps equal to the source is a no-op stride of 1, not an error.
    assert module._resolve_frame_stride(6.0, 6.0) == 1


def test_resolve_frame_stride_rejects_non_positive_target():
    with pytest.raises(ValueError, match="must be > 0"):
        module._resolve_frame_stride(30.0, 0.0)
    with pytest.raises(ValueError, match="must be > 0"):
        module._resolve_frame_stride(30.0, -1.0)


def test_resolve_frame_stride_rejects_upsampling():
    with pytest.raises(ValueError, match="exceeds"):
        module._resolve_frame_stride(6.0, 10.0)


def test_write_video_subset_with_stride_one_matches_requested_frame_range(tmp_path: Path):
    source_path = tmp_path / "source.mp4"
    _write_real_video(source_path, frame_count=20, fps=5.0)

    output_path = tmp_path / "chunk.mp4"
    written = module._write_video_subset(
        str(source_path),
        output_path,
        start_frame=5,
        end_frame=12,
        frame_stride=1,
        output_fps=5.0,
        frame_size=(32, 24),
    )

    assert written == 7
    assert module._probe_video_metadata(str(output_path))["frame_count"] == 7


def test_write_video_subset_decimates_by_stride(tmp_path: Path):
    source_path = tmp_path / "source.mp4"
    _write_real_video(source_path, frame_count=12, fps=6.0)

    output_path = tmp_path / "decimated.mp4"
    written = module._write_video_subset(
        str(source_path),
        output_path,
        start_frame=0,
        end_frame=12,
        frame_stride=3,
        output_fps=2.0,
        frame_size=(32, 24),
    )

    assert written == 4
    assert module._probe_video_metadata(str(output_path))["frame_count"] == 4


def test_write_video_subset_reports_fewer_frames_when_source_runs_out_early(tmp_path: Path):
    source_path = tmp_path / "source.mp4"
    _write_real_video(source_path, frame_count=10, fps=5.0)

    output_path = tmp_path / "chunk.mp4"
    written = module._write_video_subset(
        str(source_path),
        output_path,
        start_frame=6,
        end_frame=20,
        frame_stride=1,
        output_fps=5.0,
        frame_size=(32, 24),
    )

    assert written == 4


def test_write_video_subset_rejects_invalid_stride(tmp_path: Path):
    source_path = tmp_path / "source.mp4"
    _write_real_video(source_path, frame_count=5, fps=5.0)

    with pytest.raises(ValueError, match="frame_stride"):
        module._write_video_subset(
            str(source_path),
            tmp_path / "out.mp4",
            start_frame=0,
            end_frame=5,
            frame_stride=0,
            output_fps=5.0,
            frame_size=(32, 24),
        )


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


def test_select_cleanup_keep_indices_applies_mask_fill_ratio():
    full_masks = [
        np.pad(np.ones((8, 8), dtype=np.uint8), ((4, 28), (4, 28))),
        np.pad(np.ones((2, 2), dtype=np.uint8), ((20, 18), (20, 18))),
    ]
    frame = module.Sam3FrameArtifacts(
        frame_index=0,
        orig_img=np.zeros((40, 40, 3), dtype=np.uint8),
        path="video.mp4",
        names={0: "cow"},
        xyxy=np.asarray(
            [
                [4.0, 4.0, 14.0, 14.0],
                [20.0, 20.0, 30.0, 30.0],
            ],
            dtype=np.float32,
        ),
        conf=np.asarray([0.9, 0.9], dtype=np.float32),
        cls=np.asarray([0, 0], dtype=np.int32),
        object_ids=np.asarray([1, 2], dtype=np.int32),
        # Masks are stored packed (cropped to their tight bounding box), the
        # same representation `_extract_frame_artifacts` produces.
        masks=module._as_object_array([module._pack_mask(mask) for mask in full_masks]),
    )
    cleanup = module.TrackingCleanupConfig.from_mapping(
        {
            "enabled": True,
            "conf_threshold": 0.1,
            "nms_mode": "iou_nms",
            "nms_iou": 0.5,
            "drop_edge_boxes": False,
            "min_mask_fill_ratio": 0.25,
        }
    )

    keep_indices, removed_by_mask_fill = module._select_cleanup_keep_indices(frame, cleanup)

    assert keep_indices.tolist() == [0]
    assert removed_by_mask_fill == 1


def test_run_semantic_tracking_for_video_writes_summary_and_video(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"placeholder")
    recorded = {}

    class FakeBoxes:
        def __init__(self, object_ids):
            self.xyxy = FakeArray(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [4.0, 4.0, 16.0, 16.0],
                ][: len(object_ids)]
            )
            self.id = FakeTensor(object_ids)
            self.conf = FakeTensor([0.85 for _ in object_ids])
            self.cls = FakeTensor([0 for _ in object_ids])

        def __len__(self):
            return len(self.id.tolist())

    class FakeArray:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float32)

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self._values, dtype=np.float32)

    class FakeMasks:
        def __init__(self, count: int):
            self.data = FakeArray(np.ones((count, 24, 32), dtype=np.uint8))

    class FakeResult:
        def __init__(self, object_ids):
            self.boxes = FakeBoxes(object_ids)
            self.names = {0: "cow"}
            self.orig_img = np.zeros((24, 32, 3), dtype=np.uint8)
            self.path = str(video_path)
            self.masks = FakeMasks(len(object_ids))

        def plot(self, **_kwargs):
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
        imgsz=512,
        device="0",
        half=True,
        render_mode="all",
        max_frames=0,
        max_render_frames=600,
        dump_frame_metadata=True,
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
    assert Path(result.annotated_video_path).exists()
    assert Path(result.clean_annotated_video_path).exists()
    assert Path(result.processed_annotated_video_path).exists()
    assert Path(result.processed_clean_annotated_video_path).exists()
    assert result.processed_mean_instances_per_frame == 0.0
    assert result.processed_max_instances_per_frame == 0
    assert result.processed_tracked_object_ids == []
    assert Path(result.summary_json_path).exists()


def test_run_semantic_tracking_for_video_respects_max_frames(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"placeholder")

    class FakeBoxes:
        def __init__(self, object_ids):
            self.xyxy = FakeArray([[0.0, 0.0, 10.0, 10.0]] * len(object_ids))
            self.id = FakeTensor(object_ids)
            self.conf = FakeTensor([0.85 for _ in object_ids])
            self.cls = FakeTensor([0 for _ in object_ids])

        def __len__(self):
            return len(self.id.tolist())

    class FakeArray:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float32)

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self._values, dtype=np.float32)

    class FakeMasks:
        def __init__(self, count: int):
            self.data = FakeArray(np.ones((count, 24, 32), dtype=np.uint8))

    class FakeResult:
        def __init__(self, object_ids):
            self.boxes = FakeBoxes(object_ids)
            self.names = {0: "cow"}
            self.orig_img = np.zeros((24, 32, 3), dtype=np.uint8)
            self.path = str(video_path)
            self.masks = FakeMasks(len(object_ids))

        def plot(self, **_kwargs):
            return np.zeros((24, 32, 3), dtype=np.uint8)

    class FakePredictor:
        def __init__(self, overrides):
            self.overrides = overrides

        def __call__(self, *, source, text, stream):
            return iter([FakeResult([1]), FakeResult([2]), FakeResult([3])])

    monkeypatch.setattr(module, "SAM3VideoSemanticPredictor", FakePredictor)
    monkeypatch.setattr(
        module,
        "_probe_video_metadata",
        lambda _path: {"fps": 5.0, "width": 32, "height": 24, "frame_count": 3},
    )

    result = module._run_semantic_tracking_for_video(
        video_path=str(video_path),
        output_root=tmp_path / "out",
        prompts=["cow"],
        model_path="sam3.pt",
        conf_threshold=0.25,
        imgsz=512,
        device="0",
        half=True,
        render_mode="none",
        max_frames=1,
        max_render_frames=600,
        dump_frame_metadata=False,
        log_progress=False,
        log_every_frames=25,
    )

    assert result.frame_count == 1
    assert result.tracked_object_ids == [1]


def test_run_semantic_tracking_for_video_decimates_to_target_fps(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "input.mp4"
    _write_real_video(video_path, frame_count=6, fps=6.0)
    recorded = {}

    class FakeArray:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float32)

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self._values, dtype=np.float32)

    class FakeBoxes:
        def __init__(self):
            self.xyxy = FakeArray(np.zeros((0, 4), dtype=np.float32))
            self.id = FakeTensor([])
            self.conf = FakeTensor([])
            self.cls = FakeTensor([])

        def __len__(self):
            return 0

    class FakeMasks:
        def __init__(self):
            self.data = FakeArray(np.zeros((0, 24, 32), dtype=np.uint8))

    class FakeResult:
        def __init__(self, path):
            self.boxes = FakeBoxes()
            self.masks = FakeMasks()
            self.names = {0: "cow"}
            self.orig_img = np.zeros((24, 32, 3), dtype=np.uint8)
            self.path = path

        def plot(self, **_kwargs):
            return np.zeros((24, 32, 3), dtype=np.uint8)

    class FakePredictor:
        def __init__(self, overrides):
            recorded["overrides"] = overrides

        def __call__(self, *, source, text, stream):
            recorded["call_source"] = source
            metadata = module._probe_video_metadata(source)
            return iter([FakeResult(source) for _ in range(int(metadata["frame_count"]))])

    monkeypatch.setattr(module, "SAM3VideoSemanticPredictor", FakePredictor)

    result = module._run_semantic_tracking_for_video(
        video_path=str(video_path),
        output_root=tmp_path / "out",
        prompts=["cow"],
        model_path="sam3.pt",
        conf_threshold=0.25,
        imgsz=512,
        device=None,
        half=True,
        render_mode="none",
        max_frames=0,
        max_render_frames=0,
        dump_frame_metadata=False,
        log_progress=False,
        log_every_frames=25,
        target_fps=1.0,
    )

    # 6 frames at 6fps decimated to ~1fps -> stride 6 -> 1 frame processed.
    assert result.target_fps == 1.0
    assert result.frame_stride == 6
    assert result.frame_count == 1
    # SAM3 saw a transient decimated copy, never the original source path.
    assert recorded["call_source"] != str(video_path)
    # ... and that transient copy is gone once the run finishes.
    assert not Path(recorded["call_source"]).exists()


def test_run_semantic_tracking_for_video_skips_decimation_when_target_fps_is_none(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"placeholder")
    recorded = {}

    class FakeArray:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float32)

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self._values, dtype=np.float32)

    class FakeBoxes:
        def __init__(self):
            self.xyxy = FakeArray(np.zeros((0, 4), dtype=np.float32))
            self.id = FakeTensor([])
            self.conf = FakeTensor([])
            self.cls = FakeTensor([])

        def __len__(self):
            return 0

    class FakeMasks:
        def __init__(self):
            self.data = FakeArray(np.zeros((0, 24, 32), dtype=np.uint8))

    class FakeResult:
        def __init__(self, path):
            self.boxes = FakeBoxes()
            self.masks = FakeMasks()
            self.names = {0: "cow"}
            self.orig_img = np.zeros((24, 32, 3), dtype=np.uint8)
            self.path = path

        def plot(self, **_kwargs):
            return np.zeros((24, 32, 3), dtype=np.uint8)

    class FakePredictor:
        def __init__(self, overrides):
            recorded["overrides"] = overrides

        def __call__(self, *, source, text, stream):
            recorded["call_source"] = source
            return iter([FakeResult(source)])

    monkeypatch.setattr(module, "SAM3VideoSemanticPredictor", FakePredictor)
    monkeypatch.setattr(
        module,
        "_probe_video_metadata",
        lambda _path: {"fps": 5.0, "width": 32, "height": 24, "frame_count": 1},
    )

    result = module._run_semantic_tracking_for_video(
        video_path=str(video_path),
        output_root=tmp_path / "out",
        prompts=["cow"],
        model_path="sam3.pt",
        conf_threshold=0.25,
        imgsz=512,
        device=None,
        half=True,
        render_mode="none",
        max_frames=0,
        max_render_frames=0,
        dump_frame_metadata=False,
        log_progress=False,
        log_every_frames=25,
    )

    # No --target-fps: identical to today's behavior, no transient chunk at all.
    assert result.target_fps is None
    assert result.frame_stride == 1
    assert recorded["call_source"] == str(video_path)
