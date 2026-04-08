from __future__ import annotations

import os

import numpy as np

from cowbook.vision import preprocess_video as preprocess_module


def test_choose_channel_prefers_explicit_map():
    assert preprocess_module._choose_channel(
        "/tmp/no_hint.mp4",
        4,
        {"4": "Ch6"},
    ) == "Ch6"


def test_should_skip_uses_mask_signature_and_mtimes(tmp_path):
    src = tmp_path / "source.mp4"
    dst = tmp_path / "masked.mp4"
    mask = tmp_path / "mask.png"
    src.write_text("src", encoding="utf-8")
    dst.write_text("dst", encoding="utf-8")
    mask.write_text("mask", encoding="utf-8")

    signature = preprocess_module._build_mask_signature(str(src), str(mask), True)
    preprocess_module._write_mask_signature(str(dst), signature)

    newer = max(src.stat().st_mtime, mask.stat().st_mtime) + 5
    os.utime(dst, (newer, newer))

    assert preprocess_module._should_skip(str(src), str(dst), mask_path=str(mask), strict_half_rule=True) is True
    assert preprocess_module._should_skip(str(src), str(dst), mask_path=str(mask), strict_half_rule=False) is False


def test_process_one_video_applies_mask_and_writes_signature(monkeypatch, tmp_path):
    src_path = tmp_path / "input.mp4"
    dst_path = tmp_path / "output.mp4"

    frames = [
        np.full((2, 2, 3), 10, dtype=np.uint8),
        np.full((2, 2, 3), 20, dtype=np.uint8),
    ]

    class FakeCapture:
        def __init__(self, _path):
            self._frames = list(frames)

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == preprocess_module.cv2.CAP_PROP_FPS:
                return 6.0
            if prop == preprocess_module.cv2.CAP_PROP_FRAME_WIDTH:
                return 2
            if prop == preprocess_module.cv2.CAP_PROP_FRAME_HEIGHT:
                return 2
            return 0

        def read(self):
            if not self._frames:
                return False, None
            return True, self._frames.pop(0)

        def release(self):
            return None

    class FakeWriter:
        def __init__(self, *_args, **_kwargs):
            self.frames = []

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame.copy())

        def release(self):
            return None

    writer = FakeWriter()
    monkeypatch.setattr(preprocess_module.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(preprocess_module.cv2, "VideoWriter", lambda *args, **kwargs: writer)
    monkeypatch.setattr(preprocess_module, "_load_mask", lambda _path: (np.full((2, 2), 255, dtype=np.uint8), (2, 2)))

    src, dst, ok = preprocess_module._process_one_video(
        str(src_path),
        str(dst_path),
        "mask.png",
        strict_half_rule=True,
    )

    assert (src, dst, ok) == (str(src_path), str(dst_path), True)
    assert len(writer.frames) == 2
    assert np.array_equal(writer.frames[0], frames[0])
    metadata = preprocess_module._read_mask_signature(str(dst_path))
    assert metadata is not None
    assert metadata["mask_path"].endswith("mask.png")
