from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from cowbook.vision import preprocess_video as preprocess_module


def test_mask_bounding_box_returns_tight_box_around_nonzero_region():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:8] = 255

    assert preprocess_module._mask_bounding_box(mask) == (3, 2, 8, 5)


def test_mask_bounding_box_rejects_empty_mask():
    mask = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="no nonzero pixels"):
        preprocess_module._mask_bounding_box(mask)


def test_round_bbox_to_even_dimensions_extends_outward_when_room_available():
    # width 8->8 (already even), height 5->6 by extending y1 (room below)
    assert preprocess_module._round_bbox_to_even_dimensions((0, 0, 8, 5), frame_width=10, frame_height=10) == (
        0,
        0,
        8,
        6,
    )


def test_round_bbox_to_even_dimensions_extends_the_other_edge_when_no_room():
    # y1 already at the frame edge -> extend y0 outward instead of shrinking.
    assert preprocess_module._round_bbox_to_even_dimensions((0, 1, 8, 10), frame_width=10, frame_height=10) == (
        0,
        0,
        8,
        10,
    )


def test_round_bbox_to_even_dimensions_shrinks_only_as_a_last_resort():
    # Box already spans the full (odd) frame on both edges -> must shrink.
    assert preprocess_module._round_bbox_to_even_dimensions((0, 0, 7, 7), frame_width=7, frame_height=7) == (
        0,
        0,
        6,
        6,
    )


def test_crop_and_mask_video_produces_even_dimensions_from_an_odd_bbox(tmp_path):
    width, height = 64, 64
    src_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(src_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    writer.write(np.full((height, width, 3), 220, dtype=np.uint8))
    writer.release()

    # An odd-height, odd-width bounding box (11 wide, 21 tall).
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[10:31, 15:26] = 255
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask)

    dst_path = tmp_path / "cropped.mp4"
    x0, y0, x1, y1 = preprocess_module.crop_and_mask_video(str(src_path), str(dst_path), str(mask_path))

    assert (x1 - x0) % 2 == 0
    assert (y1 - y0) % 2 == 0
    capture = cv2.VideoCapture(str(dst_path))
    written_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    written_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    # The readback must match the bbox actually used -- an odd size here is
    # exactly what silently corrupts an mp4v-encoded frame.
    assert written_width == x1 - x0
    assert written_height == y1 - y0


def test_mask_video_keeps_full_resolution_and_blacks_out_non_mask_pixels(tmp_path):
    width, height = 64, 64
    src_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(src_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    writer.write(np.full((height, width, 3), 220, dtype=np.uint8))
    writer.release()

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[10:40, 15:55] = 255
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask)

    dst_path = tmp_path / "masked.mp4"
    preprocess_module.mask_video(str(src_path), str(dst_path), str(mask_path))

    capture = cv2.VideoCapture(str(dst_path))
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == width
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == height
    ok, frame = capture.read()
    capture.release()

    assert ok
    assert int(frame[0, 0].max()) < 100  # outside the mask -> blacked out
    assert int(frame[20, 20].min()) > 100  # inside the mask -> kept


def test_mask_video_leaves_frames_unmodified_on_size_mismatch(tmp_path, caplog):
    src_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(src_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (10, 8))
    writer.write(np.full((8, 10, 3), 220, dtype=np.uint8))
    writer.release()

    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), np.full((4, 4), 255, dtype=np.uint8))

    dst_path = tmp_path / "masked.mp4"
    preprocess_module.mask_video(str(src_path), str(dst_path), str(mask_path))

    capture = cv2.VideoCapture(str(dst_path))
    ok, frame = capture.read()
    capture.release()

    assert ok
    assert int(frame.min()) > 100  # left unmodified, not blacked out


def test_crop_and_mask_video_crops_to_bbox_and_blacks_out_holes_within_it(tmp_path):
    # A reasonably sized, high-contrast frame: mp4v is lossy enough on tiny
    # or low-brightness frames that a small hole isn't reliably
    # distinguishable from a "kept" pixel by absolute value alone.
    width, height = 64, 64
    src_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(src_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    for _ in range(2):
        writer.write(np.full((height, width, 3), 220, dtype=np.uint8))
    writer.release()

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[10:40, 15:55] = 255
    mask[10:14, 15:19] = 0  # a hole inside the bbox -- the mask isn't a solid rectangle
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask)

    dst_path = tmp_path / "cropped.mp4"
    bbox = preprocess_module.crop_and_mask_video(str(src_path), str(dst_path), str(mask_path))

    assert bbox == (15, 10, 55, 40)
    capture = cv2.VideoCapture(str(dst_path))
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 40
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 30
    ok, frame = capture.read()
    capture.release()

    assert ok
    # mp4v is lossy, so compare against a midpoint rather than exact values.
    assert int(frame[1, 1].max()) < 100  # the hole -> blacked out
    assert int(frame[20, 20].min()) > 100  # inside the mask -> kept


def test_crop_and_mask_video_can_skip_within_crop_masking(tmp_path):
    width, height = 64, 64
    src_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(src_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    writer.write(np.full((height, width, 3), 220, dtype=np.uint8))
    writer.release()

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[10:40, 15:55] = 255
    mask[10:14, 15:19] = 0  # a hole inside the bbox
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask)

    dst_path = tmp_path / "cropped.mp4"
    preprocess_module.crop_and_mask_video(
        str(src_path), str(dst_path), str(mask_path), apply_mask_within_crop=False
    )

    capture = cv2.VideoCapture(str(dst_path))
    ok, frame = capture.read()
    capture.release()

    assert ok
    # apply_mask_within_crop=False: the hole is left untouched, unlike above.
    assert int(frame[1, 1].min()) > 100


def test_crop_and_mask_video_rejects_mismatched_mask_size(tmp_path):
    src_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(src_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (10, 8))
    writer.write(np.full((8, 10, 3), 10, dtype=np.uint8))
    writer.release()

    mask = np.full((4, 4), 255, dtype=np.uint8)
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask)

    with pytest.raises(ValueError, match="does not match video size"):
        preprocess_module.crop_and_mask_video(str(src_path), str(tmp_path / "out.mp4"), str(mask_path))


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
