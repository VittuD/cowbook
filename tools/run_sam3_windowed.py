from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from cowbook.io.json_utils import dump_path_compact, dumps_pretty
from tools.benchmark_sam3_semantic_tracking import (
    Sam3VideoRunResult,
    _artifact_stem,
    _collect_runtime_info,
    _default_prompts,
    _default_videos,
    _log_progress,
    _resolve_prompts,
    _run_semantic_tracking_for_video,
    _validate_model_path,
    _validate_videos,
)
from tools.benchmark_tracking import _probe_video_metadata

DEFAULT_WINDOW_SECONDS = 600.0  # 10 minutes


@dataclass(slots=True)
class WindowBounds:
    window_index: int
    start_frame: int
    end_frame: int  # exclusive


def _compute_window_bounds(frame_count: int, fps: float, window_seconds: float) -> list[WindowBounds]:
    """Split [0, frame_count) into fixed-size windows of `window_seconds` each.

    The final window is shorter than the rest whenever frame_count isn't an
    exact multiple of the window size -- that's expected, not an error.

    Windowing exists to reset SAM3's video-tracker state periodically: its
    GPU memory footprint grows with the number of tracked objects over a
    video's duration (measured climbing toward the A40's VRAM ceiling within
    a single one-minute test clip), so feeding a multi-hour recording
    through in one unbroken pass risks running out of memory well before
    the end. A fresh window means a fresh predictor and a reset memory bank.
    """
    if frame_count <= 0 or fps <= 0 or window_seconds <= 0:
        return []
    window_frames = max(1, int(round(window_seconds * fps)))
    bounds: list[WindowBounds] = []
    start = 0
    index = 0
    while start < frame_count:
        end = min(frame_count, start + window_frames)
        bounds.append(WindowBounds(window_index=index, start_frame=start, end_frame=end))
        start = end
        index += 1
    return bounds


def _write_window_chunk(
    source_path: str,
    bounds: WindowBounds,
    output_path: Path,
    *,
    fps: float,
    frame_size: tuple[int, int],
) -> int:
    """Seek to `bounds` in `source_path` and write just that frame range to `output_path`.

    This chunk is transient by design: the caller deletes it immediately
    after SAM3 finishes with it, so a multi-hour video never holds more
    than one window's worth of re-encoded video on disk at a time -- there
    is no efficient stream-copy path available in this environment (no
    ffmpeg binary, no PyAV, no working hardware/software H.264 encoder in
    this OpenCV build), so re-encoding via mp4v is the only option, and
    keeping it transient is what keeps that acceptable.

    Returns the number of frames actually written, which can be less than
    requested if the source runs out of frames early.
    """
    capture = cv2.VideoCapture(source_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Failed to open output chunk for writing: {output_path}")

    written = 0
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(bounds.start_frame))
        for _ in range(bounds.end_frame - bounds.start_frame):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()
    return written


def _run_windowed_semantic_tracking_for_video(
    *,
    video_path: str,
    output_root: Path,
    chunk_tmp_dir: Path,
    window_seconds: float,
    prompts: list[str],
    model_path: str,
    conf_threshold: float,
    imgsz: int,
    device: str | None,
    half: bool,
    render_mode: str,
    max_render_frames: int,
    dump_frame_metadata: bool,
    log_progress: bool,
    log_every_frames: int,
) -> list[Sam3VideoRunResult]:
    metadata = _probe_video_metadata(video_path)
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    frame_count = int(metadata["frame_count"])

    window_bounds = _compute_window_bounds(frame_count, fps, window_seconds)
    stem = _artifact_stem(video_path)
    _log_progress(
        log_progress,
        f"[sam3-windowed] {video_path}: {len(window_bounds)} window(s) of ~{window_seconds:.0f}s each",
    )

    results: list[Sam3VideoRunResult] = []
    for bounds in window_bounds:
        chunk_path = chunk_tmp_dir / f"{stem}_window{bounds.window_index:03d}.mp4"
        written = _write_window_chunk(video_path, bounds, chunk_path, fps=fps, frame_size=(width, height))
        _log_progress(
            log_progress,
            f"[sam3-windowed] {video_path} window {bounds.window_index}: "
            f"frames [{bounds.start_frame}, {bounds.end_frame}) -> {written} written",
        )
        try:
            result = _run_semantic_tracking_for_video(
                video_path=str(chunk_path),
                output_root=output_root,
                prompts=prompts,
                model_path=model_path,
                conf_threshold=conf_threshold,
                imgsz=imgsz,
                device=device,
                half=half,
                render_mode=render_mode,
                max_frames=0,
                max_render_frames=max_render_frames,
                dump_frame_metadata=dump_frame_metadata,
                log_progress=log_progress,
                log_every_frames=log_every_frames,
            )
            results.append(result)
        finally:
            chunk_path.unlink(missing_ok=True)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3 semantic video tracking in fixed-duration windows "
            "instead of one pass over the whole video. Each window is a "
            "transient re-encoded chunk -- seeked directly from the "
            "source, deleted immediately after processing -- so a "
            "multi-hour video never grows disk usage and SAM3's "
            "per-video tracker memory resets on a fixed cadence."
        )
    )
    parser.add_argument("--videos", nargs="+", default=_default_videos(), help="Video paths to run.")
    parser.add_argument(
        "--model-path",
        default="sam3.pt",
        help="Local SAM3 model weights path. Ultralytics docs require downloading `sam3.pt` manually first.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=_default_prompts(),
        help="Text prompts for all-instance semantic tracking.",
    )
    parser.add_argument(
        "--output-root",
        default="var/windowed/sam3_semantic_tracking",
        help="Directory where per-window JSON summaries (and videos, if enabled) are written.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Window length in seconds (default: 600 = 10 minutes).",
    )
    parser.add_argument(
        "--chunk-tmp-dir",
        default=None,
        help="Directory for transient per-window chunk files. Defaults to <output-root>/_window_chunks_tmp.",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold forwarded to SAM3.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size forwarded to Ultralytics.")
    parser.add_argument("--device", default=None, help="Optional device override forwarded to Ultralytics.")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable half-precision inference.",
    )
    parser.add_argument(
        "--dump-frame-metadata",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include per-frame object summaries in each window's JSON output.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("all", "processed-only", "raw-only", "none"),
        default="none",
        help="Choose which annotated videos to write per window.",
    )
    parser.add_argument(
        "--max-render-frames",
        type=int,
        default=0,
        help="Maximum number of frames to render per window's annotated videos. Use 0 for no limit.",
    )
    parser.add_argument(
        "--summary-name",
        default="summary.json",
        help="Top-level summary filename under --output-root.",
    )
    parser.add_argument(
        "--log-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable coarse progress logs for console and swarm logs.",
    )
    parser.add_argument(
        "--log-every-frames",
        type=int,
        default=200,
        help="Log every N frames when --log-progress is enabled.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if float(args.window_seconds) <= 0:
        raise ValueError("--window-seconds must be > 0.")
    if int(args.imgsz) < 1:
        raise ValueError("--imgsz must be >= 1.")
    if int(args.max_render_frames) < 0:
        raise ValueError("--max-render-frames must be >= 0.")
    if int(args.log_every_frames) < 1:
        raise ValueError("--log-every-frames must be >= 1.")

    videos = _validate_videos(args.videos)
    prompts = _resolve_prompts(args.prompts)
    model_path = _validate_model_path(args.model_path)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    chunk_tmp_dir = Path(args.chunk_tmp_dir).resolve() if args.chunk_tmp_dir else output_root / "_window_chunks_tmp"
    chunk_tmp_dir.mkdir(parents=True, exist_ok=True)

    runtime_info = _collect_runtime_info(model_path, args.device)
    runtime_info["window_seconds"] = float(args.window_seconds)

    runs = {
        video_path: [
            asdict(result)
            for result in _run_windowed_semantic_tracking_for_video(
                video_path=video_path,
                output_root=output_root,
                chunk_tmp_dir=chunk_tmp_dir,
                window_seconds=float(args.window_seconds),
                prompts=prompts,
                model_path=model_path,
                conf_threshold=float(args.conf_threshold),
                imgsz=int(args.imgsz),
                device=args.device,
                half=bool(args.half),
                render_mode=str(args.render_mode),
                max_render_frames=int(args.max_render_frames),
                dump_frame_metadata=bool(args.dump_frame_metadata),
                log_progress=bool(args.log_progress),
                log_every_frames=int(args.log_every_frames),
            )
        ]
        for video_path in videos
    }

    # Each chunk is already removed as soon as its window finishes; this
    # only clears the (by then empty) transient directory itself.
    try:
        chunk_tmp_dir.rmdir()
    except OSError:
        pass

    summary_payload = {
        "tool": "run_sam3_windowed",
        "runtime": runtime_info,
        "runs": runs,
    }
    summary_path = output_root / str(args.summary_name)
    dump_path_compact(summary_path, summary_payload)
    print("SAM3 windowed semantic tracking summary")
    print(dumps_pretty(summary_payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
