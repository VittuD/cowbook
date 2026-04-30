from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from cowbook.io.json_utils import dump_path_compact, dumps_pretty
from cowbook.vision.calibration import (
    build_camera_model,
    build_undistort_maps,
    default_calibration_file,
    resolve_camera_spec,
    scale_camera_spec,
    undistort_image_with_model,
)


def _log_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


@dataclass(slots=True)
class UndistortRunResult:
    input_video_path: str
    output_video_path: str
    frame_count: int
    fps: float
    width: int
    height: int
    elapsed_s: float
    effective_fps: float
    camera_nr: int
    calibration_file: str
    calibration_image_size: list[int]


def _collect_videos(input_dir: str, pattern: str) -> list[str]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")
    paths = [str(path) for path in sorted(root.glob(pattern)) if path.is_file()]
    if not paths:
        raise FileNotFoundError(f"No videos matched pattern {pattern!r} under {input_dir}")
    return paths


def _output_path_for(video_path: str, output_dir: Path, suffix: str) -> Path:
    path = Path(video_path)
    return output_dir / f"{path.stem}{suffix}{path.suffix}"


def _open_capture(video_path: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return capture


def _video_fps(capture: cv2.VideoCapture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps if fps > 0 else 6.0


def _run_for_video(
    *,
    video_path: str,
    output_dir: Path,
    camera_nr: int,
    calibration_file: str,
    suffix: str,
    overwrite: bool,
    log_progress: bool,
) -> UndistortRunResult:
    output_path = _output_path_for(video_path, output_dir, suffix)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    capture = _open_capture(video_path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = _video_fps(capture)

        source_spec = resolve_camera_spec(camera_nr, calibration_file=calibration_file)
        scaled_spec = scale_camera_spec(source_spec, (width, height), scale_reference_points=False)
        camera_model = build_camera_model(scaled_spec)
        maps = build_undistort_maps(camera_model)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {output_path}")

        frame_count = 0
        start = time.perf_counter()
        _log_progress(log_progress, f"[undistort] start: {video_path} -> {output_path}")
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                undistorted = undistort_image_with_model(frame, camera_model, maps=maps)
                writer.write(undistorted)
                frame_count += 1
                if log_progress and (frame_count == 1 or frame_count % 300 == 0):
                    print(f"[undistort] progress: {video_path} frame={frame_count}", flush=True)
        finally:
            writer.release()
        elapsed_s = time.perf_counter() - start
    finally:
        capture.release()

    return UndistortRunResult(
        input_video_path=video_path,
        output_video_path=str(output_path),
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        elapsed_s=elapsed_s,
        effective_fps=(frame_count / elapsed_s) if elapsed_s > 0 else 0.0,
        camera_nr=camera_nr,
        calibration_file=calibration_file,
        calibration_image_size=[int(source_spec.image_size[0]), int(source_spec.image_size[1])],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Undistort every video in a folder using Cowbook camera calibration. "
            "Assumes the input videos are resized versions of the calibrated camera stream."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing input videos.")
    parser.add_argument("--camera-nr", required=True, type=int, help="Camera number to use.")
    parser.add_argument(
        "--output-dir",
        help="Directory for undistorted output videos. Defaults to <input-dir>/undistorted.",
    )
    parser.add_argument(
        "--pattern",
        default="*.mp4",
        help="Glob pattern used to select videos within --input-dir.",
    )
    parser.add_argument(
        "--suffix",
        default="_undistorted",
        help="Suffix appended to each output video stem.",
    )
    parser.add_argument(
        "--calibration-file",
        default=default_calibration_file(),
        help="Calibration bundle used for undistortion.",
    )
    parser.add_argument(
        "--summary-name",
        default="summary.json",
        help="Summary JSON filename under --output-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--log-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable progress logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    input_dir = str(Path(args.input_dir))
    output_dir = Path(args.output_dir) if args.output_dir else Path(input_dir) / "undistorted"
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = _collect_videos(input_dir, args.pattern)
    results = [
        _run_for_video(
            video_path=video_path,
            output_dir=output_dir,
            camera_nr=int(args.camera_nr),
            calibration_file=str(args.calibration_file),
            suffix=str(args.suffix),
            overwrite=bool(args.overwrite),
            log_progress=bool(args.log_progress),
        )
        for video_path in videos
    ]

    summary = {
        "input_dir": input_dir,
        "output_dir": str(output_dir),
        "pattern": str(args.pattern),
        "camera_nr": int(args.camera_nr),
        "calibration_file": str(args.calibration_file),
        "suffix": str(args.suffix),
        "runs": [asdict(result) for result in results],
    }
    summary_path = output_dir / args.summary_name
    dump_path_compact(summary_path, summary)
    print("Undistortion summary")
    print(dumps_pretty(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
