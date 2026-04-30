from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from cowbook.io.json_utils import dump_path_compact, dumps_pretty, load_path
from cowbook.io.video_processor import create_video_from_images
from cowbook.vision.calibration import default_calibration_file
from cowbook.vision.frame_processor import plot_combined_projected_centroids, process_centroids
from cowbook.vision.processing import reconstruct_json


def _log_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _infer_camera_nr(path: str) -> int:
    match = re.search(r"(?i)ch(\d+)", Path(path).name)
    if match is None:
        raise ValueError(
            f"Could not infer camera number from tracking JSON path: {path}. "
            "Provide --camera-nr explicitly."
        )
    return int(match.group(1))


def _collect_tracking_jsons(
    *,
    tracking_jsons: list[str] | None,
    tracking_json_dir: str | None,
) -> list[str]:
    if tracking_jsons:
        paths = [str(Path(path)) for path in tracking_jsons]
    elif tracking_json_dir:
        root = Path(tracking_json_dir)
        paths = [str(path) for path in sorted(root.glob("*_sam3_tracking.json"))]
    else:
        raise ValueError("Provide --tracking-jsons or --tracking-json-dir.")

    if not paths:
        raise FileNotFoundError("No SAM3 tracking JSON files were found.")

    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing SAM3 tracking JSON(s): {missing}")
    return paths


def _count_document_frames(path: str) -> int:
    payload = load_path(path)
    return len(payload.get("frames", []))


def _offset_frames_data(frames_data: list[dict[str, Any]], frame_offset: int) -> list[dict[str, Any]]:
    if frame_offset == 0:
        return frames_data
    return [
        {
            **frame,
            "frame_id": int(frame["frame_id"]) + frame_offset,
        }
        for frame in frames_data
    ]


def _offset_tracking_frames(frames: list[dict[str, Any]], frame_offset: int) -> list[dict[str, Any]]:
    if frame_offset == 0:
        return frames
    return [
        {
            **frame,
            "frame_id": int(frame["frame_id"]) + frame_offset,
        }
        for frame in frames
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project SAM3 export tracking JSONs onto the ground plane using Cowbook's "
            "camera calibration, and optionally render a combined projection video."
        )
    )
    parser.add_argument(
        "--tracking-jsons",
        nargs="+",
        help="Explicit SAM3 tracking JSON paths to process in sequence.",
    )
    parser.add_argument(
        "--tracking-json-dir",
        help="Directory containing *_sam3_tracking.json files.",
    )
    parser.add_argument(
        "--camera-nr",
        type=int,
        help="Camera number to use for all input JSONs. If omitted, infer from filenames.",
    )
    parser.add_argument(
        "--calibration-file",
        default=default_calibration_file(),
        help="Calibration bundle used for centroid projection.",
    )
    parser.add_argument(
        "--output-root",
        default="var/benchmarks/sam3_projection",
        help="Root directory for processed JSON, frames, video, and summary outputs.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="FPS for the combined projection video.",
    )
    parser.add_argument(
        "--output-video-filename",
        default="sam3_projection.mp4",
        help="Combined projection video filename under <output-root>/videos.",
    )
    parser.add_argument(
        "--output-image-format",
        default="jpg",
        choices=("jpg", "jpeg", "png"),
        help="Rendered projection frame format.",
    )
    parser.add_argument(
        "--num-plot-workers",
        type=int,
        default=0,
        help="Parallel workers for rendering projection frames.",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Only write projected JSON outputs; do not render frames or assemble a video.",
    )
    parser.add_argument(
        "--summary-name",
        default="summary.json",
        help="Projection summary filename under <output-root>.",
    )
    parser.add_argument(
        "--log-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable verbose progress logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    tracking_json_paths = _collect_tracking_jsons(
        tracking_jsons=args.tracking_jsons,
        tracking_json_dir=args.tracking_json_dir,
    )
    camera_nr = int(args.camera_nr) if args.camera_nr is not None else _infer_camera_nr(tracking_json_paths[0])

    output_root = Path(args.output_root)
    json_output_dir = output_root / "json"
    frame_output_dir = output_root / "frames"
    video_output_dir = output_root / "videos"
    json_output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_video:
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        video_output_dir.mkdir(parents=True, exist_ok=True)

    merged_frames_data: list[dict[str, Any]] = []
    processed_json_paths: list[str] = []
    chunk_summaries: list[dict[str, Any]] = []
    merged_tracking_frames: list[dict[str, Any]] = []
    frame_offset = 0

    _log_progress(bool(args.log_progress), f"[sam3-project] processing {len(tracking_json_paths)} tracking JSON(s)")
    for tracking_json_path in tracking_json_paths:
        _log_progress(bool(args.log_progress), f"[sam3-project] project: {tracking_json_path}")
        raw_document = load_path(tracking_json_path)
        merged_tracking_frames.extend(_offset_tracking_frames(raw_document.get("frames", []), frame_offset))
        frames_data = process_centroids(
            tracking_json_path,
            camera_nr,
            args.calibration_file,
            cancellation_token=None,
            show_progress=bool(args.log_progress),
        )
        processed_json_path = json_output_dir / f"{Path(tracking_json_path).stem}_processed.json"
        dump_path_compact(processed_json_path, reconstruct_json(frames_data))
        processed_json_paths.append(str(processed_json_path))

        frame_count = len(frames_data)
        merged_frames_data.extend(_offset_frames_data(frames_data, frame_offset))
        chunk_summaries.append(
            {
                "tracking_json_path": tracking_json_path,
                "processed_json_path": str(processed_json_path),
                "frame_offset": frame_offset,
                "frame_count": frame_count,
            }
        )
        frame_offset += frame_count if frame_count > 0 else _count_document_frames(tracking_json_path)

    merged_tracking_json_path = json_output_dir / "merged_sam3_tracking_global.json"
    merged_processed_json_path = json_output_dir / "merged_sam3_tracking_global_processed.json"
    dump_path_compact(merged_tracking_json_path, {"frames": merged_tracking_frames})
    merged_document = reconstruct_json(merged_frames_data)
    dump_path_compact(merged_processed_json_path, merged_document)

    projection_video_path: str | None = None
    if not args.skip_video:
        base_filename = str(frame_output_dir / "combined_projected_centroids")
        _log_progress(bool(args.log_progress), "[sam3-project] rendering projection frames")
        plot_combined_projected_centroids(
            [str(merged_processed_json_path)],
            base_filename,
            num_workers=int(args.num_plot_workers),
            image_format=str(args.output_image_format),
            cancellation_token=None,
            reporter=None,
            group_idx=None,
            log_progress=bool(args.log_progress),
        )
        projection_video_path = str(video_output_dir / args.output_video_filename)
        _log_progress(bool(args.log_progress), f"[sam3-project] assembling video -> {projection_video_path}")
        create_video_from_images(
            str(frame_output_dir),
            projection_video_path,
            fps=int(args.fps),
            reporter=None,
            group_idx=None,
            log_progress=bool(args.log_progress),
        )

    summary = {
        "tracking_json_paths": tracking_json_paths,
        "camera_nr": camera_nr,
        "calibration_file": str(args.calibration_file),
        "processed_json_paths": processed_json_paths,
        "merged_tracking_json_path": str(merged_tracking_json_path),
        "merged_processed_json_path": str(merged_processed_json_path),
        "projection_video_path": projection_video_path,
        "fps": int(args.fps),
        "output_image_format": str(args.output_image_format),
        "num_plot_workers": int(args.num_plot_workers),
        "skip_video": bool(args.skip_video),
        "chunks": chunk_summaries,
    }
    summary_path = output_root / args.summary_name
    dump_path_compact(summary_path, summary)
    print("SAM3 projection summary")
    print(dumps_pretty(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
