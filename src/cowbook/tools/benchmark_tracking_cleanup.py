from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cowbook.tools.benchmark_tracking import _prepare_benchmark_videos, _query_gpu_info
from cowbook.vision.tracking import track_video_with_yolo


@dataclass(slots=True)
class CleanupRunResult:
    video_path: str
    output_json_path: str
    elapsed_s: float
    frame_count: int
    tracking_cleanup: dict[str, Any]


def _default_videos() -> list[str]:
    return [
        "sample_data/videos/Ch1_60.mp4",
        "sample_data/videos/Ch4_60.mp4",
        "sample_data/videos/Ch6_60.mp4",
        "sample_data/videos/Ch8_60.mp4",
    ]


def _default_cleanup_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "conf_threshold": 0.15,
        "nms_mode": "hybrid_nms",
        "two_pass_prune_short_tracks": True,
        "min_track_length": 30,
        "postprocess_smoothing": True,
    }


def _validate_videos(video_paths: list[str]) -> list[str]:
    normalized = [str(Path(path)) for path in video_paths]
    missing = [path for path in normalized if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing cleanup benchmark video(s): {missing}")
    return normalized


def _count_frames_from_tracking_json(output_json_path: str) -> int:
    document = json.loads(Path(output_json_path).read_text(encoding="utf-8"))
    return len(document.get("frames", []))


def _track_cleanup_worker(
    video_path: str,
    output_json_path: str,
    model_path: str,
    save_tracking_video: bool,
    tracking_cleanup: dict[str, Any],
) -> CleanupRunResult:
    try:
        import torch  # type: ignore

        torch.set_num_threads(1)
    except Exception:
        pass

    start = time.perf_counter()
    track_video_with_yolo(
        video_path,
        output_json_path,
        model_path,
        save=save_tracking_video,
        tracking_cleanup=tracking_cleanup,
    )
    elapsed_s = time.perf_counter() - start
    return CleanupRunResult(
        video_path=video_path,
        output_json_path=output_json_path,
        elapsed_s=elapsed_s,
        frame_count=_count_frames_from_tracking_json(output_json_path),
        tracking_cleanup=tracking_cleanup,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Cowbook's tracking cleanup path on prepared benchmark videos.",
    )
    parser.add_argument("--videos", nargs="+", default=_default_videos(), help="Video paths to run.")
    parser.add_argument(
        "--model-path",
        default="models/yolov11_best.pt",
        help="Path to Ultralytics model weights.",
    )
    parser.add_argument(
        "--extend-seconds",
        type=int,
        default=300,
        help="If > 0, create longer benchmark copies of the input videos before running.",
    )
    parser.add_argument(
        "--prepared-video-dir",
        default="var/benchmarks/prepared_videos_cleanup_300s",
        help="Directory for prepared benchmark videos when --extend-seconds is used.",
    )
    parser.add_argument(
        "--output-root",
        default="var/benchmarks/tracking_cleanup_300s",
        help="Directory where cleanup benchmark outputs and the summary JSON are written.",
    )
    parser.add_argument(
        "--summary-name",
        default="summary.json",
        help="Summary JSON filename under --output-root.",
    )
    parser.add_argument(
        "--tracking-concurrency",
        type=int,
        default=1,
        help="How many videos to process concurrently in this benchmark run.",
    )
    parser.add_argument(
        "--save-tracking-video",
        action="store_true",
        help="Also save annotated tracking videos next to the JSON outputs.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.15,
        help="Cleanup confidence threshold.",
    )
    parser.add_argument(
        "--nms-mode",
        choices=("iou_nms", "footpoint_nms", "hybrid_nms"),
        default="hybrid_nms",
        help="Cleanup NMS mode.",
    )
    parser.add_argument(
        "--two-pass-prune-short-tracks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable the second pass that prunes short-lived tracks.",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=30,
        help="Minimum track length for the pruning pass.",
    )
    parser.add_argument(
        "--postprocess-smoothing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable output smoothing after tracking.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    videos = _validate_videos(args.videos)
    prepared_videos, prepared_metadata = _prepare_benchmark_videos(
        videos,
        target_duration_seconds=args.extend_seconds,
        prepared_video_dir=args.prepared_video_dir,
    )
    benchmark_videos = prepared_videos or videos

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    json_output_dir = output_root / "json"
    json_output_dir.mkdir(parents=True, exist_ok=True)

    tracking_cleanup = _default_cleanup_config()
    tracking_cleanup.update(
        {
            "conf_threshold": args.conf_threshold,
            "nms_mode": args.nms_mode,
            "two_pass_prune_short_tracks": args.two_pass_prune_short_tracks,
            "min_track_length": args.min_track_length,
            "postprocess_smoothing": args.postprocess_smoothing,
        }
    )

    tasks = [
        (
            video_path,
            str(json_output_dir / f"{Path(video_path).stem}_cleanup_tracking.json"),
            args.model_path,
            bool(args.save_tracking_video),
            tracking_cleanup,
        )
        for video_path in benchmark_videos
    ]

    requested_concurrency = max(1, int(args.tracking_concurrency))
    effective_concurrency = min(requested_concurrency, len(tasks))
    start = time.perf_counter()
    if effective_concurrency == 1:
        results = [_track_cleanup_worker(*task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=effective_concurrency) as pool:
            results = pool.starmap(_track_cleanup_worker, tasks)
    total_elapsed_s = time.perf_counter() - start

    summary = {
        "source_videos": videos,
        "benchmark_videos": benchmark_videos,
        "extend_seconds": args.extend_seconds,
        "prepared_video_dir": args.prepared_video_dir if args.extend_seconds > 0 else None,
        "prepared_videos": prepared_metadata,
        "model_path": args.model_path,
        "gpu_info": _query_gpu_info(),
        "requested_tracking_concurrency": requested_concurrency,
        "effective_tracking_concurrency": effective_concurrency,
        "save_tracking_video": bool(args.save_tracking_video),
        "tracking_cleanup": tracking_cleanup,
        "total_elapsed_s": total_elapsed_s,
        "runs": [asdict(result) for result in results],
    }

    summary_path = output_root / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Tracking cleanup benchmark summary")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
