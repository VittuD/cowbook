from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from cowbook.io.json_utils import dump_path_compact
from tools.benchmark_tracking import _probe_video_metadata
from tools.run_sam3_windowed import DEFAULT_WINDOW_SECONDS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHANNELS = ["Ch1", "Ch4", "Ch6", "Ch8"]
# Measured on a single A40: no render, single "cow" prompt, includes our
# per-frame mask-extraction cost. Override with --seconds-per-frame once
# the target node's actual throughput is known.
DEFAULT_SECONDS_PER_FRAME = 0.93


def _log(enabled: bool, message: str) -> None:
    if enabled:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)


@dataclass(slots=True)
class VideoWorkItem:
    path: str
    frame_count: int
    fps: float
    duration_s: float


@dataclass(slots=True)
class GpuAssignment:
    gpu_index: int
    videos: list[str] = field(default_factory=list)
    total_frames: int = 0

    def to_dict(self, seconds_per_frame: float) -> dict:
        return {
            "gpu_index": self.gpu_index,
            "video_count": len(self.videos),
            "videos": list(self.videos),
            "total_frames": self.total_frames,
            "estimated_seconds": self.total_frames * seconds_per_frame,
        }


def _collect_channel_videos(input_dir: str, channels: list[str]) -> list[str]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")
    matched: set[str] = set()
    for channel in channels:
        matched.update(str(path) for path in root.glob(f"{channel}_*.mp4") if path.is_file())
    if not matched:
        raise FileNotFoundError(f"No videos matched channels {channels} under {input_dir}")
    return sorted(matched)


def _probe_videos(video_paths: list[str], *, log_progress: bool) -> list[VideoWorkItem]:
    items: list[VideoWorkItem] = []
    for path in video_paths:
        metadata = _probe_video_metadata(path)
        item = VideoWorkItem(
            path=path,
            frame_count=int(metadata["frame_count"]),
            fps=float(metadata["fps"]),
            duration_s=float(metadata["duration_s"]),
        )
        items.append(item)
        _log(
            log_progress,
            f"[probe] {path}: frames={item.frame_count} fps={item.fps:.2f} duration_s={item.duration_s:.1f}",
        )
    return items


def _assign_longest_processing_time_first(items: list[VideoWorkItem], num_gpus: int) -> list[GpuAssignment]:
    """Greedy LPT bin-packing: the largest remaining job always goes to the
    currently least-loaded GPU.

    This is the standard heuristic for minimizing makespan across identical
    parallel machines when job sizes are known up front (our frame counts).
    SAM3 video tracking can't be split within a single video -- the tracker
    carries frame-to-frame memory state -- so the only axis that
    parallelizes is across videos, one full model instance per GPU. That
    holds whether or not a video is windowed internally (tools.run_sam3_
    windowed processes one video's windows sequentially on a single GPU):
    windowing bounds memory and adds a small, roughly constant per-window
    reload cost, but a video's total frame count remains an accurate
    proxy for its total processing time either way, so no separate
    weighting is needed for windowed vs. single-pass dispatch.
    """
    assignments = [GpuAssignment(gpu_index=index) for index in range(num_gpus)]
    for item in sorted(items, key=lambda work_item: work_item.frame_count, reverse=True):
        target = min(assignments, key=lambda assignment: assignment.total_frames)
        target.videos.append(item.path)
        target.total_frames += item.frame_count
    return assignments


def _detect_gpu_count() -> int:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return 1
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return max(1, len(lines))


def _print_plan(assignments: list[GpuAssignment], *, seconds_per_frame: float) -> None:
    makespan_s = max((a.total_frames for a in assignments), default=0) * seconds_per_frame
    print("SAM3 multi-GPU batch plan")
    for assignment in assignments:
        hours = assignment.total_frames * seconds_per_frame / 3600.0
        print(
            f"  gpu {assignment.gpu_index}: {len(assignment.videos)} videos, "
            f"{assignment.total_frames} frames, ~{hours:.1f}h"
        )
    print(f"estimated makespan: ~{makespan_s / 3600.0:.1f}h (bottleneck GPU, assuming perfect concurrency)")


def _launch_assignment(
    assignment: GpuAssignment,
    *,
    output_root: Path,
    model_path: str,
    prompts: list[str],
    render_mode: str,
    log_dir: Path,
    log_every_frames: int,
    windowed: bool,
    window_seconds: float,
    target_fps: float | None = None,
) -> subprocess.Popen:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"gpu{assignment.gpu_index}.log"
    module_name = "tools.run_sam3_windowed" if windowed else "tools.benchmark_sam3_semantic_tracking"
    command = [
        sys.executable,
        "-m",
        module_name,
        "--videos",
        *assignment.videos,
        "--prompts",
        *prompts,
        "--model-path",
        model_path,
        "--output-root",
        str(output_root),
        "--render-mode",
        render_mode,
        "--dump-frame-metadata",
        "--log-progress",
        "--log-every-frames",
        str(log_every_frames),
    ]
    if windowed:
        command += ["--window-seconds", str(window_seconds)]
    if target_fps is not None:
        command += ["--target-fps", str(target_fps)]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(assignment.gpu_index)
    log_file = open(log_path, "w")
    return subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Partition a folder's videos across the available GPUs (one SAM3 "
            "video-tracking process per GPU, longest-job-first bin-packing) "
            "and optionally launch them. Dispatches tools.run_sam3_windowed "
            "by default (bounded GPU memory via fixed-duration windows); "
            "pass --no-windowed for a single unbroken pass per video "
            "(tools.benchmark_sam3_semantic_tracking) instead."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing input videos.")
    parser.add_argument(
        "--channels",
        nargs="+",
        default=DEFAULT_CHANNELS,
        help="Channel prefixes to include, e.g. Ch1 Ch4 Ch6 Ch8. Matches '<channel>_*.mp4'.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs to use. 0 auto-detects via nvidia-smi (falls back to 1).",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["cow"],
        help="Text prompts forwarded to each worker.",
    )
    parser.add_argument("--model-path", default="models/sam3.pt", help="Local SAM3 model weights path.")
    parser.add_argument(
        "--output-root",
        default="var/batches/sam3_ch1468",
        help="Shared output directory for all workers (video stems are unique, so this is safe to share).",
    )
    parser.add_argument(
        "--render-mode",
        choices=("all", "processed-only", "raw-only", "none"),
        default="none",
        help="Rendering mode forwarded to each worker.",
    )
    parser.add_argument(
        "--windowed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Dispatch tools.run_sam3_windowed (fixed-duration windows, "
            "bounded GPU memory, no persistent extra disk usage) instead "
            "of tools.benchmark_sam3_semantic_tracking (single pass over "
            "the whole video)."
        ),
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Window length in seconds forwarded to each worker when --windowed is set.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help=(
            "If set, decimate each video to approximately this many frames "
            "per second before running SAM3, forwarded to each worker "
            "regardless of --windowed (frame dropping only -- must not "
            "exceed a video's own fps)."
        ),
    )
    parser.add_argument(
        "--seconds-per-frame",
        type=float,
        default=DEFAULT_SECONDS_PER_FRAME,
        help="Throughput estimate used only for the printed plan, not for scheduling.",
    )
    parser.add_argument(
        "--log-every-frames",
        type=int,
        default=200,
        help="Progress log interval forwarded to each worker.",
    )
    parser.add_argument(
        "--log-dir",
        default="var/batches/sam3_ch1468/logs",
        help="Directory for per-GPU launcher log files.",
    )
    parser.add_argument(
        "--plan-path",
        default=None,
        help="Optional path to dump the assignment plan as JSON.",
    )
    parser.add_argument(
        "--launch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Actually launch the worker processes. Without this, only prints the plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    num_gpus = int(args.num_gpus) if int(args.num_gpus) > 0 else _detect_gpu_count()
    video_paths = _collect_channel_videos(str(args.input_dir), list(args.channels))
    _log(True, f"found {len(video_paths)} videos across channels {list(args.channels)}, targeting {num_gpus} GPU(s)")
    if args.windowed:
        _log(True, f"dispatch mode: windowed ({float(args.window_seconds):.0f}s windows per worker)")
    else:
        _log(True, "dispatch mode: single pass per video (no windowing)")
    if args.target_fps is not None:
        _log(True, f"decimating to ~{float(args.target_fps):.2f}fps per worker")

    items = _probe_videos(video_paths, log_progress=True)
    assignments = _assign_longest_processing_time_first(items, num_gpus)
    _print_plan(assignments, seconds_per_frame=float(args.seconds_per_frame))

    if args.plan_path:
        plan_payload = {
            "input_dir": str(args.input_dir),
            "channels": list(args.channels),
            "num_gpus": num_gpus,
            "seconds_per_frame": float(args.seconds_per_frame),
            "windowed": bool(args.windowed),
            "window_seconds": float(args.window_seconds),
            "target_fps": float(args.target_fps) if args.target_fps is not None else None,
            "assignments": [assignment.to_dict(float(args.seconds_per_frame)) for assignment in assignments],
        }
        plan_path = Path(args.plan_path)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path_compact(plan_path, plan_payload)

    if not args.launch:
        print("Dry run only (pass --launch to start the worker processes).")
        return 0

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)

    processes = []
    for assignment in assignments:
        if not assignment.videos:
            continue
        process = _launch_assignment(
            assignment,
            output_root=output_root,
            model_path=str(args.model_path),
            prompts=list(args.prompts),
            render_mode=str(args.render_mode),
            log_dir=log_dir,
            log_every_frames=int(args.log_every_frames),
            windowed=bool(args.windowed),
            window_seconds=float(args.window_seconds),
            target_fps=float(args.target_fps) if args.target_fps is not None else None,
        )
        processes.append((assignment.gpu_index, process.pid))
        _log(True, f"launched gpu {assignment.gpu_index}: pid={process.pid} log={log_dir / f'gpu{assignment.gpu_index}.log'}")

    print("Launched worker PIDs (per GPU):")
    for gpu_index, pid in processes:
        print(f"  gpu {gpu_index}: pid {pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
