from __future__ import annotations

import argparse
import concurrent.futures as futures
import multiprocessing as mp
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from cowbook.io.json_utils import dump_path_compact
from cowbook.vision.preprocess_video import DEFAULT_CHANNEL_MASKS, crop_and_mask_video, mask_video
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


def _channel_for_video_path(path: str, channels: list[str]) -> str | None:
    name = Path(path).name
    for channel in channels:
        if name.startswith(f"{channel}_"):
            return channel
    return None


def _mask_one_video(args: tuple[str, str, str, bool]) -> str:
    """Top-level (picklable) worker for the process pool: mask, optionally
    crop, one video. Returns the produced path, so failures raise directly
    in the pool rather than being swallowed."""
    src_path, dst_path, mask_path, crop_to_mask = args
    if crop_to_mask:
        crop_and_mask_video(src_path, dst_path, mask_path)
    else:
        mask_video(src_path, dst_path, mask_path)
    return dst_path


def _is_masked_output_up_to_date(src_path: str, dst_path: str, mask_path: str) -> bool:
    """True if dst_path already reflects the current src_path and mask_path
    (mtime-based, like preprocess_video's own _should_skip). Lets a re-run
    against the same input skip the expensive full-video crop/mask pass
    entirely instead of redoing it from scratch."""
    if not os.path.exists(dst_path):
        return False
    dst_mtime = os.path.getmtime(dst_path)
    return dst_mtime >= os.path.getmtime(src_path) and dst_mtime >= os.path.getmtime(mask_path)


def _preprocess_videos_for_masking(
    video_paths: list[str],
    channels: list[str],
    *,
    output_dir: Path,
    crop_to_mask: bool,
    channel_masks: dict[str, str],
    max_workers: int,
    log_progress: bool,
) -> list[str]:
    """Produce masked (and optionally cropped-to-bbox) copies of every video
    before any SAM3 worker sees them.

    Doing this once here, per video, rather than inside each per-window
    chunk write, matters: run_sam3_windowed re-derives its window/decimated
    chunks straight from whatever path it's handed, so masking/cropping at
    that layer would re-run on every window of every video instead of once
    per video total -- for a multi-hour recording split into many windows,
    that's a lot of repeated full-resolution frame decoding for no benefit,
    since the mask never changes within a video.

    Also skips any video whose masked/cropped copy is already up to date
    (mtime-based), so re-running against the same input/output roots --
    e.g. after tuning --target-fps or another downstream flag -- doesn't
    redo this expensive full-video pass for videos already done.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    processed_paths = []
    skipped = 0
    for path in video_paths:
        channel = _channel_for_video_path(path, channels)
        mask_path = channel_masks.get(channel) if channel else None
        if not mask_path or not Path(mask_path).exists():
            raise FileNotFoundError(
                f"No usable mask for {path} (channel={channel!r}); configured channel masks: {channel_masks}"
            )
        dst_path = str(output_dir / Path(path).name)
        processed_paths.append(dst_path)
        if _is_masked_output_up_to_date(path, dst_path, mask_path):
            skipped += 1
            continue
        jobs.append((path, dst_path, mask_path, crop_to_mask))

    _log(
        log_progress,
        f"[mask] preprocessing {len(jobs)} video(s) "
        f"({'crop-to-mask-bbox' if crop_to_mask else 'mask, full resolution'}) with {max_workers} worker(s)"
        + (f", reusing {skipped} already up to date" if skipped else ""),
    )
    if jobs:
        with futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as pool:
            list(pool.map(_mask_one_video, jobs))
    _log(log_progress, f"[mask] done: {len(processed_paths)} video(s) -> {output_dir}")
    return processed_paths


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


def _build_preview_command(
    assignment: GpuAssignment,
    *,
    output_root: Path,
    model_path: str,
    prompts: list[str],
    preview_sample_count: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.preview_sam3_samples",
        "--videos",
        *assignment.videos,
        "--prompts",
        *prompts,
        "--model-path",
        model_path,
        "--output-root",
        str(output_root / "preview"),
        "--sample-count",
        str(preview_sample_count),
        "--log-progress",
    ]


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
    preview_sample_count: int | None = None,
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
    if preview_sample_count is None:
        return subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT))

    # Chained, not launched alongside: a second concurrent SAM3 predictor on
    # the same GPU while the tracking pass is still running would compete
    # for the same GPU memory instead of reusing it after the tracking pass
    # releases it.
    preview_command = _build_preview_command(
        assignment,
        output_root=output_root,
        model_path=model_path,
        prompts=prompts,
        preview_sample_count=preview_sample_count,
    )
    chained = " && ".join(shlex.join(cmd) for cmd in (command, preview_command))
    # Plain -c, not -l/login: a login shell sources /etc/profile and
    # friends before running anything, which is both unnecessary (the
    # environment is already passed explicitly via env=) and a real risk
    # on cluster images where those profile scripts commonly do
    # network-dependent or environment-module setup that can stall or
    # override variables we depend on (CUDA_VISIBLE_DEVICES included).
    return subprocess.Popen(
        ["bash", "-c", chained], stdout=log_file, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT)
    )


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
        "--preview-sample-count",
        type=int,
        default=None,
        help=(
            "If set, also run tools.preview_sam3_samples on this many "
            "equally spaced frames per video, after the main tracking "
            "pass finishes on that GPU (sequential, not concurrent, so it "
            "doesn't compete for GPU memory with the tracking pass). "
            "Independent single-frame inference, not tracking -- a cheap "
            "visual sanity check in place of full annotated-video "
            "rendering (--render-mode)."
        ),
    )
    parser.add_argument(
        "--mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply each video's channel mask before running SAM3 (same "
            "resolution, non-mask pixels blacked out). Ignored if "
            "--crop-to-mask is also set."
        ),
    )
    parser.add_argument(
        "--crop-to-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Crop each video to its channel mask's tight bounding box "
            "before running SAM3 (also masks non-mask pixels within that "
            "box). Cheaper and typically more accurate than --mask alone, "
            "since SAM3 then spends no inference on pixels outside the "
            "region of interest. Implies masking, so --mask is redundant "
            "alongside it."
        ),
    )
    parser.add_argument(
        "--mask-dir",
        default=None,
        help=(
            "Directory holding 'combined_mask_<channel-lower>.png' files "
            "(e.g. combined_mask_ch1.png), to override the built-in "
            "per-channel mask paths (DEFAULT_CHANNEL_MASKS). Only "
            "consulted with --mask or --crop-to-mask."
        ),
    )
    parser.add_argument(
        "--masked-video-dir",
        default=None,
        help="Where preprocessed (masked/cropped) video copies are written. Defaults to <output-root>/_masked_videos.",
    )
    parser.add_argument(
        "--mask-workers",
        type=int,
        default=0,
        help="Parallel workers for --mask/--crop-to-mask preprocessing. 0 uses min(detected CPUs - 1, 8).",
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
    output_root = Path(args.output_root)
    video_paths = _collect_channel_videos(str(args.input_dir), list(args.channels))
    _log(True, f"found {len(video_paths)} videos across channels {list(args.channels)}, targeting {num_gpus} GPU(s)")
    if args.windowed:
        _log(True, f"dispatch mode: windowed ({float(args.window_seconds):.0f}s windows per worker)")
    else:
        _log(True, "dispatch mode: single pass per video (no windowing)")
    if args.target_fps is not None:
        _log(True, f"decimating to ~{float(args.target_fps):.2f}fps per worker")
    if args.preview_sample_count is not None:
        _log(True, f"also generating a {int(args.preview_sample_count)}-frame preview per video")

    crop_to_mask = bool(args.crop_to_mask)
    apply_mask = bool(args.mask) or crop_to_mask
    if crop_to_mask:
        _log(True, "cropping each video to its channel mask's bounding box before SAM3")
    elif apply_mask:
        _log(True, "masking each video (full resolution) before SAM3")
    if apply_mask:
        channel_masks = (
            {
                channel: str(Path(args.mask_dir) / f"combined_mask_{channel.lower()}.png")
                for channel in args.channels
            }
            if args.mask_dir
            else dict(DEFAULT_CHANNEL_MASKS)
        )
        masked_video_dir = Path(args.masked_video_dir) if args.masked_video_dir else output_root / "_masked_videos"
        # Capped, not just cpu_count() - 1: this is CPU/IO-bound full-video
        # decode+encode, and a container's reported cpu_count() can wildly
        # overstate what's actually usable without starving other workers
        # sharing the same node/storage.
        mask_workers = int(args.mask_workers) if int(args.mask_workers) > 0 else min(max((os.cpu_count() or 2) - 1, 1), 8)
        video_paths = _preprocess_videos_for_masking(
            video_paths,
            list(args.channels),
            output_dir=masked_video_dir,
            crop_to_mask=crop_to_mask,
            channel_masks=channel_masks,
            max_workers=mask_workers,
            log_progress=True,
        )

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
            "preview_sample_count": int(args.preview_sample_count) if args.preview_sample_count is not None else None,
            "mask": bool(args.mask),
            "crop_to_mask": crop_to_mask,
            "assignments": [assignment.to_dict(float(args.seconds_per_frame)) for assignment in assignments],
        }
        plan_path = Path(args.plan_path)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path_compact(plan_path, plan_payload)

    if not args.launch:
        print("Dry run only (pass --launch to start the worker processes).")
        return 0

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
            preview_sample_count=int(args.preview_sample_count) if args.preview_sample_count is not None else None,
        )
        processes.append((assignment.gpu_index, process))
        _log(True, f"launched gpu {assignment.gpu_index}: pid={process.pid} log={log_dir / f'gpu{assignment.gpu_index}.log'}")

    print("Launched worker PIDs (per GPU):")
    for gpu_index, process in processes:
        print(f"  gpu {gpu_index}: pid {process.pid}")

    # Wait for every worker before returning: this process is the
    # container's entrypoint (PID 1) when run under Docker/Swarm, and a
    # PID-1 exit tears down the whole container -- killing every child
    # subprocess with it, typically before it even prints its first log
    # line. Fire-and-forget (returning right after Popen) only looks fine
    # outside a container, where orphaned children get reparented to init
    # and keep running; it silently discards all GPU work the instant this
    # function returns when it's actually PID 1.
    failed_gpus = []
    for gpu_index, process in processes:
        returncode = process.wait()
        if returncode == 0:
            _log(True, f"gpu {gpu_index} worker finished (pid={process.pid})")
        else:
            _log(True, f"gpu {gpu_index} worker FAILED (pid={process.pid}, exit={returncode})")
            failed_gpus.append(gpu_index)

    if failed_gpus:
        _log(True, f"one or more GPU workers failed: {failed_gpus}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
