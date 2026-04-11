from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2

from cowbook.io.json_utils import dump_path_compact, dumps_pretty
from tools.benchmark_tracking import _prepare_benchmark_videos, _probe_video_metadata, _query_gpu_info


@dataclass(slots=True)
class DevaVideoRunResult:
    video_path: str
    output_dir: str
    prepared_frames_dir: str
    primary_rendered_artifact: str
    rendered_artifacts: list[str]
    summary_json_path: str
    elapsed_s: float
    frame_count: int
    fps: float
    width: int
    height: int
    prompts: list[str]
    sam_variant: str
    size: int
    chunk_size: int
    temporal_setting: str
    amp: bool
    max_frames: int


def _default_videos() -> list[str]:
    return [
        "sample_data/videos/Ch1_60.mp4",
        "sample_data/videos/Ch4_60.mp4",
        "sample_data/videos/Ch6_60.mp4",
        "sample_data/videos/Ch8_60.mp4",
    ]


def _default_prompts() -> list[str]:
    return ["cow"]


def _default_deva_repo() -> str:
    return "/opt/Tracking-Anything-with-DEVA"


def _default_gsa_repo() -> str:
    return "/opt/Grounded-Segment-Anything"


def _default_prepared_frame_dir() -> str:
    return "var/benchmarks/prepared_frames_deva"


def _log_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _validate_videos(video_paths: Iterable[str]) -> list[str]:
    normalized = [str(Path(path)) for path in video_paths]
    missing = [path for path in normalized if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing DEVA benchmark video(s): {missing}")
    return normalized


def _resolve_prompts(prompts: Iterable[str] | None) -> list[str]:
    if prompts is None:
        return _default_prompts()
    resolved = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
    if not resolved:
        raise ValueError("At least one non-empty prompt is required.")
    return resolved


def _validate_repo_path(path: str, label: str) -> str:
    candidate = Path(path)
    if not candidate.is_dir():
        raise FileNotFoundError(f"{label} repo directory does not exist: {candidate}")
    return str(candidate)


def _artifact_stem(video_path: str) -> str:
    return Path(video_path).stem


def _extract_video_frames(
    *,
    video_path: str,
    output_dir: Path,
    max_frames: int,
) -> dict[str, int | float | str]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    width = 0
    height = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if width == 0 or height == 0:
                height = int(frame.shape[0])
                width = int(frame.shape[1])
            output_path = output_dir / f"{frame_count:07d}.jpg"
            if not cv2.imwrite(str(output_path), frame):
                raise ValueError(f"Failed to write extracted frame: {output_path}")
            frame_count += 1
            if max_frames > 0 and frame_count >= max_frames:
                break
    finally:
        capture.release()

    if frame_count == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "frames_dir": str(output_dir),
    }


def _prompt_arg(prompts: list[str]) -> str:
    return ".".join(prompts)


def _build_deva_text_command(
    *,
    python_bin: str,
    frames_dir: str,
    output_dir: str,
    prompts: list[str],
    chunk_size: int,
    size: int,
    temporal_setting: str,
    amp: bool,
    sam_variant: str,
) -> list[str]:
    command = [
        python_bin,
        "demo/demo_with_text.py",
        "--chunk_size",
        str(chunk_size),
        "--img_path",
        frames_dir,
        "--temporal_setting",
        temporal_setting,
        "--size",
        str(size),
        "--output",
        output_dir,
        "--prompt",
        _prompt_arg(prompts),
    ]
    if amp:
        command.append("--amp")
    if sam_variant != "original":
        command.extend(["--sam_variant", sam_variant])
    return command


def _collect_rendered_artifacts(output_dir: Path) -> list[str]:
    extensions = {".mp4", ".mov", ".avi", ".gif", ".webm", ".mkv"}
    artifacts = [
        str(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return artifacts


def _run_deva_text_tracking_for_video(
    *,
    video_path: str,
    output_root: Path,
    prompts: list[str],
    deva_repo: str,
    gsa_repo: str,
    python_bin: str,
    sam_variant: str,
    size: int,
    chunk_size: int,
    temporal_setting: str,
    amp: bool,
    max_frames: int,
    prepared_frame_dir: Path,
    log_progress: bool,
) -> DevaVideoRunResult:
    metadata = _probe_video_metadata(video_path)
    stem = _artifact_stem(video_path)
    json_dir = output_root / "json"
    raw_dir = output_root / "deva_raw" / stem
    frames_dir = prepared_frame_dir / stem
    summary_json_path = json_dir / f"{stem}_deva_summary.json"

    json_dir.mkdir(parents=True, exist_ok=True)
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    extracted = _extract_video_frames(
        video_path=video_path,
        output_dir=frames_dir,
        max_frames=max_frames,
    )

    command = _build_deva_text_command(
        python_bin=python_bin,
        frames_dir=str(frames_dir),
        output_dir=str(raw_dir),
        prompts=prompts,
        chunk_size=chunk_size,
        size=size,
        temporal_setting=temporal_setting,
        amp=amp,
        sam_variant=sam_variant,
    )
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_entries = [deva_repo]
    env["PYTHONPATH"] = ":".join(
        [entry for entry in pythonpath_entries + ([existing_pythonpath] if existing_pythonpath else []) if entry]
    )

    _log_progress(log_progress, f"[deva] start: {video_path} prompts={prompts}")
    start = time.perf_counter()
    subprocess.run(
        command,
        cwd=deva_repo,
        env=env,
        check=True,
    )
    elapsed_s = time.perf_counter() - start

    rendered_artifacts = _collect_rendered_artifacts(raw_dir)
    primary_rendered_artifact = rendered_artifacts[0] if rendered_artifacts else ""
    summary_payload = {
        "video_path": video_path,
        "output_dir": str(raw_dir),
        "prepared_frames_dir": str(frames_dir),
        "primary_rendered_artifact": primary_rendered_artifact,
        "rendered_artifacts": rendered_artifacts,
        "elapsed_s": elapsed_s,
        "frame_count": int(extracted["frame_count"]),
        "fps": float(extracted["fps"]),
        "width": int(extracted["width"] or metadata["width"]),
        "height": int(extracted["height"] or metadata["height"]),
        "prompts": prompts,
        "sam_variant": sam_variant,
        "size": size,
        "chunk_size": chunk_size,
        "temporal_setting": temporal_setting,
        "amp": amp,
        "max_frames": max_frames,
        "deva_command": command,
    }
    dump_path_compact(summary_json_path, summary_payload)
    _log_progress(log_progress, f"[deva] done: {video_path} in {elapsed_s:.2f}s -> {summary_json_path}")
    return DevaVideoRunResult(
        video_path=video_path,
        output_dir=str(raw_dir),
        prepared_frames_dir=str(frames_dir),
        primary_rendered_artifact=primary_rendered_artifact,
        rendered_artifacts=rendered_artifacts,
        summary_json_path=str(summary_json_path),
        elapsed_s=elapsed_s,
        frame_count=int(extracted["frame_count"]),
        fps=float(extracted["fps"]),
        width=int(extracted["width"] or metadata["width"]),
        height=int(extracted["height"] or metadata["height"]),
        prompts=prompts,
        sam_variant=sam_variant,
        size=size,
        chunk_size=chunk_size,
        temporal_setting=temporal_setting,
        amp=amp,
        max_frames=max_frames,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DEVA text-prompted video segmentation on one or more videos.",
    )
    parser.add_argument("--videos", nargs="+", default=_default_videos(), help="Video paths to run.")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=_default_prompts(),
        help="Text prompts for DEVA. Multiple prompts are joined with '.' for the upstream demo.",
    )
    parser.add_argument(
        "--output-root",
        default="var/benchmarks/deva_text_tracking",
        help="Directory where DEVA outputs and JSON summaries are written.",
    )
    parser.add_argument(
        "--summary-name",
        default="summary.json",
        help="Top-level summary filename under --output-root.",
    )
    parser.add_argument(
        "--extend-seconds",
        type=int,
        default=0,
        help="If > 0, create longer benchmark copies of the input videos before running.",
    )
    parser.add_argument(
        "--prepared-video-dir",
        default="var/benchmarks/prepared_videos_deva",
        help="Directory for prepared benchmark videos when --extend-seconds is used.",
    )
    parser.add_argument(
        "--prepared-frame-dir",
        default=_default_prepared_frame_dir(),
        help="Directory where extracted frame folders are written for DEVA.",
    )
    parser.add_argument(
        "--deva-repo",
        default=_default_deva_repo(),
        help="Path to the cloned Tracking-Anything-with-DEVA repository.",
    )
    parser.add_argument(
        "--gsa-repo",
        default=_default_gsa_repo(),
        help="Path to the cloned Grounded-Segment-Anything repository.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to invoke upstream DEVA scripts.",
    )
    parser.add_argument(
        "--sam-variant",
        choices=("original", "sam_hq", "sam_hq_light", "mobile"),
        default="original",
        help="SAM variant forwarded to DEVA text mode.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=480,
        help="Resize shorter-side target forwarded to DEVA.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4,
        help="Chunk size forwarded to DEVA.",
    )
    parser.add_argument(
        "--temporal-setting",
        choices=("online", "semionline"),
        default="semionline",
        help="Temporal setting forwarded to DEVA.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable mixed precision for DEVA.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If > 0, extract and process at most this many frames from each input video.",
    )
    parser.add_argument(
        "--log-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable coarse progress logs for console and swarm logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if int(args.size) < 1:
        raise ValueError("--size must be >= 1.")
    if int(args.chunk_size) < 1:
        raise ValueError("--chunk-size must be >= 1.")
    if int(args.max_frames) < 0:
        raise ValueError("--max-frames must be >= 0.")

    videos = _validate_videos(args.videos)
    prompts = _resolve_prompts(args.prompts)
    deva_repo = _validate_repo_path(args.deva_repo, "DEVA")
    gsa_repo = _validate_repo_path(args.gsa_repo, "Grounded-Segment-Anything")

    prepared_videos, prepared_video_metadata = _prepare_benchmark_videos(
        videos,
        target_duration_seconds=args.extend_seconds,
        prepared_video_dir=args.prepared_video_dir,
        log_progress=bool(args.log_progress),
    )
    benchmark_videos = prepared_videos or videos

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prepared_frame_dir = Path(args.prepared_frame_dir)
    prepared_frame_dir.mkdir(parents=True, exist_ok=True)

    runtime_info = {
        "gpu_info": _query_gpu_info(),
        "deva_repo": deva_repo,
        "gsa_repo": gsa_repo,
        "python_bin": args.python_bin,
    }
    runs = [
        _run_deva_text_tracking_for_video(
            video_path=video_path,
            output_root=output_root,
            prompts=prompts,
            deva_repo=deva_repo,
            gsa_repo=gsa_repo,
            python_bin=args.python_bin,
            sam_variant=args.sam_variant,
            size=int(args.size),
            chunk_size=int(args.chunk_size),
            temporal_setting=str(args.temporal_setting),
            amp=bool(args.amp),
            max_frames=int(args.max_frames),
            prepared_frame_dir=prepared_frame_dir,
            log_progress=bool(args.log_progress),
        )
        for video_path in benchmark_videos
    ]

    summary = {
        "tool": "benchmark_deva_text_tracking",
        "runtime": runtime_info,
        "videos": benchmark_videos,
        "prompts": prompts,
        "prepared_video_metadata": prepared_video_metadata,
        "runs": [asdict(run) for run in runs],
    }
    summary_path = output_root / args.summary_name
    dump_path_compact(summary_path, summary)
    if args.log_progress:
        print("DEVA text tracking summary")
        print(dumps_pretty(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
