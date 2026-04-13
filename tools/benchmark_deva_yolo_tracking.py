from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO

from cowbook.io.json_utils import dump_path_compact, dumps_pretty
from tools.benchmark_deva_text_tracking import (
    _artifact_stem,
    _collect_rendered_artifacts,
    _default_deva_model_path,
    _default_deva_repo,
    _default_prepared_frame_dir,
    _default_videos,
    _extract_video_frames,
    _log_progress,
    _prepare_benchmark_videos,
    _probe_video_metadata,
    _query_gpu_info,
    _validate_repo_path,
    _validate_videos,
)


@dataclass(slots=True)
class DevaYoloVideoRunResult:
    video_path: str
    output_dir: str
    prepared_frames_dir: str
    prepared_masks_dir: str
    primary_rendered_artifact: str
    rendered_artifacts: list[str]
    summary_json_path: str
    elapsed_s: float
    yolo_elapsed_s: float
    deva_elapsed_s: float
    frame_count: int
    fps: float
    width: int
    height: int
    yolo_model_path: str
    yolo_conf: float
    size: int
    chunk_size: int
    temporal_setting: str
    amp: bool
    max_frames: int


def _default_prepared_mask_dir() -> str:
    return "var/benchmarks/prepared_masks_deva_yolo"


def _validate_model_path(model_path: str) -> str:
    candidate = Path(model_path)
    if not candidate.exists():
        raise FileNotFoundError(f"YOLO segmentation weights do not exist: {candidate}")
    return str(candidate)


def _build_deva_detection_command(
    *,
    python_bin: str,
    frames_root: str,
    masks_root: str,
    output_root: str,
    deva_model_path: str,
    chunk_size: int,
    size: int,
    temporal_setting: str,
    amp: bool,
) -> list[str]:
    command = [
        python_bin,
        "evaluation/eval_with_detections.py",
        "--model",
        deva_model_path,
        "--img_path",
        frames_root,
        "--mask_path",
        masks_root,
        "--output",
        output_root,
        "--dataset",
        "demo",
        "--detection_every",
        "1",
        "--num_voting_frames",
        "1",
        "--chunk_size",
        str(chunk_size),
        "--size",
        str(size),
        "--temporal_setting",
        temporal_setting,
    ]
    if amp:
        command.append("--amp")
    return command


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _ordered_mask_indices(confidences: np.ndarray, count: int) -> list[int]:
    if count <= 0:
        return []
    if confidences.shape[0] != count:
        return list(range(count))
    return list(np.argsort(confidences.astype(np.float32)))


def _mask_payload_from_result(result) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    orig_shape = getattr(result, "orig_shape", None)
    if orig_shape is None or len(orig_shape) < 2:
        raise ValueError("YOLO segmentation result is missing orig_shape.")
    height = int(orig_shape[0])
    width = int(orig_shape[1])
    label_mask = np.zeros((height, width), dtype=np.uint8)

    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or getattr(masks, "data", None) is None:
        return label_mask, []

    mask_data = _tensor_to_numpy(masks.data)
    if mask_data.ndim == 2:
        mask_data = mask_data[None, ...]

    confidences = np.zeros((mask_data.shape[0],), dtype=np.float32)
    if boxes is not None and getattr(boxes, "conf", None) is not None:
        confidences = _tensor_to_numpy(boxes.conf).astype(np.float32).reshape(-1)

    segments_info: list[dict[str, float | int]] = []
    next_id = 1
    for raw_index in _ordered_mask_indices(confidences, int(mask_data.shape[0])):
        binary_mask = np.asarray(mask_data[raw_index], dtype=np.float32)
        if binary_mask.shape != (height, width):
            binary_mask = cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        active = binary_mask > 0.5
        if not np.any(active):
            continue
        if next_id > 255:
            raise ValueError("Per-frame YOLO instance count exceeds 255; grayscale DEVA mask export is insufficient.")
        label_mask[active] = np.uint8(next_id)
        score = float(confidences[raw_index]) if raw_index < confidences.shape[0] else 1.0
        segments_info.append(
            {
                "id": next_id,
                "category_id": 1,
                "score": round(score, 6),
            }
        )
        next_id += 1

    return label_mask, segments_info


def _write_detection_artifacts_for_result(*, result, masks_dir: Path) -> int:
    frame_stem = Path(str(result.path)).stem
    label_mask, segments_info = _mask_payload_from_result(result)
    png_path = masks_dir / f"{frame_stem}.png"
    json_path = masks_dir / f"{frame_stem}.json"

    masks_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(png_path), label_mask):
        raise ValueError(f"Failed to write YOLO mask PNG: {png_path}")
    json_path.write_text(json.dumps(segments_info), encoding="utf-8")
    return len(segments_info)


def _run_yolo_segmentation_export(
    *,
    frames_dir: Path,
    masks_dir: Path,
    model_path: str,
    yolo_conf: float,
    size: int,
    log_progress: bool,
    log_every_frames: int,
) -> tuple[float, int]:
    model = YOLO(model_path, task="segment")
    if masks_dir.exists():
        shutil.rmtree(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    total_instances = 0
    for frame_index, result in enumerate(
        model.predict(
            source=str(frames_dir),
            stream=True,
            verbose=False,
            conf=float(yolo_conf),
            imgsz=int(size),
            retina_masks=True,
        )
    ):
        instance_count = _write_detection_artifacts_for_result(result=result, masks_dir=masks_dir)
        total_instances += instance_count
        if log_progress and log_every_frames > 0:
            if frame_index == 0 or ((frame_index + 1) % log_every_frames == 0):
                _log_progress(
                    True,
                    f"[deva-yolo] yolo export: {frames_dir.name} frame={frame_index + 1} instances={instance_count}",
                )
    return time.perf_counter() - start, total_instances


def _run_deva_yolo_tracking_for_video(
    *,
    video_path: str,
    output_root: Path,
    deva_repo: str,
    python_bin: str,
    deva_model_path: str,
    yolo_model_path: str,
    yolo_conf: float,
    size: int,
    chunk_size: int,
    temporal_setting: str,
    amp: bool,
    max_frames: int,
    prepared_frame_dir: Path,
    prepared_mask_dir: Path,
    log_progress: bool,
    log_every_frames: int,
) -> DevaYoloVideoRunResult:
    metadata = _probe_video_metadata(video_path)
    stem = _artifact_stem(video_path)
    json_dir = output_root / "json"
    raw_root = output_root / "deva_raw"
    raw_dir = raw_root / stem
    frames_dir = prepared_frame_dir / stem
    masks_dir = prepared_mask_dir / stem
    summary_json_path = json_dir / f"{stem}_deva_yolo_summary.json"

    json_dir.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    extracted = _extract_video_frames(
        video_path=video_path,
        output_dir=frames_dir,
        max_frames=max_frames,
    )
    _log_progress(log_progress, f"[deva-yolo] start: {video_path}")
    total_start = time.perf_counter()
    yolo_elapsed_s, _ = _run_yolo_segmentation_export(
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        model_path=yolo_model_path,
        yolo_conf=yolo_conf,
        size=size,
        log_progress=log_progress,
        log_every_frames=log_every_frames,
    )

    command = _build_deva_detection_command(
        python_bin=python_bin,
        frames_root=str(prepared_frame_dir),
        masks_root=str(prepared_mask_dir),
        output_root=str(raw_root),
        deva_model_path=deva_model_path,
        chunk_size=chunk_size,
        size=size,
        temporal_setting=temporal_setting,
        amp=amp,
    )
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(
        entry for entry in [deva_repo, existing_pythonpath] if entry
    )

    deva_start = time.perf_counter()
    subprocess.run(
        command,
        cwd=deva_repo,
        env=env,
        check=True,
    )
    deva_elapsed_s = time.perf_counter() - deva_start
    elapsed_s = time.perf_counter() - total_start

    rendered_artifacts = _collect_rendered_artifacts(raw_dir)
    primary_rendered_artifact = rendered_artifacts[0] if rendered_artifacts else ""
    summary_payload = {
        "video_path": video_path,
        "output_dir": str(raw_dir),
        "prepared_frames_dir": str(frames_dir),
        "prepared_masks_dir": str(masks_dir),
        "primary_rendered_artifact": primary_rendered_artifact,
        "rendered_artifacts": rendered_artifacts,
        "elapsed_s": elapsed_s,
        "yolo_elapsed_s": yolo_elapsed_s,
        "deva_elapsed_s": deva_elapsed_s,
        "frame_count": int(extracted["frame_count"]),
        "fps": float(extracted["fps"]),
        "width": int(extracted["width"] or metadata["width"]),
        "height": int(extracted["height"] or metadata["height"]),
        "yolo_model_path": yolo_model_path,
        "yolo_conf": float(yolo_conf),
        "size": size,
        "chunk_size": chunk_size,
        "temporal_setting": temporal_setting,
        "amp": amp,
        "max_frames": max_frames,
        "deva_command": command,
        "effective_fps": (int(extracted["frame_count"]) / elapsed_s) if elapsed_s > 0 else 0.0,
    }
    dump_path_compact(summary_json_path, summary_payload)
    _log_progress(log_progress, f"[deva-yolo] done: {video_path} in {elapsed_s:.2f}s -> {summary_json_path}")
    return DevaYoloVideoRunResult(
        video_path=video_path,
        output_dir=str(raw_dir),
        prepared_frames_dir=str(frames_dir),
        prepared_masks_dir=str(masks_dir),
        primary_rendered_artifact=primary_rendered_artifact,
        rendered_artifacts=rendered_artifacts,
        summary_json_path=str(summary_json_path),
        elapsed_s=elapsed_s,
        yolo_elapsed_s=yolo_elapsed_s,
        deva_elapsed_s=deva_elapsed_s,
        frame_count=int(extracted["frame_count"]),
        fps=float(extracted["fps"]),
        width=int(extracted["width"] or metadata["width"]),
        height=int(extracted["height"] or metadata["height"]),
        yolo_model_path=yolo_model_path,
        yolo_conf=float(yolo_conf),
        size=size,
        chunk_size=chunk_size,
        temporal_setting=temporal_setting,
        amp=amp,
        max_frames=max_frames,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-class YOLO11-seg proposals through DEVA video propagation.",
    )
    parser.add_argument("--videos", nargs="+", default=_default_videos(), help="Video paths to run.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Absolute or container-local path to the trained YOLO11 segmentation weights.",
    )
    parser.add_argument(
        "--output-root",
        default="var/benchmarks/deva_yolo_tracking",
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
        default="var/benchmarks/prepared_videos_deva_yolo",
        help="Directory for prepared benchmark videos when --extend-seconds is used.",
    )
    parser.add_argument(
        "--prepared-frame-dir",
        default=_default_prepared_frame_dir(),
        help="Directory where extracted frame folders are written for DEVA.",
    )
    parser.add_argument(
        "--prepared-mask-dir",
        default=_default_prepared_mask_dir(),
        help="Directory where YOLO-generated mask folders are written for DEVA.",
    )
    parser.add_argument(
        "--deva-repo",
        default=_default_deva_repo(),
        help="Path to the cloned Tracking-Anything-with-DEVA repository.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to invoke upstream DEVA scripts.",
    )
    parser.add_argument(
        "--deva-model-path",
        default=_default_deva_model_path(),
        help="Absolute path to the DEVA propagation checkpoint.",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="Confidence threshold forwarded to YOLO segmentation inference.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=480,
        help="Resize target used for both YOLO inference and DEVA evaluation.",
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
    parser.add_argument(
        "--log-every-frames",
        type=int,
        default=25,
        help="When logging is enabled, emit progress every N exported YOLO frames.",
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
    if int(args.log_every_frames) < 1:
        raise ValueError("--log-every-frames must be >= 1.")
    if not (0.0 <= float(args.yolo_conf) <= 1.0):
        raise ValueError("--yolo-conf must be between 0 and 1.")

    videos = _validate_videos(args.videos)
    deva_repo = _validate_repo_path(args.deva_repo, "DEVA")
    deva_model_path = str(Path(args.deva_model_path))
    if not Path(deva_model_path).exists():
        raise FileNotFoundError(f"DEVA propagation checkpoint does not exist: {deva_model_path}")
    yolo_model_path = _validate_model_path(args.model_path)

    prepared_videos, prepared_video_metadata = _prepare_benchmark_videos(
        videos,
        target_duration_seconds=args.extend_seconds,
        prepared_video_dir=args.prepared_video_dir,
        log_progress=bool(args.log_progress),
    )
    benchmark_videos = prepared_videos or videos

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prepared_frame_dir = Path(args.prepared_frame_dir).resolve()
    prepared_frame_dir.mkdir(parents=True, exist_ok=True)
    prepared_mask_dir = Path(args.prepared_mask_dir).resolve()
    prepared_mask_dir.mkdir(parents=True, exist_ok=True)

    runtime_info = {
        "gpu_info": _query_gpu_info(),
        "deva_repo": deva_repo,
        "python_bin": args.python_bin,
        "ultralytics_version": ultralytics.__version__,
    }
    runs = [
        _run_deva_yolo_tracking_for_video(
            video_path=video_path,
            output_root=output_root,
            deva_repo=deva_repo,
            python_bin=args.python_bin,
            deva_model_path=deva_model_path,
            yolo_model_path=yolo_model_path,
            yolo_conf=float(args.yolo_conf),
            size=int(args.size),
            chunk_size=int(args.chunk_size),
            temporal_setting=str(args.temporal_setting),
            amp=bool(args.amp),
            max_frames=int(args.max_frames),
            prepared_frame_dir=prepared_frame_dir,
            prepared_mask_dir=prepared_mask_dir,
            log_progress=bool(args.log_progress),
            log_every_frames=int(args.log_every_frames),
        )
        for video_path in benchmark_videos
    ]

    summary = {
        "tool": "benchmark_deva_yolo_tracking",
        "runtime": runtime_info,
        "videos": benchmark_videos,
        "prepared_video_metadata": prepared_video_metadata,
        "runs": [asdict(run) for run in runs],
    }
    summary_path = output_root / args.summary_name
    dump_path_compact(summary_path, summary)
    if args.log_progress:
        print("DEVA YOLO tracking summary")
        print(dumps_pretty(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
