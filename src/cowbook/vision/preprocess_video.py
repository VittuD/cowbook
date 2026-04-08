# preprocess_video.py
import concurrent.futures as futures
import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
from typing import Any, Dict, List, Tuple

import cv2

from cowbook.execution import JobReporter, StageProgressReporter

logger = logging.getLogger(__name__)


# ---------- Helpers to choose and load masks ----------
def _infer_channel_from_name(path: str) -> str | None:
    """Try to guess channel from filename (Ch1/Ch4/Ch6/Ch8), fallback to Ch8 rule used in dataset code."""
    name = os.path.basename(path)
    for ch in ("Ch1", "Ch4", "Ch6", "Ch8"):
        if ch in name:
            return ch
    # dataset rule: if frame_*_jpg pattern existed, use Ch8
    if re.match(r"^frame_\d+_.*_jpg", name):
        return "Ch8"
    return None


def _map_camera_to_channel(camera_nr: int, camera_to_mask_map: Dict[str, str] | None) -> str | None:
    if not camera_to_mask_map:
        return None
    # keys may be strings in JSON
    return camera_to_mask_map.get(str(camera_nr)) or camera_to_mask_map.get(camera_nr)


def _choose_channel(video_path: str, camera_nr: int, camera_to_mask_map: Dict[str, str] | None) -> str | None:
    # Prefer explicit mapping, then filename heuristic
    return _map_camera_to_channel(camera_nr, camera_to_mask_map) or _infer_channel_from_name(video_path)


def _load_mask(mask_path: str) -> tuple[Any, tuple[int, int]]:
    """
    Load mask as single-channel uint8 (0..255).
    Returns (mask, (w, h)).
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")
    h, w = mask.shape[:2]
    return mask, (w, h)


def _resize_mask_if_half(mask, mask_size: Tuple[int, int], target_size: Tuple[int, int]):
    """Resize mask with NEAREST if the target is exactly half of mask dims. Else, return original."""
    mw, mh = mask_size
    tw, th = target_size
    if tw == mw // 2 and th == mh // 2:
        resized = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        return resized, True
    return mask, False


def _ensure_mask_size(mask, target_size: Tuple[int, int]):
    """Apply the same size logic used in your dataset script."""
    h, w = mask.shape[:2]
    tw, th = target_size
    if (w, h) == (tw, th):
        return mask, "match"
    resized, did_half = _resize_mask_if_half(mask, (w, h), (tw, th))
    if did_half:
        return resized, "half"
    # no resize -> will result in no masking (identity)
    return mask, "mismatch"


# ---------- Video processing ----------
def _derive_masked_path(masked_root: str, source_path: str) -> str:
    """
    Keep filenames unique even across different folders by appending a short hash.
    """
    base = os.path.basename(source_path)
    stem, ext = os.path.splitext(base)
    short = hashlib.sha1(os.path.abspath(source_path).encode("utf-8")).hexdigest()[:8]
    out_name = f"{stem}_{short}{ext if ext else '.mp4'}"
    os.makedirs(masked_root, exist_ok=True)
    return os.path.join(masked_root, out_name)


def _pick_fourcc(output_path: str) -> int:
    ext = os.path.splitext(output_path)[1].lower()
    # Keep it simple and reliable; mp4v works broadly, avi uses XVID
    if ext in {".mp4", ".m4v", ".mov"}:
        return cv2.VideoWriter_fourcc(*"mp4v")
    if ext in {".avi"}:
        return cv2.VideoWriter_fourcc(*"XVID")
    # default
    return cv2.VideoWriter_fourcc(*"mp4v")


def _metadata_path(dst: str) -> str:
    return f"{dst}.maskmeta.json"


def _build_mask_signature(
    src_path: str,
    mask_path: str | None,
    strict_half_rule: bool,
) -> Dict[str, Any]:
    mask_mtime_ns = None
    if mask_path and os.path.exists(mask_path):
        mask_mtime_ns = os.stat(mask_path).st_mtime_ns

    return {
        "src_path": os.path.abspath(src_path),
        "mask_path": os.path.abspath(mask_path) if mask_path else None,
        "mask_mtime_ns": mask_mtime_ns,
        "strict_half_rule": bool(strict_half_rule),
    }


def _read_mask_signature(dst: str) -> Dict[str, Any] | None:
    metadata_path = _metadata_path(dst)
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, "r") as metadata_file:
        return json.load(metadata_file)


def _write_mask_signature(dst: str, signature: Dict[str, Any]) -> None:
    metadata_path = _metadata_path(dst)
    with open(metadata_path, "w") as metadata_file:
        json.dump(signature, metadata_file, indent=2, sort_keys=True)


def _should_skip(
    src: str,
    dst: str,
    *,
    mask_path: str | None,
    strict_half_rule: bool,
) -> bool:
    if not os.path.exists(dst):
        return False
    if os.path.getmtime(dst) < os.path.getmtime(src):
        return False
    if mask_path and os.path.exists(mask_path) and os.path.getmtime(dst) < os.path.getmtime(mask_path):
        return False

    current_signature = _build_mask_signature(src, mask_path, strict_half_rule)
    previous_signature = _read_mask_signature(dst)
    return previous_signature == current_signature


def _process_one_video(
    src_path: str,
    dst_path: str,
    mask_path: str | None,
    strict_half_rule: bool = True,
) -> Tuple[str, str, bool]:
    """
    Process a single video. Returns (src, dst, succeeded).
    If mask_path is None, we just copy-encode the video unchanged.
    """
    try:
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {src_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = _pick_fourcc(dst_path)
        out = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise RuntimeError(f"Failed to create writer: {dst_path}")

        # Load mask (if any)
        if mask_path:
            raw_mask, (mw, mh) = _load_mask(mask_path)
            mask, size_state = _ensure_mask_size(raw_mask, (width, height))
            if size_state == "mismatch":
                # Respect your dataset logic: don't resize unless it's exactly half; produce unmodified frames.
                logger.warning(
                    "Mask size mismatch (mask=%sx%s, frame=%sx%s) for %s. Frames will be left unmodified.",
                    mw, mh, width, height, os.path.basename(src_path),
                )
                mask = None
        else:
            mask = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mask is not None:
                # Keep pixels where mask > 0; else black (0)
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                out.write(masked_frame)
            else:
                out.write(frame)

        cap.release()
        out.release()
        _write_mask_signature(
            dst_path,
            _build_mask_signature(src_path, mask_path, strict_half_rule),
        )
        return (src_path, dst_path, True)
    except Exception as e:
        logger.exception("Error masking %s -> %s: %s", src_path, dst_path, e)
        return (src_path, dst_path, False)


# ---------- Public entrypoint ----------
def preprocess_videos(
    config: Dict,
    *,
    reporter: JobReporter | None = None,
    log_progress: bool = False,
) -> List[List[Dict]]:
    """
    - Reads config["video_groups"] and config["masks"].
    - Produces masked copies of every referenced video in parallel (if needed).
    - Returns a new copy of video_groups with paths replaced by masked paths.
    """
    masked_root = config.get("masked_video_folder", "masked_videos")
    os.makedirs(masked_root, exist_ok=True)

    # Where are the masks?
    masks_cfg: Dict[str, str] = config.get("masks", {
        "Ch1": "assets/masks/combined_mask_ch1.png",
        "Ch4": "assets/masks/combined_mask_ch4.png",
        "Ch6": "assets/masks/combined_mask_ch6.png",
        "Ch8": "assets/masks/combined_mask_ch8.png",
    })

    # Optional explicit camera -> channel mapping, e.g. {"1":"Ch1","4":"Ch4","6":"Ch6","8":"Ch8"}
    camera_to_mask_map = config.get("camera_to_mask_map", None)

    groups = config.get("video_groups", [])
    if not groups:
        logger.info("No video_groups present; nothing to mask.")
        return groups

    # Build a unique worklist across all groups
    work_items: Dict[str, Dict] = {}  # by src_path
    for group in groups:
        for item in group:
            src_path = item["path"]
            camera_nr = int(item["camera_nr"])
            ch = _choose_channel(src_path, camera_nr, camera_to_mask_map)
            mask_path = masks_cfg.get(ch) if ch else None
            dst_path = _derive_masked_path(masked_root, src_path)
            work_items[src_path] = {
                "src": src_path,
                "dst": dst_path,
                "mask_path": mask_path,
            }

    # Process in parallel
    max_workers = int(config.get("num_mask_workers", max(os.cpu_count() - 1, 1)))
    strict_half_rule = bool(config.get("mask_strict_half_rule", True))

    to_process = []
    for w in work_items.values():
        if _should_skip(
            w["src"],
            w["dst"],
            mask_path=w["mask_path"],
            strict_half_rule=strict_half_rule,
        ):
            logger.info("Skipping up-to-date masked video: %s", os.path.basename(w["dst"]))
            continue
        to_process.append(w)

    logger.info("Masking %d/%d videos with %d workers...", len(to_process), len(work_items), max_workers)
    progress = StageProgressReporter(
        event_prefix="masking",
        reporter_stage="masking",
        stage_name="mask_videos",
        total=len(to_process),
        log_progress=log_progress,
        reporter=reporter,
        extra_payload={"masked_video_folder": masked_root},
    )
    progress.stage_started()

    if to_process:
        # Use "spawn" explicitly to avoid fork-related issues in multi-threaded runtimes.
        with futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
        ) as ex:
            jobs = [
                ex.submit(_process_one_video, w["src"], w["dst"], w["mask_path"], strict_half_rule)
                for w in to_process
            ]
            completed = 0
            for j in futures.as_completed(jobs):
                src, dst, ok = j.result()
                completed += 1
                progress.step_progress(completed, len(to_process))
                if ok:
                    logger.info("Masked: %s -> %s", os.path.basename(src), os.path.basename(dst))
                else:
                    logger.error("Failed: %s", src)
    progress.stage_completed()

    # Build a new groups object with replaced paths
    new_groups: List[List[Dict]] = []
    for group in groups:
        new_group = []
        for item in group:
            src_path = item["path"]
            masked_path = work_items[src_path]["dst"]
            # Even if masking failed or was skipped, we still point to the masked file if it exists;
            # otherwise we fall back to original.
            use_path = masked_path if os.path.exists(masked_path) else src_path
            new_item = dict(item)
            new_item["path"] = use_path
            new_group.append(new_item)
        new_groups.append(new_group)

    return new_groups
