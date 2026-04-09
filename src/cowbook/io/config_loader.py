import copy
import json
import os
from pathlib import Path
from typing import Any

from cowbook.core.contracts import PipelineConfig
from cowbook.io.directory_manager import resolve_output_paths
from cowbook.io.json_utils import dump_path_pretty, load_path
from cowbook.vision.calibration import default_calibration_file


def _copy_config_mapping(config: PipelineConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, PipelineConfig):
        return config.to_dict()
    if not isinstance(config, dict):
        raise TypeError("'config' must be a PipelineConfig or a mapping.")
    return copy.deepcopy(config)


def _normalize_positive_int(config: dict, key: str) -> None:
    try:
        value = int(config[key])
    except Exception as e:
        raise ValueError(f"'{key}' must be a positive integer (got {config.get(key)!r}).") from e
    if value < 1:
        raise ValueError(f"'{key}' must be >= 1 (got {value!r}).")
    config[key] = value


def _optional_float(value, key: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception as e:
        raise ValueError(f"'{key}' must be a number or null (got {value!r}).") from e


def _normalize_tracking_cleanup(config: dict) -> None:
    cleanup = config.get("tracking_cleanup")
    if cleanup is None:
        cleanup = {}
    if not isinstance(cleanup, dict):
        raise ValueError("'tracking_cleanup' must be an object when provided.")

    defaults = {
        "enabled": False,
        "conf_threshold": 0.15,
        "nms_mode": "hybrid_nms",
        "nms_iou": 0.75,
        "footpoint_dist_k": 0.18,
        "footpoint_dist_min_px": 10.0,
        "footpoint_iou_guard": 0.15,
        "hybrid_iou_hard": 0.92,
        "hybrid_iou_guard": 0.15,
        "hybrid_footpoint_dist_k": 0.18,
        "hybrid_footpoint_dist_min_px": 10.0,
        "min_area_px": None,
        "max_area_px": None,
        "min_aspect_ratio": None,
        "max_aspect_ratio": None,
        "drop_edge_boxes": False,
        "edge_margin_px": 10,
        "roi": None,
        "two_pass_prune_short_tracks": False,
        "min_track_length": 30,
        "min_track_total_observations": None,
        "short_track_gap_tolerance": 6,
        "postprocess_gap_fill": False,
        "postprocess_smoothing": False,
        "smoothing_alpha": 0.65,
        "gap_fill_max_frames": 3,
        "max_center_speed_px_per_frame": 80.0,
        "max_relative_area_change": 0.80,
        "max_relative_aspect_change": 0.80,
    }
    cleanup = {**defaults, **cleanup}

    cleanup["enabled"] = bool(cleanup["enabled"])
    cleanup["drop_edge_boxes"] = bool(cleanup["drop_edge_boxes"])
    cleanup["two_pass_prune_short_tracks"] = bool(cleanup["two_pass_prune_short_tracks"])
    cleanup["postprocess_gap_fill"] = bool(cleanup["postprocess_gap_fill"])
    cleanup["postprocess_smoothing"] = bool(cleanup["postprocess_smoothing"])

    for key in (
        "conf_threshold",
        "nms_iou",
        "footpoint_dist_k",
        "footpoint_dist_min_px",
        "hybrid_iou_hard",
        "hybrid_iou_guard",
        "hybrid_footpoint_dist_k",
        "hybrid_footpoint_dist_min_px",
        "smoothing_alpha",
        "max_center_speed_px_per_frame",
        "max_relative_area_change",
        "max_relative_aspect_change",
    ):
        cleanup[key] = float(cleanup[key])

    cleanup["footpoint_iou_guard"] = _optional_float(
        cleanup.get("footpoint_iou_guard"), "tracking_cleanup.footpoint_iou_guard"
    )
    cleanup["min_area_px"] = _optional_float(
        cleanup.get("min_area_px"), "tracking_cleanup.min_area_px"
    )
    cleanup["max_area_px"] = _optional_float(
        cleanup.get("max_area_px"), "tracking_cleanup.max_area_px"
    )
    cleanup["min_aspect_ratio"] = _optional_float(
        cleanup.get("min_aspect_ratio"), "tracking_cleanup.min_aspect_ratio"
    )
    cleanup["max_aspect_ratio"] = _optional_float(
        cleanup.get("max_aspect_ratio"), "tracking_cleanup.max_aspect_ratio"
    )
    if cleanup.get("min_track_total_observations") is None:
        cleanup["min_track_total_observations"] = None
    else:
        try:
            cleanup["min_track_total_observations"] = int(cleanup["min_track_total_observations"])
        except Exception as e:
            raise ValueError(
                "tracking_cleanup.min_track_total_observations must be an integer or null."
            ) from e

    try:
        cleanup["edge_margin_px"] = int(cleanup["edge_margin_px"])
        cleanup["min_track_length"] = int(cleanup["min_track_length"])
        cleanup["short_track_gap_tolerance"] = int(cleanup["short_track_gap_tolerance"])
        cleanup["gap_fill_max_frames"] = int(cleanup["gap_fill_max_frames"])
    except Exception as e:
        raise ValueError("tracking_cleanup integer fields must be integers.") from e

    if cleanup["nms_mode"] not in {"iou_nms", "footpoint_nms", "hybrid_nms"}:
        raise ValueError(
            "tracking_cleanup.nms_mode must be one of 'iou_nms', 'footpoint_nms', or 'hybrid_nms'."
        )
    if cleanup["conf_threshold"] < 0:
        raise ValueError("tracking_cleanup.conf_threshold must be >= 0.")
    if cleanup["nms_iou"] <= 0 or cleanup["nms_iou"] > 1:
        raise ValueError("tracking_cleanup.nms_iou must be in (0, 1].")
    if cleanup["edge_margin_px"] < 0:
        raise ValueError("tracking_cleanup.edge_margin_px must be >= 0.")
    if cleanup["min_track_length"] < 1:
        raise ValueError("tracking_cleanup.min_track_length must be >= 1.")
    if (
        cleanup["min_track_total_observations"] is not None
        and cleanup["min_track_total_observations"] < 1
    ):
        raise ValueError("tracking_cleanup.min_track_total_observations must be >= 1 when provided.")
    if cleanup["short_track_gap_tolerance"] < 0:
        raise ValueError("tracking_cleanup.short_track_gap_tolerance must be >= 0.")
    if cleanup["gap_fill_max_frames"] < 0:
        raise ValueError("tracking_cleanup.gap_fill_max_frames must be >= 0.")
    if cleanup["smoothing_alpha"] <= 0 or cleanup["smoothing_alpha"] >= 1:
        raise ValueError("tracking_cleanup.smoothing_alpha must be in (0, 1).")
    if cleanup["max_center_speed_px_per_frame"] < 0:
        raise ValueError("tracking_cleanup.max_center_speed_px_per_frame must be >= 0.")
    if cleanup["max_relative_area_change"] < 0:
        raise ValueError("tracking_cleanup.max_relative_area_change must be >= 0.")
    if cleanup["max_relative_aspect_change"] < 0:
        raise ValueError("tracking_cleanup.max_relative_aspect_change must be >= 0.")
    if cleanup["min_area_px"] is not None and cleanup["min_area_px"] < 0:
        raise ValueError("tracking_cleanup.min_area_px must be >= 0.")
    if cleanup["max_area_px"] is not None and cleanup["max_area_px"] < 0:
        raise ValueError("tracking_cleanup.max_area_px must be >= 0.")
    if (
        cleanup["min_area_px"] is not None
        and cleanup["max_area_px"] is not None
        and cleanup["min_area_px"] > cleanup["max_area_px"]
    ):
        raise ValueError("tracking_cleanup.min_area_px must be <= tracking_cleanup.max_area_px.")
    if (
        cleanup["min_aspect_ratio"] is not None
        and cleanup["max_aspect_ratio"] is not None
        and cleanup["min_aspect_ratio"] > cleanup["max_aspect_ratio"]
    ):
        raise ValueError(
            "tracking_cleanup.min_aspect_ratio must be <= tracking_cleanup.max_aspect_ratio."
        )

    roi = cleanup.get("roi")
    if roi is not None:
        if not isinstance(roi, list) or len(roi) < 3:
            raise ValueError("tracking_cleanup.roi must be a polygon with at least 3 points.")
        normalized_roi = []
        for point in roi:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("tracking_cleanup.roi points must be [x, y] pairs.")
            normalized_roi.append([float(point[0]), float(point[1])])
        cleanup["roi"] = normalized_roi

    config["tracking_cleanup"] = cleanup


def normalize_config_mapping(
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a config mapping into Cowbook's validated runtime shape."""

    config = _copy_config_mapping(config)

    # ---- Defaults (centralized) ----
    config.setdefault("model_path", "models/best.pt")
    config.setdefault("fps", 6)
    config.setdefault("save_tracking_video", False)
    config.setdefault("create_projection_video", True)
    config.setdefault("video_groups", [])
    config.setdefault("calibration_file", default_calibration_file())
    config.setdefault("runtime_root", "var")
    config.setdefault("run_name", "default")
    # Parallel rendering & image format
    config.setdefault("num_plot_workers", max(1, os.cpu_count() - 1) if hasattr(os, 'cpu_count') else 0)
    config.setdefault("output_image_format", "jpg")
    # Output directories & filename
    config.setdefault("output_video_filename", "combined_projection.mp4")
    # CSV conversion
    config.setdefault("convert_to_csv", True)
    # Clean-up frames after assembling the video (default ON)
    config.setdefault("clean_frames_after_video", True)
    # Tracking concurrency (default to 1 to avoid GPU contention)
    config.setdefault("tracking_concurrency", 1)
    config.setdefault("log_progress", False)
    config.setdefault("tracking_cleanup", {})
    # ---- Masking at inference ----
    config.setdefault("mask_videos", False)
    config.setdefault("num_mask_workers", max(1, os.cpu_count() - 1) if hasattr(os, 'cpu_count') else 0)
    config.setdefault("mask_strict_half_rule", True)
    # Per-channel mask image paths
    config.setdefault("masks", {
        "Ch1": "assets/masks/combined_mask_ch1.png",
        "Ch4": "assets/masks/combined_mask_ch4.png",
        "Ch6": "assets/masks/combined_mask_ch6.png",
        "Ch8": "assets/masks/combined_mask_ch8.png",
    })
    config.setdefault("camera_to_mask_map", {})

    # ---- Apply optional overrides (from CLI or caller) ----
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                config[k] = v

    # ---- Normalization ----
    try:
        config["fps"] = int(config["fps"])
    except Exception as e:
        raise ValueError(f"'fps' must be an integer (got {config.get('fps')!r}).") from e

    _normalize_positive_int(config, "tracking_concurrency")
    config["log_progress"] = bool(config.get("log_progress", False))
    _normalize_tracking_cleanup(config)

    fmt = str(config.get("output_image_format", "jpg")).lower()
    if fmt not in {"png", "jpg", "jpeg"}:
        raise ValueError(f"Invalid output_image_format: {fmt}. Use 'png' or 'jpg'.")
    config["output_image_format"] = "jpg" if fmt in {"jpg", "jpeg"} else "png"

    (
        runtime_root,
        output_root,
        output_image_folder,
        output_video_folder,
        output_json_folder,
        output_masked_folder,
    ) = resolve_output_paths(config)
    config["runtime_root"] = runtime_root
    config["output_root"] = output_root
    config["output_image_folder"] = output_image_folder
    config["output_video_folder"] = output_video_folder
    config["output_json_folder"] = output_json_folder
    config["masked_video_folder"] = output_masked_folder

    groups = config["video_groups"]
    if not isinstance(groups, list):
        raise ValueError("'video_groups' must be a list of groups.")

    for gi, group in enumerate(groups, start=1):
        if not isinstance(group, list):
            raise ValueError(f"Group {gi} must be a list.")
        num_videos = len(group)
        if num_videos == 0 or num_videos > 4:
            raise ValueError("Each video group must contain between 1 and 4 videos.")

        camera_ids = []
        for vi, video in enumerate(group, start=1):
            if not isinstance(video, dict):
                raise ValueError(f"Group {gi}, item {vi} must be an object with 'path' and 'camera_nr'.")
            if "path" not in video or "camera_nr" not in video:
                raise ValueError(f"Group {gi}, item {vi} missing 'path' or 'camera_nr'.")
            try:
                video["camera_nr"] = int(video["camera_nr"])
            except Exception as e:
                raise ValueError(f"Group {gi}, item {vi} has non-integer camera_nr.") from e
            camera_ids.append(video["camera_nr"])

        if len(set(camera_ids)) < num_videos:
            raise ValueError(f"Each video in group {gi} must be associated with a unique camera.")

    return PipelineConfig.from_mapping(config).to_dict()


def load_config_file(
    config_path: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load, normalize, and validate a config file into Cowbook's runtime shape."""

    config = load_path(config_path)
    return normalize_config_mapping(config, overrides=overrides)


def write_config_file(
    config: PipelineConfig | dict[str, Any],
    output_path: str,
    overrides: dict[str, Any] | None = None,
) -> str:
    """Write a normalized config mapping to disk and return the output path."""

    normalized = normalize_config_mapping(config, overrides=overrides)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dump_path_pretty(destination, normalized, trailing_newline=True)
    return str(destination)
