from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _to_float_list(values: list[Any]) -> list[float]:
    return [float(v) for v in values]


def _to_optional_float_list(values: list[Any] | None) -> list[float] | None:
    if values is None:
        return None
    return [float(v) for v in values]


@dataclass(slots=True)
class VideoGroupItem:
    path: str
    camera_nr: int

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "VideoGroupItem":
        return cls(path=str(data["path"]), camera_nr=int(data["camera_nr"]))

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "camera_nr": self.camera_nr}


@dataclass(slots=True)
class TrackingLabel:
    class_id: int | None
    id: int | None
    camera_nr: int | None = None
    local_track_id: int | None = None
    global_id: int | None = None
    det_idx: int | None = None
    real: int | None = None
    src: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TrackingLabel":
        class_id = data.get("class_id")
        det_id = data.get("id")
        camera_nr = data.get("camera_nr")
        local_track_id = data.get("local_track_id")
        global_id = data.get("global_id")
        det_idx = data.get("det_idx")
        real = data.get("real")
        src = data.get("src")
        return cls(
            class_id=int(class_id) if class_id is not None else None,
            id=int(det_id) if det_id is not None else None,
            camera_nr=int(camera_nr) if camera_nr is not None else None,
            local_track_id=int(local_track_id) if local_track_id is not None else None,
            global_id=int(global_id) if global_id is not None else None,
            det_idx=int(det_idx) if det_idx is not None else None,
            real=int(real) if real is not None else None,
            src=str(src) if src is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out = {"class_id": self.class_id}
        if self.id is not None:
            out["id"] = self.id
        if self.camera_nr is not None:
            out["camera_nr"] = self.camera_nr
        if self.local_track_id is not None:
            out["local_track_id"] = self.local_track_id
        if (
            self.camera_nr is not None
            or self.local_track_id is not None
            or self.global_id is not None
        ):
            out["global_id"] = self.global_id
        if self.det_idx is not None:
            out["det_idx"] = self.det_idx
        if self.real is not None:
            out["real"] = self.real
        if self.src is not None:
            out["src"] = self.src
        return out


@dataclass(slots=True)
class TrackingCleanupConfig:
    enabled: bool = False
    conf_threshold: float = 0.15
    nms_mode: str = "hybrid_nms"
    nms_iou: float = 0.75
    footpoint_dist_k: float = 0.18
    footpoint_dist_min_px: float = 10.0
    footpoint_iou_guard: float | None = 0.15
    hybrid_iou_hard: float = 0.92
    hybrid_iou_guard: float = 0.15
    hybrid_footpoint_dist_k: float = 0.18
    hybrid_footpoint_dist_min_px: float = 10.0
    min_area_px: float | None = None
    max_area_px: float | None = None
    min_aspect_ratio: float | None = None
    max_aspect_ratio: float | None = None
    drop_edge_boxes: bool = False
    edge_margin_px: int = 10
    roi: list[list[float]] | None = None
    two_pass_prune_short_tracks: bool = False
    min_track_length: int = 30
    min_track_total_observations: int | None = None
    short_track_gap_tolerance: int = 6
    postprocess_smoothing: bool = False
    smoothing_alpha: float = 0.65
    gap_fill_max_frames: int = 3
    max_center_speed_px_per_frame: float = 80.0
    max_relative_area_change: float = 0.80
    max_relative_aspect_change: float = 0.80

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "TrackingCleanupConfig":
        data = data or {}
        roi_raw = data.get("roi")
        roi = None
        if roi_raw is not None:
            roi = [_to_float_list(point) for point in roi_raw]
        return cls(
            enabled=bool(data.get("enabled", False)),
            conf_threshold=float(data.get("conf_threshold", 0.15)),
            nms_mode=str(data.get("nms_mode", "hybrid_nms")),
            nms_iou=float(data.get("nms_iou", 0.75)),
            footpoint_dist_k=float(data.get("footpoint_dist_k", 0.18)),
            footpoint_dist_min_px=float(data.get("footpoint_dist_min_px", 10.0)),
            footpoint_iou_guard=(
                float(data["footpoint_iou_guard"])
                if data.get("footpoint_iou_guard") is not None
                else None
            ),
            hybrid_iou_hard=float(data.get("hybrid_iou_hard", 0.92)),
            hybrid_iou_guard=float(data.get("hybrid_iou_guard", 0.15)),
            hybrid_footpoint_dist_k=float(data.get("hybrid_footpoint_dist_k", 0.18)),
            hybrid_footpoint_dist_min_px=float(data.get("hybrid_footpoint_dist_min_px", 10.0)),
            min_area_px=(
                float(data["min_area_px"]) if data.get("min_area_px") is not None else None
            ),
            max_area_px=(
                float(data["max_area_px"]) if data.get("max_area_px") is not None else None
            ),
            min_aspect_ratio=(
                float(data["min_aspect_ratio"])
                if data.get("min_aspect_ratio") is not None
                else None
            ),
            max_aspect_ratio=(
                float(data["max_aspect_ratio"])
                if data.get("max_aspect_ratio") is not None
                else None
            ),
            drop_edge_boxes=bool(data.get("drop_edge_boxes", False)),
            edge_margin_px=int(data.get("edge_margin_px", 10)),
            roi=roi,
            two_pass_prune_short_tracks=bool(data.get("two_pass_prune_short_tracks", False)),
            min_track_length=int(data.get("min_track_length", 30)),
            min_track_total_observations=(
                int(data["min_track_total_observations"])
                if data.get("min_track_total_observations") is not None
                else None
            ),
            short_track_gap_tolerance=int(data.get("short_track_gap_tolerance", 6)),
            postprocess_smoothing=bool(data.get("postprocess_smoothing", False)),
            smoothing_alpha=float(data.get("smoothing_alpha", 0.65)),
            gap_fill_max_frames=int(data.get("gap_fill_max_frames", 3)),
            max_center_speed_px_per_frame=float(
                data.get("max_center_speed_px_per_frame", 80.0)
            ),
            max_relative_area_change=float(data.get("max_relative_area_change", 0.80)),
            max_relative_aspect_change=float(data.get("max_relative_aspect_change", 0.80)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "conf_threshold": self.conf_threshold,
            "nms_mode": self.nms_mode,
            "nms_iou": self.nms_iou,
            "footpoint_dist_k": self.footpoint_dist_k,
            "footpoint_dist_min_px": self.footpoint_dist_min_px,
            "footpoint_iou_guard": self.footpoint_iou_guard,
            "hybrid_iou_hard": self.hybrid_iou_hard,
            "hybrid_iou_guard": self.hybrid_iou_guard,
            "hybrid_footpoint_dist_k": self.hybrid_footpoint_dist_k,
            "hybrid_footpoint_dist_min_px": self.hybrid_footpoint_dist_min_px,
            "min_area_px": self.min_area_px,
            "max_area_px": self.max_area_px,
            "min_aspect_ratio": self.min_aspect_ratio,
            "max_aspect_ratio": self.max_aspect_ratio,
            "drop_edge_boxes": self.drop_edge_boxes,
            "edge_margin_px": self.edge_margin_px,
            "roi": self.roi,
            "two_pass_prune_short_tracks": self.two_pass_prune_short_tracks,
            "min_track_length": self.min_track_length,
            "min_track_total_observations": self.min_track_total_observations,
            "short_track_gap_tolerance": self.short_track_gap_tolerance,
            "postprocess_smoothing": self.postprocess_smoothing,
            "smoothing_alpha": self.smoothing_alpha,
            "gap_fill_max_frames": self.gap_fill_max_frames,
            "max_center_speed_px_per_frame": self.max_center_speed_px_per_frame,
            "max_relative_area_change": self.max_relative_area_change,
            "max_relative_aspect_change": self.max_relative_aspect_change,
        }


@dataclass(slots=True)
class Detections:
    xyxy: list[list[float]]
    centroids: list[list[float]] | None = None
    projected_centroids: list[list[float] | None] | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Detections":
        xyxy = [_to_float_list(box) for box in data.get("xyxy", []) or []]
        centroids_raw = data.get("centroids")
        projected_raw = data.get("projected_centroids")
        centroids = None
        projected = None
        if centroids_raw is not None:
            centroids = [_to_float_list(point) for point in centroids_raw]
        if projected_raw is not None:
            projected = [_to_optional_float_list(point) for point in projected_raw]
        return cls(xyxy=xyxy, centroids=centroids, projected_centroids=projected)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"xyxy": self.xyxy}
        if self.centroids is not None:
            out["centroids"] = self.centroids
        if self.projected_centroids is not None:
            out["projected_centroids"] = self.projected_centroids
        return out


@dataclass(slots=True)
class TrackingFrame:
    frame_id: int
    detections: Detections
    labels: list[TrackingLabel] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TrackingFrame":
        return cls(
            frame_id=int(data["frame_id"]),
            detections=Detections.from_mapping(data.get("detections", {}) or {}),
            labels=[TrackingLabel.from_mapping(label) for label in data.get("labels", []) or []],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "detections": self.detections.to_dict(),
            "labels": [label.to_dict() for label in self.labels],
        }


@dataclass(slots=True)
class TrackingDocument:
    frames: list[TrackingFrame]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TrackingDocument":
        return cls(frames=[TrackingFrame.from_mapping(frame) for frame in data.get("frames", []) or []])

    def to_dict(self) -> dict[str, Any]:
        return {"frames": [frame.to_dict() for frame in self.frames]}


DEFAULT_MASKS = {
    "Ch1": "assets/masks/combined_mask_ch1.png",
    "Ch4": "assets/masks/combined_mask_ch4.png",
    "Ch6": "assets/masks/combined_mask_ch6.png",
    "Ch8": "assets/masks/combined_mask_ch8.png",
}


@dataclass(slots=True)
class PipelineConfig:
    """Typed runtime configuration for a cowbook pipeline run."""

    model_path: str = "models/best.pt"
    fps: int = 6
    save_tracking_video: bool = False
    create_projection_video: bool = True
    video_groups: list[list[VideoGroupItem]] = field(default_factory=list)
    calibration_file: str = "assets/calibration/camera_system.json"
    runtime_root: str = "var"
    run_name: str = "default"
    output_root: str = "var/runs/default"
    num_plot_workers: int = 0
    output_image_format: str = "jpg"
    output_image_folder: str = "var/runs/default/frames"
    output_video_folder: str = "var/runs/default/videos"
    output_json_folder: str = "var/runs/default/json"
    output_video_filename: str = "combined_projection.mp4"
    convert_to_csv: bool = True
    clean_frames_after_video: bool = True
    tracking_concurrency: int = 1
    log_progress: bool = False
    tracking_cleanup: TrackingCleanupConfig = field(default_factory=TrackingCleanupConfig)
    mask_videos: bool = False
    masked_video_folder: str = "var/cache/masked_videos"
    num_mask_workers: int = 0
    mask_strict_half_rule: bool = True
    masks: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MASKS))
    camera_to_mask_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PipelineConfig":
        groups = [
            [VideoGroupItem.from_mapping(item) for item in group]
            for group in data.get("video_groups", []) or []
        ]
        return cls(
            model_path=str(data.get("model_path", "models/best.pt")),
            fps=int(data.get("fps", 6)),
            save_tracking_video=bool(data.get("save_tracking_video", False)),
            create_projection_video=bool(data.get("create_projection_video", True)),
            video_groups=groups,
            calibration_file=str(
                data.get("calibration_file", "assets/calibration/camera_system.json")
            ),
            runtime_root=str(data.get("runtime_root", "var")),
            run_name=str(data.get("run_name", "default")),
            output_root=str(data.get("output_root", "var/runs/default")),
            num_plot_workers=int(data.get("num_plot_workers", 0)),
            output_image_format=str(data.get("output_image_format", "jpg")),
            output_image_folder=str(data.get("output_image_folder", "var/runs/default/frames")),
            output_video_folder=str(data.get("output_video_folder", "var/runs/default/videos")),
            output_json_folder=str(data.get("output_json_folder", "var/runs/default/json")),
            output_video_filename=str(data.get("output_video_filename", "combined_projection.mp4")),
            convert_to_csv=bool(data.get("convert_to_csv", True)),
            clean_frames_after_video=bool(data.get("clean_frames_after_video", True)),
            tracking_concurrency=int(data.get("tracking_concurrency", 1)),
            log_progress=bool(data.get("log_progress", False)),
            tracking_cleanup=TrackingCleanupConfig.from_mapping(data.get("tracking_cleanup")),
            mask_videos=bool(data.get("mask_videos", False)),
            masked_video_folder=str(data.get("masked_video_folder", "var/cache/masked_videos")),
            num_mask_workers=int(data.get("num_mask_workers", 0)),
            mask_strict_half_rule=bool(data.get("mask_strict_half_rule", True)),
            masks={str(k): str(v) for k, v in (data.get("masks", DEFAULT_MASKS) or {}).items()},
            camera_to_mask_map={
                str(k): str(v) for k, v in (data.get("camera_to_mask_map", {}) or {}).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "fps": self.fps,
            "save_tracking_video": self.save_tracking_video,
            "create_projection_video": self.create_projection_video,
            "video_groups": [[item.to_dict() for item in group] for group in self.video_groups],
            "calibration_file": self.calibration_file,
            "runtime_root": self.runtime_root,
            "run_name": self.run_name,
            "output_root": self.output_root,
            "num_plot_workers": self.num_plot_workers,
            "output_image_format": self.output_image_format,
            "output_image_folder": self.output_image_folder,
            "output_video_folder": self.output_video_folder,
            "output_json_folder": self.output_json_folder,
            "output_video_filename": self.output_video_filename,
            "convert_to_csv": self.convert_to_csv,
            "clean_frames_after_video": self.clean_frames_after_video,
            "tracking_concurrency": self.tracking_concurrency,
            "log_progress": self.log_progress,
            "tracking_cleanup": self.tracking_cleanup.to_dict(),
            "mask_videos": self.mask_videos,
            "masked_video_folder": self.masked_video_folder,
            "num_mask_workers": self.num_mask_workers,
            "mask_strict_half_rule": self.mask_strict_half_rule,
            "masks": dict(self.masks),
            "camera_to_mask_map": dict(self.camera_to_mask_map),
        }


@dataclass(slots=True)
class RunRequest:
    """Typed programmatic submission contract for one pipeline run."""

    config_path: str | None = None
    config: PipelineConfig | dict[str, Any] | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_path = self.config_path is not None
        has_config = self.config is not None
        if has_path == has_config:
            raise ValueError("Exactly one of 'config_path' or 'config' must be provided.")
        if self.config_path is not None:
            self.config_path = str(self.config_path)
        if self.config is not None and not isinstance(self.config, (PipelineConfig, dict)):
            raise TypeError("'config' must be a PipelineConfig or a mapping.")
        self.overrides = dict(self.overrides)

    @property
    def config_reference(self) -> str:
        """Return a stable config label for job snapshots and events."""

        return self.config_path or "<in-memory>"
