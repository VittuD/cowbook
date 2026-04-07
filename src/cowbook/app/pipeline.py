from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from cowbook.app.services import (
    ConfigService,
    DirectoryService,
    GroupProcessingService,
    MaskingService,
    VideoService,
)
from cowbook.execution import (
    CancellationToken,
    CompositeObserver,
    InMemoryJobStore,
    JobCancelledError,
    JobObserver,
    JobReporter,
    JobRun,
    new_job_id,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineRunner:
    """Synchronous orchestration entrypoint for one pipeline run.

    The runner loads config, prepares output folders, optionally applies masking,
    processes all configured groups, renders the projection video, and emits
    structured execution events through the attached observer.
    """

    config_service: ConfigService = field(default_factory=ConfigService)
    directory_service: DirectoryService = field(default_factory=DirectoryService)
    masking_service: MaskingService = field(default_factory=MaskingService)
    group_processing_service: GroupProcessingService = field(default_factory=GroupProcessingService)
    video_service: VideoService = field(default_factory=VideoService)

    def run(
        self,
        config_path: str,
        overrides: dict[str, object] | None = None,
        *,
        observer: JobObserver | None = None,
        job_id: str | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> JobRun | None:
        """Execute one pipeline run from a config file.

        Args:
            config_path: Path to the JSON configuration file.
            overrides: Optional runtime overrides applied after config load.
            observer: Optional observer for structured job events.
            job_id: Optional externally supplied job identifier.
            cancellation_token: Optional cooperative cancellation token.

        Returns:
            A final :class:`cowbook.execution.models.JobRun` snapshot when available.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

        run_store = InMemoryJobStore()
        observers = [run_store]
        if observer is not None:
            observers.append(observer)
        reporter = JobReporter(
            job_id=job_id or new_job_id(),
            config_path=config_path,
            observer=CompositeObserver(observers),
        )
        reporter.emit(
            "job_started",
            status="running",
            stage="config",
            payload={"overrides": dict(overrides or {})},
        )
        if self._cancel_if_requested(reporter, cancellation_token, stage="config"):
            return run_store.get(reporter.job_id)

        config = self.config_service.load(config_path, overrides=overrides)
        if not config:
            logger.error("Failed to load config from %s", config_path)
            reporter.emit(
                "job_failed",
                status="failed",
                stage="config",
                message=f"Failed to load config from {config_path}",
                payload={"error": "config_load_failed"},
            )
            return run_store.get(reporter.job_id)

        reporter.emit(
            "config_loaded",
            stage="config",
            payload={
                "run_name": config.get("run_name"),
                "runtime_root": config.get("runtime_root"),
                "output_root": config.get("output_root"),
            },
        )

        (
            output_image_folder,
            output_video_folder,
            output_json_folder,
            _output_masked_folder,
        ) = self.directory_service.prepare_output_dirs(config)
        reporter.emit("output_dirs_prepared", stage="setup")
        reporter.artifact("output_dir", output_image_folder, metadata={"role": "frames"})
        reporter.artifact("output_dir", output_video_folder, metadata={"role": "videos"})
        reporter.artifact("output_dir", output_json_folder, metadata={"role": "json"})

        self.directory_service.clear_output_directory(output_image_folder)
        reporter.emit(
            "output_frames_cleared",
            stage="setup",
            payload={"path": output_image_folder},
        )
        if self._cancel_if_requested(reporter, cancellation_token, stage="setup"):
            return run_store.get(reporter.job_id)

        groups = config.get("video_groups", [])
        reporter.emit("groups_discovered", stage="groups", payload={"count": len(groups)})
        if config.get("mask_videos", False):
            logger.info("mask_videos=true -> generating masked copies before inference...")
            reporter.emit(
                "masking_started",
                stage="masking",
                payload={"group_count": len(groups)},
            )
            try:
                groups = self.masking_service.preprocess(config)
                logger.info("Masked video groups prepared.")
                reporter.emit(
                    "masking_completed",
                    stage="masking",
                    payload={"group_count": len(groups)},
                )
            except Exception as exc:
                logger.exception("Video masking failed. Falling back to original videos: %s", exc)
                reporter.emit(
                    "masking_failed",
                    stage="masking",
                    message="Video masking failed. Falling back to original videos.",
                    payload={"error": str(exc), "fallback": "original_videos"},
                )
        if self._cancel_if_requested(reporter, cancellation_token, stage="groups"):
            return run_store.get(reporter.job_id)

        if not groups:
            logger.warning("No video groups specified in config.")
            reporter.emit(
                "groups_empty",
                stage="groups",
                message="No video groups specified in config.",
            )
        else:
            for idx, video_group in enumerate(groups, start=1):
                if self._cancel_if_requested(reporter, cancellation_token, stage="group", group_idx=idx):
                    return run_store.get(reporter.job_id)
                logger.info("=== Group %d/%d ===", idx, len(groups))
                try:
                    self.group_processing_service.process_group(
                        idx,
                        video_group,
                        config["model_path"],
                        config,
                        output_json_folder,
                        output_image_folder,
                        reporter=reporter,
                        cancellation_token=cancellation_token,
                    )
                except JobCancelledError:
                    reporter.emit(
                        "group_cancelled",
                        status="cancelling",
                        stage="group",
                        group_idx=idx,
                        message=f"Group {idx} cancelled.",
                    )
                    self._cancel_if_requested(reporter, cancellation_token, stage="group", group_idx=idx)
                    return run_store.get(reporter.job_id)
                except Exception as exc:
                    logger.exception("Group %d failed: %s", idx, exc)
                    reporter.emit(
                        "group_failed",
                        stage="group",
                        group_idx=idx,
                        message=f"Group {idx} failed.",
                        payload={"error": str(exc)},
                    )

        if config.get("create_projection_video", True):
            if self._cancel_if_requested(reporter, cancellation_token, stage="video"):
                return run_store.get(reporter.job_id)
            output_video_path = os.path.join(
                output_video_folder,
                config.get("output_video_filename", "combined_projection.mp4"),
            )
            fps = config["fps"]
            logger.info("Generating combined projection video at %d FPS -> %s", fps, output_video_path)
            reporter.emit(
                "video_started",
                stage="video",
                payload={"path": output_video_path, "fps": fps},
            )
            try:
                self.video_service.create_projection_video(output_image_folder, output_video_path, fps)
                logger.info("Combined projection video generated successfully.")
                reporter.artifact("projection_video", output_video_path)
                reporter.emit(
                    "video_completed",
                    stage="video",
                    payload={"path": output_video_path, "fps": fps},
                )
                if config.get("clean_frames_after_video", True):
                    logger.info("Cleaning up intermediate frames in %s ...", output_image_folder)
                    self.directory_service.clear_output_directory(output_image_folder)
                    reporter.emit(
                        "output_frames_cleared",
                        stage="video",
                        payload={"path": output_image_folder, "reason": "post_video_cleanup"},
                    )
                else:
                    logger.info("Keeping intermediate frames (clean_frames_after_video=false).")
                    reporter.emit(
                        "output_frames_retained",
                        stage="video",
                        payload={"path": output_image_folder},
                    )
            except Exception as exc:
                logger.exception("Failed to create video: %s", exc)
                reporter.emit(
                    "video_failed",
                    stage="video",
                    message="Failed to create projection video.",
                    payload={"error": str(exc), "path": output_video_path},
                )

        snapshot = run_store.get(reporter.job_id)
        reporter.emit(
            "job_completed",
            status="completed",
            stage="pipeline",
            payload={"had_errors": bool(snapshot.errors) if snapshot else False},
        )
        return run_store.get(reporter.job_id)

    def _cancel_if_requested(
        self,
        reporter: JobReporter,
        cancellation_token: CancellationToken | None,
        *,
        stage: str,
        group_idx: int | None = None,
    ) -> bool:
        if cancellation_token is None or not cancellation_token.is_cancelled():
            return False
        reporter.emit(
            "job_cancelled",
            status="cancelled",
            stage=stage,
            group_idx=group_idx,
            message="Job cancelled.",
        )
        return True
