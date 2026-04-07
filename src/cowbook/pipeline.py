from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from cowbook.services import (
    ConfigService,
    DirectoryService,
    GroupProcessingService,
    MaskingService,
    VideoService,
)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineRunner:
    config_service: ConfigService = field(default_factory=ConfigService)
    directory_service: DirectoryService = field(default_factory=DirectoryService)
    masking_service: MaskingService = field(default_factory=MaskingService)
    group_processing_service: GroupProcessingService = field(default_factory=GroupProcessingService)
    video_service: VideoService = field(default_factory=VideoService)

    def run(self, config_path: str, overrides: dict[str, object] | None = None) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

        config = self.config_service.load(config_path, overrides=overrides)
        if not config:
            logger.error("Failed to load config from %s", config_path)
            return

        (
            output_image_folder,
            output_video_folder,
            output_json_folder,
            _output_masked_folder,
        ) = self.directory_service.prepare_output_dirs(config)

        self.directory_service.clear_output_directory(output_image_folder)

        groups = config.get("video_groups", [])
        if config.get("mask_videos", False):
            logger.info("mask_videos=true -> generating masked copies before inference...")
            try:
                groups = self.masking_service.preprocess(config)
                logger.info("Masked video groups prepared.")
            except Exception as exc:
                logger.exception("Video masking failed. Falling back to original videos: %s", exc)

        if not groups:
            logger.warning("No video groups specified in config.")
        else:
            for idx, video_group in enumerate(groups, start=1):
                logger.info("=== Group %d/%d ===", idx, len(groups))
                try:
                    self.group_processing_service.process_group(
                        idx,
                        video_group,
                        config["model_path"],
                        config,
                        output_json_folder,
                        output_image_folder,
                    )
                except Exception as exc:
                    logger.exception("Group %d failed: %s", idx, exc)

        if config.get("create_projection_video", True):
            output_video_path = os.path.join(
                output_video_folder,
                config.get("output_video_filename", "combined_projection.mp4"),
            )
            fps = config["fps"]
            logger.info("Generating combined projection video at %d FPS -> %s", fps, output_video_path)
            try:
                self.video_service.create_projection_video(output_image_folder, output_video_path, fps)
                logger.info("Combined projection video generated successfully.")
                if config.get("clean_frames_after_video", True):
                    logger.info("Cleaning up intermediate frames in %s ...", output_image_folder)
                    self.directory_service.clear_output_directory(output_image_folder)
                else:
                    logger.info("Keeping intermediate frames (clean_frames_after_video=false).")
            except Exception as exc:
                logger.exception("Failed to create video: %s", exc)
