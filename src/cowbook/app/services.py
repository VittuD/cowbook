from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from cowbook.core.contracts import PipelineConfig
from cowbook.core.runtime import ensure_repo_root_on_path
from cowbook.execution import CancellationToken, JobReporter


def _import_package_module(name: str):
    ensure_repo_root_on_path()
    return importlib.import_module(name)


@dataclass(slots=True)
class ConfigService:
    def load(self, config_path: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.load_file(config_path, overrides=overrides)

    def load_file(self, config_path: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        module = _import_package_module("cowbook.io.config_loader")
        return module.load_config_file(config_path, overrides=overrides)

    def normalize(
        self,
        config: PipelineConfig | dict[str, Any],
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        module = _import_package_module("cowbook.io.config_loader")
        config_mapping = config.to_dict() if isinstance(config, PipelineConfig) else dict(config)
        return module.normalize_config_mapping(config_mapping, overrides=overrides)

    def materialize(
        self,
        config: PipelineConfig | dict[str, Any],
        output_path: str,
        overrides: dict[str, Any] | None = None,
    ) -> str:
        module = _import_package_module("cowbook.io.config_loader")
        return module.write_config_file(config, output_path, overrides=overrides)


@dataclass(slots=True)
class DirectoryService:
    def prepare_output_dirs(self, config: dict[str, Any]):
        module = _import_package_module("cowbook.io.directory_manager")
        return module.prepare_output_dirs(config)

    def clear_output_directory(self, directory_path: str) -> None:
        module = _import_package_module("cowbook.io.directory_manager")
        module.clear_output_directory(directory_path)


@dataclass(slots=True)
class MaskingService:
    def preprocess(
        self,
        config: dict[str, Any],
        *,
        reporter: JobReporter | None = None,
        log_progress: bool = False,
    ):
        module = _import_package_module("cowbook.vision.preprocess_video")
        return module.preprocess_videos(config, reporter=reporter, log_progress=log_progress)


@dataclass(slots=True)
class GroupProcessingService:
    def process_group(
        self,
        group_idx: int,
        video_group: list[dict[str, Any]],
        model_ref: str,
        config: dict[str, Any],
        output_json_folder: str,
        output_image_folder: str,
        reporter: JobReporter | None = None,
        cancellation_token: CancellationToken | None = None,
    ):
        module = _import_package_module("cowbook.workflows.group_processor")
        return module.process_video_group(
            group_idx,
            video_group,
            model_ref,
            config,
            output_json_folder,
            output_image_folder,
            reporter=reporter,
            cancellation_token=cancellation_token,
        )


@dataclass(slots=True)
class VideoService:
    def create_projection_video(
        self,
        image_folder: str,
        output_video_path: str,
        fps: int,
        *,
        reporter: JobReporter | None = None,
        group_idx: int | None = None,
        log_progress: bool = False,
    ) -> None:
        module = _import_package_module("cowbook.io.video_processor")
        module.create_video_from_images(
            image_folder,
            output_video_path,
            fps,
            reporter=reporter,
            group_idx=group_idx,
            log_progress=log_progress,
        )
