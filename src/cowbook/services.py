from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from cowbook.runtime import ensure_repo_root_on_path


def _import_legacy_module(name: str):
    ensure_repo_root_on_path()
    return importlib.import_module(name)


@dataclass(slots=True)
class ConfigService:
    def load(self, config_path: str) -> dict[str, Any]:
        module = _import_legacy_module("config_loader")
        return module.load_config(config_path)


@dataclass(slots=True)
class DirectoryService:
    def prepare_output_dirs(self, config: dict[str, Any]):
        module = _import_legacy_module("directory_manager")
        return module.prepare_output_dirs(config)

    def clear_output_directory(self, directory_path: str) -> None:
        module = _import_legacy_module("directory_manager")
        module.clear_output_directory(directory_path)


@dataclass(slots=True)
class MaskingService:
    def preprocess(self, config: dict[str, Any]):
        module = _import_legacy_module("preprocess_video")
        return module.preprocess_videos(config)


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
    ):
        module = _import_legacy_module("group_processor")
        return module.process_video_group(
            group_idx,
            video_group,
            model_ref,
            config,
            output_json_folder,
            output_image_folder,
        )


@dataclass(slots=True)
class VideoService:
    def create_projection_video(self, image_folder: str, output_video_path: str, fps: int) -> None:
        module = _import_legacy_module("video_processor")
        module.create_video_from_images(image_folder, output_video_path, fps)
