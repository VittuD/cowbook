from __future__ import annotations

from dataclasses import dataclass, field

from cowbook.core.contracts import PipelineConfig
from cowbook.execution.models import JobArtifact, JobRun


def _unique_paths(artifacts: list[JobArtifact], kind: str) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for artifact in artifacts:
        if artifact.kind != kind or artifact.path in seen:
            continue
        seen.add(artifact.path)
        paths.append(artifact.path)
    return paths


@dataclass(slots=True)
class RunResult:
    """Normalized package-facing summary for one completed pipeline request."""

    job_run: JobRun
    resolved_config: PipelineConfig | None = None
    materialized_config_path: str | None = None
    output_root: str | None = None
    output_image_folder: str | None = None
    output_video_folder: str | None = None
    output_json_folder: str | None = None
    projection_video_path: str | None = None
    tracking_json_paths: list[str] = field(default_factory=list)
    processed_json_paths: list[str] = field(default_factory=list)
    merged_json_paths: list[str] = field(default_factory=list)
    csv_paths: list[str] = field(default_factory=list)

    @property
    def job_id(self) -> str:
        return self.job_run.job_id

    @property
    def status(self) -> str:
        return self.job_run.status

    def to_dict(self) -> dict[str, object]:
        return {
            "job_run": self.job_run.to_dict(),
            "resolved_config": self.resolved_config.to_dict() if self.resolved_config is not None else None,
            "materialized_config_path": self.materialized_config_path,
            "output_root": self.output_root,
            "output_image_folder": self.output_image_folder,
            "output_video_folder": self.output_video_folder,
            "output_json_folder": self.output_json_folder,
            "projection_video_path": self.projection_video_path,
            "tracking_json_paths": list(self.tracking_json_paths),
            "processed_json_paths": list(self.processed_json_paths),
            "merged_json_paths": list(self.merged_json_paths),
            "csv_paths": list(self.csv_paths),
        }


def build_run_result(
    job_run: JobRun,
    *,
    resolved_config: PipelineConfig | None = None,
    materialized_config_path: str | None = None,
) -> RunResult:
    """Build a normalized run summary from the aggregated job snapshot."""

    output_root = resolved_config.output_root if resolved_config is not None else None
    output_image_folder = resolved_config.output_image_folder if resolved_config is not None else None
    output_video_folder = resolved_config.output_video_folder if resolved_config is not None else None
    output_json_folder = resolved_config.output_json_folder if resolved_config is not None else None

    for artifact in job_run.artifacts:
        if artifact.kind != "output_dir":
            continue
        role = artifact.metadata.get("role")
        if role == "frames":
            output_image_folder = artifact.path
        elif role == "videos":
            output_video_folder = artifact.path
        elif role == "json":
            output_json_folder = artifact.path

    projection_video_path = next(
        (artifact.path for artifact in job_run.artifacts if artifact.kind == "projection_video"),
        None,
    )

    return RunResult(
        job_run=job_run,
        resolved_config=resolved_config,
        materialized_config_path=materialized_config_path,
        output_root=output_root,
        output_image_folder=output_image_folder,
        output_video_folder=output_video_folder,
        output_json_folder=output_json_folder,
        projection_video_path=projection_video_path,
        tracking_json_paths=_unique_paths(job_run.artifacts, "tracking_json"),
        processed_json_paths=_unique_paths(job_run.artifacts, "processed_json"),
        merged_json_paths=_unique_paths(job_run.artifacts, "merged_json"),
        csv_paths=_unique_paths(job_run.artifacts, "csv"),
    )
