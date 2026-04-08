from __future__ import annotations

"""Public runtime interface for using cowbook as a Python package."""

from typing import Any

from cowbook.app.pipeline import PipelineRunner
from cowbook.core.contracts import PipelineConfig, RunRequest
from cowbook.execution.control import CancellationToken, JobCancelledError
from cowbook.execution.models import JobArtifact, JobEvent, JobRun
from cowbook.execution.observers import JobObserver
from cowbook.execution.results import RunResult
from cowbook.io.config_loader import (
    load_config_file,
    normalize_config_mapping,
    write_config_file,
)


def load_pipeline_config(
    config_path: str,
    overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Load, normalize, and validate a pipeline config file.

    Args:
        config_path: Path to the JSON configuration file.
        overrides: Optional runtime overrides applied after loading.

    Returns:
        A typed :class:`PipelineConfig` instance.

    Raises:
        ValueError: If the config cannot be loaded or validated.
    """
    config = load_config_file(config_path, overrides=overrides)
    if not config:
        raise ValueError(f"Failed to load pipeline config from {config_path}")
    return PipelineConfig.from_mapping(config)


def load_pipeline_config_object(
    config: PipelineConfig | dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Load, normalize, and validate a pipeline config from an in-memory object."""

    config_mapping = config.to_dict() if isinstance(config, PipelineConfig) else dict(config)
    normalized = normalize_config_mapping(config_mapping, overrides=overrides)
    return PipelineConfig.from_mapping(normalized)


def materialize_pipeline_config(
    config: PipelineConfig | dict[str, Any],
    output_path: str,
    overrides: dict[str, Any] | None = None,
) -> str:
    """Write a normalized pipeline config to disk and return the destination path."""

    return write_config_file(config, output_path, overrides=overrides)


def run_pipeline(
    config_path: str,
    overrides: dict[str, Any] | None = None,
    *,
    observer: JobObserver | None = None,
    job_id: str | None = None,
    cancellation_token: CancellationToken | None = None,
    runner: PipelineRunner | None = None,
) -> RunResult | None:
    """Run one cowbook pipeline job synchronously.

    Args:
        config_path: Path to the JSON configuration file.
        overrides: Optional runtime overrides applied after loading.
        observer: Optional execution observer for structured events.
        job_id: Optional externally provided job identifier.
        cancellation_token: Optional cooperative cancellation token.
        runner: Optional pipeline runner instance for custom wiring or tests.

    Returns:
        A :class:`RunResult` when available.
    """
    active_runner = runner or PipelineRunner()
    return active_runner.run_request(
        RunRequest(config_path=config_path, overrides=dict(overrides or {})),
        observer=observer,
        job_id=job_id,
        cancellation_token=cancellation_token,
    )


def run_pipeline_request(
    request: RunRequest,
    *,
    observer: JobObserver | None = None,
    job_id: str | None = None,
    cancellation_token: CancellationToken | None = None,
    runner: PipelineRunner | None = None,
) -> RunResult | None:
    """Run one cowbook pipeline job synchronously from a typed request."""

    active_runner = runner or PipelineRunner()
    return active_runner.run_request(
        request,
        observer=observer,
        job_id=job_id,
        cancellation_token=cancellation_token,
    )


__all__ = [
    "CancellationToken",
    "JobArtifact",
    "JobCancelledError",
    "JobEvent",
    "JobObserver",
    "JobRun",
    "PipelineConfig",
    "PipelineRunner",
    "RunRequest",
    "RunResult",
    "load_pipeline_config",
    "load_pipeline_config_object",
    "materialize_pipeline_config",
    "run_pipeline",
    "run_pipeline_request",
]
