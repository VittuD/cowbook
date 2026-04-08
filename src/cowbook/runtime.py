from __future__ import annotations

"""Public runtime interface for using cowbook as a Python package."""

from typing import Any

from cowbook.app.pipeline import PipelineRunner
from cowbook.core.contracts import PipelineConfig, RunRequest
from cowbook.execution.control import CancellationToken, JobCancelledError
from cowbook.execution.models import JobArtifact, JobEvent, JobRun
from cowbook.execution.observers import JobObserver
from cowbook.io.config_loader import load_config, normalize_config_mapping


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
    config = load_config(config_path, overrides=overrides)
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


def run_pipeline(
    config_path: str,
    overrides: dict[str, Any] | None = None,
    *,
    observer: JobObserver | None = None,
    job_id: str | None = None,
    cancellation_token: CancellationToken | None = None,
    runner: PipelineRunner | None = None,
) -> JobRun | None:
    """Run one cowbook pipeline job synchronously.

    Args:
        config_path: Path to the JSON configuration file.
        overrides: Optional runtime overrides applied after loading.
        observer: Optional execution observer for structured events.
        job_id: Optional externally provided job identifier.
        cancellation_token: Optional cooperative cancellation token.
        runner: Optional pipeline runner instance for custom wiring or tests.

    Returns:
        A :class:`JobRun` snapshot when available.
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
) -> JobRun | None:
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
    "load_pipeline_config",
    "load_pipeline_config_object",
    "run_pipeline",
    "run_pipeline_request",
]
