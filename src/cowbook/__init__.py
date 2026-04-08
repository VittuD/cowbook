"""Cowbook package entrypoints, metadata, and stable contracts."""

from cowbook.runtime import (
    CancellationToken,
    JobArtifact,
    JobCancelledError,
    JobEvent,
    JobObserver,
    JobRun,
    PipelineConfig,
    PipelineRunner,
    RunRequest,
    RunResult,
    load_pipeline_config,
    load_pipeline_config_object,
    materialize_pipeline_config,
    run_pipeline,
    run_pipeline_request,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
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
