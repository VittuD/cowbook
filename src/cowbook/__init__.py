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
    load_pipeline_config,
    run_pipeline,
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
    "load_pipeline_config",
    "run_pipeline",
]
