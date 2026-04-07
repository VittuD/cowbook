from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

JobStatus = str


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_job_id() -> str:
    return uuid4().hex


@dataclass(slots=True)
class JobArtifact:
    """Artifact produced during a pipeline run.

    Attributes:
        kind: Logical artifact type such as ``output_dir``, ``projection_video``,
            ``json`` or ``csv``.
        path: Filesystem path to the produced artifact.
        group_idx: Optional 1-based group index when the artifact belongs to a
            specific camera group.
        metadata: Free-form extra data attached by the caller.
    """

    kind: str
    path: str
    group_idx: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the artifact."""

        return {
            "kind": self.kind,
            "path": self.path,
            "group_idx": self.group_idx,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class JobEvent:
    """Structured event emitted during job execution.

    A job event is the smallest execution-trace unit exposed by Cowbook.
    Pipeline stages emit these events as work progresses. Observers can log
    them, store them, or transform them into another status view.

    Attributes:
        job_id: Stable run identifier for the event stream.
        event_type: Short event name such as ``job_started``,
            ``groups_discovered`` or ``artifact_created``.
        timestamp: UTC ISO-8601 timestamp generated when the event is created.
        status: Optional coarse-grained run status such as ``running``,
            ``completed``, ``failed`` or ``cancelled``.
        stage: Optional pipeline stage name such as ``config``, ``group`` or
            ``video``.
        message: Optional human-readable message for logs or UI.
        group_idx: Optional 1-based group index for group-scoped events.
        payload: Extra structured data carried with the event.
    """

    job_id: str
    event_type: str
    timestamp: str = field(default_factory=utc_now_iso)
    status: JobStatus | None = None
    stage: str | None = None
    message: str | None = None
    group_idx: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the event."""

        return {
            "job_id": self.job_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "group_idx": self.group_idx,
            "payload": dict(self.payload),
        }


@dataclass(slots=True)
class JobRun:
    """Aggregated execution snapshot for one pipeline run.

    ``JobRun`` is a derived state object built from the event stream. It is
    useful when callers want the latest known execution state without replaying
    every raw event themselves.

    Attributes:
        job_id: Stable identifier for the run.
        config_path: Config file path associated with the run.
        status: Current coarse-grained run status.
        cancel_requested: Whether cooperative cancellation has been requested.
        cancel_requested_at: UTC timestamp for the cancellation request.
        current_stage: Latest known stage name.
        started_at: UTC timestamp of the first ``job_started`` event.
        finished_at: UTC timestamp of the terminal event, if any.
        groups_total: Total number of configured groups discovered for the run.
        groups_completed: Number of groups that completed successfully.
        groups_failed: Number of groups that failed.
        error_count: Number of error payloads observed so far.
        artifacts: Produced artifacts accumulated from ``artifact_created`` events.
        errors: Error messages accumulated from failed events.
        events: Full event history currently stored for the run.
    """

    job_id: str
    config_path: str
    status: JobStatus = "queued"
    cancel_requested: bool = False
    cancel_requested_at: str | None = None
    current_stage: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    groups_total: int = 0
    groups_completed: int = 0
    groups_failed: int = 0
    error_count: int = 0
    artifacts: list[JobArtifact] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    events: list[JobEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the aggregated run state."""

        return {
            "job_id": self.job_id,
            "config_path": self.config_path,
            "status": self.status,
            "cancel_requested": self.cancel_requested,
            "cancel_requested_at": self.cancel_requested_at,
            "current_stage": self.current_stage,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "groups_total": self.groups_total,
            "groups_completed": self.groups_completed,
            "groups_failed": self.groups_failed,
            "error_count": self.error_count,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "errors": list(self.errors),
            "events": [event.to_dict() for event in self.events],
        }
