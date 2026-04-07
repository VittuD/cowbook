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
    kind: str
    path: str
    group_idx: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "group_idx": self.group_idx,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class JobEvent:
    job_id: str
    event_type: str
    timestamp: str = field(default_factory=utc_now_iso)
    status: JobStatus | None = None
    stage: str | None = None
    message: str | None = None
    group_idx: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
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
