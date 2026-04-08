from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from cowbook.execution.models import JobArtifact, JobEvent, JobRun


class JobObserver(Protocol):
    """Protocol for sinks that consume execution events.

    Implementations can log events, accumulate them into a run snapshot, or
    forward them to another system. The pipeline only depends on this protocol,
    not on any concrete storage or transport.
    """

    def emit(self, event: JobEvent) -> None: ...


@dataclass(slots=True)
class NullObserver:
    """Observer that ignores every event."""

    def emit(self, event: JobEvent) -> None:
        """Discard ``event``."""

        return None


@dataclass(slots=True)
class CompositeObserver:
    """Fan out each event to multiple observers."""

    observers: list[JobObserver] = field(default_factory=list)

    def emit(self, event: JobEvent) -> None:
        """Forward ``event`` to every configured observer in order."""

        for observer in self.observers:
            observer.emit(event)


@dataclass(slots=True)
class InMemoryJobStore:
    """In-memory run snapshot store built from the event stream.

    This store is useful in tests, local tools, or lightweight embedding
    scenarios where callers want the latest run state without maintaining their
    own event reducer.
    """

    jobs: dict[str, JobRun] = field(default_factory=dict)

    def emit(self, event: JobEvent) -> None:
        """Update the stored :class:`JobRun` for ``event.job_id``."""

        payload = dict(event.payload)
        config_path = str(payload.get("config_path", ""))
        run = self.jobs.setdefault(
            event.job_id,
            JobRun(job_id=event.job_id, config_path=config_path),
        )

        if config_path:
            run.config_path = config_path
        if event.status is not None:
            run.status = event.status
        if event.event_type == "job_cancel_requested":
            run.cancel_requested = True
            run.cancel_requested_at = event.timestamp
        if event.stage is not None:
            run.current_stage = event.stage
        if event.event_type == "job_started" and run.started_at is None:
            run.started_at = event.timestamp
        if event.event_type in {"job_completed", "job_failed", "job_cancelled"}:
            run.finished_at = event.timestamp
        if event.event_type == "groups_discovered":
            run.groups_total = int(payload.get("count", 0))
        if event.event_type == "group_completed":
            run.groups_completed += 1
        if event.event_type == "group_failed":
            run.groups_failed += 1

        error_message = (
            payload.get("error_detail")
            or payload.get("error")
            or (event.message if event.status == "failed" else None)
        )
        if error_message:
            run.error_count += 1
            run.errors.append(str(error_message))

        if event.event_type == "artifact_created":
            artifact = JobArtifact(
                kind=str(payload.get("kind", "artifact")),
                path=str(payload.get("path", "")),
                group_idx=event.group_idx,
                metadata={k: v for k, v in payload.items() if k not in {"kind", "path", "config_path"}},
            )
            run.artifacts.append(artifact)

        run.events.append(event)

    def get(self, job_id: str) -> JobRun | None:
        """Return the latest stored run snapshot for ``job_id``."""

        return self.jobs.get(job_id)


@dataclass(slots=True)
class JobReporter:
    """Convenience emitter bound to one run.

    ``JobReporter`` reduces boilerplate in the pipeline by attaching the job id
    and config path to every event. Callers normally use :meth:`emit` for stage
    transitions and :meth:`artifact` for produced files.
    """

    job_id: str
    config_path: str
    observer: JobObserver = field(default_factory=NullObserver)

    def emit(
        self,
        event_type: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        message: str | None = None,
        group_idx: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit one structured execution event for the bound run."""

        event_payload = {"config_path": self.config_path}
        if payload:
            event_payload.update(payload)
        self.observer.emit(
            JobEvent(
                job_id=self.job_id,
                event_type=event_type,
                status=status,
                stage=stage,
                message=message,
                group_idx=group_idx,
                payload=event_payload,
            )
        )

    def artifact(
        self,
        kind: str,
        path: str,
        *,
        group_idx: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit an ``artifact_created`` event for a produced file."""

        payload = {"kind": kind, "path": path}
        if metadata:
            payload.update(metadata)
        self.emit("artifact_created", group_idx=group_idx, payload=payload)
