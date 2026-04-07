from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from cowbook.execution.models import JobArtifact, JobEvent, JobRun


class JobObserver(Protocol):
    def emit(self, event: JobEvent) -> None: ...


@dataclass(slots=True)
class NullObserver:
    def emit(self, event: JobEvent) -> None:
        return None


@dataclass(slots=True)
class CompositeObserver:
    observers: list[JobObserver] = field(default_factory=list)

    def emit(self, event: JobEvent) -> None:
        for observer in self.observers:
            observer.emit(event)


@dataclass(slots=True)
class InMemoryJobStore:
    jobs: dict[str, JobRun] = field(default_factory=dict)

    def emit(self, event: JobEvent) -> None:
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

        error_message = payload.get("error") or (event.message if event.status == "failed" else None)
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
        return self.jobs.get(job_id)


@dataclass(slots=True)
class JobReporter:
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
        payload = {"kind": kind, "path": path}
        if metadata:
            payload.update(metadata)
        self.emit("artifact_created", group_idx=group_idx, payload=payload)
