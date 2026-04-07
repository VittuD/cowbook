from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from cowbook.app.pipeline import PipelineRunner
from cowbook.execution.control import CancellationToken
from cowbook.execution.models import JobEvent, JobRun, new_job_id
from cowbook.execution.observers import CompositeObserver, InMemoryJobStore, JobObserver

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class JobManagerService:
    runner: PipelineRunner = field(default_factory=PipelineRunner)
    store: InMemoryJobStore = field(default_factory=InMemoryJobStore)
    observer: JobObserver | None = None
    _threads: dict[str, threading.Thread] = field(default_factory=dict, init=False, repr=False)
    _tokens: dict[str, CancellationToken] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def submit_run(
        self,
        config_path: str,
        overrides: dict[str, Any] | None = None,
        *,
        job_id: str | None = None,
        background: bool = True,
        observer: JobObserver | None = None,
    ) -> str:
        resolved_job_id = job_id or new_job_id()
        cancellation_token = CancellationToken()
        combined_observer = self._build_observer(observer)

        combined_observer.emit(
            JobEvent(
                job_id=resolved_job_id,
                event_type="job_submitted",
                status="queued",
                stage="queue",
                payload={
                    "config_path": config_path,
                    "overrides": dict(overrides or {}),
                    "background": background,
                },
            )
        )

        with self._lock:
            self._tokens[resolved_job_id] = cancellation_token

        if background:
            thread = threading.Thread(
                target=self._run_job,
                args=(resolved_job_id, config_path, overrides, cancellation_token, combined_observer),
                daemon=True,
                name=f"cowbook-job-{resolved_job_id[:8]}",
            )
            with self._lock:
                self._threads[resolved_job_id] = thread
            thread.start()
        else:
            self._run_job(
                resolved_job_id,
                config_path,
                overrides,
                cancellation_token,
                combined_observer,
            )

        return resolved_job_id

    def get_job(self, job_id: str) -> JobRun | None:
        return self.store.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            token = self._tokens.get(job_id)
        snapshot = self.store.get(job_id)
        if token is None or snapshot is None:
            return False
        if snapshot.status in {"completed", "failed", "cancelled"}:
            return False

        token.cancel()
        self._build_observer(None).emit(
            JobEvent(
                job_id=job_id,
                event_type="job_cancel_requested",
                status="cancelling",
                stage=snapshot.current_stage,
                payload={"config_path": snapshot.config_path},
            )
        )
        return True

    def _build_observer(self, extra_observer: JobObserver | None) -> JobObserver:
        observers: list[JobObserver] = [self.store]
        if self.observer is not None:
            observers.append(self.observer)
        if extra_observer is not None:
            observers.append(extra_observer)
        return CompositeObserver(observers)

    def _run_job(
        self,
        job_id: str,
        config_path: str,
        overrides: dict[str, Any] | None,
        cancellation_token: CancellationToken,
        observer: JobObserver,
    ) -> None:
        try:
            self.runner.run(
                config_path,
                overrides=overrides,
                observer=observer,
                job_id=job_id,
                cancellation_token=cancellation_token,
            )
        except Exception as exc:
            logger.exception("Background job %s crashed: %s", job_id, exc)
            observer.emit(
                JobEvent(
                    job_id=job_id,
                    event_type="job_failed",
                    status="failed",
                    stage="pipeline",
                    message="Background job crashed.",
                    payload={"config_path": config_path, "error": str(exc)},
                )
            )
        finally:
            with self._lock:
                self._threads.pop(job_id, None)
