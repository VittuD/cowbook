from __future__ import annotations

import threading
import time

from cowbook.execution import CancellationToken, InMemoryJobStore, JobReporter
from cowbook.execution.manager import JobManagerService


class FakeRunner:
    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.allow_finish = threading.Event()

    def run(
        self,
        config_path,
        overrides=None,
        *,
        observer=None,
        job_id=None,
        cancellation_token: CancellationToken | None = None,
    ):
        self.calls.append((config_path, overrides, job_id))
        reporter = JobReporter(job_id=job_id or "job", config_path=config_path, observer=observer)
        reporter.emit("job_started", status="running", stage="config")
        self.started.set()

        while not self.allow_finish.is_set():
            if cancellation_token is not None and cancellation_token.is_cancelled():
                reporter.emit("job_cancelled", status="cancelled", stage="group")
                return
            time.sleep(0.01)

        reporter.emit("job_completed", status="completed", stage="pipeline")


def test_job_manager_submit_run_tracks_job_in_store():
    runner = FakeRunner()
    runner.allow_finish.set()
    manager = JobManagerService(runner=runner, store=InMemoryJobStore())

    job_id = manager.submit_run("config.json", overrides={"fps": 8}, background=False)
    snapshot = manager.get_job(job_id)

    assert snapshot is not None
    assert snapshot.status == "completed"
    assert snapshot.config_path == "config.json"
    assert [event.event_type for event in snapshot.events] == [
        "job_submitted",
        "job_started",
        "job_completed",
    ]


def test_job_manager_cancel_job_marks_run_cancelled():
    runner = FakeRunner()
    manager = JobManagerService(runner=runner, store=InMemoryJobStore())

    job_id = manager.submit_run("config.json", background=True)
    assert runner.started.wait(timeout=2.0)
    assert manager.cancel_job(job_id) is True

    deadline = time.time() + 2.0
    snapshot = manager.get_job(job_id)
    while snapshot is not None and snapshot.status != "cancelled" and time.time() < deadline:
        time.sleep(0.01)
        snapshot = manager.get_job(job_id)

    assert snapshot is not None
    assert snapshot.cancel_requested is True
    assert snapshot.status == "cancelled"
    assert any(event.event_type == "job_cancel_requested" for event in snapshot.events)
    assert any(event.event_type == "job_cancelled" for event in snapshot.events)
