from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from cowbook.execution.observers import JobReporter

ProgressEventSink = Callable[[str, dict[str, Any]], None]


@dataclass(slots=True)
class StageProgressReporter:
    """Shared milestone-based progress adapter for long-running stages."""

    event_prefix: str
    reporter_stage: str
    stage_name: str
    path_value: str | None = None
    path_key: str = "path"
    camera_nr: int | None = None
    total: int | None = None
    log_progress: bool = False
    reporter: JobReporter | None = None
    group_idx: int | None = None
    event_sink: ProgressEventSink | None = None
    extra_payload: dict[str, Any] = field(default_factory=dict)
    current_key: str = "current"
    total_key: str = "total"
    _last_progress_value: int = field(default=0, init=False)

    def stage_started(self) -> None:
        payload = self._base_payload()
        if self.total is not None:
            payload[self.total_key] = self.total
        self._emit_event(f"{self.event_prefix}_stage_started", payload)
        if self.log_progress:
            print(self._format_message("started"), flush=True)

    def step_progress(self, current: int, total: int | None = None) -> None:
        if total is not None:
            self.total = total
        if not self._should_emit_milestone(current):
            return
        payload = self._base_payload()
        payload[self.current_key] = current
        if self.total is not None:
            payload[self.total_key] = self.total
            payload["progress_fraction"] = min(1.0, current / self.total)
        self._emit_event(f"{self.event_prefix}_stage_progress", payload)
        if self.log_progress:
            print(self._format_message("progress", current=current), flush=True)
        self._last_progress_value = current

    def stage_completed(self) -> None:
        payload = self._base_payload()
        if self.total is not None:
            payload[self.total_key] = self.total
        self._emit_event(f"{self.event_prefix}_stage_completed", payload)
        if self.log_progress:
            print(self._format_message("completed"), flush=True)

    def _base_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"stage_name": self.stage_name}
        if self.path_value is not None:
            payload[self.path_key] = self.path_value
        if self.camera_nr is not None:
            payload["camera_nr"] = self.camera_nr
        if self.extra_payload:
            payload.update(self.extra_payload)
        return payload

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(event_type, payload)
            return
        if self.reporter is not None:
            self.reporter.emit(
                event_type,
                stage=self.reporter_stage,
                group_idx=self.group_idx,
                payload=payload,
            )

    def _should_emit_milestone(self, current: int) -> bool:
        if current <= 1:
            return True
        if self.total is not None and self.total > 0:
            if current >= self.total:
                return True
            interval = max(1, self.total // 20)
            return current - self._last_progress_value >= interval
        interval = 10
        return current - self._last_progress_value >= interval

    def _format_message(self, state: str, *, current: int | None = None) -> str:
        prefix = f"[{self.event_prefix}] {self.stage_name}:"
        target = self.path_value or self.reporter_stage
        if state == "progress":
            if current is not None and self.total is not None:
                return f"{prefix} {target} {current}/{self.total}"
            if current is not None:
                return f"{prefix} {target} {current}"
        return f"{prefix} {target} {state}"


@dataclass(slots=True)
class TrackingProgressReporter:
    """Compatibility wrapper around :class:`StageProgressReporter` for tracking."""

    tracking_mode: str
    stage_name: str
    video_path: str
    camera_nr: int | None = None
    frame_total: int | None = None
    log_progress: bool = False
    reporter: JobReporter | None = None
    group_idx: int | None = None
    event_sink: ProgressEventSink | None = None
    _reporter: StageProgressReporter = field(init=False)

    def __post_init__(self) -> None:
        self._reporter = StageProgressReporter(
            event_prefix="tracking",
            reporter_stage="tracking",
            stage_name=self.stage_name,
            path_value=self.video_path,
            path_key="video_path",
            camera_nr=self.camera_nr,
            total=self.frame_total,
            log_progress=self.log_progress,
            reporter=self.reporter,
            group_idx=self.group_idx,
            event_sink=self.event_sink,
            extra_payload={"tracking_mode": self.tracking_mode},
            current_key="frame_current",
            total_key="frame_total",
        )

    def stage_started(self) -> None:
        self._reporter.stage_started()

    def frame_progress(self, frame_current: int, frame_total: int | None = None) -> None:
        self._reporter.step_progress(frame_current, frame_total)

    def stage_completed(self) -> None:
        self._reporter.stage_completed()
