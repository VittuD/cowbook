from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from cowbook.execution.observers import JobReporter

ProgressEventSink = Callable[[str, dict[str, Any]], None]


@dataclass(slots=True)
class TrackingProgressReporter:
    tracking_mode: str
    stage_name: str
    video_path: str
    camera_nr: int | None = None
    frame_total: int | None = None
    log_progress: bool = False
    reporter: JobReporter | None = None
    group_idx: int | None = None
    event_sink: ProgressEventSink | None = None
    _last_progress_frame: int = field(default=0, init=False)

    def stage_started(self) -> None:
        payload = self._base_payload()
        if self.frame_total is not None:
            payload["frame_total"] = self.frame_total
        self._emit_event("tracking_stage_started", payload)
        if self.log_progress:
            print(
                f"[tracking] {self.stage_name}: {self.video_path} started",
                flush=True,
            )

    def frame_progress(self, frame_current: int, frame_total: int | None = None) -> None:
        if frame_total is not None:
            self.frame_total = frame_total
        if not self._should_emit_milestone(frame_current):
            return
        payload = self._base_payload()
        payload["frame_current"] = frame_current
        if self.frame_total is not None:
            payload["frame_total"] = self.frame_total
            payload["progress_fraction"] = min(1.0, frame_current / self.frame_total)
        self._emit_event("tracking_stage_progress", payload)
        if self.log_progress:
            if self.frame_total is not None:
                print(
                    f"[tracking] {self.stage_name}: {self.video_path} frame {frame_current}/{self.frame_total}",
                    flush=True,
                )
            else:
                print(
                    f"[tracking] {self.stage_name}: {self.video_path} frame {frame_current}",
                    flush=True,
                )
        self._last_progress_frame = frame_current

    def stage_completed(self) -> None:
        payload = self._base_payload()
        if self.frame_total is not None:
            payload["frame_total"] = self.frame_total
        self._emit_event("tracking_stage_completed", payload)
        if self.log_progress:
            print(
                f"[tracking] {self.stage_name}: {self.video_path} completed",
                flush=True,
            )

    def _base_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tracking_mode": self.tracking_mode,
            "stage_name": self.stage_name,
            "video_path": self.video_path,
        }
        if self.camera_nr is not None:
            payload["camera_nr"] = self.camera_nr
        return payload

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(event_type, payload)
            return
        if self.reporter is not None:
            self.reporter.emit(
                event_type,
                stage="tracking",
                group_idx=self.group_idx,
                payload=payload,
            )

    def _should_emit_milestone(self, frame_current: int) -> bool:
        if frame_current <= 1:
            return True
        if self.frame_total is not None and self.frame_total > 0:
            if frame_current >= self.frame_total:
                return True
            interval = max(100, self.frame_total // 20)
            return frame_current - self._last_progress_frame >= interval
        interval = 300
        return frame_current - self._last_progress_frame >= interval
