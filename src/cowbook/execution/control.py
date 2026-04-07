from __future__ import annotations

import threading
from dataclasses import dataclass, field


class JobCancelledError(RuntimeError):
    pass


@dataclass(slots=True)
class CancellationToken:
    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise JobCancelledError("Job cancellation requested.")
