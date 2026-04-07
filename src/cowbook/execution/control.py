from __future__ import annotations

import threading
from dataclasses import dataclass, field


class JobCancelledError(RuntimeError):
    """Raised when cooperative cancellation stops an in-flight run."""

    pass


@dataclass(slots=True)
class CancellationToken:
    """Thread-safe cooperative cancellation flag.

    The pipeline checks this token at stage boundaries and inside selected
    long-running loops. Cancellation is cooperative: the caller requests a
    stop, and the pipeline raises :class:`JobCancelledError` the next time it
    reaches a cancellation check.
    """

    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        """Mark the token as cancelled."""

        self._event.set()

    def is_cancelled(self) -> bool:
        """Return ``True`` after cancellation has been requested."""

        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise :class:`JobCancelledError` when the token is cancelled."""

        if self.is_cancelled():
            raise JobCancelledError("Job cancellation requested.")
