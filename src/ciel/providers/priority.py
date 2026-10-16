"""Give foreground speech priority without a shared request lock or FIFO queue."""

import threading
from contextlib import contextmanager


class BackgroundDeferred(RuntimeError):
    """Retry a background summary later; do not save a partial/canceled summary."""


class RequestPriority:
    """Cancel expendable background reads when foreground work begins."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._foreground = 0
        self._background: set[threading.Event] = set()

    @contextmanager
    def foreground(self):
        with self._lock:
            self._foreground += 1
            for canceled in self._background:
                canceled.set()
        try:
            yield
        finally:
            with self._lock:
                self._foreground -= 1

    @contextmanager
    def background(self):
        canceled = threading.Event()
        with self._lock:
            if self._foreground:
                raise BackgroundDeferred("Foreground conversation is active")
            self._background.add(canceled)
        try:
            yield canceled
            if canceled.is_set():
                raise BackgroundDeferred("Foreground conversation interrupted summarization")
        finally:
            with self._lock:
                self._background.discard(canceled)
