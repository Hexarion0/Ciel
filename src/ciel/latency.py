"""Privacy-safe, monotonic timing of one spoken turn across worker threads."""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from ciel.logging import get_logger

logger = get_logger("latency")


@dataclass
class TurnTrace:
    """Record stage boundaries, never transcripts, prompts, audio or HTTP headers."""

    origin: float = field(default_factory=time.monotonic)
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    events: dict[str, dict] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def mark(self, stage: str, *, at: float | None = None, **details) -> None:
        """Keep the first occurrence of a stage using the same process clock."""
        stamp = time.monotonic() if at is None else at
        with self._lock:
            self.events.setdefault(stage, {"seconds": stamp - self.origin, **details})

    def report(self) -> dict:
        with self._lock:
            return {"turn_id": self.turn_id, "events": dict(self.events)}

    def log(self) -> None:
        logger.info("Turn latency: %s", json.dumps(self.report(), sort_keys=True))


active_trace: ContextVar[TurnTrace | None] = ContextVar("ciel_turn_trace", default=None)
_phase: ContextVar[str] = ContextVar("ciel_latency_phase", default="")


@contextmanager
def latency_phase(name: str):
    """Keep speech-gate network timestamps distinct from reply-network timestamps."""
    token = _phase.set(name + ".")
    try:
        yield
    finally:
        _phase.reset(token)


def mark(stage: str, **details) -> None:
    """Record a boundary only when the caller belongs to a traced spoken turn."""
    trace = active_trace.get()
    if trace is not None:
        trace.mark(_phase.get() + stage, **details)


def http_trace(name: str, _info: dict) -> None:
    """HTTPX transport hook: discard potentially sensitive connection/request data."""
    mark("transport." + name)
    if name.endswith("send_request_body.complete"):
        mark("provider_request_sent")


def http_extensions() -> dict:
    """Avoid transport tracing overhead for untraced calls."""
    return {"trace": http_trace} if active_trace.get() is not None else {}
