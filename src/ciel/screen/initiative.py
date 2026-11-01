"""Session-only observation policy with foreground priority and bounded background work."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from ciel.logging import get_logger
from ciel.providers.vision import VisionError
from ciel.screen.observation import (
    CodingObservation,
    evidence_digest,
    same_view,
    visual_signature,
)

logger = get_logger("screen.initiative")


class ObservationCancellation:
    """Let capture, HTTP and speech observe permission changes while work is blocked."""

    def __init__(self, valid: Callable[[], bool], parent: Any = None) -> None:
        self.valid, self.parent = valid, parent

    def is_set(self) -> bool:
        return bool(self.parent is not None and self.parent.is_set()) or not self.valid()


class ScreenInitiative:
    """Watch only after opt-in and deliver at most one hint for an unresolved error.

    One worker owns transient evidence. Delivery goes through the listener's ordinary
    serialized output worker, without creating a synthetic user turn or saving pixels.
    """

    scan_interval = 20.0
    repeat_seconds = 120.0
    cooldown_seconds = 900.0
    freshness_seconds = 15.0

    def __init__(
        self, *, backend: Any, analyzer: Any, scope: Callable[[], Any],
        ready: Callable[[], bool], local_only: Callable[[], bool],
        owner_activity: Callable[[], int], submit: Callable[[Callable], bool],
        deliver: Callable[[CodingObservation, Any, Callable[[], bool]], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend, self.analyzer = backend, analyzer
        self.scope, self.ready, self.local_only = scope, ready, local_only
        self.owner_activity, self.submit, self.deliver = owner_activity, submit, deliver
        self.clock = clock
        self._lock = threading.RLock()
        self._scan_lock = threading.Lock()
        self._closed = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = False
        self._paused = False
        self._epoch = 0
        self._scope: Any = None
        self._next_scan = 0.0
        self._last_delivered = float("-inf")
        self._waiting = False
        self._activity = 0
        self._failures = 0
        self._problem = ""
        self._notified_key: tuple[str, str] | None = None
        self._clear_evidence()

    def _clear_evidence(self) -> None:
        self._observation: CodingObservation | None = None
        self._overview = b""
        self._first_seen = 0.0
        self._last_seen = 0.0
        self._queued = False
        self._queued_at = 0.0

    def control(self, operation: str) -> str:
        """Change this session only. Resuming cannot erase cooldown or ignored hints."""
        with self._lock:
            if operation == "status":
                return self.status()
            self._epoch += 1
            self._clear_evidence()
            self._notified_key = None
            self._problem = ""
            if operation in {"on", "resume"} and not self._closed.is_set():
                self._enabled, self._paused = True, False
                self._scope = self.scope()
                self._next_scan = self.clock() + self.scan_interval
                if self._thread is None:
                    self._thread = threading.Thread(target=self._run,
                                                    name="ciel-observation", daemon=True)
                    self._thread.start()
            elif operation == "pause":
                self._paused = self._enabled
            elif operation == "off":
                self._enabled, self._paused = False, False
                self._scope = None
            self._wake.set()
            return self.status()

    def status(self) -> str:
        """Keep the enabled state and any quiet suppression visible in local controls."""
        with self._lock:
            if not self._enabled:
                return "Screen observation: off. Use /observe on to enable coding hints."
            if self._paused:
                return "Screen observation: paused. Use /observe resume to continue."
            if self._problem:
                return f"Screen observation: on; retrying after {self._problem}."
            if self._waiting:
                return "Screen observation: on; quiet until you speak or type to CIEL again."
            return "Screen observation: on; coding hints only. Use /observe pause to pause."

    def _valid(self, epoch: int, *, scanning: bool = False) -> bool:
        with self._lock:
            valid = (self._enabled and not self._paused and not self._closed.is_set()
                     and self._epoch == epoch)
            scope = self._scope
        return bool(valid and self.scope() == scope and (not scanning or self.ready()))

    def tick(self) -> None:
        """Perform one eligible scan; injectable clocks make silence policy testable."""
        if not self._scan_lock.acquire(blocking=False):
            return
        try:
            self._tick()
        finally:
            self._scan_lock.release()

    def _tick(self) -> None:
        now = self.clock()
        with self._lock:
            if not self._enabled or self._closed.is_set():
                return
            if self.scope() != self._scope:
                self._waiting = False
                self.control("off")
                return
            if self._waiting and self.owner_activity() > self._activity:
                self._waiting = False
            if self._queued and now - self._queued_at > self.freshness_seconds:
                self._clear_evidence()
            if (self._paused or self._waiting or self._queued or now < self._next_scan
                    or now - self._last_delivered < self.cooldown_seconds):
                return
            if not self.ready():
                # An interruption breaks continuous observation of the problem.
                self._clear_evidence()
                return
            epoch = self._epoch
            activity = self.owner_activity()
            self._next_scan = now + self.scan_interval
        cancel = ObservationCancellation(lambda: self._valid(epoch, scanning=True)
                                         and self.owner_activity() == activity)
        try:
            if cancel.is_set():
                return
            local_only = self.local_only()
            # Fail before capture when protected context cannot use the selected model.
            vision = getattr(self.analyzer, "vision", None)
            if local_only and vision is not None and vision.provider != "ollama":
                raise VisionError("protected local conversation needs a local image model")
            capture = self.backend.capture(cancel=cancel)
            if cancel.is_set():
                return
            overview = visual_signature(capture)
            with self._lock:
                previous = self._observation
                stable = same_view(self._overview, overview)
                due = previous is not None and now - self._first_seen >= self.repeat_seconds
                if now - self._last_seen > self.scan_interval * 3:
                    previous = None
                    self._clear_evidence()
                self._last_seen = now
            if stable and not due:
                return
            observation = self.analyzer.analyze(capture, cancel, local_only=local_only,
                                                focus=previous if due else None)
            if cancel.is_set():
                return
            with self._lock:
                if not self._valid(epoch, scanning=True):
                    return
                self._failures = 0
                self._problem = ""
                self._overview = overview
                if observation is None:
                    self._observation = None
                    self._first_seen = 0.0
                    self._notified_key = None
                    return
                if observation.key == self._notified_key:
                    self._observation = None
                    return
                self._observation = observation
                if previous is None or observation.key != previous.key:
                    self._first_seen = now
                    return
                if not due:
                    return
                verified_at = self.clock()
                signature = evidence_digest(capture, observation)
                self._queued = True
                self._queued_at = verified_at

            def speak(cancel_event: Any) -> None:
                self._deliver(observation, signature, epoch, verified_at, cancel_event,
                              local_only=local_only, activity=activity)

            if not self.submit(speak):
                with self._lock:
                    self._queued = False
        except Exception as error:
            if not cancel.is_set():
                with self._lock:
                    self._failures += 1
                    self._clear_evidence()
                    self._next_scan = self.clock() + min(300.0, 20 * 2 ** min(self._failures, 4))
                    # Never log raw inference errors, screenshots or extracted error text.
                    self._problem = ("a screen connection problem" if isinstance(error, VisionError)
                                     else "a capture or analysis problem")
                logger.debug("Background observation deferred (%s).", type(error).__name__)
        finally:
            if cancel.is_set():
                with self._lock:
                    if epoch == self._epoch:
                        self._clear_evidence()

    def _deliver(self, observation: CodingObservation, signature: bytes, epoch: int,
                 verified_at: float, parent: Any, *, local_only: bool, activity: int) -> None:
        """Recheck actual current pixels and permission immediately before output."""
        cancel = ObservationCancellation(lambda: self._valid(epoch)
                                         and self.owner_activity() == activity
                                         and self.clock() - verified_at <= self.freshness_seconds,
                                         parent)
        try:
            if (cancel.is_set() or self.clock() - verified_at > self.freshness_seconds
                    or self.local_only() != local_only):
                return
            fresh = self.backend.capture(cancel=cancel)
            if cancel.is_set() or signature != evidence_digest(fresh, observation):
                return

            def started() -> bool:
                with self._lock:
                    if cancel.is_set():
                        return False
                    self._last_delivered = self.clock()
                    self._activity = self.owner_activity()
                    self._waiting = True
                    self._notified_key = observation.key
                    return True

            self.deliver(observation, cancel, started)
        finally:
            with self._lock:
                if epoch == self._epoch:
                    self._clear_evidence()

    def _run(self) -> None:
        while not self._closed.is_set():
            self._wake.wait(1.0)
            self._wake.clear()
            if not self._closed.is_set():
                try:
                    self.tick()
                except Exception as error:
                    logger.debug("Observation state unavailable (%s).", type(error).__name__)

    def close(self) -> None:
        """Invalidate in-flight inference/output before releasing runtime resources."""
        self._closed.set()
        self.control("off")
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.5)
