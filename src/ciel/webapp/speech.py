"""Origin-only browser speech with a bounded producer queue and single-tab claim."""

import queue
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ciel.audio.text_reply import spoken_text_reply
from ciel.webapp.audio import encode_audio_wav
from ciel.webapp.requests import WebRequest

ECHO_MARGIN_SECONDS = 0.8
"""After handed-out audio should have finished, keep treating Ciel as audible this long.

Covers network delivery, decoding, and device output latency, so a speaker's tail is not
picked up by any microphone, in this tab, another tab, or another device.
"""

class _Cancellation(threading.Event):
    """Let the voice engine observe parent cancellation and revocation immediately."""

    def __init__(self, parent: threading.Event, current: Callable[[], bool]) -> None:
        super().__init__()
        self.parent, self.current = parent, current

    def is_set(self) -> bool:
        return super().is_set() or self.parent.is_set() or not self.current()


@dataclass
class _Playback:
    request: WebRequest
    text: str
    current: Callable[[], bool]
    canceled: threading.Event
    parent: threading.Event
    chunks: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=4))
    client: str = ""
    done: bool = False
    error: str = ""
    expires: float = field(default_factory=lambda: time.monotonic() + 120)


class BrowserSpeech:
    """Synthesize through CIEL's configured voice, without opening PC audio output."""

    def __init__(self, tts: Any, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.tts = tts
        self.clock = clock
        self._lock = threading.RLock()
        self._jobs: dict[str, _Playback] = {}
        self._audible_until = 0.0

    def audible(self) -> bool:
        """Whether some browser is, or may still be, playing Ciel's voice."""
        with self._lock:
            return self.clock() < self._audible_until + ECHO_MARGIN_SECONDS

    def overlaps_playback(self, clip_seconds: float) -> bool:
        """Whether a just-finished recording of this length began while Ciel was audible."""
        with self._lock:
            started = self.clock() - max(0.0, clip_seconds)
            return started < self._audible_until + ECHO_MARGIN_SECONDS

    def speak(
        self,
        text: str,
        request: WebRequest,
        *,
        cancel_event: threading.Event,
        current: Callable[[], bool],
        full: bool = False,
    ) -> str:
        """Start bounded synthesis; playback success is acknowledged separately by the browser."""
        with self._lock:
            self._prune()
            if len(self._jobs) >= 8:
                raise ValueError("Finish or stop the current reply before requesting more audio.")
            handle = secrets.token_urlsafe(18)
            job = _Playback(
                request, text[:64000], current, _Cancellation(cancel_event, current), cancel_event
            )
            self._jobs[handle] = job
        audible = job.text if full else spoken_text_reply(job.text)

        def produce() -> None:
            try:
                for audio, rate in self.tts.synthesize_stream(audible, cancel_event=job.canceled):
                    if not 8000 <= rate <= 96000:
                        raise ValueError("Unsupported voice sample rate.")
                    for start in range(0, len(audio), rate):
                        if not self._valid(job):
                            return
                        samples = audio[start : start + rate]
                        chunk = (encode_audio_wav(samples, rate), len(samples) / rate)
                        deadline = time.monotonic() + 30
                        while self._valid(job):
                            try:
                                job.chunks.put(chunk, timeout=0.1)
                                break
                            except queue.Full:
                                if time.monotonic() > deadline:
                                    job.canceled.set()
            except Exception:
                job.error = "Audio is unavailable. You can read the answer below."
            finally:
                job.done = True

        threading.Thread(target=produce, name="ciel-browser-speech", daemon=True).start()
        return handle

    @staticmethod
    def _valid(job: _Playback) -> bool:
        valid = (
            not job.canceled.is_set()
            and not job.parent.is_set()
            and job.expires > time.monotonic()
            and job.current()
        )
        if not valid:
            job.canceled.set()
        return valid

    def _prune(self) -> None:
        self._jobs = {h: j for h, j in self._jobs.items() if self._valid(j)}

    def list(self, device_id: str, revision: int) -> list[dict]:
        """Advertise only this device's current, unexpired jobs; never old chat audio."""
        with self._lock:
            self._prune()
            return [
                {"id": h, "error": j.error, "excerpt": spoken_text_reply(j.text) != j.text}
                for h, j in self._jobs.items()
                if j.request.device_id == device_id and j.request.revision == revision
            ]

    def chunk(self, handle: str, device_id: str, revision: int, client: str) -> bytes | None:
        """Claim once across tabs and drain one chunk; None means synthesis is still pending."""
        with self._lock:
            job = self._jobs.get(handle)
            if (
                job is None
                or not self._valid(job)
                or job.request.device_id != device_id
                or job.request.revision != revision
                or job.client
                and job.client != client
            ):
                raise ValueError("This playback is no longer available.")
            job.client = client
            try:
                data, seconds = job.chunks.get_nowait()
            except queue.Empty:
                if job.done:
                    raise EOFError(job.error) from None
                return None
            # Browsers play chunks back to back, so audibility extends from the later of now
            # and the end of the audio already handed out.
            self._audible_until = max(self.clock(), self._audible_until) + seconds
            return data

    def replay(self, handle: str, device_id: str, revision: int) -> str:
        """Read the already generated full answer; never make another model request."""
        with self._lock:
            job = self._jobs.get(handle)
            if (
                job is None
                or not self._valid(job)
                or job.request.device_id != device_id
                or job.request.revision != revision
            ):
                raise ValueError("This answer is no longer available for playback.")
            return self.speak(
                job.text, job.request, cancel_event=job.parent, current=job.current, full=True
            )

    def cancel(self) -> None:
        """Invalidate queued chunks and stop producers on Stop, switch, or shutdown."""
        with self._lock:
            for job in self._jobs.values():
                job.canceled.set()
            self._jobs.clear()
            # Stop silences the browser at once, so the mic is free immediately afterwards.
            self._audible_until = min(self._audible_until, self.clock() - ECHO_MARGIN_SECONDS)
