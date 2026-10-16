"""Bounded audio-chunk playback with prompt cancellation and no inference in callbacks."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator

import numpy as np

from ciel.audio.native import sd
from ciel.logging import get_logger

logger = get_logger("audio.chunk_playback")


class ChunkPlayback:
    """One producer and one PortAudio stream; cancel cooperatively on every exit."""

    def __init__(self, stop: threading.Event, *, queue_size: int, timeout: float) -> None:
        self.stop = stop
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.timeout = timeout
        self.finished = threading.Event()
        self.total_samples = 0
        self.played_samples = 0
        self.sample_rate = 0
        self.underruns = 0
        self.error: Exception | None = None
        self._end = object()
        self._pending = np.zeros(0, dtype=np.float32)
        self._last_chunk_at = time.monotonic()

    def _offer(self, item: object) -> None:
        while not self.stop.is_set():
            try:
                self.queue.put(item, timeout=0.02)
                return
            except queue.Full:
                continue

    def produce(self, chunks: Iterator[tuple[np.ndarray, int]]) -> None:
        """Consume and close the generator in its owning thread."""
        try:
            for wave, rate in chunks:
                if self.stop.is_set():
                    break
                if self.sample_rate and self.sample_rate != rate:
                    raise ValueError("Sample rate changed during speech")
                self.sample_rate = rate
                self.total_samples += wave.size
                self._last_chunk_at = time.monotonic()
                self._offer(wave)
            self._offer(self._end)
        except Exception as err:
            self.error = err
            self._offer(self._end)
        finally:
            chunks.close()

    def _callback(self, outdata, frames, _time_info, status) -> None:
        outdata.fill(0)
        if self.stop.is_set() or self.error is not None:
            self.finished.set()
            raise sd.CallbackAbort
        if status.output_underflow:
            self.underruns += 1
        offset = 0
        while offset < frames:
            if not self._pending.size:
                try:
                    item = self.queue.get_nowait()
                except queue.Empty:
                    self.underruns += 1
                    return
                if item is self._end:
                    self.finished.set()
                    raise sd.CallbackStop
                self._pending = item
            count = min(frames - offset, self._pending.size)
            outdata[offset:offset + count, 0] = self._pending[:count]
            self._pending = self._pending[count:]
            self.played_samples += count
            offset += count

    def play(
        self, *, device: int | str | None, latency: float,
        current: Callable[[], bool], on_start: Callable[[], None],
        on_tick: Callable[[float, float], None],
    ) -> bool:
        """Start at the first usable chunk while synthesis continues in the producer."""
        try:
            while current() and not self.stop.is_set():
                if time.monotonic() - self._last_chunk_at > self.timeout:
                    raise TimeoutError("TTS did not provide audio before the synthesis timeout")
                try:
                    first = self.queue.get(timeout=0.02)
                    break
                except queue.Empty:
                    continue
            else:
                return False
            if self.error:
                raise self.error
            if first is self._end:
                return False
            self._pending = first
            with sd.OutputStream(
                samplerate=self.sample_rate, channels=1, dtype="float32", device=device,
                latency=latency, blocksize=1024, callback=self._callback,
                finished_callback=self.finished.set,
            ) as stream:
                on_start()
                while current() and not self.stop.is_set():
                    on_tick(
                        self.played_samples / self.sample_rate,
                        self.total_samples / self.sample_rate,
                    )
                    if self.finished.wait(0.01):
                        break
                    if time.monotonic() - self._last_chunk_at > self.timeout:
                        raise TimeoutError("TTS audio generation stalled")
                if not current() or self.stop.is_set() or self.error:
                    stream.abort()
                    if self.error:
                        raise self.error
                    return False
            if self.underruns:
                logger.warning("Streaming TTS had %d empty/underflow callbacks.", self.underruns)
            return current() and self.finished.is_set()
        finally:
            self.stop.set()
