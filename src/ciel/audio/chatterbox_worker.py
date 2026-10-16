"""Persistent client for the isolated pinned Chatterbox Turbo Python environment."""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ciel.logging import get_logger

logger = get_logger("audio.chatterbox")


class ChatterboxWorkerError(RuntimeError):
    """The model worker could not initialize or complete a local synthesis request."""


class ChatterboxTurboWorker:
    """Keep one local CUDA model loaded behind a small line-oriented IPC protocol."""

    def __init__(
        self,
        python: str | Path,
        script: str | Path,
        model: str | Path,
        reference: str | Path,
        reference_text: str,
        *,
        startup_timeout: float = 180.0,
        min_free_vram_mib: int = 3000,
        synthesis_timeout: float = 60.0,
        event_reference: str | Path | None = None,
        cue_audio: dict[str, str | Path] | None = None,
    ) -> None:
        # Preserve virtualenv launchers: resolve() follows the venv's python symlink
        # to the base interpreter, which then cannot see venv site-packages.
        self.python = Path(os.path.abspath(Path(python).expanduser()))
        self.script = Path(script).resolve()
        self.model = Path(model).expanduser().resolve()
        self.reference = Path(reference).expanduser().resolve()
        self.event_reference = (
            Path(event_reference).expanduser().resolve() if event_reference is not None else None
        )
        self.cue_audio = {key: Path(path).expanduser().resolve()
                          for key, path in (cue_audio or {}).items()}
        self.reference_text = reference_text.strip()
        self.startup_timeout = startup_timeout
        if not math.isfinite(synthesis_timeout) or synthesis_timeout <= 0:
            raise ValueError("Synthesis timeout must be finite and positive")
        self.synthesis_timeout = synthesis_timeout
        self.min_free_vram_mib = min_free_vram_mib
        self._process: subprocess.Popen[str] | None = None
        self._ready = threading.Event()
        self._startup_result: dict = {}
        self._pending: dict[str, queue.Queue[dict]] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._closed = False
        self.startup_metrics: dict = {}

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _start(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise ChatterboxWorkerError("Chatterbox worker is closed")
            if self.is_running:
                process = None
            else:
                for path in [self.python, self.script, self.model, self.reference]:
                    if not path.exists():
                        raise ChatterboxWorkerError(
                            f"Required Chatterbox asset is missing: {path}"
                        )
                if not self.reference_text:
                    raise ChatterboxWorkerError("The CIEL reference transcript is empty")
                env = os.environ.copy()
                env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
                self._ready.clear()
                self._startup_result = {}
                command = [
                    str(self.python), str(self.script),
                    "--model", str(self.model),
                    "--reference", str(self.reference),
                    "--min-free-vram-mib", str(self.min_free_vram_mib),
                ]
                if self.event_reference is not None:
                    command += ["--event-reference", str(self.event_reference)]
                for cue in ("sigh", "groan"):
                    if cue in self.cue_audio:
                        command += [f"--{cue}-audio", str(self.cue_audio[cue])]
                self._process = subprocess.Popen(
                    command,
                    cwd=self.script.parents[1],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                )
                process = self._process
                self._reader = threading.Thread(
                    target=self._read_stdout, args=(process,),
                    name="ciel-chatterbox-rx", daemon=True,
                )
                self._stderr_reader = threading.Thread(
                    target=self._read_stderr, args=(process,),
                    name="ciel-chatterbox-log", daemon=True,
                )
                self._reader.start()
                self._stderr_reader.start()
        if not self._ready.wait(timeout=self.startup_timeout):
            self.close(force=True)
            raise ChatterboxWorkerError("Timed out loading Chatterbox Turbo")
        ready = self._startup_result
        if not ready.get("ready"):
            self.close(force=True)
            raise ChatterboxWorkerError(ready.get("error", "Chatterbox failed to initialize"))
        logger.info(
            "Chatterbox Turbo ready (pid=%s, model load %.2fs, reference %.2fs, "
            "free VRAM before load %s MiB).",
            ready.get("pid"),
            ready.get("model_load_seconds", 0.0),
            ready.get("conditioning_seconds", 0.0),
            ready.get("free_vram_before_load_mib", "unknown"),
        )
        self.startup_metrics = ready
        if self.event_reference is not None and not ready.get("vocal_cues_enabled"):
            logger.warning("Vocal effect reference unavailable; using native inline cues.")

    def start(self) -> dict:
        """Start the persistent model worker and return measured cold initialization data."""
        self._start()
        return self.startup_metrics.copy()

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ignoring malformed Chatterbox worker output")
                continue
            request_id = message.get("id")
            if request_id is None:
                self._startup_result = message
                self._ready.set()
                continue
            with self._pending_lock:
                response = self._pending.get(request_id)
            if response is None:
                self._remove_audio_file(message)
                continue
            try:
                response.put_nowait(message)
            except queue.Full:
                self._remove_audio_file(message)
        error = ChatterboxWorkerError("Chatterbox worker exited unexpectedly")
        self._startup_result = {"ready": False, "error": str(error)}
        self._ready.set()
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for response in pending:
            try:
                response.put_nowait({"error": str(error)})
            except queue.Full:
                pass

    @staticmethod
    def _read_stderr(process: subprocess.Popen[str]) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            logger.info("Chatterbox: %s", line.rstrip())

    @staticmethod
    def _remove_audio_file(message: dict) -> None:
        audio_path = message.get("audio_path")
        if audio_path:
            try:
                Path(audio_path).unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not remove stale Chatterbox audio file")

    def synthesize(
        self,
        text: str,
        *,
        canceled: Callable[[], bool] = lambda: False,
    ) -> tuple[np.ndarray, int] | None:
        """Generate one finite PCM waveform; canceled responses are deleted, never played."""
        self._start()
        if canceled():
            return None
        request_id = uuid.uuid4().hex
        response: queue.Queue[dict] = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = response
        process = self._process
        try:
            assert process is not None and process.stdin is not None
            with self._write_lock:
                process.stdin.write(json.dumps({"id": request_id, "text": text}) + "\n")
                process.stdin.flush()
            deadline = time.monotonic() + self.synthesis_timeout
            while not canceled():
                try:
                    message = response.get(timeout=0.025)
                except queue.Empty:
                    if time.monotonic() >= deadline:
                        self.close(force=True)
                        raise ChatterboxWorkerError("Timed out synthesizing Chatterbox speech")
                    if process.poll() is not None:
                        raise ChatterboxWorkerError("Chatterbox worker exited during synthesis")
                    continue
                if message.get("error"):
                    raise ChatterboxWorkerError(message["error"])
                path = Path(message["audio_path"])
                try:
                    audio = np.load(path, allow_pickle=False)
                finally:
                    path.unlink(missing_ok=True)
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                if not audio.size or not np.isfinite(audio).all():
                    raise ChatterboxWorkerError("Chatterbox returned empty or non-finite audio")
                return audio, int(message["sample_rate"])
            return None
        except (OSError, ValueError, KeyError, queue.Full) as err:
            raise ChatterboxWorkerError(str(err)) from err
        finally:
            with self._pending_lock:
                try:
                    response.put_nowait({"cancelled": True})
                except queue.Full:
                    pass
                self._pending.pop(request_id, None)
                try:
                    late = response.get_nowait()
                except queue.Empty:
                    late = None
            if late is not None and not late.get("cancelled"):
                self._remove_audio_file(late)

    def close(self, *, force: bool = False) -> None:
        """Ask the persistent model process to exit, then bound shutdown time."""
        with self._lifecycle_lock:
            self._closed = True
            process = self._process
            if process is None or process.poll() is not None:
                return
            if not force and process.stdin is not None:
                try:
                    with self._write_lock:
                        process.stdin.write('{"command":"shutdown"}\n')
                        process.stdin.flush()
                except OSError:
                    pass
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        logger.info("Chatterbox Turbo worker stopped.")
