#!/usr/bin/env python3
"""Compare local recognition on a supplied, transcribed corpus; never record or download.

Each invocation measures one backend in its own process. A manifest contains a
``samples`` array with ``id``, ``path`` and ``text`` fields. Streaming replay is
paced like a microphone: report finalization separately from total inference work.
Results describe these recordings, not the owner's microphone or accent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import psutil
import soundfile as sf
from scipy.signal import resample_poly


def word_error(reference: str, actual: str) -> dict[str, int | float]:
    """Return edit distance after case/punctuation normalization, preserving contractions."""
    expected = re.findall(r"[\w]+(?:'[\w]+)*", reference.casefold().replace("’", "'"))
    words = re.findall(r"[\w]+(?:'[\w]+)*", actual.casefold().replace("’", "'"))
    previous = list(range(len(words) + 1))
    for index, word in enumerate(expected, 1):
        current = [index]
        for offset, candidate in enumerate(words, 1):
            current.append(min(current[-1] + 1, previous[offset] + 1,
                               previous[offset - 1] + (word != candidate)))
        previous = current
    return {"errors": previous[-1], "words": len(expected),
            "wer": previous[-1] / max(1, len(expected))}


def load_audio(path: Path) -> np.ndarray:
    """Load finite mono speech and resample to the recognizers' common 16 kHz input."""
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite audio: {path}")
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor)
    return audio.astype(np.float32)


class ResourceSampler:
    """Sample this process only; other desktop applications do not count as savings."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.peak_uss_mib = 0.0
        self.peak_pss_mib = 0.0
        self.peak_gpu_mib = 0
        self.gpu_probe_available = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            memory = self.process.memory_full_info()
            self.peak_uss_mib = max(self.peak_uss_mib, memory.uss / 2**20)
            self.peak_pss_mib = max(self.peak_pss_mib, getattr(memory, "pss", 0) / 2**20)
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1, check=True,
                )
                self.gpu_probe_available = True
                for line in result.stdout.splitlines():
                    pid, used = line.split(",", 1)
                    if int(pid) == os.getpid():
                        self.peak_gpu_mib = max(self.peak_gpu_mib, int(used.strip()))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop.wait(0.25)

    def __enter__(self) -> ResourceSampler:
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=2)


def stream_audio(transcriber: Any, audio: np.ndarray) -> dict[str, Any]:
    """Replay at real-time cadence and retain updated lines by their stable IDs."""
    lines: dict[int, tuple[float, str]] = {}
    first_text = None
    errors = []
    from moonshine_voice import Error

    started = time.perf_counter()
    cpu_started = time.process_time()

    def receive(event: Any) -> None:
        nonlocal first_text
        if isinstance(event, Error):
            errors.append(str(event))
        line = getattr(event, "line", None)
        if line is not None:
            lines[line.line_id] = (line.start_time, line.text)
            if line.text and first_text is None:
                first_text = time.perf_counter() - started

    stream = transcriber.create_stream(update_interval=0.5)
    stream.add_listener(receive)
    max_lag = 0.0
    stream.start()
    try:
        for index in range(0, audio.size, 1280):
            frame = audio[index:index + 1280]
            due = started + (index + frame.size) / 16000
            time.sleep(max(0, due - time.perf_counter()))
            max_lag = max(max_lag, time.perf_counter() - due)
            stream.add_audio(frame.tolist(), sample_rate=16000)
        final_started = time.perf_counter()
        final = stream.stop()
        final_seconds = time.perf_counter() - final_started
        if final is None or errors:
            raise RuntimeError(f"Moonshine failed to finalize: {errors}")
        for line in final.lines:
            lines[line.line_id] = (line.start_time, line.text)
        finished = time.perf_counter()
        return {
            "text": " ".join(text for _, text in sorted(lines.values()) if text),
            "wall_seconds": finished - started,
            "cpu_seconds": time.process_time() - cpu_started,
            "first_partial_seconds": first_text,
            "finalize_seconds": final_seconds,
            "after_last_frame_seconds": max(0, finished - started - audio.size / 16000),
            "max_ingestion_lag_seconds": max_lag,
        }
    finally:
        stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["whisper", "moonshine-batch", "moonshine-stream"],
                        required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-arch", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.backend.startswith("moonshine") and args.model_path is None:
        parser.error("Moonshine needs an existing --model-path; this script never downloads")
    manifest = json.loads(args.manifest.read_text())
    if not manifest.get("samples"):
        parser.error("The manifest must contain at least one transcribed sample")
    samples = [(sample, load_audio(Path(sample["path"]))) for sample in manifest["samples"]]
    report: dict[str, Any] = {
        "backend": args.backend, "manifest_sha256": hashlib.sha256(
            args.manifest.read_bytes()).hexdigest(), "rows": [],
        "moonshine_single_thread": os.getenv("MOONSHINE_ORT_SINGLE_THREAD", "0"),
        "notes": ["Local replay; no microphone, playback, upload or LLM request.",
                  "First repeat includes the first inference; subsequent repeats are warm."],
    }
    with ResourceSampler() as resources:
        started = time.perf_counter()
        if args.backend == "whisper":
            from ciel.audio.stt import SpeechToText
            from ciel.config import get_settings

            cfg = get_settings().audio
            recognizer = SpeechToText(
                model_size=cfg.stt_model, device=cfg.stt_device,
                compute_type=cfg.stt_compute_type, language=cfg.stt_language,
                min_confidence=cfg.stt_min_confidence, retry_confidence=cfg.stt_retry_confidence,
                retry_model=cfg.stt_retry_model, retry_min_free_mb=cfg.stt_retry_min_free_mb,
            )
            if not recognizer.wait_ready(120):
                raise RuntimeError("Configured recognition did not become ready")
            report["model"] = {"name": cfg.stt_model, "device": recognizer.device,
                               "compute_type": recognizer.compute_type}
        else:
            from moonshine_voice import ModelArch, Transcriber, __version__

            recognizer = Transcriber(args.model_path, ModelArch(args.model_arch))
            report["model"] = {"path": str(args.model_path), "arch": args.model_arch,
                               "version": __version__, "device": "cpu"}
        report["load_seconds"] = time.perf_counter() - started
        for repeat in range(args.repeats):
            for sample, audio in samples:
                started = time.perf_counter()
                cpu_started = time.process_time()
                if args.backend == "whisper":
                    result = recognizer.transcribe_detailed(audio, beam_size=cfg.stt_beam_size)
                    result = {key: result.get(key) for key in (
                        "text", "confidence", "retried", "needs_repeat")}
                elif args.backend == "moonshine-batch":
                    transcript = recognizer.transcribe_without_streaming(audio.tolist(), 16000)
                    result = {"text": " ".join(line.text for line in transcript.lines)}
                else:
                    result = stream_audio(recognizer, audio)
                result.setdefault("wall_seconds", time.perf_counter() - started)
                result.setdefault("cpu_seconds", time.process_time() - cpu_started)
                result.update(id=sample["id"], repeat=repeat, duration_seconds=audio.size / 16000,
                              accuracy=word_error(sample["text"], result["text"]))
                report["rows"].append(result)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({k: result[k] for k in ("id", "repeat", "wall_seconds",
                                                       "accuracy")}), flush=True)
        if args.backend != "whisper":
            recognizer.close()
    rows = report["rows"]
    report["summary"] = {
        "weighted_wer": sum(row["accuracy"]["errors"] for row in rows)
        / max(1, sum(row["accuracy"]["words"] for row in rows)),
        "median_wall_seconds": median(row["wall_seconds"] for row in rows),
        "peak_private_ram_mib": resources.peak_uss_mib,
        "peak_proportional_ram_mib": resources.peak_pss_mib,
        "peak_process_gpu_mib": (
            resources.peak_gpu_mib if resources.gpu_probe_available else None),
        "cpu_seconds_per_audio_second": sum(row["cpu_seconds"] for row in rows)
        / max(0.001, sum(row["duration_seconds"] for row in rows)),
    }
    if args.backend == "moonshine-stream":
        report["summary"]["median_after_last_frame_seconds"] = median(
            row["after_last_frame_seconds"] for row in rows)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
