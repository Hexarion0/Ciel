"""Paced CPU-only provisional-reference stability check, without playing private audio."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from .analyze import dump

LINES = (
    "Everything is running normally. I can check the rest of the system too.",
    "Let's review the configuration together before restarting the service.",
    "The environment variable is missing. Fix that first, then restart the process.",
    "You don't have to rush. Show me what happened and we'll work through it together.",
    "You changed it again? I was wondering how long that would take.",
    "The driver is loaded correctly. We can check whether the settings survive a reboot.",
)


def run(directory: Path, reference_id: str, duration: float = 300) -> Path:
    """Simulate wall-clock conversation cadence and periodically cancel live inference.

    This validates synthesis streaming/caching and memory observations. Real
    speaker playback/barge-in is covered by Ciel's existing integration tests;
    this tool does not claim microphone or perceptual voice-quality acceptance.
    """
    import torch

    from ciel.audio.pocket import PocketTTS
    from ciel.audio.voice_style import TTSRequest
    from scripts.benchmark_tts import gpu_snapshot

    manifest = json.loads((directory / "manifest.json").read_text())
    reference = next(row for row in manifest["candidates"] if row["id"] == reference_id)
    state_path = Path(reference["states"]["fp32"]["path"])
    provider = PocketTTS(directory, neutral_reference=state_path, quantization="none")
    provider.warmup()
    model = provider._model
    state_loads = []
    original_state = model.get_state_for_audio_prompt

    def counted(*args, **kwargs):
        state_loads.append(1)
        return original_state(*args, **kwargs)

    model.get_state_for_audio_prompt = counted
    process = psutil.Process()
    baseline_threads = {thread.ident for thread in threading.enumerate()}
    rows = []
    started = time.monotonic()
    try:
        while time.monotonic() - started < duration:
            index = len(rows)
            stop = threading.Event()
            text = LINES[index % len(LINES)]
            cancel = index > 0 and index % 7 == 0
            first = None
            samples = 0
            start = time.monotonic()
            for wave, rate in provider.stream(TTSRequest(text), stop):
                if first is None:
                    first = time.monotonic() - start
                samples += wave.size
                if cancel and samples >= rate * 0.24:
                    stop.set()
            elapsed = time.monotonic() - start
            if not samples and not cancel:
                raise RuntimeError("Repeated Pocket synthesis returned no audio")
            if torch.cuda.is_initialized():
                raise RuntimeError("Pocket unexpectedly initialized CUDA")
            extra_threads = [thread.name for thread in threading.enumerate()
                             if thread.ident not in baseline_threads]
            if extra_threads:
                raise RuntimeError(f"Pocket worker threads did not unwind: {extra_threads}")
            rows.append({
                "turn": index + 1, "elapsed_seconds": time.monotonic() - started,
                "first_audio_seconds": first, "synthesis_seconds": elapsed,
                "generated_audio_seconds": samples / rate, "canceled": cancel,
                "rss_mib": process.memory_info().rss / 1024**2,
                "cached_reference_states": len(provider._states),
                "additional_reference_encoding_or_loading_calls": len(state_loads),
                "cuda_initialized": False, "worker_threads_remaining": extra_threads,
                "gpu": gpu_snapshot({process.pid}) if index % 5 == 0 else {},
            })
            # Pace with synthetic audio duration + a short response gap, without
            # blocking a controller tool call: this runs in a retained exec session.
            pause = min(max(0, samples / rate - elapsed) + 0.3,
                        max(0, duration - (time.monotonic() - started)))
            stop.wait(pause)
            if (index + 1) % 10 == 0:
                print("CPU conversation turns:", index + 1, "elapsed:",
                      round(time.monotonic() - started), "seconds", flush=True)
    finally:
        provider.close()
    warm = [row["rss_mib"] for row in rows[5:] if not row["canceled"]]
    width = max(1, len(warm) // 4)
    report = {
        "candidate": reference_id, "reference_state": str(state_path),
        "provisional": not bool(manifest.get("human_acceptance")),
        "wall_seconds": time.monotonic() - started,
        "total_generated_audio_seconds": sum(row["generated_audio_seconds"] for row in rows),
        "turns": len(rows), "interrupted_generations": sum(row["canceled"] for row in rows),
        "warm_rss_first_quartile_median_mib": float(np.median(warm[:width])) if warm else None,
        "warm_rss_last_quartile_median_mib": float(np.median(warm[-width:])) if warm else None,
        "rss_peak_mib": max(row["rss_mib"] for row in rows),
        "cuda_initialized": False, "additional_reference_loads": len(state_loads),
        "device": "cpu", "rows": rows,
        "engine_vram_peak_mib": max(
            (row["gpu"]["engine_vram_mib"] for row in rows
             if "engine_vram_mib" in row["gpu"]), default=None,
        ),
        "limits": "Synthesis streaming/cancellation only; no STT, audible playback, "
                  "subjective listening, or proof against long-term leaks. GPU activity "
                  "and power describe the entire device; VRAM is attributed by process.",
    }
    path = directory / f"stress-{reference_id}.json"
    if path.exists():
        raise FileExistsError(path)
    dump(path, report)
    return path
