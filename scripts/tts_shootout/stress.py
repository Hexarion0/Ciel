"""Paced sustained synthesis and idle recovery observations for provisional finalists."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import psutil

from .common import CONFIGURATIONS, SENTENCES, dump
from .providers import load
from .telemetry import Sampler
from .worker import measure


def run(directory: Path, identifier: str, seconds: float, recovery: float) -> Path:
    """Simulate response/audio pacing without playing audio or loading Ciel STT/LLM."""
    import torch

    directory = directory.resolve()
    destination = directory / f"sustained-{identifier}.json"
    if destination.exists():
        raise FileExistsError(destination)
    configuration = CONFIGURATIONS[identifier]
    reference = directory / "reference.wav"
    cache = Path("data/voice/states/shootout").resolve()
    torch.set_num_threads(4)
    provider = load(configuration, reference, cache)
    try:
        measure(provider, SENTENCES["short"])
        direct_cuda = configuration["device"] == "cuda" and configuration["engine"] != "chatterbox"
        if direct_cuda:
            torch.cuda.reset_peak_memory_stats()
        calls = []
        if configuration["engine"] == "styletts2":
            original = provider.module.compute_style

            def counted(*args, **kwargs):
                calls.append(1)
                return original(*args, **kwargs)

            provider.module.compute_style = counted
        elif configuration["engine"] == "xtts":
            original = provider.model.get_conditioning_latents

            def counted(*args, **kwargs):
                calls.append(1)
                return original(*args, **kwargs)

            provider.model.get_conditioning_latents = counted
        elif configuration["engine"] == "neutts":
            original = provider.model.encode_reference

            def counted(*args, **kwargs):
                calls.append(1)
                return original(*args, **kwargs)

            provider.model.encode_reference = counted
        process = psutil.Process()
        worker_pids = [child.pid for child in process.children(recursive=True)]
        rows = []
        # A separate sampler captures idle baselines and post-load recovery.
        with Sampler() as baseline:
            threading.Event().wait(2)
        started = time.perf_counter()
        with Sampler() as workload:
            while time.perf_counter() - started < seconds:
                name = list(SENTENCES)[len(rows) % len(SENTENCES)]
                torch.manual_seed(73 + len(rows))
                row = {"sentence": name, "turn": len(rows) + 1,
                       **measure(provider, SENTENCES[name])}
                tree = [process, *process.children(recursive=True)]
                row["rss_after_mib"] = sum(item.memory_info().rss for item in tree) / 1024**2
                row["cuda_initialized"] = torch.cuda.is_initialized()
                row["engine_cuda_initialized"] = (
                    row["cuda_initialized"] or configuration["engine"] == "chatterbox"
                )
                row["additional_conditioning_calls"] = len(calls)
                rows.append(row)
                # Retain natural audio cadence rather than driving continuous max GPU load.
                remaining = seconds - (time.perf_counter() - started)
                pause = min(max(0, row["audio_seconds"] - row["synthesis_seconds"]) + 0.35,
                            max(0, remaining))
                threading.Event().wait(pause)
                if len(rows) % 5 == 0:
                    print(identifier, len(rows), "turns",
                          round(time.perf_counter() - started), "seconds", flush=True)
        wall = time.perf_counter() - started
        with Sampler() as idle:
            threading.Event().wait(recovery)
        dump(destination, {
            "configuration_id": identifier, "configuration": configuration,
            "selection": "provisional technical finalist; subjective winner not chosen",
            "wall_seconds": wall, "turns": len(rows), "rows": rows,
            "baseline_with_model_loaded": baseline.metrics,
            "workload": workload.metrics, "idle_recovery_seconds": recovery,
            "idle_recovery": idle.metrics, "telemetry": workload.rows,
            "recovery_telemetry": idle.rows,
            "conditioning_calls_after_warmup": (
                len(calls) if configuration["engine"] != "chatterbox" else None
            ),
            "conditioning_measurement": (
                "native encoding entry point instrumented"
                if configuration["engine"] != "chatterbox"
                else "existing child worker not instrumented; inspect stable worker PIDs"
            ),
            "worker_pids_at_start": worker_pids,
            "worker_pids_at_end": [child.pid for child in process.children(recursive=True)],
            "reference_state": provider.state_path,
            "torch_cuda_peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / 1024**2 if direct_cuda else None
            ),
            "torch_cuda_peak_reserved_mib": (
                torch.cuda.max_memory_reserved() / 1024**2 if direct_cuda else None
            ),
            "limits": "Synthetic dialogue pacing, no full audible conversation, STT, or LLM. "
                      "Global GPU power/fan percentages include unrelated activity. RPM and "
                      "audible fan comfort unmeasured. A five-minute check is not a leak proof.",
        })
    finally:
        provider.close()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--recovery", type=float, default=30)
    args = parser.parse_args()
    if args.seconds < 1 or args.recovery < 1:
        parser.error("Durations must be positive")
    print(run(args.directory, args.configuration, args.seconds, args.recovery))


if __name__ == "__main__":
    main()
