"""One isolated engine/configuration per process, with multiple warm trials."""

from __future__ import annotations

import argparse
import os
import statistics
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import CONFIGURATIONS, SENTENCES, digest, dump
from .telemetry import Sampler

BOOT_STARTED = float(os.environ.get("CIEL_SHOOTOUT_STARTED", str(time.perf_counter())))


def measure(provider, text: str, destination: Path | None = None) -> dict:
    """Count first usable samples and full generation separately; exclude WAV writes."""
    waves, first, rate = [], None, 0
    with Sampler() as sampler:
        started = time.perf_counter()
        for wave, rate in provider.stream(text):
            wave = np.asarray(wave, dtype=np.float32).reshape(-1)
            if not wave.size or not np.isfinite(wave).all():
                raise ValueError("Engine returned empty/non-finite audio")
            if first is None:
                first = time.perf_counter() - started
            waves.append(wave)
        elapsed = time.perf_counter() - started
    if not waves:
        raise ValueError("Engine produced no audio; no fallback is permitted in a shootout")
    audio = np.concatenate(waves)
    if np.max(np.abs(audio)) > 5:
        raise ValueError("Engine produced abnormal amplitude; preserve logs, do not audition")
    duration = audio.size / rate
    if destination:
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, audio, rate, subtype="FLOAT")
    return {"first_audio_seconds": first, "synthesis_seconds": elapsed,
            "audio_seconds": duration, "rtf": elapsed / duration, "sample_rate": rate,
            "chunks": len(waves), "peak_amplitude": float(np.max(np.abs(audio))),
            "rms_dbfs": float(20 * np.log10(max(1e-12, np.sqrt(np.mean(audio**2))))),
            "clipping_fraction": float(np.mean(np.abs(audio) >= 1)),
            "audio": str(destination) if destination else None, "resources": sampler.metrics}


def run(args) -> dict:
    import torch

    from .providers import load

    configuration = CONFIGURATIONS[args.configuration]
    torch.set_num_threads(4)
    result = {"configuration_id": args.configuration, "configuration": configuration,
              "reference": str(args.reference), "reference_sha256": digest(args.reference),
              "status": "running", "rows": [], "versions": {}, "threads": 4,
              "listening_acceptance": None}
    names = args.sentences or list(SENTENCES)
    if args.sentences is None and args.configuration == "xtts-cpu-fp32":
        names = ["short", "conversational"]
        result["coverage_reason"] = (
            "Initial CPU probe produced RTF 3.2–5.3 on this i5; not practical for conversation. "
            "Repeat short/conversational trials; all seven texts are covered by XTTS CUDA."
        )
    result["test_sentences"] = names
    for package in ("torch", "torchaudio", "coqui-tts", "transformers", "phonemizer",
                    "neutts", "neucodec", "llama-cpp-python", "resemble-perth"):
        try:
            result["versions"][package] = version(package)
        except PackageNotFoundError:
            pass
    provider = None
    try:
        if configuration["device"] == "cuda" and configuration["engine"] != "chatterbox":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; requested configuration was not substituted")
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_capability"] = torch.cuda.get_device_capability(0)
        with Sampler() as sampler:
            started = time.perf_counter()
            provider = load(configuration, args.reference, args.cache)
            result["model_and_conditioning_seconds"] = time.perf_counter() - started
            result["cold_startup_seconds"] = time.perf_counter() - BOOT_STARTED
        result["startup_resources"] = sampler.metrics
        result["conditioning"] = {
            "state": provider.state_path, "encoded_this_run": provider.conditioning_encoded,
            "seconds": provider.conditioning_seconds, "loads": provider.conditioning_loads,
        }
        direct_cuda = configuration["device"] == "cuda" and configuration["engine"] != "chatterbox"
        if direct_cuda:
            torch.cuda.reset_peak_memory_stats()
        result["streaming"] = provider.streaming
        result["runtime_details"] = getattr(provider, "runtime_details", {})
        torch.manual_seed(73)
        result["cold_synthesis"] = measure(provider, SENTENCES["short"])
        dump(args.output, result)
        for iteration in range(args.iterations):
            for name in names:
                text = SENTENCES[name]
                torch.manual_seed(73 + iteration)
                audio = (args.output.parent / "audio" / args.configuration / f"{name}.wav"
                         if iteration == 0 else None)
                row = {"sentence": name, "iteration": iteration + 1,
                       **measure(provider, text, audio)}
                if direct_cuda:
                    allocated = torch.cuda.max_memory_allocated() / 1024**2
                    row["torch_cuda_peak_allocated_mib"] = allocated
                    row["torch_cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
                result["rows"].append(row)
                dump(args.output, result)
                print(args.configuration, name, iteration + 1, flush=True)
        result["cuda_initialized"] = torch.cuda.is_initialized()
        result["engine_cuda_initialized"] = (
            result["cuda_initialized"] or configuration["engine"] == "chatterbox"
        )
        result["conditioning"]["loads_after_repeated_synthesis"] = provider.conditioning_loads
        result["status"] = "ok"
        result["summary"] = {}
        for name in names:
            rows = [row for row in result["rows"] if row["sentence"] == name]
            result["summary"][name] = {
                key + "_median": statistics.median(row[key] for row in rows)
                for key in ("first_audio_seconds", "synthesis_seconds", "rtf")
            }
    except Exception as error:
        result["status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        if provider is not None:
            provider.close()
        result["cuda_initialized"] = torch.cuda.is_initialized()
        dump(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--sentences", nargs="+", choices=SENTENCES)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("At least two warm iterations are required")
    args.reference, args.cache, args.output = (
        path.resolve() for path in (args.reference, args.cache, args.output)
    )
    result = run(args)
    print(result["status"], result.get("error", "complete"), flush=True)


if __name__ == "__main__":
    main()
