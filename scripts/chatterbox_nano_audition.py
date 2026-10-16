"""Offline CPU-only Nano audition; never registers or changes Ciel's runtime voice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from importlib.metadata import version
from pathlib import Path

from scripts.tts_finetune.common import EVALUATION_TEXTS, ROOT, digest, dump

MODEL_ID = "ResembleAI/chatterbox-nano"
REVISION = "71ccd1d0081b430592cea481f4307e764e07bc64"
MODEL_DIR = ROOT / "data/voice/models/chatterbox-nano"
REFERENCE = ROOT / "data/voice/references/ciel/reference-1.wav"
BASELINE = ROOT / "data/voice/shootout-research/chatterbox-accepted-baseline"
FILES = (
    "ve.safetensors", "t3_nano_v1.safetensors", "s3gen_meanflow.safetensors",
    "added_tokens.json", "merges.txt", "special_tokens_map.json",
    "tokenizer_config.json", "vocab.json", "t3_nano_v1.yaml", "README.md",
)


def inventory(directory: Path) -> Path:
    """Fingerprint the pinned download before using cached speaker conditioning."""
    missing = [name for name in FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run download first; missing Nano files: {missing}")
    path = directory / "inventory.json"
    dump(path, {"model_id": MODEL_ID, "revision": REVISION, "license": "MIT",
                "files": {name: digest(directory / name) for name in FILES}})
    return path


class NanoProvider:
    """Native Nano API, one persistent CPU model and one cached reference state."""

    streaming = "full utterance; upstream generate() does not yield audio chunks"

    def __init__(self, directory: Path, reference: Path, cache: Path, threads: int = 4):
        os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                          TRANSFORMERS_OFFLINE="1")
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals

        self.torch = torch
        self.assert_cpu()
        torch.set_num_threads(threads)
        self.model = ChatterboxTurboTTS.from_local(str(directory), device="cpu", nano=True)
        for module in (self.model.t3, self.model.s3gen, self.model.ve):
            if any(parameter.device.type != "cpu" for parameter in module.parameters()):
                raise RuntimeError("Nano audition requires every model parameter on CPU")
        fingerprint = {"model": digest(directory / "inventory.json"),
                       "reference": digest(reference), "exaggeration": 0.0,
                       "chatterbox": version("chatterbox-tts"), "torch": torch.__version__}
        key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        cache.mkdir(parents=True, exist_ok=True)
        state = cache / f"{key}.pt"
        self.conditioning_encoded = not state.is_file()
        started = time.perf_counter()
        with torch.inference_mode():
            if state.is_file():
                self.model.conds = Conditionals.load(state, map_location="cpu").to("cpu")
            else:
                self.model.prepare_conditionals(str(reference), exaggeration=0.0)
                temporary = state.with_suffix(".pt.tmp")
                self.model.conds.save(temporary)
                temporary.replace(state)
        self.conditioning_seconds = time.perf_counter() - started
        self.conditioning_loads = 1
        self.state_path = str(state)
        self.assert_cpu()

    def assert_cpu(self):
        """Checking initialization does not initialize CUDA."""
        if self.torch.cuda.is_initialized():
            raise RuntimeError("CUDA was initialized; use a fresh CPU-only audition process")

    def stream(self, text: str):
        self.assert_cpu()
        with self.torch.inference_mode():
            wave = self.model.generate(text)
        self.assert_cpu()
        yield wave.detach().cpu().numpy().reshape(-1), self.model.sr

    def close(self):
        self.model = None


def run(output: Path, iterations: int = 3, threads: int = 4) -> Path:
    """Save seven raw test clips and multiple warm trials, without a provider fallback."""
    if iterations < 3 or not 1 <= threads <= 12:
        raise ValueError("Use at least three warm iterations and 1–12 CPU threads")
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from scripts.tts_shootout.telemetry import Sampler
    from scripts.tts_shootout.worker import measure

    inventory(MODEL_DIR)
    output.mkdir(parents=True)
    result = {"status": "running", "label": "Chatterbox Nano · CPU FP32",
              "model_id": MODEL_ID, "revision": REVISION, "checkpoint_sha256": None,
              "reference_sha256": digest(REFERENCE), "texts": EVALUATION_TEXTS,
              "production_enabled": False, "threads": threads, "rows": [],
              "versions": {name: version(name) for name in
                           ("torch", "chatterbox-tts", "transformers", "resemble-perth")},
              "resource_scope": "CPU/RAM and engine VRAM: process tree. GPU power, fan %, "
                                "utilization and temperature: whole device, not attribution.",
              "streaming": NanoProvider.streaming, "listening_acceptance": None}
    path = output / "result.json"
    provider = None
    try:
        with Sampler() as sampler:
            provider = NanoProvider(MODEL_DIR, REFERENCE, ROOT / "data/voice/states/nano", threads)
        result["cold_startup_seconds"] = time.perf_counter() - started
        result["startup_resources"] = sampler.metrics
        result["conditioning"] = {"state": provider.state_path,
                                  "encoded_this_run": provider.conditioning_encoded,
                                  "seconds": provider.conditioning_seconds, "loads": 1}
        provider.torch.manual_seed(73)
        result["cold_synthesis"] = measure(provider, EVALUATION_TEXTS["short"])
        dump(path, result)
        for iteration in range(1, iterations + 1):
            for name, text in EVALUATION_TEXTS.items():
                provider.torch.manual_seed(73 + iteration)
                audio = output / "audio" / f"{name}.wav" if iteration == 1 else None
                row = {"sentence": name, "text": text, "iteration": iteration,
                       **measure(provider, text, audio)}
                result["rows"].append(row)
                dump(path, result)
                print(json.dumps({"sentence": name, "iteration": iteration,
                                  "seconds": round(row["synthesis_seconds"], 3),
                                  "rtf": round(row["rtf"], 3)}), flush=True)
        result["summary"] = {
            name: {key + "_median": statistics.median(
                row[key] for row in result["rows"] if row["sentence"] == name)
                for key in ("first_audio_seconds", "synthesis_seconds", "rtf")}
            for name in EVALUATION_TEXTS
        }
        result["cuda_initialized"] = provider.torch.cuda.is_initialized()
        result["conditioning"]["loads_after_synthesis"] = provider.conditioning_loads
        result["status"] = "complete_awaiting_listening"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if provider:
            provider.close()
        dump(path, result)
    return path


def page(result_path: Path, output: Path) -> Path:
    """Attenuate to the quieter clip's loudness for each pair; preserve original WAVs."""
    import numpy as np
    import pyloudnorm as ln
    import soundfile as sf

    from scripts.tts_finetune.evaluate import listening

    if output.exists():
        raise FileExistsError(output)
    nano = json.loads(result_path.read_text())
    baseline = json.loads((BASELINE / "manifest.json").read_text())
    if (nano["status"] != "complete_awaiting_listening"
            or nano["texts"] != EVALUATION_TEXTS or baseline["texts"] != EVALUATION_TEXTS
            or baseline["reference_sha256"] != nano["reference_sha256"]):
        raise ValueError("Comparison requires complete synthesis, identical reference and texts")
    results = []
    for label in ("nano", "turbo"):
        value = {"status": "complete_awaiting_listening", "texts": EVALUATION_TEXTS,
                 "reference_sha256": nano["reference_sha256"], "checkpoint_sha256": None,
                 "label": nano["label"] if label == "nano" else "Accepted Chatterbox Turbo",
                 "rows": []}
        results.append((output / "matched" / label / "result.json", value))
    for name, text in EVALUATION_TEXTS.items():
        row = next(row for row in nano["rows"]
                   if row["sentence"] == name and row["iteration"] == 1)
        nano_audio = Path(row["audio"])
        if not nano_audio.is_absolute():
            nano_audio = result_path.parent / nano_audio
        baseline_audio = BASELINE / baseline["audio"][name]["file"]
        if digest(baseline_audio) != baseline["audio"][name]["sha256"]:
            raise ValueError("Accepted baseline audio hash changed")
        clips = [sf.read(path, dtype="float32") for path in (nano_audio, baseline_audio)]
        loudness = [float(ln.Meter(rate).integrated_loudness(wave)) for wave, rate in clips]
        if not np.isfinite(loudness).all():
            raise ValueError("Cannot level-match silent/non-finite audio")
        target = min(loudness)
        for (wave, rate), level in zip(clips, loudness, strict=True):
            target = min(target, level - 1 - 20 * np.log10(max(1e-12, np.max(np.abs(wave)))))
        for (path, value), (wave, rate), level in zip(results, clips, loudness, strict=True):
            gain_db = target - level
            destination = path.parent / f"{name}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(destination, wave * 10**(gain_db / 20), rate, subtype="FLOAT")
            value["rows"].append({"sentence": name, "text": text, "iteration": 1,
                                  "audio": destination.name, "gain_db": gain_db,
                                  "original_lufs": level, "matched_lufs": target})
    for path, value in results:
        dump(path, value)
    return listening(output / "listening", [path for path, _ in results])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    runner = sub.add_parser("run")
    runner.add_argument("--output", required=True, type=Path)
    runner.add_argument("--iterations", default=3, type=int)
    runner.add_argument("--threads", default=4, type=int)
    listener = sub.add_parser("page")
    listener.add_argument("--result", required=True, type=Path)
    listener.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "download":
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=MODEL_ID, revision=REVISION, allow_patterns=list(FILES),
                          local_dir=str(MODEL_DIR), max_workers=3)
        print(inventory(MODEL_DIR))
    elif args.command == "run":
        print(run(args.output.resolve(), args.iterations, args.threads))
    else:
        print(page(args.result.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
