"""Generate Pocket-only comparisons and reusable exported reference states on CPU."""

from __future__ import annotations

import importlib.metadata
import json
import secrets
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .analyze import dump
from .audio import sha256

COMMON_LINE = (
    "You changed the configuration again, didn't you? "
    "Show me what you changed and I'll take a look."
)
STYLE_LINES = {
    "neutral": "Everything is running normally. I can check the rest of the system if you'd like.",
    "warm": "It's alright. Show me what happened and we'll work through it together.",
    "amused": "You changed it again? I suppose I shouldn't be surprised anymore.",
    "firm": "No. We're verifying that configuration before applying anything.",
    "focused": "The service is running correctly, but the environment variable is missing.",
    "soft": "You're still awake? That's alright. I'll stay here for a little while.",
}


def cached_state(model, reference: Path, cache: Path, *, language: str, quantize: bool):
    """Export once per waveform/model/quantization; do not re-encode each test sentence."""
    from pocket_tts import export_model_state

    fingerprint = {
        "reference_sha256": sha256(reference),
        "pocket_tts_version": importlib.metadata.version("pocket-tts"),
        "language": language, "quantization": "int8" if quantize else "fp32",
    }
    import hashlib

    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    destination = cache / f"{key}.safetensors"
    metadata = destination.with_suffix(".json")
    cache.mkdir(parents=True, exist_ok=True)
    if destination.exists() and metadata.exists():
        if json.loads(metadata.read_text()) != fingerprint:
            raise RuntimeError("Cached voice state provenance mismatch")
        state = model.get_state_for_audio_prompt(destination)
        encoded = False
    else:
        if destination.exists() or metadata.exists():
            raise RuntimeError(f"Incomplete cached state; preserve and inspect: {destination}")
        state = model.get_state_for_audio_prompt(reference, truncate=True)
        export_model_state(state, destination)
        dump(metadata, fingerprint)
        encoded = True
    return state, destination, encoded


def synthesize(model, state, text: str, path: Path, seed: int = 73) -> dict:
    """Use CPU RNG only and identical text/seed for the two numerical modes."""
    import torch

    if path.exists():
        raise FileExistsError(path)
    torch.random.default_generator.manual_seed(seed)
    start = time.perf_counter()
    first = None
    waves = []
    for tensor in model.generate_audio_stream(
        state, text, copy_state=True, stop=threading.Event(),
    ):
        if first is None:
            first = time.perf_counter() - start
        waves.append(tensor.detach().numpy())
    elapsed = time.perf_counter() - start
    if not waves:
        raise RuntimeError("Pocket returned no audio")
    wave = np.concatenate(waves).astype(np.float32)
    if not np.isfinite(wave).all():
        raise RuntimeError("Pocket produced invalid audio")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave, model.sample_rate, subtype="FLOAT")
    return {"audio": str(path), "text": text, "seed": seed,
            "audio_ready_seconds": first, "synthesis_seconds": elapsed,
            "duration_seconds": wave.size / model.sample_rate,
            "sample_rate": model.sample_rate, "cuda_initialized": torch.cuda.is_initialized()}


def run(directory: Path, cache: Path, language: str = "english_2026-09") -> None:
    """Prepare both modes, plus anonymous A/B files; never update runtime config."""
    import torch
    from pocket_tts import TTSModel

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("human_acceptance"):
        raise ValueError("Use a new run for additional previews of an accepted bank")
    for quantize in (False, True):
        mode = "int8" if quantize else "fp32"
        pending = [row for row in manifest["candidates"] if mode not in row.get("previews", {})]
        if not pending:
            continue
        model = TTSModel.load_model(language=language, quantize=quantize)
        if str(model.device) != "cpu":
            raise RuntimeError("Voice preview inference must stay on CPU")
        for row in pending:
            reference = directory / row["reference"]
            state, destination, encoded = cached_state(
                model, reference, cache, language=language, quantize=quantize,
            )
            row.setdefault("states", {})[mode] = {
                "path": str(destination.resolve()), "encoded_this_run": encoded,
                "reference_sha256": sha256(reference), "language": language,
                "pocket_tts_version": importlib.metadata.version("pocket-tts"),
            }
            outputs = {}
            for kind, text in (("common", COMMON_LINE),
                               ("style", STYLE_LINES[row["intended_style"]])):
                path = directory / "previews" / row["id"] / f"{kind}-{mode}.wav"
                outputs[kind] = synthesize(model, state, text, path)
                outputs[kind]["audio"] = str(path.relative_to(directory))
            row.setdefault("previews", {})[mode] = outputs
            dump(manifest_path, manifest)
            print(mode, row["id"], "reference encoded" if encoded else "state reused", flush=True)
        del model
    for row in manifest["candidates"]:
        if row.get("blind"):
            continue
        modes = ["fp32", "int8"]
        if secrets.randbelow(2):
            modes.reverse()
        mapping = {}
        for label, mode in zip(("A", "B"), modes):
            path = directory / "blind" / row["id"] / f"sample-{label}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            source = directory / row["previews"][mode]["common"]["audio"]
            shutil.copy2(source, path)
            mapping[label] = {"mode": mode, "audio": str(path.relative_to(directory)),
                              "audio_sha256": sha256(path)}
        row["blind"] = mapping
    manifest["preview_configuration"] = {
        "engine": "Pocket TTS", "device": "cpu", "language": language,
        "seed": 73, "common_text": COMMON_LINE, "style_texts": STYLE_LINES,
        "cuda_initialized": torch.cuda.is_initialized(),
        "quality_acceptance": "Pending human listening; speed does not choose the default mode.",
    }
    dump(manifest_path, manifest)
    print("Prepared previews and blind comparisons; runtime configuration preserved.")
