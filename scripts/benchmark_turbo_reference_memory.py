#!/usr/bin/env python3
"""Check actual Turbo decoder parity and RAM after releasing reference-only networks.

Run with the existing Turbo Python environment. This stays on CPU, never plays
audio, and never downloads models. It does not start another full voice generator
on a GPU already occupied by the desktop assistant.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("data/voice/models/chatterbox-turbo"))
    parser.add_argument("--reference", type=Path,
                        default=Path("data/voice/references/ciel/reference-1.wav"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/diagnostics/speech-performance/decoder-release.json"))
    parser.add_argument("--reference-gpu-only", action="store_true", help=(
        "Measure only the real reference networks on GPU; leave the decoder on CPU. "
        "Use a separate --output to preserve the CPU parity report."))
    args = parser.parse_args()

    import numpy as np
    import psutil
    import soundfile as sf
    import torch
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.voice_encoder import VoiceEncoder
    from safetensors.torch import load_file

    worker_path = Path(__file__).resolve().parents[1] / "src/ciel/audio/turbo_worker.py"
    spec = importlib.util.spec_from_file_location("turbo_memory_probe", worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    torch.set_num_threads(2)
    process = psutil.Process()

    def memory() -> float:
        worker.release_unused_memory(torch)
        return process.memory_full_info().uss / 2**20

    decoder = S3Gen(meanflow=True).eval()
    weights = load_file(args.model / "s3gen_meanflow.safetensors")
    decoder.load_state_dict(weights, strict=True)
    del weights
    encoder = VoiceEncoder().eval()
    encoder.load_state_dict(load_file(args.model / "ve.safetensors"))
    audio, rate = sf.read(args.reference, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    with torch.inference_mode():
        reference = decoder.embed_ref(audio[:rate * 10], rate, device="cpu")
        tokens = [reference["prompt_token"][:, :length] for length in (32, 64)]

    if args.reference_gpu_only:
        free, _total = torch.cuda.mem_get_info()
        if free < 1024 * 2**20:
            raise SystemExit("Reference-only GPU probe needs at least 1 GiB free")
        for module in (encoder, decoder.tokenizer, decoder.speaker_encoder):
            module.to("cuda")
        del module
        wrapper = SimpleNamespace(s3gen=decoder, ve=encoder, conds=reference)
        del encoder
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        released = worker.release_reference_encoders(wrapper, torch)
        worker.release_unused_memory(torch)
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        report = {"cuda_allocated_before_mib": before / 2**20,
                  "cuda_allocated_after_mib": after / 2**20,
                  "released_cuda_mib": (before - after) / 2**20, **released,
                  "scope": "Isolated real reference networks on CUDA. Decoder stayed on CPU; "
                           "not a full GPU synthesis or live desktop memory sample."}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return

    def synthesize() -> tuple[list, list[float]]:
        waves, elapsed = [], []
        for index, speech in enumerate(tokens):
            with torch.inference_mode():
                torch.manual_seed(123 + index)
                started = time.perf_counter()
                wave = decoder.inference(
                    speech_tokens=speech, ref_dict=reference, n_cfm_timesteps=2,
                )[0].detach().clone()
                elapsed.append(time.perf_counter() - started)
                waves.append(wave)
        return waves, elapsed

    before, before_seconds = synthesize()
    wrapper = SimpleNamespace(s3gen=decoder, ve=encoder, conds=reference)
    del encoder
    baseline_ram = memory()
    released = worker.release_reference_encoders(wrapper, torch)
    after_ram = memory()
    after, after_seconds = synthesize()
    comparisons = [{
        "samples": a.numel(), "bitwise_equal": bool(torch.equal(a, b)),
        "max_absolute_difference": float(torch.max(torch.abs(a - b))),
        "before_sha256": hashlib.sha256(a.numpy().tobytes()).hexdigest(),
        "after_sha256": hashlib.sha256(b.numpy().tobytes()).hexdigest(),
    } for a, b in zip(before, after)]
    if not all(np.isfinite(wave.numpy()).all() for wave in after):
        raise ValueError("Decoder returned non-finite audio")
    report = {
        "private_ram_before_mib": baseline_ram, "private_ram_after_mib": after_ram,
        "released_ram_mib": baseline_ram - after_ram, **released,
        "before_seconds": before_seconds, "after_seconds": after_seconds,
        "comparisons": comparisons,
        "scope": "Actual pinned S3 decoder on CPU, fixed tokens, conditioning and seeds. "
                 "Not full T3 synthesis, live desktop memory, or an audible voice evaluation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not all(case["bitwise_equal"] for case in comparisons):
        raise SystemExit("Decoder parity failed")


if __name__ == "__main__":
    main()
