"""Persistent local inference process for the pinned Chatterbox Turbo checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path


def emit(message: dict) -> None:
    """Write one protocol response without exposing text/audio content in logs."""
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--event-reference", type=Path)
    parser.add_argument("--sigh-audio", type=Path)
    parser.add_argument("--groan-audio", type=Path)
    parser.add_argument("--min-free-vram-mib", type=int, default=3000)
    args = parser.parse_args()
    model_dir, reference = args.model.resolve(), args.reference.resolve()
    if not reference.is_file():
        emit({"ready": False, "error": f"Reference WAV not found: {reference}"})
        return 2
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    protocol = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            import torch
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            # This Python environment contains Turbo, not necessarily the CIEL app.
            # Load the adjacent helper without importing CIEL or its audio dependencies.
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from vocal_cues import create_turbo_renderer

            torch.set_num_threads(4)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_mib = int(free_bytes / 2**20)
            if free_mib < args.min_free_vram_mib:
                raise RuntimeError(
                    f"Chatterbox startup needs {args.min_free_vram_mib} MiB free; "
                    f"only {free_mib} MiB is available"
                )
            inventory = json.loads((model_dir / "inventory.json").read_text())
            if inventory.get("revision") != "749d1c1a46eb10492095d68fbcf55691ccf137cd":
                raise ValueError("The local Turbo model is not the selected pinned revision")
            token_path = model_dir / "added_tokens.json"
            native_tokens = json.loads(token_path.read_text())
            supported = {
                "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
                "[sniff]", "[gasp]", "[chuckle]", "[laugh]",
            }
            if not supported <= native_tokens.keys():
                raise ValueError("The selected native Turbo event tokens are unavailable")
            load_started = time.perf_counter()
            model = ChatterboxTurboTTS.from_local(str(model_dir), device="cuda")
            load_seconds = time.perf_counter() - load_started
            condition_started = time.perf_counter()
            model.prepare_conditionals(str(reference), exaggeration=0.0)
            renderer = None
            cue_audio = {key: path for key, path in
                         (("sigh", args.sigh_audio), ("groan", args.groan_audio))
                         if path is not None}
            if args.event_reference is not None and (
                args.event_reference.is_file() or any(path.is_file() for path in cue_audio.values())
            ):
                try:
                    renderer = create_turbo_renderer(model, args.event_reference, cue_audio)
                except Exception:
                    traceback.print_exc(file=sys.stderr)
            torch.cuda.synchronize()
            condition_seconds = time.perf_counter() - condition_started
            pid = os.getpid()
        sys.stdout = protocol
        emit({
            "ready": True,
            "pid": pid,
            "sample_rate": model.sr,
            "model_load_seconds": load_seconds,
            "conditioning_seconds": condition_seconds,
            "free_vram_before_load_mib": free_mib,
            "total_vram_mib": int(total_bytes / 2**20),
            "vocal_cues_enabled": renderer is not None,
        })
    except Exception as err:
        sys.stdout = protocol
        emit({"ready": False, "error": f"{type(err).__name__}: {err}"})
        traceback.print_exc(file=sys.stderr)
        return 1

    for line in sys.stdin:
        output = None
        request = {}
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                break
            request_id, text = request["id"], request["text"]
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                audio = renderer.render(text) if renderer is not None else model.generate(text)
                torch.cuda.synchronize()
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if not audio.size or not np.isfinite(audio).all():
                raise ValueError("Model returned empty or non-finite audio")
            with tempfile.NamedTemporaryFile(
                prefix="ciel-chatterbox-", suffix=".npy", delete=False
            ) as temporary:
                output = Path(temporary.name)
            np.save(output, audio, allow_pickle=False)
            emit(
                {
                    "id": request_id,
                    "audio_path": str(output),
                    "sample_rate": int(model.sr),
                    "synthesis_seconds": time.perf_counter() - started,
                }
            )
        except Exception as err:
            if output is not None:
                output.unlink(missing_ok=True)
            emit({"id": request.get("id"), "error": f"{type(err).__name__}: {err}"})
            traceback.print_exc(file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
