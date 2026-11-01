"""Persistent local inference process for the pinned Chatterbox Turbo checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import queue
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any


def prepare_voice_reference(
    model: Any, torch: Any, reference: Path, event_reference: Path | None,
    cue_audio: dict[str, Path], create_renderer: Callable[..., Any],
) -> Any:
    """Keep both normal and event conditioning free of retained training activations."""
    with torch.inference_mode():
        model.prepare_conditionals(str(reference), exaggeration=0.0)
        if event_reference is not None and (
            event_reference.is_file() or any(path.is_file() for path in cue_audio.values())
        ):
            try:
                return create_renderer(model, event_reference, cue_audio)
            except Exception:
                traceback.print_exc(file=sys.stderr)
    return None


def release_reference_encoders(model: Any, torch: Any) -> dict[str, int]:
    """Release reference-only networks in this pinned, fixed-voice inference worker.

    Turbo's generation uses prepared ``conds``, not the voice encoder, speech
    tokenizer or reference speaker encoder. S3Gen still discovers its device
    through the tokenizer's first parameter, so preserve a zero-element device
    anchor. The worker never accepts a new voice reference after startup.
    """
    if model.conds is None:
        raise ValueError("Voice conditioning must be prepared before releasing encoders")
    encoders = (model.ve, model.s3gen.tokenizer, model.s3gen.speaker_encoder)
    # Validate and account for the complete layout before changing any module.
    first = next(model.s3gen.tokenizer.parameters())
    device, dtype = first.device, first.dtype
    size = sum(
        tensor.numel() * tensor.element_size()
        for encoder in encoders
        for tensors in (encoder.parameters(), encoder.buffers())
        for tensor in tensors
    )

    class PreparedReference(torch.nn.Module):
        """Preserve upstream S3Gen.device without retaining reference model weights."""

        def __init__(self) -> None:
            super().__init__()
            self.register_parameter("device_anchor", torch.nn.Parameter(
                torch.empty(0, device=device, dtype=dtype), requires_grad=False,
            ))

        def forward(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("This worker's voice reference is already prepared")

    anchor = PreparedReference()
    model.s3gen.tokenizer = anchor
    model.s3gen.speaker_encoder = None
    model.ve = None
    return {"reference_encoder_bytes_released": size}


def release_unused_memory(torch: Any) -> dict[str, float | str]:
    """Release only unused allocator pages; keep model weights and voice tensors resident."""
    gc.collect()
    try:
        before = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_reserved()
        metrics = {
            "cuda_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 2),
            "cuda_reserved_mib": round(after / 2**20, 2),
            "released_cuda_cache_mib": round(max(0, before - after) / 2**20, 2),
        }
    except RuntimeError:
        # Optional maintenance must not turn an otherwise working voice into fallback.
        traceback.print_exc(file=sys.stderr)
        metrics = {"memory_cleanup_error": "RuntimeError"}
    # Large temporary CPU checkpoint/reference allocations can leave free heap
    # pages resident. glibc can return those pages without unloading live tensors.
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            trim = ctypes.CDLL(None).malloc_trim
            trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
            trim(0)
        except (AttributeError, OSError):
            pass
    return metrics


def idle_requests(
    stream: Iterable[str], cleanup: Callable[[], object], *, timeout: float = 30.0,
) -> Iterator[str]:
    """Trim once per idle period without delaying buffered requests or active synthesis."""
    requests = queue.Queue(maxsize=2)

    def read() -> None:
        try:
            for line in stream:
                requests.put(line)
        finally:
            requests.put(None)

    threading.Thread(target=read, name="ciel-voice-requests", daemon=True).start()
    cleaned = False
    while True:
        try:
            line = requests.get(timeout=None if cleaned else timeout)
        except queue.Empty:
            try:
                cleanup()
            except Exception:
                traceback.print_exc(file=sys.stderr)
            cleaned = True
            continue
        if line is None:
            return
        yield line
        cleaned = False


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
            cue_audio = {key: path for key, path in
                         (("sigh", args.sigh_audio), ("groan", args.groan_audio))
                         if path is not None}
            renderer = prepare_voice_reference(
                model, torch, reference, args.event_reference, cue_audio, create_turbo_renderer,
            )
            reference_memory = release_reference_encoders(model, torch)
            torch.cuda.synchronize()
            condition_seconds = time.perf_counter() - condition_started
            memory = release_unused_memory(torch)
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
            "reference_inference_mode": True,
            **reference_memory,
            **memory,
        })
    except Exception as err:
        sys.stdout = protocol
        emit({"ready": False, "error": f"{type(err).__name__}: {err}"})
        traceback.print_exc(file=sys.stderr)
        return 1

    for line in idle_requests(sys.stdin, lambda: release_unused_memory(torch)):
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
