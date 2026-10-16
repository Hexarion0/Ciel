"""CPU ASR and edge checks for references, supplementing rather than replacing listening."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .analyze import dump
from .audio import resample, sha256


def run(directory: Path) -> Path:
    """Inspect prepared clip boundaries with the already cached Whisper model."""
    from faster_whisper import WhisperModel

    model = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=2,
                         num_workers=1, local_files_only=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = []
    for row in manifest["candidates"]:
        path = directory / row["reference"]
        digest = sha256(path)
        if row.get("boundary_review", {}).get("reference_sha256") == digest:
            rows.append({"candidate": row["id"], **row["boundary_review"]})
            continue
        wave, rate = sf.read(path, dtype="float32")
        mono = wave.mean(axis=1) if wave.ndim == 2 else wave
        # Use the existing NumPy input API: this installed PyAV version no longer
        # accepts the file loader's metadata_errors argument. No dependency patch.
        parts, _info = model.transcribe(resample(mono, rate, 16000), language="en", beam_size=3,
                                       word_timestamps=True, vad_filter=False)
        segments = list(parts)
        words = [word for segment in segments for word in (segment.words or [])]
        total_rms = float(np.sqrt(np.mean(mono ** 2)))
        edge = round(rate * 0.08)
        row["boundary_review"] = {
            "reference_sha256": digest,
            "asr_text": " ".join(segment.text.strip() for segment in segments),
            "asr_mean_word_probability": float(np.mean([w.probability for w in words]))
            if words else None,
            "last_asr_word_end": words[-1].end if words else None,
            "reference_duration_seconds": len(mono) / rate,
            "start_80ms_rms": float(np.sqrt(np.mean(mono[:edge] ** 2))),
            "end_80ms_rms": float(np.sqrt(np.mean(mono[-edge:] ** 2))),
            "tail_speech_flag_proxy": bool(np.sqrt(np.mean(mono[-edge:] ** 2)) > total_rms * 0.5),
            "interpretation": "ASR/edge-energy proxies only. Listen for clipped words or effects.",
        }
        rows.append({"candidate": row["id"], **row["boundary_review"]})
        print(row["id"], row["boundary_review"]["asr_text"][:100], flush=True)
    dump(directory / "manifest.json", manifest)
    path = directory / "boundary-review.json"
    dump(path, {"device": "cpu", "model": "cached faster-whisper small.en", "rows": rows})
    return path
