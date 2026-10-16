"""Add an explicitly chosen source window as a new immutable audition candidate."""

from __future__ import annotations

import json
import re
from pathlib import Path

import soundfile as sf

from .analyze import dump, permission
from .audio import SOURCE_IDS, STYLES, metrics, prepare, sha256, timestamp, write_new_audio


def run(directory: Path, source_root: Path, *, identifier: str, source_id: str,
        start: float, end: float, style: str, notes: str = "") -> Path:
    """Keep previous candidates and the runtime baseline untouched."""
    permission(source_root)
    if source_id not in SOURCE_IDS or style not in STYLES:
        raise ValueError("Unknown source or style")
    if not re.fullmatch(r"candidate-[a-z]+-\d{2}", identifier):
        raise ValueError("Use an identifier such as candidate-warm-03")
    if not 0 <= start < end or not 5 <= end - start <= 15:
        raise ValueError("Choose a source interval between 5 and 15 seconds")
    path = directory / f"{identifier}.wav"
    if path.exists():
        raise FileExistsError(path)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("human_acceptance"):
        raise ValueError("Use a new run after a bank is accepted")
    decoded = source_root / f"source-{source_id}.decoded.wav"
    source_digest = sha256(source_root / f"source-{source_id}.webm")
    with sf.SoundFile(decoded) as audio:
        if end > audio.frames / audio.samplerate:
            raise ValueError("Interval extends beyond the source")
        rate = audio.samplerate
        audio.seek(round(start * rate))
        raw = audio.read(round((end - start) * rate), dtype="float32", always_2d=True)
    original = directory / "screening" / "raw" / f"{identifier}.wav"
    write_new_audio(original, raw, rate, "FLOAT")
    prepared, operations = prepare(raw, rate)
    write_new_audio(path, prepared, 24000)
    row = {
        "id": identifier, "source_id": source_id,
        "source_url": f"https://www.youtube.com/watch?v={source_id}",
        "source_timestamp": f"{timestamp(start)}–{timestamp(end)}", "start": start, "end": end,
        "source_sha256": source_digest,
        "original_duration_seconds": end - start,
        "processed_duration_seconds": len(prepared) / 24000,
        "unprocessed": str(original.relative_to(directory)), "reference": path.name,
        "reference_sha256": sha256(path), "preprocessing": operations,
        "intended_style": style, "style_status": "provisional; listening pending",
        "speaker_cosine_to_anchor": None, "identity_status": "CPU recheck pending",
        "metrics": metrics(raw, rate), "notes": notes,
    }
    manifest["candidates"].append(row)
    for source in manifest["sources"]:
        if source["id"] == source_id:
            source["shortlisted_candidates"].append(identifier)
    dump(directory / "manifest.json", manifest)
    return path
