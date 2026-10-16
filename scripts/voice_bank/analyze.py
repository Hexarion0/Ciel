"""Archive provenance, screen complete sources, and extract reviewable candidates."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import soundfile as sf

from .audio import SOURCE_IDS, metrics, prepare, sha256, timestamp, vtt_words, write_new_audio

# Observed sponsor-to-roleplay transitions in the downloaded original captions.
# Whole sources are retained; only these promotional introductions are excluded.
INTRO_END = {"3mZ2ZbciDPM": 95.0, "lSaNBdJbhds": 118.0}
STYLE_TERMS = {
    "warm": ("it's alright", "it's okay", "don't worry", "help you", "together",
             "care about", "thank you", "you don't have to"),
    "amused": ("funny", "teasing", "joking", "predictable", "lost puppy",
               "did you really", "miss me that much", "ridiculous"),
    "soft": ("sleep", "tired", "relax", "awake", "stay", "quiet", "rest"),
}


def dump(path: Path, value: dict) -> None:
    """Write analysis metadata, with real line breaks."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def permission(source_root: Path) -> dict:
    """Require the user's recorded permission scope before handling source audio."""
    record = json.loads((source_root / "permission.json").read_text())
    ids = {row["id"] for row in record.get("sources", [])}
    if record.get("status") != "user_confirmed" or not set(SOURCE_IDS).issubset(ids):
        raise ValueError("Permission for all four sources has not been confirmed")
    return record


def windows(words: list[dict], duration: float, minimum: float) -> list[dict]:
    """Consider caption sentence boundaries throughout the source, not its opening."""
    rows = []
    for i, word in enumerate(words):
        begin = word["start"]
        previous = words[i - 1] if i else None
        if begin < minimum or begin > duration - 6:
            continue
        if previous and not re.search(r"[.!?]$", previous["text"]) and (
            begin - previous["start"] < 1.2
        ):
            continue
        endings = [j for j in range(i + 1, min(i + 45, len(words)))
                   if 6 <= words[j]["start"] - begin <= 13.5
                   and re.search(r"[.!?]$", words[j]["text"])]
        if not endings:
            continue
        j = min(endings, key=lambda j: abs(words[j]["start"] - begin - 10))
        end = min(duration, words[j]["start"] + 0.5)
        text = " ".join(item["text"] for item in words[i:j + 1]).replace(">>", "").strip()
        if re.search(r"\[[^]]+\]|\b(kiss|kissing|moan|screaming|shouting)\b", text, re.I):
            continue
        rows.append({"start": max(minimum, begin - 0.12), "end": end, "caption": text})
    return rows


def style_hint(text: str, style: str) -> int:
    """Text is a shortlist hint only; delivery/identity need listening confirmation."""
    lower = text.casefold()
    return sum(term in lower for term in STYLE_TERMS.get(style, ()))


def run(source_root: Path, output: Path) -> Path:
    """Create a fresh immutable screening run with up to four windows/style/source."""
    rights = permission(source_root)
    if output.exists():
        raise FileExistsError(f"Use a new candidate run directory: {output}")
    output.mkdir(parents=True)
    report = {
        "status": "screening_pending_identity_and_audition", "permission": rights,
        "sources": [], "pool": [],
        "limits": (
            "Caption-assisted boundaries and waveform proxies only. These cannot verify "
            "music, overlap, reverb, age, emotional delivery, or perceptual quality. "
            "No automatic promotion to the runtime bank."
        ),
    }
    for sid in SOURCE_IDS:
        media = source_root / f"source-{sid}.webm"
        info_path = source_root / f"source-{sid}.info.json"
        info = json.loads(info_path.read_text())
        # Decode once for screening; retain original rate and stereo in this lossless file.
        decoded = source_root / f"source-{sid}.decoded.wav"
        command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-threads", "2",
                   "-i", str(media), "-c:a", "pcm_s16le", str(decoded)]
        if not decoded.exists():
            subprocess.run(command, check=True)
        original, rate = sf.read(decoded, dtype="float32", always_2d=True)
        mono = original.mean(axis=1)
        words = vtt_words(source_root / f"source-{sid}.en-orig.vtt")
        proposals = windows(words, original.shape[0] / rate, INTRO_END.get(sid, 12.0))
        for row in proposals:
            wave = mono[round(row["start"] * rate):round(row["end"] * rate)]
            row["metrics"] = metrics(wave, rate)
            m = row["metrics"]
            row["score"] = (
                min(m["level_spread_db"] / 30, 1) * 0.35
                + m["voiced_fraction_proxy"] * 0.45
                + m["active_fraction_proxy"] * 0.2
                - m["clipping_fraction"] * 1000
                - max(0, m["quiet_floor_dbfs_proxy"] + 36) / 30
            )
        picked = []
        for style in ("neutral", "warm", "amused", "soft"):
            ordered = sorted(proposals, key=lambda row: (
                row["score"] + min(style_hint(row["caption"], style), 2) * 0.1
            ), reverse=True)
            style_rows = []
            for row in ordered:
                if row["metrics"]["clipping_fraction"] > 0.001:
                    continue
                if any(abs(row["start"] - other["start"]) < 16 for other in style_rows):
                    continue
                copy = {**row, "intended_style": style, "source_id": sid}
                style_rows.append(copy)
                if len(style_rows) == 4:
                    break
            picked.extend(style_rows)
        source_hash = sha256(media)
        for row in picked:
            key = f"{sid}-{round(row['start'] * 1000)}"
            raw = output / "screening" / "raw" / f"{key}.wav"
            reference = output / "screening" / "prepared" / f"{key}.wav"
            if not raw.exists():
                chunk = original[round(row["start"] * rate):round(row["end"] * rate)]
                write_new_audio(raw, chunk, rate, "FLOAT")
                wave, operations = prepare(chunk, rate)
                write_new_audio(reference, wave, 24000)
                dump(reference.with_suffix(".processing.json"), operations)
            operations = json.loads(reference.with_suffix(".processing.json").read_text())
            row.update({
                "id": key, "source_url": f"https://www.youtube.com/watch?v={sid}",
                "source_sha256": source_hash,
                "source_timestamp": f"{timestamp(row['start'])}–{timestamp(row['end'])}",
                "original_duration_seconds": row["end"] - row["start"],
                "processed_duration_seconds": sf.info(reference).duration,
                "unprocessed": str(raw.relative_to(output)),
                "reference": str(reference.relative_to(output)),
                "reference_sha256": sha256(reference), "preprocessing": operations,
                "style_status": "tentative context hint; not listening-confirmed",
                "notes": "Waveform shortlist only; compare raw speech before accepting.",
            })
            report["pool"].append(row)
        report["sources"].append({
            "id": sid, "url": f"https://www.youtube.com/watch?v={sid}",
            "title": info["title"], "channel": info.get("channel"),
            "channel_id": info.get("channel_id"), "duration_seconds": info["duration"],
            "audio_format": info.get("format"), "codec": info.get("acodec"),
            "archive": str(media), "sha256": source_hash, "decode_command": command,
            "excluded_intro_seconds": INTRO_END.get(sid, 12.0),
            "windows_screened": len(proposals),
            "known_context": (
                "Thunderstorm roleplay; inspect rain/transient contamination"
                if sid == "iP0AY_BnlHE" else "ASMR roleplay; inspect breathiness and acting"
            ),
        })
        dump(output / "screening.json", report)
        print(sid, "screened", len(proposals), "retained", len(picked), flush=True)
    return output / "screening.json"
