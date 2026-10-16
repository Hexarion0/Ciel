"""Conservative audio preparation and transparent waveform screening."""

from __future__ import annotations

import hashlib
import html
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SOURCE_IDS = ("3mZ2ZbciDPM", "iP0AY_BnlHE", "cv0ovU6kb-s", "lSaNBdJbhds")
STYLES = ("neutral", "warm", "amused", "firm", "focused", "soft")


def sha256(path: Path) -> str:
    """Fingerprint originals without reading a large source into RAM at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seconds(stamp: str) -> float:
    """Parse a WebVTT timestamp."""
    hours, minutes, secs = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(secs)


def timestamp(value: float) -> str:
    """Human-readable source timestamp, retaining milliseconds."""
    minutes, secs = divmod(value, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"


def vtt_words(path: Path) -> list[dict]:
    """Read YouTube's inline word timestamps, discarding repeated rolling captions.

    Captions are only a boundary/context aid, not verified transcription or
    evidence of speaker identity. No subtitle text is executed as instructions.
    """
    cue_start = 0.0
    words = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if " --> " in line:
            cue_start = seconds(line.split(" --> ")[0])
            continue
        matches = list(re.finditer(r"<(\d\d:\d\d:\d\d\.\d+)>", line))
        if not matches:
            continue  # A repeated rendered line does not introduce new words.
        prefix = html.unescape(re.sub(r"<[^>]+>", "", line[:matches[0].start()])).strip()
        if prefix and len(prefix.split()) <= 3:
            words[(round(cue_start, 3), prefix)] = {"start": cue_start, "text": prefix}
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
            text = html.unescape(re.sub(r"<[^>]+>", "", line[match.end():end])).strip()
            start = seconds(match.group(1))
            if text:
                words[(round(start, 3), text)] = {"start": start, "text": text}
    return sorted(words.values(), key=lambda row: row["start"])


def resample(audio: np.ndarray, rate: int, target: int = 24000) -> np.ndarray:
    """Downmix and polyphase resample without changing pitch or tempo."""
    wave = audio.mean(axis=1) if audio.ndim == 2 else audio
    if rate == target:
        return np.asarray(wave, dtype=np.float32)
    divisor = math.gcd(rate, target)
    return resample_poly(wave, target // divisor, rate // divisor).astype(np.float32)


def metrics(audio: np.ndarray, rate: int) -> dict:
    """Waveform proxies; these cannot prove absence of music, overlap, or whispering."""
    wave = resample(audio, rate, 16000)
    if not wave.size or not np.isfinite(wave).all():
        raise ValueError("Audio is empty or non-finite")
    frames = wave[:wave.size // 320 * 320].reshape(-1, 320)
    if not frames.size:
        raise ValueError("Audio is shorter than 20ms")
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    floor = float(np.percentile(rms, 10))
    active = rms > max(0.003, floor * 2)
    pitches = []
    voiced = 0
    # Autocorrelation is a screening proxy, not a calibrated pitch/whisper detector.
    for frame in frames[::4]:
        if np.sqrt(np.mean(frame ** 2)) < 0.003:
            continue
        centered = frame.astype(np.float64) - frame.mean()
        correlation = np.correlate(centered, centered, "full")[319:]
        lag = int(np.argmax(correlation[45:160])) + 45
        strength = correlation[lag] / max(correlation[0], 1e-12)
        if strength > 0.45:
            pitches.append(16000 / lag)
            voiced += 1
    peak = float(np.max(np.abs(wave)))
    return {
        "duration_seconds": round(wave.size / 16000, 3),
        "peak": peak,
        "rms_dbfs": float(20 * np.log10(max(float(np.sqrt(np.mean(wave ** 2))), 1e-9))),
        "clipping_fraction": float(np.mean(np.abs(wave) >= 0.995)),
        "quiet_floor_dbfs_proxy": float(20 * np.log10(max(floor, 1e-9))),
        "active_fraction_proxy": float(np.mean(active)),
        "voiced_fraction_proxy": voiced / max(1, len(frames[::4])),
        "pitch_median_hz_proxy": float(np.median(pitches)) if pitches else None,
        "level_spread_db": float(20 * np.log10(
            max(float(np.percentile(rms, 90)), 1e-9) / max(floor, 1e-9)
        )),
        "qualification": "Uncalibrated waveform screening; human listening is required.",
    }


def prepare(audio: np.ndarray, rate: int) -> tuple[np.ndarray, dict]:
    """Trim only quiet edges, resample to 24k mono, and apply bounded linear gain.

    No denoiser, separation, pitch/formant transform, compression, high-pass
    filter, or click removal is applied by default.
    """
    wave = resample(audio, rate)
    if not wave.size or not np.isfinite(wave).all():
        raise ValueError("Audio is empty or non-finite")
    # Keep 100ms padding around edge activity to protect initial/final consonants.
    width = 480
    frame_count = wave.size // width
    if not frame_count:
        raise ValueError("Candidate too short")
    rms = np.sqrt(np.mean(wave[:frame_count * width].reshape(-1, width) ** 2, axis=1))
    threshold = max(0.001, min(0.004, float(np.percentile(rms, 10)) * 1.5))
    active = np.flatnonzero(rms > threshold)
    if not active.size:
        raise ValueError("No usable speech energy in candidate")
    first = max(0, int(active[0] * width) - 2400)
    last = min(wave.size, int((active[-1] + 1) * width) + 2400)
    wave = wave[first:last]
    rms_value = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(wave)))
    gain = min(10 ** (-23 / 20) / max(rms_value, 1e-9), 2.0, 0.85 / max(peak, 1e-9))
    wave = (wave * gain).astype(np.float32)
    return wave, {
        "operations": ["quiet-edge trim with 100ms padding", "mono downmix",
                       "polyphase resample to 24000 Hz", "bounded linear level normalization"],
        "trim_start_seconds": first / 24000,
        "trim_end_seconds": (audio.shape[0] / rate) - last / 24000,
        "linear_gain": gain, "gain_db": 20 * math.log10(max(gain, 1e-9)),
        "sample_rate": 24000, "channels": 1, "wav_subtype": "PCM_16",
        "denoising": False, "separation": False, "compression": False,
        "pitch_or_formant_change": False,
    }


def write_new_audio(path: Path, audio: np.ndarray, rate: int, subtype: str = "PCM_16") -> None:
    """Candidate runs are immutable: never silently replace an existing sample."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, rate, subtype=subtype)
