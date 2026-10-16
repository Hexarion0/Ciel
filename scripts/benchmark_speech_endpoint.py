#!/usr/bin/env python3
"""Replay existing human speech through CIEL VAD; optionally measure actual STT.

This is a repeatable proxy, not a measurement of this user's microphone or intent
to finish speaking. A later spoken word in a passage flags a potential early cut.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ciel.audio.activity import SpeechActivity
from ciel.audio.recorder import VoiceRecorder
from ciel.config import get_settings


def load_audio(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16000:
        audio = resample_poly(audio, 16000, rate).astype("float32")
    return audio


def replay(audio: np.ndarray, silence: float) -> dict:
    detector = SpeechActivity()
    # Cache the VAD labels, so every endpoint gets identical evidence.
    padded = np.pad(audio, (0, 16000 * 3))
    frames = [padded[i:i + 1280] for i in range(0, padded.size, 1280)]
    labels = [detector(frame) for frame in frames]
    decisions = iter(labels)
    recorder = VoiceRecorder(
        silence_duration_seconds=silence, pause_grace_seconds=0,
        speech_detector=lambda _: next(decisions),
    )
    recorder.start()
    end_index = len(frames) - 1
    for i, frame in enumerate(frames):
        if not recorder.add_frame(frame):
            end_index = i
            break
    last = max((i for i, label in enumerate(labels[:end_index + 1]) if label), default=0)
    return {
        "endpoint_seconds": (end_index + 1) * 0.08,
        "last_vad_speech_seconds": (last + 1) * 0.08,
        "trailing_silence_seconds": (end_index - last) * 0.08,
        "speech_frames_after_cut": sum(labels[end_index + 1:]),
        "passage_has_later_speech": any(labels[end_index + 1:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stt", action="store_true")
    args = parser.parse_args()
    root = Path("data/diagnostics/conversation-latency")
    root.mkdir(parents=True, exist_ok=True)
    refs = Path("data/voice/references/ciel")
    rows = []
    stt = None
    if args.stt:
        from ciel.audio.stt import SpeechToText

        cfg = get_settings().audio
        began = time.monotonic()
        stt = SpeechToText(
            model_size=cfg.stt_model, device=cfg.stt_device, compute_type=cfg.stt_compute_type,
            language=cfg.stt_language, min_confidence=cfg.stt_min_confidence,
            retry_confidence=cfg.stt_retry_confidence, retry_model=cfg.stt_retry_model,
            retry_min_free_mb=cfg.stt_retry_min_free_mb, wake_phrase=cfg.wake_word,
        )
        if not stt.wait_ready(120):
            raise TimeoutError("STT did not load")
        print(json.dumps({"stt_load_seconds": time.monotonic() - began,
                          "device": stt.device, "compute_type": stt.compute_type}), flush=True)
    for index in range(1, 4):
        path = refs / f"reference-{index}.wav"
        audio = load_audio(path)
        row = {"reference": str(path), "duration": audio.size / 16000,
               "endpoint": {str(s): replay(audio, s) for s in [1.8, 1.0, 0.8, 0.6]}}
        if stt is not None:
            samples = [audio, audio[:int(min(4, len(audio) / 16000) * 16000)]]
            row["stt"] = []
            for repeat in range(2):
                for sample in samples:
                    began = time.monotonic()
                    result = stt.transcribe_detailed(sample, beam_size=cfg.stt_beam_size)
                    row["stt"].append({
                        "repeat": repeat, "duration": sample.size / 16000,
                        "wall_seconds": time.monotonic() - began,
                        "transcript": result["text"], "confidence": result.get("confidence"),
                        "retried": result.get("retried"),
                    })
        rows.append(row)
        (root / ("endpoint-stt.json" if stt else "endpoint.json")).write_text(
            json.dumps(rows, indent=2)
        )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
