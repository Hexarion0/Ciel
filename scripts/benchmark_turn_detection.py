#!/usr/bin/env python3
"""Replay labeled speech pauses against a local Smart Turn CPU model; no downloads.

Later recorded speech is a potential cutoff proxy, not ground truth about human
turn-taking. Compare that proxy with CIEL's current silence setting before choosing
to enable an earlier endpoint. This script does not change CIEL's configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from statistics import median

import numpy as np

from ciel.audio.activity import SpeechActivity
from ciel.config import get_settings
from scripts.benchmark_speech_backends import load_audio


def turn_features(audio: np.ndarray, extractor) -> np.ndarray:
    """Apply Smart Turn's normalized, left-padded 8-second Whisper input contract."""
    tail = np.asarray(audio, dtype=np.float32).reshape(-1)[-128000:]
    if not np.isfinite(tail).all():
        raise ValueError("Turn audio must contain only finite samples")
    padded = np.pad(tail, (128000 - tail.size, 0))
    padded = (padded - padded.mean()) / np.sqrt(padded.var() + 1e-7)
    return extractor(padded, padding=0)[None].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import onnxruntime as ort
    from faster_whisper.feature_extractor import FeatureExtractor

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(args.model), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    extractor = FeatureExtractor(chunk_length=8)
    cfg = get_settings().audio
    silence = cfg.utterance_silence_seconds + cfg.utterance_pause_grace_seconds
    rows = []
    for sample in json.loads(args.manifest.read_text())["samples"]:
        audio = np.pad(load_audio(Path(sample["path"])), (0, 16000 * 3))
        vad = SpeechActivity()
        labels = [vad(audio[index:index + 1280]) for index in range(0, audio.size, 1280)]
        for index in range(8, len(labels)):
            # First candidate after four silent frames (320 ms) following speech.
            if any(labels[index - 3:index + 1]) or not labels[index - 4]:
                continue
            next_speech = next((n for n in range(index + 1, len(labels)) if labels[n]), None)
            gap = ((next_speech - (index - 4) - 1) * 0.08
                   if next_speech is not None else None)
            started = time.perf_counter()
            inputs = turn_features(audio[:(index + 1) * 1280], extractor)
            probability = float(session.run(None, {"input_features": inputs})[0].reshape(-1)[0])
            rows.append({
                "id": sample["id"], "at_seconds": (index + 1) * 0.08,
                "later_recorded_speech": next_speech is not None,
                "silence_before_next_speech_seconds": gap,
                "baseline_would_cut": gap is not None and gap >= silence,
                "completion_probability": probability,
                "inference_seconds": time.perf_counter() - started,
            })
    if not rows:
        raise ValueError("The corpus contained no eligible speech/silence transitions")
    report = {
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "baseline_silence_seconds": silence, "rows": rows,
        "summary": {
            "complete_clip_candidates": sum(not row["later_recorded_speech"] for row in rows),
            "complete_clips_detected": sum(
                not row["later_recorded_speech"] and row["completion_probability"] > 0.5
                for row in rows),
            "additional_potential_cutoffs": sum(
                row["later_recorded_speech"] and not row["baseline_would_cut"]
                and row["completion_probability"] > 0.5 for row in rows),
            "median_inference_seconds": median(row["inference_seconds"] for row in rows),
        },
        "limitation": "Pauses in read speech are a cutoff proxy, not labeled conversation turns. "
                      "Inference time excludes the 320 ms silence and capture-worker scheduling.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
