"""CPU transcript checks for incomplete/repeated output; listening decides pronunciation quality."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .common import SENTENCES, digest, dump
from .page import available


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace("’", "'"))


def check(directory: Path) -> Path:
    from faster_whisper import WhisperModel

    directory = directory.resolve()
    path = directory / "transcript-checks.json"
    stored = json.loads(path.read_text()) if path.exists() else {"rows": []}
    existing = {row["audio_sha256"]: row for row in stored["rows"]}
    models = list((Path.home() / ".cache/huggingface/hub/"
                   "models--Systran--faster-whisper-small.en/snapshots").glob("*/model.bin"))
    if not models:
        raise FileNotFoundError("Whisper small.en is not cached; no model is downloaded implicitly")
    model = WhisperModel(str(models[0].parent), device="cpu", compute_type="int8", cpu_threads=2)
    results = []
    for identifier, result in available(directory).items():
        for row in result["rows"]:
            if row["iteration"] != 1:
                continue
            audio = Path(row["audio"])
            key = digest(audio)
            if key in existing:
                results.append(existing[key])
                continue
            wave, rate = sf.read(audio, dtype="float32")
            if wave.ndim == 2:
                wave = wave.mean(axis=1)
            import math

            factor = math.gcd(rate, 16000)
            wave = resample_poly(wave, 16000 // factor, rate // factor).astype(np.float32)
            # NumPy bypasses the installed file-decoder/PyAV metadata_errors incompatibility.
            segments, _info = model.transcribe(wave, language="en", beam_size=5,
                                              vad_filter=False, condition_on_previous_text=False)
            transcript = " ".join(segment.text.strip() for segment in segments)
            expected = SENTENCES[row["sentence"]]
            actual_words, expected_words = words(transcript), words(expected)
            tail = expected_words[-4:]
            tail_present = any(actual_words[index:index + len(tail)] == tail
                               for index in range(max(0, len(actual_words) - len(tail) + 1)))
            repeats = [" ".join(actual_words[index:index + 3])
                       for index in range(max(0, len(actual_words) - 5))
                       if actual_words[index:index + 3] == actual_words[index + 3:index + 6]]
            results.append({
                "configuration": identifier, "sentence": row["sentence"],
                "audio_sha256": key, "transcript": transcript,
                "expected_tail_present": tail_present, "adjacent_repeated_trigrams": repeats,
                "review_flag": not tail_present or bool(repeats),
                "interpretation": "ASR review aid, not a listening score or proof of correctness",
            })
            dump(path, {"engine": "cached Whisper small.en CPU INT8", "rows": results})
            status = "review" if results[-1]["review_flag"] else "tail found"
            print(identifier, row["sentence"], status, flush=True)
    dump(path, {"engine": "cached Whisper small.en CPU INT8", "rows": results})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(check(parser.parse_args().directory))


if __name__ == "__main__":
    main()
