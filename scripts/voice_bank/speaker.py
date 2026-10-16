"""CPU-only reference identity proxy using the existing cached speaker encoder.

Runs in the existing clone environment. This does not generate Chatterbox audio.
Cosines are uncalibrated and cannot establish real-world identity or voice quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    import librosa
    import numpy as np
    import torch
    from chatterbox.models.voice_encoder import VoiceEncoder
    from safetensors.torch import load_file

    from .audio import sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", action="store_true", help="Recheck the final audition set")
    args = parser.parse_args()
    torch.set_num_threads(2)
    encoder = VoiceEncoder()
    encoder.load_state_dict(load_file(args.weights, device="cpu"))
    encoder.eval()
    path = args.directory / ("manifest.json" if args.candidates else "screening.json")
    screening = json.loads(path.read_text())
    collection = screening["candidates"] if args.candidates else screening["pool"]
    unique = {row["id"]: row for row in collection if row.get("source_id") != "baseline"}
    vectors = {}
    for key, row in unique.items():
        wave, _rate = librosa.load(args.directory / row["reference"], sr=16000)
        with torch.inference_mode():
            halves = [wave, wave[:len(wave) // 2], wave[len(wave) // 2:]]
            embeds = encoder.embeds_from_wavs(halves, sample_rate=16000, trim_top_db=None)
        embeds /= np.maximum(np.linalg.norm(embeds, axis=1, keepdims=True), 1e-12)
        vectors[key] = embeds[0].tolist()
        row["within_clip_cosine"] = float(embeds[1] @ embeds[2])
    wave, _rate = librosa.load(args.baseline, sr=16000)
    with torch.inference_mode():
        baseline = encoder.embeds_from_wavs([wave], sample_rate=16000, trim_top_db=None)[0]
    baseline /= max(np.linalg.norm(baseline), 1e-12)
    report = {
        "encoder": "existing Chatterbox speaker encoder; inference only, CPU",
        "encoder_sha256": sha256(args.weights), "baseline_sha256": sha256(args.baseline),
        "vectors": vectors, "baseline_vector": baseline.tolist(),
        "within_clip_cosines": {key: row["within_clip_cosine"] for key, row in unique.items()},
        "cuda_initialized": torch.cuda.is_initialized(),
        "interpretation": "Uncalibrated screening proxies, not identity proof or listening tests.",
    }
    if args.candidates:
        anchor = np.asarray(vectors["candidate-neutral-01"])
        for key, row in unique.items():
            row["speaker_cosine_to_anchor"] = float(np.asarray(vectors[key]) @ anchor)
            row["speaker_cosine_to_baseline"] = float(np.asarray(vectors[key]) @ baseline)
            row["within_clip_cosine"] = report["within_clip_cosines"][key]
            row["identity_status"] = "CPU proxy rechecked; human identity acceptance pending"
        report["pairwise_cosines"] = {
            key: {other: float(np.asarray(vector) @ np.asarray(other_vector))
                  for other, other_vector in vectors.items()}
            for key, vector in vectors.items()
        }
        path.write_text(json.dumps(screening, indent=2) + "\n")
    destination = "speaker-candidates.json" if args.candidates else "speaker-screening.json"
    (args.directory / destination).write_text(json.dumps(report, indent=2) + "\n")
    print("CPU speaker screening complete:", len(vectors), "unique references", flush=True)


if __name__ == "__main__":
    main()
