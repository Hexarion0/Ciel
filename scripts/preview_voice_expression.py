"""Generate short, offline voice previews on CPU without taking the assistant's GPU."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def main() -> int:
    """Use the separate clone environment to save WAVs and measured synthesis metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cues-only", action="store_true", help="Save standalone effect clips")
    parser.add_argument("--expressiveness", choices=("expressive", "natural", "off"),
                        default="expressive")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    import numpy as np
    import soundfile as sf
    import torch
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    from ciel.audio.speech_events import SpeechEventLimiter
    from ciel.audio.vocal_cues import create_turbo_renderer

    out = args.output or root / "data/voice/expressive-previews" / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    started = time.monotonic()
    with contextlib.redirect_stdout(sys.stderr):
        model = ChatterboxTurboTTS.from_local(
            str(root / "data/voice/models/chatterbox-turbo"), device="cpu",
        )
        model.prepare_conditionals(
            str(root / "data/voice/references/ciel/reference-1.wav"), exaggeration=0.0,
        )
        event_reference = root / "data/voice/references/ciel/event-style.wav"
        cue_audio = {key: root / f"data/voice/references/ciel/{key}.wav"
                     for key in ("sigh", "groan")}
        renderer = (
            create_turbo_renderer(model, event_reference, cue_audio)
            if event_reference.is_file() or any(path.is_file() for path in cue_audio.values())
            else None
        )
    report: dict = {"device": "cpu", "expressiveness": args.expressiveness,
                    "seed": 42, "load_seconds": round(time.monotonic() - started, 2),
                    "notes": "Offline synthesis only; speaker playback and quality not judged.",
                    "vocal_cues_enabled": renderer is not None, "samples": []}
    for name, text in (
        ("chuckle", "[chuckle] You got me."),
        ("sigh", "[sigh] That's a relief."),
        ("moan-alias", "Oh! [moan] Not again."),
    ):
        if args.cues_only:
            text = {"chuckle": "[chuckle]", "sigh": "[sigh]", "moan-alias": "[moan]"}[name]
        normalized = SpeechEventLimiter(args.expressiveness).apply(text)
        if not normalized:
            continue
        started = time.monotonic()
        with contextlib.redirect_stdout(sys.stderr):
            audio = (renderer.render(normalized) if renderer is not None else
                     model.generate(normalized).squeeze().numpy())
        if not audio.size or not np.isfinite(audio).all():
            raise RuntimeError(f"Invalid preview waveform: {name}")
        sf.write(out / f"{name}.wav", audio, model.sr)
        row = {"name": name, "text": normalized, "sample_rate": model.sr,
               "duration_seconds": round(len(audio) / model.sr, 2),
               "synthesis_seconds": round(time.monotonic() - started, 2)}
        report["samples"].append(row)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)
    print(f"Saved previews to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
