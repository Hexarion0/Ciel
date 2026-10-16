"""Render faint Turbo effects with separate style conditioning and bounded caching."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
_CUES = re.compile(r"(\[sigh\]|\[groan\])", re.IGNORECASE)
_CUE_SEEDS = {"[sigh]": 73, "[groan]": 42}


def balance_vocal_cue(audio: np.ndarray) -> np.ndarray | None:
    """Balance an audible cue gently; never greatly amplify near-silence or clip peaks."""
    wave = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not wave.size or not np.isfinite(wave).all():
        return None
    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(wave)))
    if rms < 0.002 or peak <= 0:
        return None
    gain = min(0.025 / rms, 0.5 / peak, 3.0)
    return wave * gain


def generate_styled_cue(model: Any, style: Any, text: str) -> Any:
    """Use a separate speaking pattern while retaining the clone's speaker and decoder."""
    original = model.conds
    model.conds = replace(original, t3=replace(
        original.t3,
        cond_prompt_speech_tokens=style.t3.cond_prompt_speech_tokens,
        cond_prompt_speech_emb=None,
    ))
    try:
        return model.generate(text)
    finally:
        model.conds = original


class VocalCueRenderer:
    """Join separately rendered sighs/groans to normal speech at their requested position."""

    def __init__(
        self, generate: Callable[[str], np.ndarray],
        generate_event: Callable[[str], np.ndarray], sample_rate: int,
    ) -> None:
        self.generate = generate
        self.generate_event = generate_event
        self.sample_rate = sample_rate
        self._cache: dict[str, np.ndarray | None] = {}

    def render(self, text: str) -> np.ndarray:
        """Keep normal requests on the original path; cache effects for this voice worker."""
        if not _CUES.search(text):
            return self.generate(text)
        waves = []
        for part in _CUES.split(text):
            part = part.strip()
            if not part:
                continue
            if _CUES.fullmatch(part):
                key = part.casefold()
                if key not in self._cache:
                    try:
                        cue = balance_vocal_cue(self.generate_event(key))
                    except Exception:
                        logger.warning("Native cue %s generation failed; omitted.", key)
                        cue = None
                    if cue is not None and cue.size > self.sample_rate * 4:
                        cue = None
                    self._cache[key] = cue
                    if cue is None:
                        logger.warning("Native cue %s was too faint or invalid; omitted.", key)
                wave = self._cache[key]
            else:
                wave = self.generate(part)
            if wave is not None and wave.size:
                if waves:
                    waves.append(np.zeros(round(self.sample_rate * 0.06), dtype=np.float32))
                waves.append(wave)
        return np.concatenate(waves) if waves else np.zeros(0, dtype=np.float32)


def load_cue_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Load a local cue recording at the voice worker's sample rate."""
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32")
    if rate != sample_rate:
        raise ValueError("Vocal cue sample rate must match the voice worker")
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def create_turbo_renderer(
    model: Any, event_reference: Path, cue_audio: dict[str, Path] | None = None,
) -> VocalCueRenderer:
    """Prepare optional cue style once, without replacing normal voice conditionals."""
    original = model.conds
    style = None
    try:
        if event_reference.is_file():
            model.prepare_conditionals(str(event_reference), exaggeration=0.0)
            style = model.conds
    finally:
        model.conds = original

    def generate(text: str) -> np.ndarray:
        return np.asarray(model.generate(text), dtype=np.float32).reshape(-1)

    def event(text: str) -> np.ndarray:
        path = (cue_audio or {}).get(text.strip("[]"))
        if path is not None and path.is_file():
            try:
                return load_cue_audio(path, model.sr)
            except Exception:
                logger.warning("Local %s cue could not be loaded; generating a replacement.", text)
        if style is None:
            return generate(text)
        import torch

        devices = [torch.cuda.current_device()] if str(model.device).startswith("cuda") else []
        # Repeatable cached cues; preserve sampling state for all normal speech.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(_CUE_SEEDS[text])
            return np.asarray(generate_styled_cue(model, style, text), dtype=np.float32).reshape(-1)

    return VocalCueRenderer(generate, event, model.sr)
