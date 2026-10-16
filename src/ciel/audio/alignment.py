"""Caption character schedules using native spoken-word spans where available.

Piper phoneme durations locate words in the generated waveform. Written letters
are interpolated within those word spans; spelling and phonemes are not equivalent.
Unmatched text normalization and engines without timings use a labelled estimate.
"""

import math
import re
import unicodedata
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

_WORDS = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_PAUSE_PHONEMES = frozenset(".,!?;:—–-\"“”()[]")
_SPECIAL_PHONEMES = frozenset({"^", "$", "_"})


@dataclass(frozen=True)
class CharacterTimeline:
    """Reveal timestamps for each codepoint in the original caption text."""

    text: str
    character_times: tuple[float, ...]
    source: str

    def visible_count(self, elapsed_seconds: float) -> int:
        """Return how much original text should be visible at the playback cursor."""
        if math.isnan(elapsed_seconds) or elapsed_seconds <= 0:
            return 0
        return bisect_right(self.character_times, elapsed_seconds)

    def prefix(self, elapsed_seconds: float) -> str:
        """Return the visible caption while preserving its original spelling."""
        return self.text[:self.visible_count(elapsed_seconds)]


def estimated_timeline(text: str, duration_seconds: float) -> CharacterTimeline:
    """Distribute written characters across known audio duration, labelled estimated."""
    duration = duration_seconds if math.isfinite(duration_seconds) else 0.0
    duration = max(0.0, duration)
    units = sum(not char.isspace() and not unicodedata.combining(char) for char in text)
    times: list[float] = []
    consumed = 0
    for char in text:
        if not char.isspace() and not unicodedata.combining(char):
            consumed += 1
        times.append(duration * consumed / units if units else 0.0)
    return CharacterTimeline(text, tuple(times), "estimated")


def _boundary(phoneme: str) -> bool:
    return (
        phoneme.isspace() or phoneme in _PAUSE_PHONEMES or phoneme in _SPECIAL_PHONEMES
    )


def _spoken_group_count(sentences: Sequence[Sequence[str]]) -> int:
    """Count spoken words after token normalization using the voice's own phonemizer."""
    if isinstance(sentences, str):
        raise ValueError("Invalid phonemizer result")
    count = 0
    for sentence in sentences:
        if isinstance(sentence, str):
            raise ValueError("Invalid phonemizer sentence")
        active = False
        for phoneme in sentence:
            if not isinstance(phoneme, str) or not phoneme:
                raise ValueError("Invalid phonemizer phoneme")
            if _boundary(phoneme):
                active = False
            elif not active:
                count += 1
                active = True
    return count


def _written_tokens(text: str) -> list[tuple[int, int]]:
    """Keep expansions such as 5:32 and 88°C in one original token."""
    tokens = []
    for match in re.finditer(r"\S+", text):
        positions = [i for i in range(match.start(), match.end()) if text[i].isalnum()]
        if positions:
            tokens.append((positions[0], positions[-1] + 1))
    return tokens


def _word_spans(chunks: Sequence[Any], sample_rate: int) -> tuple[list[tuple[int, int]], int]:
    """Collect native word boundaries and ensure every sample belongs to the waveform."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for chunk in chunks:
        if chunk.sample_rate != sample_rate:
            raise ValueError("Alignment sample rate differs from playback audio")
        samples = len(chunk.audio_float_array)
        alignments = chunk.phoneme_alignments
        if not alignments:
            raise ValueError("Native phoneme timings are missing")
        cursor = 0
        word_start: int | None = None
        word_end = 0
        for alignment in alignments:
            count = int(alignment.num_samples)
            if count < 0 or count != alignment.num_samples:
                raise ValueError("Invalid phoneme sample count")
            phoneme = alignment.phoneme
            if not isinstance(phoneme, str) or not phoneme:
                raise ValueError("Invalid aligned phoneme")
            if _boundary(phoneme):
                if word_start is not None:
                    if word_end <= word_start:
                        raise ValueError("A spoken word has no audio duration")
                    spans.append((offset + word_start, offset + word_end))
                    word_start = None
            else:
                if word_start is None:
                    word_start = cursor
                word_end = cursor + count
            cursor += count
        if word_start is not None:
            if word_end <= word_start:
                raise ValueError("A spoken word has no audio duration")
            spans.append((offset + word_start, offset + word_end))
        if cursor != samples:
            raise ValueError("Native durations do not sum to the generated audio length")
        offset += samples
    return spans, offset


def character_timeline(
    text: str, chunks: Sequence[Any], sample_rate: int, audio_samples: int,
    *, phonemize: Callable[[str], Sequence[Sequence[str]]] | None = None,
) -> CharacterTimeline:
    """Map original text to native word times, falling back when mapping is uncertain.

    The optional voice phonemizer maps original tokens to their expanded spoken-word
    counts, preserving numeric spellings such as 5:32 without another audio inference.
    Without it, numeric text is estimated because normalization may change word counts.
    Native timings must cover the exact audio supplied for playback, including pauses.
    This examines existing synthesis results and never performs additional inference.
    """
    duration = audio_samples / sample_rate if sample_rate > 0 and audio_samples > 0 else 0.0
    fallback = estimated_timeline(text, duration)
    if not text or sample_rate <= 0 or audio_samples <= 0:
        return fallback
    if phonemize is None and any(c.isdigit() for c in text):
        return fallback
    try:
        spans, native_samples = _word_spans(chunks, sample_rate)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return fallback
    if native_samples != audio_samples:
        return fallback
    if phonemize is None:
        words = [(match.start(), match.end()) for match in _WORDS.finditer(text)]
        if not words or len(words) != len(spans):
            return fallback
    else:
        words = _written_tokens(text)
        mapped_spans = []
        cursor = 0
        try:
            for start, end in words:
                count = _spoken_group_count(phonemize(text[start:end]))
                if count <= 0 or cursor + count > len(spans):
                    return fallback
                mapped_spans.append((spans[cursor][0], spans[cursor + count - 1][1]))
                cursor += count
        except Exception:
            # A phonemizer failure must not break playback or claim native alignment.
            return fallback
        if not words or cursor != len(spans):
            return fallback
        spans = mapped_spans

    times: list[float] = [0.0] * len(text)
    preceding_end = 0
    text_cursor = 0
    for (word_start, word_end), (start, end) in zip(words, spans):
        for index in range(text_cursor, word_start):
            times[index] = preceding_end / sample_rate
        units = sum(not unicodedata.combining(char) for char in text[word_start:word_end])
        consumed = 0
        for index in range(word_start, word_end):
            if not unicodedata.combining(text[index]):
                consumed += 1
            times[index] = (start + (end - start) * consumed / units) / sample_rate
        preceding_end = end
        text_cursor = word_end
    for index in range(text_cursor, len(text)):
        times[index] = preceding_end / sample_rate
    return CharacterTimeline(text, tuple(times), "phoneme")
