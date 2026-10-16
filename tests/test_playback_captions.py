"""Voice captions follow playback timing and stop at the audible interruption."""

from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from ciel.audio.alignment import CharacterTimeline
from ciel.audio.tts import TextToSpeech


def timed_playback(monkeypatch, *, stop_at=None):
    """Model a one-second output buffer with a known device latency."""
    clock = [100.0]
    tts = TextToSpeech(engine="piper", enabled=False)
    tts.enabled = True
    audio = np.ones(100, dtype=np.float32)
    timeline = CharacterTimeline('ABC', (0.2, 0.4, 0.8), 'phoneme')
    monkeypatch.setattr(tts, '_prepare_playback', lambda text: (audio, 100, timeline))
    monkeypatch.setattr('ciel.audio.tts.time.monotonic', lambda: clock[0])

    class OutputStream:
        latency = 0.1

        @property
        def active(self):
            return clock[0] < 101.1

    class AudioClock(Event):
        def wait(self, timeout=None):
            clock[0] += 0.05
            if stop_at is not None and clock[0] >= stop_at and not self.is_set():
                tts.stop()
            return self.is_set()

    tts._stop_event = AudioClock()
    played, stopped = [], []
    monkeypatch.setattr('ciel.audio.tts.sd', SimpleNamespace(
        play=lambda *args, **kwargs: played.append(1),
        get_stream=lambda: OutputStream(), stop=lambda: stopped.append(1),
    ))
    return tts, clock, played, stopped


@pytest.mark.parametrize('block', [False, True])
def test_each_character_follows_native_timing_and_device_latency(monkeypatch, block):
    tts, clock, played, stopped = timed_playback(monkeypatch)
    updates, done = [], Event()

    def progress(prefix, finished, interrupted):
        updates.append((prefix, finished, interrupted, clock[0]))
        if finished:
            done.set()

    assert tts.speak('ABC', block=block, on_progress=progress)
    assert done.wait(1)
    characters = [(prefix, at) for prefix, finished, _cut, at in updates if not finished]
    assert [prefix for prefix, _at in characters] == ['A', 'AB', 'ABC']
    for (_prefix, at), expected in zip(characters, [100.3, 100.5, 100.9]):
        assert at >= expected - 1e-8
        assert at <= expected + 0.05 + 1e-8
    assert updates[-1][:3] == ('ABC', True, False)
    assert played == [1] and stopped == []
    if block:
        assert not tts.is_speaking()


def test_barge_in_preserves_only_the_letters_reached_before_audio_stopped(monkeypatch):
    tts, _clock, played, stopped = timed_playback(monkeypatch, stop_at=100.55)
    updates = []
    assert not tts.speak('ABC', on_progress=lambda *args: updates.append(args))
    assert [prefix for prefix, finished, _cut in updates if not finished] == ['A', 'AB']
    assert updates[-1] == ('AB', True, True)
    assert not any(prefix == 'ABC' for prefix, _finished, _cut in updates)
    assert played == [1] and stopped == [1]


def test_caption_writer_failure_does_not_cancel_voice_playback(monkeypatch):
    tts, _clock, played, stopped = timed_playback(monkeypatch)
    calls = []

    def progress(*args):
        calls.append(args)
        raise OSError('Caption output is unavailable')

    assert tts.speak('ABC', on_progress=progress)
    assert len(calls) == 1
    assert played == [1] and stopped == []
    assert not tts.is_speaking()


def test_synthesis_without_audio_never_emits_caption_progress(monkeypatch):
    tts = TextToSpeech(enabled=False)
    tts.enabled = True
    monkeypatch.setattr(tts, 'synthesize', lambda _text: None)
    updates = []
    assert not tts.speak('Nothing to play.', on_progress=lambda *args: updates.append(args))
    assert updates == []


def test_timing_is_attached_to_the_same_synthesis_audio(monkeypatch):
    tts = TextToSpeech(enabled=False)
    tts.enabled = True
    audio = np.ones(100, dtype=np.float32)
    stale = np.ones(200, dtype=np.float32)
    native = CharacterTimeline('ABC', (0.2, 0.4, 0.8), 'phoneme')

    def synthesize(_text):
        tts._synthesis_timing.result = stale, native
        return audio, 100

    monkeypatch.setattr(tts, 'synthesize', synthesize)
    prepared = tts._prepare_playback('ABC')
    assert prepared[0] is audio
    assert prepared[2].source == 'estimated'
    assert prepared[2] is not native


def test_replaced_audio_cannot_finalize_the_newer_caption(monkeypatch):
    tts, _clock, _played, _stopped = timed_playback(monkeypatch)
    updates = []

    def progress(prefix, finished, interrupted):
        updates.append((prefix, finished, interrupted))
        if prefix == 'ABC':
            tts._playback_serial += 1  # A different output buffer replaced this playback.

    assert not tts.speak('ABC', on_progress=progress)
    assert [prefix for prefix, _finished, _cut in updates] == ['A', 'AB', 'ABC']
    assert not any(finished for _prefix, finished, _cut in updates)
