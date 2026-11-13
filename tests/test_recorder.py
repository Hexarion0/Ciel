"""Utterance endpointing: short answers end promptly, transient clicks do not."""

import numpy as np
import pytest

from ciel.audio.recorder import VoiceRecorder

FRAME = 1280  # 80 ms at 16 kHz, the listener's frame size.


def seconds_until_endpoint(speech_frames: int) -> float:
    recorder = VoiceRecorder(
        sample_rate=16000, silence_duration_seconds=0.7, pause_grace_seconds=0.2,
        min_speech_duration_seconds=0.12, initial_silence_timeout=5.0,
        speech_detector=lambda frame: bool(frame.max() > 0.5),
    )
    recorder.start()
    frames = [np.ones(FRAME, dtype=np.float32)] * speech_frames
    for index in range(200):
        frame = frames[index] if index < len(frames) else np.zeros(FRAME, dtype=np.float32)
        if not recorder.add_frame(frame):
            return (index + 1) * FRAME / 16000
    raise AssertionError("recording never ended")


@pytest.mark.parametrize("speech_frames", [2, 3, 6])
def test_a_short_spoken_answer_ends_after_the_pause_not_the_timeout(speech_frames):
    # "yes" or "no" can be two 80 ms frames of detected speech: 160 ms is above the
    # 120 ms minimum, so it must not be discarded as a transient click.
    assert seconds_until_endpoint(speech_frames) < 1.6


def test_a_single_click_is_still_ignored_as_a_transient():
    assert seconds_until_endpoint(1) > 4.0
