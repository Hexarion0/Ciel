"""Recognition comparisons need honest transcript scoring and isolated input handling."""

import numpy as np
import pytest
import soundfile as sf

from scripts.benchmark_speech_backends import load_audio, word_error
from scripts.benchmark_turn_detection import turn_features


def test_word_error_normalizes_punctuation_case_and_curly_apostrophes():
    assert word_error("No, I’ve seen Ciel.", "no i've seen ciel!") == {
        "errors": 0, "words": 4, "wer": 0,
    }


@pytest.mark.parametrize(("expected", "actual", "errors"), [
    ("stop listening now", "stop now", 1),
    ("stop now", "please stop now", 1),
    ("stop", "start", 1),
    ("", "phantom speech", 2),
    ("", "", 0),
])
def test_word_error_counts_deletions_insertions_and_substitutions(expected, actual, errors):
    assert word_error(expected, actual)["errors"] == errors


def test_stereo_resampling_preserves_duration_and_mono_shape(tmp_path):
    path = tmp_path / "sample.wav"
    sf.write(path, np.full((24000, 2), 0.1, dtype=np.float32), 24000, subtype="FLOAT")
    audio = load_audio(path)
    assert audio.shape == (16000,)
    assert audio.dtype == np.float32
    assert abs(float(audio[8000]) - 0.1) < 0.001


def test_nonfinite_recording_is_rejected_before_recognition(tmp_path):
    path = tmp_path / "invalid.wav"
    sf.write(path, np.array([0.1, np.nan], dtype=np.float32), 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="Non-finite"):
        load_audio(path)


def test_turn_features_keep_recent_audio_at_end_of_eight_second_window():
    original = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    saved = original.copy()
    captured = []

    def extract(audio, padding):
        captured.append(audio)
        assert padding == 0
        return np.zeros((80, 800), dtype=np.float32)

    result = turn_features(original, extract)
    assert result.shape == (1, 80, 800) and result.dtype == np.float32
    assert captured[0].shape == (128000,)
    assert np.allclose(captured[0][:112000], 0, atol=1e-6)
    assert captured[0][-1] > 0 and captured[0][-16000] < 0
    np.testing.assert_array_equal(original, saved)


def test_turn_features_discard_old_audio_and_reject_invalid_samples():
    captures = []

    def extract(audio, padding):
        captures.append(audio)
        return np.zeros((80, 800), dtype=np.float32)

    turn_features(np.concatenate((np.ones(16000), np.zeros(128000))), extract)
    assert not captures[0].any(), "Old audio leaked into the model's recent window"
    with pytest.raises(ValueError, match="finite"):
        turn_features(np.array([np.nan]), extract)
