"""Tests for faster-whisper speech-to-text module."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ciel.audio.stt import SpeechToText


@pytest.fixture(autouse=True)
def model_without_downloads(monkeypatch):
    model = MagicMock()
    model.transcribe.return_value = ([], SimpleNamespace(language="en", duration=1.0))
    monkeypatch.setattr(SpeechToText, "_load_model", lambda *_args: model)
    monkeypatch.setattr("ciel.audio.stt.WhisperModel", MagicMock(return_value=model))
    monkeypatch.setattr("ciel.audio.stt._preload_cuda_libraries", lambda: None)
    monkeypatch.setattr("ciel.audio.stt.free_cuda_memory_mb", lambda: 8192)
    return model


def test_stt_initialization():
    """Verify SpeechToText initializes model cleanly on CPU."""
    stt = SpeechToText(model_size="tiny.en", device="cpu", compute_type="int8")
    assert stt.model_size == "tiny.en"
    assert stt.device == "cpu"


@pytest.mark.parametrize("free_mb", [128, None])
def test_training_load_retries_primary_without_allocating_another_cuda_model(
    model_without_downloads, monkeypatch, free_mb,
):
    monkeypatch.setattr("ciel.audio.stt.free_cuda_memory_mb", lambda: free_mb)
    factory = MagicMock(side_effect=AssertionError("Must not allocate under GPU pressure"))
    monkeypatch.setattr("ciel.audio.stt.WhisperModel", factory)
    model_without_downloads.transcribe.side_effect = [
        decoding("And good.", -0.62), decoding("And include examples.", -0.1),
    ]
    stt = SpeechToText(device="cuda", retry_model="medium.en")
    result = stt.transcribe_detailed(np.ones(16000, dtype=np.float32) * 0.1)
    assert result["text"] == "And include examples."
    assert result["retried"] and stt.device == "cuda"
    factory.assert_not_called()


def test_larger_retry_becomes_available_after_training_releases_vram(
    model_without_downloads, monkeypatch,
):
    free_mb = [128]
    monkeypatch.setattr("ciel.audio.stt.free_cuda_memory_mb", lambda: free_mb[0])
    model_without_downloads.transcribe.return_value = decoding("And good.", -0.62)
    larger = MagicMock()
    larger.transcribe.return_value = decoding("And include examples.", -0.1)
    factory = MagicMock(return_value=larger)
    monkeypatch.setattr("ciel.audio.stt.WhisperModel", factory)
    stt = SpeechToText(device="cuda", retry_model="medium.en")
    audio = np.ones(16000, dtype=np.float32) * 0.1
    assert stt.transcribe(audio) == "And good."
    factory.assert_not_called()
    free_mb[0] = 4096
    assert stt.transcribe(audio) == "And include examples."
    factory.assert_called_once()
    assert stt._retry_model is None  # Do not retain the temporary training VRAM allocation.


def test_identical_retry_reuses_primary_model(model_without_downloads, monkeypatch):
    factory = MagicMock(side_effect=AssertionError("Duplicate model"))
    monkeypatch.setattr("ciel.audio.stt.WhisperModel", factory)
    model_without_downloads.transcribe.side_effect = [
        decoding("And good.", -0.62), decoding("And include examples.", -0.1),
    ]
    stt = SpeechToText(model_size="medium.en", retry_model="medium.en", device="cuda")
    assert stt.transcribe(np.ones(16000, dtype=np.float32) * 0.1) == "And include examples."
    factory.assert_not_called()


def test_stt_empty_audio():
    """Verify STT handles empty audio arrays without crashing."""
    stt = SpeechToText(model_size="tiny.en", device="cpu", compute_type="int8")
    empty_audio = np.empty((0,), dtype=np.float32)
    result = stt.transcribe(empty_audio)
    assert result == ""

    detailed = stt.transcribe_detailed(empty_audio)
    assert detailed["text"] == ""
    assert detailed["segments"] == []


def test_stt_silence_audio():
    """Verify STT with 1 second of silence produces clean empty output."""
    stt = SpeechToText(model_size="tiny.en", device="cpu", compute_type="int8")
    silent_audio = np.zeros(16000, dtype=np.float32)
    result = stt.transcribe(silent_audio)
    assert isinstance(result, str)


@pytest.mark.parametrize("text", ["Bye.", "Okay.", "Thank you.", "Oh.", "You."])
def test_confident_short_speech_is_not_discarded(text):
    assert not SpeechToText._is_hallucination(text, no_speech_prob=0.01, avg_logprob=-0.1)
    assert SpeechToText._is_hallucination(text, no_speech_prob=0.5, avg_logprob=-0.9)


def test_decode_preserves_spoken_corrections(model_without_downloads):
    model_without_downloads.transcribe.return_value = (
        [
            SimpleNamespace(
                text="No, no. Five days ago.",
                start=0,
                end=2,
                avg_logprob=-0.1,
                no_speech_prob=0.01,
            )
        ],
        SimpleNamespace(language="en", duration=2),
    )
    stt = SpeechToText()
    assert stt.transcribe(np.ones(32000, dtype=np.float32) * 0.1) == "No, no. Five days ago."
    options = model_without_downloads.transcribe.call_args.kwargs
    assert options["beam_size"] == 5
    assert options["repetition_penalty"] == 1.0
    assert options["no_repeat_ngram_size"] == 0
    assert options["condition_on_previous_text"] is False
    assert "Words like:" not in options["initial_prompt"]


def test_digital_silence_does_not_invoke_model(model_without_downloads):
    stt = SpeechToText()
    assert stt.transcribe(np.zeros(16000, dtype=np.float32)) == ""
    model_without_downloads.transcribe.assert_not_called()


def decoding(text, logprob, word_probability=1.0):
    return (
        [
            SimpleNamespace(
                text=text,
                start=0.0,
                end=1.0,
                avg_logprob=logprob,
                no_speech_prob=0.01,
                words=[SimpleNamespace(probability=word_probability)],
            )
        ],
        SimpleNamespace(language="en", duration=1.0),
    )


def test_ambiguous_short_speech_uses_one_stronger_decode(model_without_downloads):
    model_without_downloads.transcribe.side_effect = [
        decoding("I hate Ciel.", -0.95),
        decoding("Hey Ciel.", -0.15),
    ]
    result = SpeechToText().transcribe_detailed(np.ones(16000, dtype=np.float32) * 0.1)
    assert result["text"] == "Hey Ciel."
    assert result["retried"] and not result["needs_repeat"]
    assert model_without_downloads.transcribe.call_count == 2
    assert model_without_downloads.transcribe.call_args.kwargs["beam_size"] == 8


def test_uncertain_transcription_is_not_returned_as_command(model_without_downloads):
    model_without_downloads.transcribe.return_value = decoding("I hate Ciel.", -1.2)
    result = SpeechToText().transcribe_detailed(np.ones(16000, dtype=np.float32) * 0.1)
    assert result["text"] == ""
    assert result["needs_repeat"] and result["retried"]
    assert model_without_downloads.transcribe.call_count == 2


def test_rejected_speech_keeps_raw_text_for_a_repeat_not_a_command(model_without_downloads):
    model_without_downloads.transcribe.return_value = decoding("Hey Ciel.", -1.5)
    result = SpeechToText().transcribe_detailed(np.ones(16000, dtype=np.float32) * 0.1)
    assert result["text"] == ""
    assert result["raw_text"] == "Hey Ciel."
    assert result["needs_repeat"]


def test_clear_bare_wake_does_not_load_a_large_retry_model(model_without_downloads, monkeypatch):
    model_without_downloads.transcribe.return_value = decoding("Hey Ciel.", -0.63)
    factory = MagicMock(side_effect=AssertionError("Unnecessary medium model load"))
    monkeypatch.setattr("ciel.audio.stt.WhisperModel", factory)
    result = SpeechToText(retry_model="medium.en").transcribe_detailed(
        np.ones(16000, dtype=np.float32) * 0.1
    )
    assert result["text"] == "Hey Ciel."
    assert not result["retried"] and not result["needs_repeat"]
    model_without_downloads.transcribe.assert_called_once()
    factory.assert_not_called()


def test_word_confidence_can_trigger_retry(model_without_downloads):
    model_without_downloads.transcribe.side_effect = [
        decoding("Unclear command", -0.1, 0.3),
        decoding("Clear command", -0.1, 0.9),
    ]
    result = SpeechToText().transcribe_detailed(np.ones(16000, dtype=np.float32) * 0.1)
    assert result["text"] == "Clear command"
    assert result["retried"]


def test_long_uncertain_speech_does_not_repeat_expensive_inference(model_without_downloads):
    model_without_downloads.transcribe.return_value = decoding("Unclear command", -1.2)
    result = SpeechToText().transcribe_detailed(np.ones(16000 * 13, dtype=np.float32) * 0.1)
    assert result["needs_repeat"] and not result["retried"]
    model_without_downloads.transcribe.assert_called_once()


def test_confident_negative_statement_is_preserved(model_without_downloads):
    model_without_downloads.transcribe.return_value = decoding("I hate Ciel.", -0.1)
    assert SpeechToText().transcribe(np.ones(16000, dtype=np.float32) * 0.1) == "I hate Ciel."
    model_without_downloads.transcribe.assert_called_once()


def test_larger_retry_model_is_loaded_only_for_ambiguous_speech(
    model_without_downloads, monkeypatch
):
    larger = MagicMock()
    larger.transcribe.return_value = decoding("Correct command", -0.1)
    factory = MagicMock(return_value=larger)
    monkeypatch.setattr("ciel.audio.stt.WhisperModel", factory)
    stt = SpeechToText(retry_model="medium.en", device="cpu", compute_type="int8")
    model_without_downloads.transcribe.return_value = decoding("Clear command", -0.1)
    audio = np.ones(16000, dtype=np.float32) * 0.1
    assert stt.transcribe(audio) == "Clear command"
    factory.assert_not_called()
    model_without_downloads.transcribe.return_value = decoding("Wrong command", -1.2)
    assert stt.transcribe(audio) == "Correct command"
    assert stt.transcribe(audio) == "Correct command"
    factory.assert_called_once_with("medium.en", device="cpu", compute_type="int8")
    assert larger.transcribe.call_count == 2


def test_failed_larger_retry_still_requests_repeat(model_without_downloads, monkeypatch):
    model_without_downloads.transcribe.return_value = decoding("Unclear command", -1.2)
    monkeypatch.setattr(
        "ciel.audio.stt.WhisperModel", MagicMock(side_effect=RuntimeError("offline"))
    )
    result = SpeechToText(retry_model="medium.en").transcribe_detailed(
        np.ones(16000, dtype=np.float32) * 0.1
    )
    assert result["needs_repeat"] and result["text"] == ""
    assert result["raw_text"] == "Unclear command"


@pytest.mark.parametrize('fails_during_load', [False, True])
def test_retry_oom_is_not_repeated_and_releases_cached_model(
    model_without_downloads, monkeypatch, fails_during_load
):
    model_without_downloads.transcribe.return_value = decoding('And good.', -0.62)
    larger = MagicMock()
    larger.transcribe.side_effect = RuntimeError('CUDA failed with error out of memory')
    factory = MagicMock(
        side_effect=RuntimeError('CUDA failed with error out of memory')
    ) if fails_during_load else MagicMock(return_value=larger)
    monkeypatch.setattr('ciel.audio.stt.WhisperModel', factory)
    stt = SpeechToText(device='cuda', compute_type='int8_float16', retry_model='medium.en')
    audio = np.ones(16000, dtype=np.float32) * 0.1
    first = stt.transcribe_detailed(audio)
    second = stt.transcribe_detailed(audio)
    assert first['text'] == second['text'] == 'And good.'
    assert first['retried'] and not second['retried']
    assert not first['needs_repeat']
    assert stt.device == 'cuda'
    assert stt._retry_model is None
    factory.assert_called_once()
    assert larger.transcribe.call_count == (0 if fails_during_load else 1)


def test_cpu_fallback_does_not_reuse_a_cached_cuda_retry(model_without_downloads, monkeypatch):
    model_without_downloads.transcribe.side_effect = RuntimeError('CUDA out of memory')
    cpu_primary, cpu_retry, cached_cuda = MagicMock(), MagicMock(), MagicMock()
    cpu_primary.transcribe.return_value = decoding('And good.', -0.62)
    cpu_retry.transcribe.return_value = decoding('And good.', -0.1)
    factory = MagicMock(side_effect=[cpu_primary, cpu_retry])
    monkeypatch.setattr('ciel.audio.stt.WhisperModel', factory)
    stt = SpeechToText(model_size='small.en', device='cuda', retry_model='medium.en')
    stt._retry_model = cached_cuda
    result = stt.transcribe_detailed(np.ones(16000, dtype=np.float32) * 0.1)
    assert result['text'] == 'And good.' and not result['needs_repeat']
    assert stt.device == 'cpu'
    assert [call.args[0] for call in factory.call_args_list] == ['small.en', 'medium.en']
    assert all(call.kwargs['device'] == 'cpu' for call in factory.call_args_list)
    cached_cuda.transcribe.assert_not_called()


def test_repetition_hallucination_requests_repeat(model_without_downloads):
    result = decoding('la ' * 40, -0.1)
    result[0][0].compression_ratio = 5.0
    model_without_downloads.transcribe.return_value = result
    detailed = SpeechToText().transcribe_detailed(np.ones(16000, dtype=np.float32) * 0.1)
    assert detailed['needs_repeat'] and detailed['text'] == ''
