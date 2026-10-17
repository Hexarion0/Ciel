"""Validate lazy CUDA inference before declaring STT ready, without hardware."""

from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.audio.stt import SpeechToText


@pytest.mark.parametrize('cuda_failure', [False, True])
def test_startup_consumes_cuda_warmup_before_reporting_ready(monkeypatch, cuda_failure):
    stt = object.__new__(SpeechToText)
    stt.model_size, stt.device, stt.compute_type = 'small.en', 'cuda', 'int8_float16'
    stt.language = 'en'
    stt._ready = Event()
    stt._load_error = None
    gpu_model, cpu_model = MagicMock(), MagicMock()
    inference = []

    def segments():
        assert not stt._ready.is_set()
        inference.append('ran')
        if cuda_failure:
            raise RuntimeError('Library libcublas.so.12 is not found or cannot be loaded')
        yield MagicMock()

    gpu_model.transcribe.return_value = (segments(), MagicMock())
    factory = MagicMock(side_effect=[gpu_model, cpu_model])
    monkeypatch.setattr('ciel.audio.stt.WhisperModel', factory)
    stt._load_model_bg()
    assert inference == ['ran']  # Creating the generator alone does not load cuBLAS.
    assert stt.is_ready() and stt._load_error is None
    if cuda_failure:
        assert stt.model is cpu_model and stt.device == 'cpu' and stt.compute_type == 'int8'
        assert factory.call_count == 2
    else:
        assert stt.model is gpu_model and stt.device == 'cuda'
        factory.assert_called_once()
    cpu_model.transcribe.assert_not_called()


def test_standalone_stt_defaults_match_command_config_without_changing_wake_defaults(monkeypatch):
    from ciel.config import AudioConfig

    defaults = AudioConfig()
    model = MagicMock()
    load = MagicMock(return_value=model)
    monkeypatch.setattr(SpeechToText, "_load_model", load)
    monkeypatch.setattr("ciel.audio.stt._preload_cuda_libraries", lambda: None)

    stt = SpeechToText()

    assert stt.wait_ready(timeout=1)
    assert (stt.model_size, stt.device, stt.compute_type) == (
        defaults.stt_model, defaults.stt_device, defaults.stt_compute_type,
    )
    assert stt.retry_model_size == defaults.stt_retry_model
    assert stt.retry_min_free_mb == defaults.stt_retry_min_free_mb
    assert defaults.wake_stt_model == "base.en"
    load.assert_called_once_with(defaults.stt_model, defaults.stt_device, defaults.stt_compute_type)


def test_standalone_stt_keeps_explicit_cpu_and_disabled_larger_retry(monkeypatch):
    load = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(SpeechToText, "_load_model", load)
    monkeypatch.setattr("ciel.audio.stt._preload_cuda_libraries", lambda: None)

    stt = SpeechToText(
        model_size="tiny.en", device="cpu", compute_type="int8", retry_model=None,
    )

    assert stt.wait_ready(timeout=1)
    assert stt.retry_model_size is None
    load.assert_called_once_with("tiny.en", "cpu", "int8")
