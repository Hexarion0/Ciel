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
