"""Pocket/provider/streaming regressions; ordinary tests never download a model."""

import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from ciel.audio.pocket import PocketTTS
from ciel.audio.tts import TextToSpeech, resolve_tts_engine
from ciel.audio.voice_style import TTSRequest, select_voice_style
from ciel.config import AudioConfig, Settings


class Tensor:
    """Minimal CPU tensor double without a torch import."""

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.ones(1920, dtype=np.float32)


@pytest.fixture
def model(monkeypatch):
    model = SimpleNamespace(device="cpu", sample_rate=24000)
    model.get_state_for_audio_prompt = MagicMock(return_value={"cached": True})
    model.generate_audio_stream = MagicMock(
        side_effect=lambda *_args, **_kwargs: iter([Tensor(), Tensor()])
    )
    load = MagicMock(return_value=model)
    monkeypatch.setitem(sys.modules, "pocket_tts", SimpleNamespace(
        TTSModel=SimpleNamespace(load_model=load),
    ))
    model.load = load
    return model


@pytest.fixture
def tts(monkeypatch, tmp_path):
    settings = Settings(_env_file=None, tts_pocket_references=str(tmp_path))
    monkeypatch.setattr("ciel.audio.tts.get_settings", lambda: settings)
    return TextToSpeech(engine="pocket", voice_name="ciel")


@pytest.mark.parametrize("engine,expected", [
    ("pocket", "pocket"), ("hq", "chatterbox_turbo"),
    ("chatterbox", "chatterbox_turbo"), ("piper", "piper"), ("auto", "piper"),
])
def test_provider_selection(engine, expected):
    assert resolve_tts_engine("ciel", engine) == expected


@pytest.mark.parametrize("config", [AudioConfig, Settings])
@pytest.mark.parametrize("values", [
    {"tts_engine": "invented"}, {"tts_pocket_device": "cuda"},
    {"tts_pocket_quantization": "int4"}, {"tts_pocket_language": ""},
])
def test_invalid_configuration(config, values):
    with pytest.raises(ValidationError):
        config(**values)


def test_configuration_reaches_audio(monkeypatch, tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "TTS_ENGINE=pocket\nTTS_POCKET_QUANTIZATION=int8\nTTS_POCKET_STREAMING=false\n"
        f"TTS_POCKET_REFERENCES={tmp_path}\nTTS_POCKET_NEUTRAL_REFERENCE={tmp_path}/neutral.wav\n"
    )
    for name in ("TTS_ENGINE", "TTS_POCKET_QUANTIZATION", "TTS_POCKET_STREAMING",
                 "TTS_POCKET_REFERENCES", "TTS_POCKET_NEUTRAL_REFERENCE"):
        monkeypatch.delenv(name, raising=False)
    audio = Settings(_env_file=path).audio
    assert audio.tts_engine == "pocket"
    assert audio.tts_pocket_quantization == "int8"
    assert not audio.tts_pocket_streaming
    assert audio.tts_pocket_references == str(tmp_path)
    assert audio.tts_pocket_neutral_reference == str(tmp_path / "neutral.wav")


def test_lazy_cpu_init_model_and_reference_reused(tts, model):
    assert tts._pocket is None
    assert not model.load.called
    assert tts.warmup()
    for _ in range(2):
        audio, rate = tts.synthesize("Hello.")
        assert audio.dtype == np.float32 and rate == 24000
    model.load.assert_called_once_with(language="english_2026-09", quantize=False)
    model.get_state_for_audio_prompt.assert_called_once_with("alba", truncate=True)
    assert model.generate_audio_stream.call_count == 2
    assert all(call.kwargs["copy_state"] for call in model.generate_audio_stream.call_args_list)
    assert tts.last_engine == "pocket"
    tts.close()


def test_supported_int8_and_stable_fp32_recovery(tmp_path, model):
    provider = PocketTTS(tmp_path, quantization="int8")
    model.load.side_effect = [RuntimeError("quantization unavailable"), model]
    provider.warmup()
    assert provider.effective_quantization == "none"
    assert [call.kwargs["quantize"] for call in model.load.call_args_list] == [True, False]
    model.load.reset_mock(side_effect=True)
    model.load.return_value = model
    provider = PocketTTS(tmp_path, quantization="int8")
    provider.warmup()
    assert provider.effective_quantization == "int8"
    model.load.assert_called_once_with(language="english_2026-09", quantize=True)


def test_style_router_preserves_neutral_default_and_reuses_cues():
    assert select_voice_style("Technical configuration") == "neutral"
    assert select_voice_style("[chuckle] Again?") == "amused"
    for style in ("neutral", "warm", "amused", "firm", "focused", "soft"):
        assert select_voice_style("Anything", style) == style
    with pytest.raises(ValueError):
        TTSRequest("Hello", "angry")


def test_styles_use_same_bank_and_missing_style_shares_neutral(tmp_path, model, caplog):
    neutral = tmp_path / "neutral.wav"
    warm = tmp_path / "warm.wav"
    neutral.touch()
    warm.touch()
    provider = PocketTTS(tmp_path)
    for style in ("neutral", "soft", "warm", "soft"):
        list(provider.stream(TTSRequest("Hello", style), threading.Event()))
    assert model.get_state_for_audio_prompt.call_count == 2
    refs = [call.args[0] for call in model.get_state_for_audio_prompt.call_args_list]
    assert refs == [neutral, warm]
    assert "style 'soft' missing" in caplog.text


def test_missing_neutral_keeps_one_builtin_identity(tmp_path, model):
    (tmp_path / "warm.wav").touch()
    provider = PocketTTS(tmp_path)
    list(provider.stream(TTSRequest("Hello", "warm"), threading.Event()))
    model.get_state_for_audio_prompt.assert_called_once_with("alba", truncate=True)


def test_explicit_missing_reference_fails(tmp_path, model):
    provider = PocketTTS(tmp_path, neutral_reference=tmp_path / "missing.wav")
    with pytest.raises(FileNotFoundError):
        provider.warmup()


def test_pocket_to_piper_on_initialization_or_inference_failure(tts, model, monkeypatch):
    model.load.side_effect = ImportError("dependency missing")
    fallback = MagicMock(return_value=(np.ones(100, dtype=np.float32), 22050))
    monkeypatch.setattr(tts, "_synthesize_piper_fallback", fallback)
    audio, rate = tts.synthesize("[chuckle] Hello.")
    assert rate == 22050 and audio.size == 100
    assert tts.last_engine == "piper"
    fallback.assert_called_once_with("Hello.")
    model.load.side_effect = None
    tts._pocket_retry_at = time.monotonic() - 1
    model.generate_audio_stream.side_effect = RuntimeError("inference failed")
    assert tts.synthesize("Hello.") is not None
    assert fallback.call_count == 2
    # Recover on a subsequent turn without reloading the successfully loaded model.
    tts._pocket_retry_at = time.monotonic() - 1
    model.generate_audio_stream.side_effect = lambda *_a, **_k: iter([Tensor()])
    assert tts.synthesize("Hello.")[1] == 24000
    assert tts.last_engine == "pocket"


def test_hq_chain_chatterbox_to_pocket_to_piper(tts, model, monkeypatch):
    tts.engine = "chatterbox_turbo"
    monkeypatch.setattr(tts, "_load_chatterbox_worker", MagicMock(
        side_effect=RuntimeError("HQ unavailable"),
    ))
    assert tts.synthesize("Hello.")[1] == 24000
    assert tts._chatterbox_failed and tts.last_engine == "pocket"
    model.generate_audio_stream.side_effect = RuntimeError("CPU inference failed")
    monkeypatch.setattr(tts, "_synthesize_piper_fallback", lambda _t: (np.ones(10), 22050))
    assert tts.synthesize("Hello.")[1] == 22050
    assert tts.last_engine == "piper"


def test_partial_failure_stops_without_repeating_text(tts, model, monkeypatch):
    def fail(*_a, **_k):
        yield Tensor()
        raise RuntimeError("late error")
    model.generate_audio_stream.side_effect = fail
    fallback = MagicMock()
    monkeypatch.setattr(tts, "_synthesize_piper_fallback", fallback)
    iterator = tts.synthesize_stream("Hello.")
    assert next(iterator)[0].size == 1920
    with pytest.raises(RuntimeError, match="after audio started"):
        next(iterator)
    fallback.assert_not_called()


def test_provider_drains_and_releases_after_interruption(tmp_path, model):
    drained = threading.Event()
    def stream(*_a, stop, **_k):
        try:
            while not stop.is_set():
                yield Tensor()
        finally:
            drained.set()
    model.generate_audio_stream.side_effect = stream
    provider = PocketTTS(tmp_path)
    stop = threading.Event()
    iterator = provider.stream(TTSRequest("Hello"), stop)
    next(iterator)
    stop.set()
    iterator.close()
    assert drained.is_set()
    assert provider._lock.acquire(blocking=False)
    provider._lock.release()
    provider.close()
    assert provider._model is None and not provider._states


def install_output(monkeypatch, played, *, started=None):
    """PortAudio double runs real callbacks at a bounded accelerated audio clock."""
    class Abort(Exception):
        pass
    class Finished(Exception):
        pass
    class Output:
        def __init__(self, callback, finished_callback, **kwargs):
            self.callback = callback
            self.finished = finished_callback
            self.halt = threading.Event()
            self.kwargs = kwargs
        def __enter__(self):
            def run():
                while not self.halt.is_set():
                    wave = np.zeros((1024, 1), dtype=np.float32)
                    try:
                        self.callback(wave, 1024, None, SimpleNamespace(output_underflow=False))
                    except (Abort, Finished):
                        played.append(wave.copy())
                        break
                    played.append(wave.copy())
                    if started is not None and wave.any():
                        started.set()
                    self.halt.wait(0.002)
                self.finished()
            self.thread = threading.Thread(target=run)
            self.thread.start()
            return self
        def abort(self):
            self.halt.set()
        def __exit__(self, *_args):
            self.halt.set()
            self.thread.join(1)
    monkeypatch.setattr("ciel.audio.chunk_playback.sd", SimpleNamespace(
        OutputStream=Output, CallbackAbort=Abort, CallbackStop=Finished,
    ))
    monkeypatch.setattr("ciel.audio.tts.sd", SimpleNamespace(stop=lambda: None))


def test_first_chunk_plays_before_generation_completes(tts, model, monkeypatch):
    audio_started = threading.Event()
    generated = threading.Event()
    def stream(*_a, stop, **_k):
        yield Tensor()
        assert audio_started.wait(1)
        if not stop.is_set():
            yield Tensor()
            generated.set()
    model.generate_audio_stream.side_effect = stream
    played = []
    install_output(monkeypatch, played)
    progress = []
    assert tts.speak(
        "Hello.", on_start=audio_started.set,
        on_progress=lambda *args: progress.append(args),
    )
    assert generated.is_set()
    assert sum(int(np.count_nonzero(wave)) for wave in played) == 3840
    assert progress[-1] == ("Hello.", True, False)
    assert not tts.is_speaking()


def test_stop_cancels_synthesis_and_playback_promptly(tts, model, monkeypatch):
    active = threading.Event()
    drained = threading.Event()
    def stream(*_a, stop, **_k):
        try:
            while not stop.is_set():
                yield Tensor()
        finally:
            drained.set()
    model.generate_audio_stream.side_effect = stream
    played = []
    install_output(monkeypatch, played, started=active)
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(tts.speak("A longer reply.")))
    thread.start()
    assert active.wait(1)
    start = time.monotonic()
    interruption = tts.stop()
    thread.join(1)
    assert time.monotonic() - start < 1
    assert outcome == [False]
    assert drained.is_set() and not tts.is_speaking()
    assert interruption["full_text"] == "A longer reply."


def test_cancellation_before_first_audio_prevents_playback(tts, model, monkeypatch):
    entered = threading.Event()
    def stream(*_a, stop, **_k):
        entered.set()
        while not stop.is_set():
            time.sleep(0.005)
        return
        yield  # make a generator with no audio
    model.generate_audio_stream.side_effect = stream
    played = []
    install_output(monkeypatch, played)
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(tts.speak("Hello.")))
    thread.start()
    assert entered.wait(1)
    tts.stop()
    thread.join(1)
    assert outcome == [False] and not played


def test_batch_mode_honors_explicit_style(tts, model, monkeypatch, tmp_path):
    (tmp_path / "neutral.wav").touch()
    (tmp_path / "warm.wav").touch()
    tts._pocket_streaming = False
    monkeypatch.setattr("ciel.audio.tts.sd", SimpleNamespace(
        play=lambda *_a, **_k: None, get_stream=lambda: SimpleNamespace(active=False),
    ))
    assert tts.speak("Hello.", style="warm")
    assert model.get_state_for_audio_prompt.call_args.args[0] == tmp_path / "warm.wav"


@pytest.mark.integration
def test_pocket_streamed_reply_real_playback():
    """Exercise Ciel's normal sentence/token path with real inference and PortAudio."""
    from ciel.audio.streaming import stream_reply
    from ciel.providers.base import LLMStreamChunk

    class LocalReply:
        def stream(self, *_args, **_kwargs):
            text = "Everything is running normally. I can check the rest of the system too."
            for word in text.split():
                yield LLMStreamChunk(word + " ", "local-smoke", "fixture")

    speech = TextToSpeech(engine="pocket", voice_name="ciel")
    try:
        reply = stream_reply(LocalReply(), [], speech, lambda: True)
        assert reply.spoken and not reply.canceled and reply.error is None
        assert reply.first_audio_seconds is not None
        assert speech.last_engine == "pocket", "Fallback does not count as a Pocket smoke test"
    finally:
        speech.close()


@pytest.mark.integration
def test_pocket_real_barge_in():
    """Verify actual Pocket workers unwind promptly after real speaker interruption."""
    speech = TextToSpeech(engine="pocket", voice_name="ciel")
    started = threading.Event()
    outcomes = []
    worker = threading.Thread(target=lambda: outcomes.append(speech.speak(
        "Let's check each part of the configuration together. "
        "The service is running, and the remaining settings look steady. "
        "We can review the environment variables before restarting anything.",
        on_start=started.set,
    )))
    try:
        assert speech.warmup()
        worker.start()
        assert started.wait(30)
        assert speech.last_engine == "pocket"
        begin = time.monotonic()
        speech.stop()
        worker.join(2)
        assert time.monotonic() - begin < 2
        assert outcomes == [False]
        assert not speech.is_speaking()
        assert not any(t.name == "ciel-pocket-audio" for t in threading.enumerate())
    finally:
        speech.stop()
        if worker.ident is not None:
            worker.join(2)
        speech.close()


def test_unusable_model_released_and_initialization_failure_latched(tts, model, monkeypatch):
    model.get_state_for_audio_prompt.side_effect = ValueError("cloning access required")
    monkeypatch.setattr(tts, "_synthesize_piper_fallback", lambda _t: (np.ones(10), 22050))
    assert tts.synthesize("Hello.")[1] == 22050
    assert tts._pocket._model is None and not tts._pocket._states
    assert tts._pocket_failed and not tts._pocket_retry_at
    assert tts.synthesize("Again.")[1] == 22050
    assert model.load.call_count == 1


def test_batch_late_failure_can_fallback_before_playback(tts, model, monkeypatch):
    def fail(*_a, **_k):
        yield Tensor()
        raise RuntimeError("late batch error")
    model.generate_audio_stream.side_effect = fail
    monkeypatch.setattr(tts, "_synthesize_piper_fallback", lambda _t: (np.ones(10), 22050))
    audio, rate = tts.synthesize("Hello.")
    assert audio.size == 10 and rate == 22050
    assert tts.last_engine == "piper"


def test_stream_playback_timeout_cancels_worker(tts, model, monkeypatch):
    from ciel.config import Settings

    monkeypatch.setattr("ciel.audio.tts.get_settings", lambda: Settings(
        _env_file=None, tts_synthesis_timeout_seconds=0.1,
    ))
    drained = threading.Event()
    def stream(*_a, stop, **_k):
        try:
            while not stop.is_set():
                time.sleep(0.005)
            return
            yield
        finally:
            drained.set()
    model.generate_audio_stream.side_effect = stream
    assert not tts.speak("Hello.")
    assert drained.is_set() and not tts.is_speaking()
