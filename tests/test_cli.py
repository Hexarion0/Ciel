import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

SOURCE = str(Path(__file__).resolve().parents[1] / "src")


@pytest.mark.parametrize("launch_args", [[], ["run"], ["start"], ["listen"], ["--listen"]])
@pytest.mark.parametrize("enabled", [True, False])
def test_fresh_launch_uses_voice_defaults_and_explicit_opt_outs(
    tmp_path, monkeypatch, launch_args, enabled,
):
    from ciel import __main__, audio, config, platform, providers
    from ciel.audio import captions

    for field in (
        "ambient_listening", "show_transcripts", "show_ai_transcripts", "barge_in_mode",
        "stt_model", "stt_device", "stt_compute_type", "stt_beam_size", "stt_retry_model",
    ):
        monkeypatch.delenv(field.upper(), raising=False)
    settings = config.Settings(_env_file=None, memory_db_path=str(tmp_path / "memory.db"))
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(platform, "get_audio_backend", MagicMock())
    monkeypatch.setattr(providers, "get_model_router", MagicMock())
    for component in ("WakeWordDetector", "SpeechToText", "VoiceRecorder", "TextToSpeech"):
        monkeypatch.setattr(audio, component, MagicMock())
    caption_constructor = MagicMock()
    monkeypatch.setattr(captions, "TerminalCaptions", caption_constructor)
    listener = MagicMock(return_value=SimpleNamespace(
        is_running=True, start=lambda: None, stop=lambda: None,
    ))
    monkeypatch.setattr(audio, "WakeListenerLoop", listener)

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(__main__, "time", SimpleNamespace(
        sleep=interrupt, monotonic=__main__.time.monotonic,
    ))
    overrides = [] if enabled else ["--no-ambient", "--no-show-ai-transcripts"]
    monkeypatch.setattr(sys, "argv", ["ciel", *launch_args, *overrides, "--no-text-input"])

    assert __main__.main() == 0
    assert listener.call_args.kwargs["ambient"] is enabled
    assert listener.call_args.kwargs["show_transcripts"] is False
    assert listener.call_args.kwargs["barge_in_mode"] == "speech"
    assert listener.call_args.kwargs["stt_beam_size"] == 5
    assert audio.SpeechToText.call_args.kwargs["model_size"] == "small.en"
    assert audio.SpeechToText.call_args.kwargs["device"] == "cuda"
    assert audio.SpeechToText.call_args.kwargs["compute_type"] == "int8_float16"
    assert audio.SpeechToText.call_args.kwargs["retry_model"] == "medium.en"
    assert caption_constructor.call_count == int(enabled)
    assert audio.TextToSpeech.call_args.kwargs.get("include_alignments", False) is enabled


def test_plain_ciel_launches_even_when_ambient_is_disabled_in_settings(monkeypatch):
    from ciel import __main__, config, platform, runtime

    settings = config.Settings(_env_file=None, ambient_listening=False)
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(platform, "get_audio_backend", MagicMock())
    launch = MagicMock(return_value=0)
    monkeypatch.setattr(runtime, "run_desktop", launch)
    monkeypatch.setattr(sys, "argv", ["ciel"])

    assert __main__.main() == 0
    launch.assert_called_once()
    assert launch.call_args.args[1].audio.ambient_listening is False


@pytest.mark.parametrize('arguments, enabled', [([], True), (
    ['--no-ambient', '--no-show-transcripts'], False,
)])
def test_env_listening_defaults_and_cli_overrides(tmp_path, monkeypatch, arguments, enabled):
    from ciel import __main__, audio, config, platform, providers

    settings = config.Settings(
        _env_file=None, memory_db_path=str(tmp_path / 'memory.db'),
        ambient_listening=True, show_transcripts=True, barge_in_mode='speech',
    )
    monkeypatch.setattr(config, 'get_settings', lambda: settings)
    monkeypatch.setattr(platform, 'get_audio_backend', MagicMock())
    monkeypatch.setattr(providers, 'get_model_router', MagicMock())
    for component in ('WakeWordDetector', 'SpeechToText', 'VoiceRecorder', 'TextToSpeech'):
        monkeypatch.setattr(audio, component, MagicMock())
    constructor = MagicMock(return_value=SimpleNamespace(
        is_running=True, start=lambda: None, stop=lambda: None,
    ))
    monkeypatch.setattr(audio, 'WakeListenerLoop', constructor)

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(__main__, 'time', SimpleNamespace(
        sleep=interrupt, monotonic=__main__.time.monotonic,
    ))
    monkeypatch.setattr(sys, 'argv', ['ciel', *arguments])
    assert __main__.main() == 0
    assert constructor.call_args.kwargs['ambient'] is enabled
    assert constructor.call_args.kwargs['show_transcripts'] is enabled
    assert constructor.call_args.kwargs['barge_in_mode'] == 'speech'


@pytest.mark.parametrize(
    "text, wake_info, expected_reply",
    [
        ("Hey Ciel.", {}, "Hey! What's on your mind?"),
        ("", {"stt_needs_repeat": True}, "I didn't catch that clearly. Could you say it again?"),
        ("So what's the time right now?", {}, "It's 3:16 AM."),
        ("My favorite game is CS2.", {},
         "Should I remember this? Your favorite game is CS2. Say yes to save it or no to skip it."),
    ],
)
@pytest.mark.parametrize("launch_args", [["--listen"], ["start"], ["start", "dev"]])
def test_legacy_listen_wake_or_uncertain_speech_does_not_call_provider(
    tmp_path, monkeypatch, text, wake_info, expected_reply, launch_args
):
    from datetime import datetime

    from ciel import __main__, audio, config, platform, providers
    from ciel.providers import intents

    clock_answer = intents.answer_clock_query
    monkeypatch.setattr(
        intents,
        "answer_clock_query",
        lambda text: clock_answer(text, datetime(2026, 10, 3, 3, 16)),
    )

    settings = config.Settings(
        _env_file=None, memory_db_path=str(tmp_path / "memory.db"), controller_enabled=False,
        ambient_listening=False, show_ai_transcripts=False,
    )
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(platform, "get_audio_backend", MagicMock())
    router = MagicMock()
    monkeypatch.setattr(providers, "get_model_router", lambda: router)
    tts = MagicMock()
    for component in ("WakeWordDetector", "SpeechToText", "VoiceRecorder"):
        monkeypatch.setattr(audio, component, MagicMock())
    monkeypatch.setattr(audio, "TextToSpeech", MagicMock(return_value=tts))
    followup = []

    def listener(**kwargs):
        loop = SimpleNamespace(is_running=True, stop=MagicMock())
        loop.start = lambda: followup.append(kwargs["on_transcription"](text, wake_info, None))
        return loop

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(audio, "WakeListenerLoop", listener)
    monkeypatch.setattr(__main__, "time", SimpleNamespace(
        sleep=interrupt, monotonic=__main__.time.monotonic,
    ))
    monkeypatch.setattr(sys, "argv", ["ciel", *launch_args])
    assert __main__.main() == 0
    assert followup == [True]
    router.send.assert_not_called()
    tts.speak.assert_called_once_with(expected_reply, block=True)


def test_help_does_not_import_audio_or_training(tmp_path):
    script = """
import runpy, sys
sys.argv = ["ciel", "--help"]
try:
    runpy.run_module("ciel", run_name="__main__")
except SystemExit as error:
    assert error.code == 0
assert "sounddevice" not in sys.modules
assert "ciel.audio.trainer" not in sys.modules
assert "faster_whisper" not in sys.modules
assert "onnx" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": SOURCE},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert "--listen" in result.stdout


def test_keyboard_exit_cleans_up_listener_and_tts(tmp_path, monkeypatch, capsys):
    from ciel import __main__, audio, config, platform, providers, terminal

    settings = config.Settings(_env_file=None, memory_db_path=str(tmp_path / "memory.db"))
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(platform, "get_audio_backend", MagicMock())
    monkeypatch.setattr(providers, "get_model_router", MagicMock())
    for component in ("WakeWordDetector", "SpeechToText", "VoiceRecorder"):
        monkeypatch.setattr(audio, component, MagicMock())
    tts = MagicMock()
    monkeypatch.setattr(audio, "TextToSpeech", MagicMock(return_value=tts))
    loop = SimpleNamespace(
        is_running=True, start=MagicMock(), stop=MagicMock(), submit_text=MagicMock(),
        cancel_response=MagicMock(), toggle_microphone_mute=MagicMock(return_value=True),
    )
    monkeypatch.setattr(audio, "WakeListenerLoop", MagicMock(return_value=loop))
    inputs = []

    class ImmediateTerminal(terminal.TerminalInput):
        def start(self):
            inputs.append(self)
            for line in ["/mute", "/stop", "/exit"]:
                self.handle_line(line)

    monkeypatch.setattr(terminal, "TerminalInput", ImmediateTerminal)
    monkeypatch.setattr(sys, "argv", ["ciel", "start", "--text-input"])
    assert __main__.main() == 0
    loop.toggle_microphone_mute.assert_called_once()
    assert loop.cancel_response.call_count == 2
    loop.submit_text.assert_not_called()
    loop.stop.assert_called_once()
    tts.close.assert_called_once()
    assert inputs[0]._closed.is_set()
    output = capsys.readouterr().out
    assert "[system]: Microphone muted" in output
    assert "[system]: Stopped the current reply." in output
    assert "[system]: Exiting CIEL" in output


def test_web_mode_reuses_voice_callbacks_and_controls(tmp_path, monkeypatch):
    from threading import Event

    from ciel import __main__, audio, config, platform, providers, web
    from ciel.providers.base import LLMResponse

    settings = config.Settings(
        _env_file=None, memory_db_path=str(tmp_path / "memory.db"), tts_streaming=False,
        show_ai_transcripts=False, ciel_persona_file="", ciel_preferred_name="owner",
    )
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(platform, "get_audio_backend", MagicMock())
    router = MagicMock()
    router.send.return_value = LLMResponse("Hi there.", "test", "test")
    monkeypatch.setattr(providers, "get_model_router", lambda: router)
    for component in ("WakeWordDetector", "SpeechToText", "VoiceRecorder"):
        monkeypatch.setattr(audio, component, MagicMock())
    tts = MagicMock()
    tts.clean_text_for_speech.side_effect = lambda text: text

    def speak(text, **controls):
        controls["on_start"]()
        controls["on_progress"](text, True, False)
        return True

    tts.speak.side_effect = speak
    monkeypatch.setattr(audio, "TextToSpeech", MagicMock(return_value=tts))
    loop = SimpleNamespace(
        is_running=True, start=MagicMock(), start_text_only=MagicMock(),
        stop=MagicMock(), cancel_response=MagicMock(),
        toggle_microphone_mute=MagicMock(return_value=True),
    )

    def make_loop(**kwargs):
        loop.submit_text = lambda text: kwargs["on_transcription"](
            text, {"input_source": "keyboard", "cancel_event": Event()}, None,
        )
        return loop

    monkeypatch.setattr(audio, "WakeListenerLoop", make_loop)
    ui = web.DesktopWebUI("owner", "hey ciel")
    monkeypatch.setattr(web, "DesktopWebUI", lambda *_args, **_kwargs: ui)

    def start(submit, **_kwargs):
        for text in ["hello there", "/mute", "/stop", "/exit"]:
            assert submit(text)
        return "http://127.0.0.1:8765"

    ui.start = start
    ui.close = MagicMock()
    monkeypatch.setattr(sys, "argv", ["ciel", "web", "--no-open-browser"])
    assert __main__.main() == 0
    assert [entry["text"] for entry in ui.snapshot()["messages"]] == ["hello there", "Hi there."]
    assert ui.snapshot()["pc_capture"] is False
    tts.speak.assert_not_called()
    assert ui.snapshot()["provider"] == "test / test"
    loop.stop.assert_called_once()
    tts.close.assert_called_once()
    ui.close.assert_called_once()


def test_listen_archives_legacy_style_but_keeps_saved_facts(tmp_path, monkeypatch):
    """Exercise the provider request after switching away from the legacy persona."""
    from ciel import __main__, audio, config, platform, providers
    from ciel.memory import ConversationManager, MemoryItem, MemoryStore, MemoryType
    from ciel.persona import PERSONA_CONTEXT_VERSION
    from ciel.providers.base import LLMResponse

    settings = config.Settings(
        _env_file=None, memory_db_path=str(tmp_path / "memory.db"), tts_streaming=False,
        ambient_listening=False,
    )
    store = MemoryStore(settings.memory.db_path)
    legacy = ConversationManager(store=store, session_id="desktop_session")
    legacy_answer = "I follow protocol. Speak clearly. I am waiting."
    legacy.add_turn(role="user", content="What do you do?")
    legacy.add_turn(role="assistant", content=legacy_answer)
    fact = "You prefer patient explanations."
    store.save_memory(MemoryItem(content=fact, memory_type=MemoryType.FACT))
    store.save_memory(
        MemoryItem(
            content="You requested direct, cold replies. " + legacy_answer,
            memory_type=MemoryType.CONVERSATION,
            source="auto_summarizer",
            metadata={"session_id": "desktop_session", "last_turn_id": 2},
        )
    )
    store.close()

    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(platform, "get_audio_backend", MagicMock())
    router = MagicMock()
    router.send.return_value = LLMResponse("I'm here to help.", "test", "test")
    monkeypatch.setattr(providers, "get_model_router", lambda: router)
    for component in ("WakeWordDetector", "SpeechToText", "VoiceRecorder", "TextToSpeech"):
        monkeypatch.setattr(audio, component, MagicMock())

    def listener(**kwargs):
        loop = SimpleNamespace(is_running=True, stop=MagicMock())
        loop.start = lambda: kwargs["on_transcription"](
            "What do you remember about patient explanations?",
            {},
            None,
        )
        return loop

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(audio, "WakeListenerLoop", listener)
    monkeypatch.setattr(__main__, "time", SimpleNamespace(
        sleep=interrupt, monotonic=__main__.time.monotonic,
    ))
    monkeypatch.setattr(sys, "argv", ["ciel", "--listen"])

    assert __main__.main() == 0
    messages = router.send.call_args.args[0]
    assert [message.role for message in messages if message.role != "system"] == ["user"]
    assert router.send.call_count == 1
    assert router.send.call_args.kwargs["purpose"] == "conversation"
    assert fact in messages[0].content
    assert legacy_answer not in messages[0].content
    assert messages[1].content == "What do you remember about patient explanations?"
    restored = MemoryStore(settings.memory.db_path)
    try:
        assert len(restored.get_recent_turns("desktop_session")) == 2
        current = restored.get_setting(f"active_desktop_conversation:{PERSONA_CONTEXT_VERSION}")
        assert current.startswith(f"desktop_session:{PERSONA_CONTEXT_VERSION}:run:")
        assert len(restored.get_recent_turns(current)) == 2
        assert restored.get_session_summary("desktop_session") is not None
    finally:
        restored.close()


def test_training_imports_are_optional_and_missing_extras_are_actionable(tmp_path):
    script = """
import importlib.abc, sys
class NoTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"onnx", "sklearn"}:
            raise ImportError("training extra not installed")
sys.meta_path.insert(0, NoTraining())
from ciel.audio import VoiceRecorder
from ciel.audio.trainer import train_custom_wakeword
assert "sounddevice" not in sys.modules
try:
    train_custom_wakeword()
except RuntimeError as error:
    assert 'pip install -e ".[train]"' in str(error)
else:
    raise AssertionError("Missing dependency was not reported")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": SOURCE},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("engine", ["pocket", "chatterbox_turbo", "piper"])
def test_tts_prepares_once_before_normal_listening(tmp_path, monkeypatch, engine):
    from ciel import __main__, audio, config, platform, providers

    settings = config.Settings(
        _env_file=None, memory_db_path=str(tmp_path / "memory.db"), tts_engine=engine,
    )
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(platform, "get_audio_backend", MagicMock())
    monkeypatch.setattr(providers, "get_model_router", MagicMock())
    for component in ("WakeWordDetector", "VoiceRecorder"):
        monkeypatch.setattr(audio, component, MagicMock())
    stt = MagicMock()
    stt.wait_ready.return_value = True
    monkeypatch.setattr(audio, "SpeechToText", MagicMock(return_value=stt))
    events = []
    speech = MagicMock(engine=engine, enabled=True)
    speech.warmup.side_effect = lambda: events.append("warmup") or True
    monkeypatch.setattr(audio, "TextToSpeech", MagicMock(return_value=speech))
    loop = SimpleNamespace(
        is_running=True, start=lambda: events.append("listen"), stop=lambda: None,
    )
    monkeypatch.setattr(audio, "WakeListenerLoop", MagicMock(return_value=loop))

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(__main__, "time", SimpleNamespace(
        sleep=interrupt, monotonic=__main__.time.monotonic,
    ))
    monkeypatch.setattr(sys, "argv", ["ciel", "--listen"])
    assert __main__.main() == 0
    assert events == (["listen"] if engine == "piper" else ["warmup", "listen"])
    speech.close.assert_called_once()
