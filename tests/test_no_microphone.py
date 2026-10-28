"""A missing microphone must preserve the real typed conversation pipeline."""

import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ciel.platform.base import AudioInputError
from ciel.providers.base import LLMResponse

QUESTION = "Tell me a tiny bedtime story about a fox."
ANSWER = "A fox curled beneath the stars and dreamed of warm summer days."


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    """Use the real listener workers and runtime with isolated memory and fake devices."""
    from ciel import audio, config, platform, providers, terminal_ui
    from ciel.audio.listener import WakeListenerLoop
    from ciel.chat import autostart

    settings = config.Settings(
        _env_file=None, memory_db_path=str(tmp_path / "memory.db"),
        tts_enabled=True, tts_streaming=False, controller_enabled=True,
        show_ai_transcripts=False, ciel_persona_file="",
    )
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(autostart, "reconcile_chat_startup", MagicMock(
        return_value=SimpleNamespace(success=True),
    ))
    backend = MagicMock()
    backend.start_stream.side_effect = AudioInputError("No microphone is connected")
    monkeypatch.setattr(platform, "get_audio_backend", lambda: backend)
    router = MagicMock()
    router.send.return_value = LLMResponse(ANSWER, "test", "test")
    monkeypatch.setattr(providers, "get_model_router", lambda: router)
    for component in ("WakeWordDetector", "SpeechToText", "VoiceRecorder"):
        monkeypatch.setattr(audio, component, MagicMock())
    tts = MagicMock(engine="piper", enabled=True)
    tts.speak.return_value = True
    tts.clean_text_for_speech.side_effect = lambda text: text
    monkeypatch.setattr(audio, "TextToSpeech", lambda **_kwargs: tts)
    console = MagicMock()
    monkeypatch.setattr(terminal_ui, "ConsoleUI", lambda **_kwargs: console)
    finished = Event()
    transcripts, loops = [], []

    def listener(**kwargs):
        callback = kwargs["on_transcription"]

        def transcribe(text, info, recorded_audio):
            try:
                result = callback(text, info, recorded_audio)
                transcripts.append((text, info.copy()))
                return result
            finally:
                finished.set()

        kwargs["on_transcription"] = transcribe
        kwargs["speech_detector"] = lambda _audio: False
        loop = WakeListenerLoop(**kwargs)
        loops.append(loop)
        return loop

    monkeypatch.setattr(audio, "WakeListenerLoop", listener)
    return SimpleNamespace(
        settings=settings, backend=backend, router=router, tts=tts,
        console=console, finished=finished, transcripts=transcripts, loops=loops,
    )


def _assert_cleanup(desktop):
    desktop.backend.start_stream.assert_called_once()
    desktop.tts.close.assert_called_once()
    assert len(desktop.loops) == 1
    assert not desktop.loops[0].is_running
    assert not any(thread.is_alive() for thread in desktop.loops[0]._threads)


@pytest.mark.parametrize("arguments", [[], ["--text-input"]])
def test_microphone_failure_preserves_interactive_keyboard_reply_and_exit(
    desktop, monkeypatch, capsys, arguments,
):
    from ciel import __main__, terminal
    from ciel.memory import MemoryStore
    from ciel.persona import PERSONA_CONTEXT_VERSION

    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    inputs, statuses = [], []

    class TypedSession(terminal.TerminalInput):
        def start(self):
            inputs.append(self)
            assert desktop.loops[0].is_running
            assert desktop.loops[0].microphone_muted
            self.handle_line("/unmute")
            self.handle_line("/mode wake")
            statuses.append(self.status())
            self.handle_line(QUESTION)
            assert desktop.finished.wait(3), "Typed message never reached the reply callback"
            self.handle_line("/exit")

    monkeypatch.setattr(terminal, "TerminalInput", TypedSession)

    assert __main__.main(arguments) == 0

    assert [item[0] for item in desktop.transcripts] == [QUESTION]
    assert desktop.transcripts[0][1]["input_source"] == "keyboard"
    desktop.router.send.assert_called_once()
    assert [message.content for message in desktop.router.send.call_args.args[0]
            if message.role == "user"][-1] == QUESTION
    assert inputs[0]._closed.is_set()
    assert desktop.loops[0].microphone_muted
    assert "unavailable" in statuses[0].casefold() or "text" in statuses[0].casefold()
    desktop.console.panel.assert_called_once()
    readiness = dict(desktop.console.panel.call_args.args[0])
    assert readiness["○ Text chat"] == "Microphone unavailable"
    assert readiness["Recognition"] == "Off · microphone unavailable"
    assert readiness["Reasoning"] == desktop.settings.nim_reasoning
    desktop.console.work_status.assert_not_called()
    assert ANSWER in capsys.readouterr().out
    desktop.tts.speak.assert_called_once()
    assert desktop.tts.speak.call_args.args[0] == ANSWER
    store = MemoryStore(desktop.settings.memory.db_path)
    try:
        session = store.get_setting(f"active_desktop_conversation:{PERSONA_CONTEXT_VERSION}")
        assert [turn.content for turn in store.get_recent_turns(session)] == [QUESTION, ANSWER]
    finally:
        store.close()
    _assert_cleanup(desktop)


def test_web_chat_works_without_microphone_or_terminal_input(desktop, monkeypatch):
    from ciel import __main__, terminal, web

    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    ui = web.DesktopWebUI("owner", "hey ciel", voice_enabled=False)
    monkeypatch.setattr(web, "DesktopWebUI", lambda *_args, **_kwargs: ui)
    terminal_start = MagicMock()
    monkeypatch.setattr(terminal.TerminalInput, "start", terminal_start)

    def start(submit, **_kwargs):
        assert desktop.loops[0].is_running
        assert submit(QUESTION)
        assert desktop.finished.wait(3), "Web message never reached the reply callback"
        assert submit("/exit")
        return "http://127.0.0.1:8765"

    ui.start = start
    ui.close = MagicMock()

    assert __main__.main(["web", "--no-text-input", "--no-open-browser"]) == 0

    assert [entry["text"] for entry in ui.snapshot()["messages"]] == [QUESTION, ANSWER]
    assert ui.snapshot()["muted"]
    assert [item[0] for item in desktop.transcripts] == [QUESTION]
    assert desktop.transcripts[0][1]["input_source"] == "keyboard"
    terminal_start.assert_not_called()
    ui.close.assert_called_once()
    _assert_cleanup(desktop)


@pytest.mark.parametrize("arguments", [[], ["--no-text-input"]])
def test_missing_microphone_without_usable_text_frontend_still_fails(
    desktop, monkeypatch, arguments,
):
    from ciel import __main__, terminal

    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    terminal_start = MagicMock()
    monkeypatch.setattr(terminal.TerminalInput, "start", terminal_start)

    assert __main__.main(arguments) == 1

    terminal_start.assert_not_called()
    desktop.router.send.assert_not_called()
    assert desktop.transcripts == []
    _assert_cleanup(desktop)
