"""Exercise shared in-session controls through terminal and web startup wiring."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ciel.config import Settings
from ciel.memory import ConversationManager, MemoryItem, MemoryStore
from ciel.persona import PERSONA_CONTEXT_VERSION


class SessionLoop:
    """Keep listener state real while leaving audio devices and workers out of tests."""

    def __init__(self, *, ambient, **_kwargs):
        self.ambient = ambient
        self.on_transcription = _kwargs["on_transcription"]
        self.tts = _kwargs["tts"]
        self.microphone_muted = False
        self.is_running = True
        self.state = "listening_wake"
        self.mute_transitions = []
        self.start = MagicMock()
        self.start_text_only = MagicMock()
        self.stop = MagicMock()
        self.submit_text = MagicMock(return_value=True)
        self.submit_background = MagicMock(return_value=True)
        self.background_ready = MagicMock(return_value=True)
        self.cancel_response = MagicMock()
        self.set_ambient = MagicMock(side_effect=self._set_ambient)
        self.set_microphone_mute = MagicMock(side_effect=self._set_mute)
        self.toggle_microphone_mute = MagicMock(
            side_effect=lambda: self._set_mute(not self.microphone_muted),
        )

    def _set_ambient(self, enabled):
        self.ambient = enabled

    def _set_mute(self, muted):
        if self.microphone_muted != muted:
            self.microphone_muted = muted
            self.mute_transitions.append(muted)
        return self.microphone_muted


@pytest.fixture
def launch_session(tmp_path, monkeypatch):
    """Launch immediately driven sessions with real settings, memory, and feedback."""
    from ciel import __main__, audio, providers, runtime, terminal, web

    monkeypatch.chdir(tmp_path)
    environment_file = tmp_path / ".env"
    environment_text = "AMBIENT_LISTENING=true\nMEMORY_RECALL_ENABLED=true\n"
    environment_file.write_text(environment_text)
    settings = Settings(
        _env_file=None, memory_db_path=str(tmp_path / "memory.db"),
        ambient_listening=True, show_ai_transcripts=False, tts_engine="piper",
        ciel_persona_file="", memory_learning_mode="automatic",
    )
    session_id = f"desktop_session:{PERSONA_CONTEXT_VERSION}"
    store = MemoryStore(settings.memory.db_path)
    fact = "Your favorite game is CS2."
    store.save_memory(MemoryItem(content=fact))
    ConversationManager(store=store, session_id=session_id).add_turn(
        role="user", content="A conversation worth keeping.",
    )
    store.close()

    router = MagicMock()
    monkeypatch.setattr(providers, "get_model_router", lambda: router)
    for component in ("WakeWordDetector", "VoiceRecorder"):
        monkeypatch.setattr(audio, component, MagicMock())
    recognition = SimpleNamespace(
        is_ready=lambda: True, wait_ready=lambda **_kwargs: True,
        model_size="small.en", device="cpu",
    )
    monkeypatch.setattr(audio, "SpeechToText", lambda **_kwargs: recognition)
    original_terminal = terminal.TerminalInput
    original_web_ui = web.DesktopWebUI

    def inspect_memory():
        current_store = MemoryStore(settings.memory.db_path)
        try:
            return {
                "recall": current_store.get_setting("recall_enabled", True),
                "facts": [item.content for item in current_store.list_memories()],
                "conversation": [
                    turn.content for turn in current_store.get_recent_turns(session_id)
                ],
            }
        finally:
            current_store.close()

    def launch(surface, commands, *, before_commands=None, controller_enabled=True):
        settings.controller_enabled = controller_enabled
        loops = []
        feedback = []
        states = []
        speech = MagicMock(engine="piper", enabled=True, last_engine=None)
        monkeypatch.setattr(audio, "TextToSpeech", lambda **_kwargs: speech)

        def create_loop(**kwargs):
            loop = SessionLoop(**kwargs)
            loops.append(loop)
            return loop

        monkeypatch.setattr(audio, "WakeListenerLoop", create_loop)
        ui = original_web_ui("owner", "hey ciel")

        def drive(submit):
            if before_commands is not None:
                before_commands(submit, loops[0])
            for command in commands:
                assert submit(command) is not False
                states.append({
                    "command": command, "muted": loops[0].microphone_muted,
                    "ambient": loops[0].ambient, "web_muted": ui.snapshot()["muted"],
                    **inspect_memory(),
                })

        class ImmediateTerminal(original_terminal):
            def __init__(self, **kwargs):
                original_write = kwargs["write"]

                def record(message):
                    feedback.append(message)
                    original_write(message)

                kwargs["write"] = record
                super().__init__(**kwargs)

            def start(self):
                drive(self.handle_line)

        monkeypatch.setattr(terminal, "TerminalInput", ImmediateTerminal)
        monkeypatch.setattr(web, "DesktopWebUI", lambda *_args, **_kwargs: ui)

        def start_web(submit, **_kwargs):
            drive(submit)
            return "http://127.0.0.1:8765"

        ui.start = start_web
        ui.close = MagicMock()
        arguments = (["web", "--no-open-browser", "--no-text-input"] if surface == "web"
                     else ["start", "--text-input"])
        clock = SimpleNamespace(
            monotonic=lambda: 1.0,
            sleep=MagicMock(side_effect=AssertionError("Expected an exit command")),
        )
        assert runtime.run_desktop(
            __main__._build_parser().parse_args(arguments), settings, MagicMock(), MagicMock(),
            clock=clock,
        ) == 0
        router.send.assert_not_called()
        speech.close.assert_called_once()
        loops[0].stop.assert_called_once()
        assert environment_file.read_text() == environment_text
        assert settings.audio.ambient_listening is True
        assert settings.memory.recall_enabled is True
        assert settings.memory.learning_mode == "automatic"
        return SimpleNamespace(
            loop=loops[0], feedback=feedback, states=states, ui=ui, settings=settings,
        )

    return launch


@pytest.mark.parametrize("surface", ["terminal", "web"])
def test_observation_controls_are_local_session_only(launch_session, monkeypatch, surface):
    from ciel.providers.vision import VisionClient

    monkeypatch.setattr(VisionClient, "_connection", lambda _self: ("synthetic", {}))
    result = launch_session(surface, [
        "/observe", "/observe on", "/observe pause", "/observe resume", "/observe off", "/exit",
    ])
    messages = result.feedback if surface == "terminal" else [
        event["text"] for event in result.ui.snapshot()["activity"]
    ]
    assert any("Screen observation: off" in message for message in messages)
    assert any("Screen observation: on" in message for message in messages)
    assert any("Screen observation: paused" in message for message in messages)
    result.loop.submit_text.assert_not_called()
    result.loop.submit_background.assert_not_called()
    assert "off" in result.ui.snapshot()["observation"]


@pytest.mark.parametrize("controller_enabled", [True, False])
def test_direct_voice_observation_control_bypasses_model(
    launch_session, monkeypatch, controller_enabled,
):
    from ciel.providers.vision import VisionClient

    monkeypatch.setattr(VisionClient, "_connection", lambda _self: ("synthetic", {}))

    def spoken_controls(_submit, loop):
        for text in ("hey ciel, start watching my screen", "hey ciel, stop watching my screen"):
            assert loop.on_transcription(text, {
                "wake_verified": True, "cancel_event": threading.Event(), "stt_confidence": 1.0,
            }, None)

    result = launch_session("web", ["/exit"], before_commands=spoken_controls,
                            controller_enabled=controller_enabled)
    store = MemoryStore(result.settings.memory.db_path)
    try:
        session = store.get_setting(f"active_desktop_conversation:{PERSONA_CONTEXT_VERSION}")
        turns = store.get_recent_turns(session)
        assert [turn.role for turn in turns] == ["user", "assistant"] * 2
        assert "Screen observation: on" in turns[1].content
        assert "Screen observation: off" in turns[3].content
    finally:
        store.close()


def test_background_hint_archives_only_delivered_text(launch_session, monkeypatch):
    from ciel.providers.vision import VisionClient, synthetic_test_image
    from ciel.screen.capture import Capture, Cursor, HyprlandCaptureBackend, Monitor, ScreenImage
    from ciel.screen.initiative import ScreenInitiative
    from ciel.screen.observation import CodingAnalyzer, CodingObservation

    now, coordinators = [0.0], []
    original_init = ScreenInitiative.__init__

    def create(self, **kwargs):
        original_init(self, **kwargs)
        self.clock = lambda: now[0]
        coordinators.append(self)

    monkeypatch.setattr(ScreenInitiative, "__init__", create)
    monkeypatch.setattr(ScreenInitiative, "_run", lambda _self: None)
    monkeypatch.setattr(VisionClient, "_connection", lambda _self: ("synthetic", {}))
    image = synthetic_test_image().data
    pixels = Capture("synthetic", 100.0, (Monitor("coding", 0, 0, 128, 64),),
                     Cursor(0, 0, 100.0, True),
                     (ScreenImage("coding", image, 128, 64),
                      ScreenImage("desktop", image, 128, 64)))
    monkeypatch.setattr(HyprlandCaptureBackend, "capture", lambda *_args, **_kwargs: pixels)
    hint = "Try passing the name argument."
    monkeypatch.setattr(CodingAnalyzer, "analyze", lambda *_args, **_kwargs: CodingObservation(
        "coding", "Secret visible TypeError that must not enter the archive",
        hint, "fake", "vision",
    ))

    def drive(submit, loop):
        loop.tts.speak.return_value = False
        loop.tts.clean_text_for_speech.side_effect = lambda text: text

        def background(callback):
            callback(threading.Event())
            return True

        loop.submit_background.side_effect = background
        submit("/observe on")
        for at in range(20, 141, 20):
            now[0] = at
            coordinators[0].tick()

    result = launch_session("web", ["/exit"], before_commands=drive)
    assert [(message["role"], message["text"]) for message in result.ui.snapshot()["messages"]] == [
        ("assistant", hint),
    ]
    store = MemoryStore(result.settings.memory.db_path)
    try:
        session = store.get_setting(f"active_desktop_conversation:{PERSONA_CONTEXT_VERSION}")
        turns = store.get_recent_turns(session)
        assert [(turn.role, turn.content) for turn in turns] == [("assistant", hint)]
        assert turns[0].provider == "fake"
        assert turns[0].model == "vision"
    finally:
        store.close()


@pytest.mark.parametrize("surface", ["terminal", "web"])
def test_session_controls_share_live_state_and_preserve_saved_data(launch_session, surface):
    commands = [
        "/status", "/mute on", "/mute on", "/status", "/unmute", "/mute off",
        "/mode", "/mode wake", "/mode", "/status", "/mode ambient",
        "/memory", "/memory pause", "/memory status", "/memory resume",
        "/help memory", "/quit",
    ]
    result = launch_session(surface, commands)
    assert [state["muted"] for state in result.states[:6]] == (
        [False] * 6 if surface == 'web' else [False, True, True, True, False, False]
    )
    assert result.loop.mute_transitions == ([] if surface == 'web' else [True, False])
    assert result.states[7]["ambient"] is (surface == 'web')
    assert result.states[8]["ambient"] is (surface == 'web')
    assert result.states[10]["ambient"] is True
    assert result.states[12]["recall"] is False
    assert result.states[13]["recall"] is False
    assert result.states[14]["recall"] is True
    for state in result.states:
        assert state["facts"] == ["Your favorite game is CS2."]
        assert state["conversation"] == ["A conversation worth keeping."]
        if surface == "web":
            assert state["web_muted"] is state["muted"]
    assert result.ui.snapshot()["provider"] == ""
    result.loop.submit_text.assert_not_called()
    output = "\n".join(result.feedback).casefold()
    if surface == 'terminal':
        assert "unavailable" not in output
    assert "could not run" not in output
    assert "microphone" in output
    assert "paused" in output
    assert "automatic" in output
    assert "/memory" in output


@pytest.mark.parametrize("surface", ["terminal", "web"])
def test_memory_pause_persists_but_microphone_and_mode_reset(launch_session, surface):
    first = launch_session(surface, ["/mute on", "/mode wake", "/memory pause", "/exit"])
    assert first.states[-1]["muted"] is (surface == 'terminal')
    assert first.states[-1]["ambient"] is (surface == 'web')
    assert first.states[-1]["recall"] is False

    second = launch_session(surface, ["/status", "/memory status", "/memory resume", "/exit"])
    assert second.states[0]["muted"] is False
    assert second.states[0]["ambient"] is True
    assert second.states[0]["recall"] is False
    assert second.states[2]["recall"] is True
    assert any("paused" in message.casefold() for message in second.feedback[:2])


@pytest.mark.parametrize("surface", ["terminal", "web"])
def test_provider_overrides_keep_their_message_routes(launch_session, surface):
    messages = [f"/{route} Explain this clearly." for route in ("fast", "strong", "deep")]
    result = launch_session(surface, [*messages, "/exit"])
    assert [call.args[0] for call in result.loop.submit_text.call_args_list] == messages


@pytest.mark.parametrize("surface", ["terminal", "web"])
def test_removed_local_command_never_enters_listener_or_changes_session(launch_session, surface):
    result = launch_session(surface, [
        "/local", "/LOCAL remember my name is Example",
        "Hello\n/local Tell me a story", "Hello\n  /LOCAL remember my name is Example", "/exit",
    ])
    result.loop.submit_text.assert_not_called()
    for state in result.states:
        assert state["facts"] == ["Your favorite game is CS2."]
        assert state["conversation"] == ["A conversation worth keeping."]
    assert sum("no longer available" in message for message in result.feedback) == 4


@pytest.mark.parametrize("surface", ["terminal", "web"])
def test_invalid_local_controls_leave_session_state_unchanged(launch_session, surface):
    result = launch_session(surface, [
        "/mode listen", "/memory delete", "/mute all", "/status secrets", "/quit no", "/exit",
    ])
    result.loop.submit_text.assert_not_called()
    result.loop.set_ambient.assert_not_called()
    result.loop.set_microphone_mute.assert_not_called()
    result.loop.cancel_response.assert_called_once()
    assert all(state["recall"] is True for state in result.states)
    assert sum("Usage:" in message for message in result.feedback) == 5


@pytest.mark.parametrize("controller_enabled", [False, True])
@pytest.mark.parametrize("surface", ["terminal"])
def test_concurrent_voice_and_slash_modes_keep_capture_and_status_consistent(
    launch_session, monkeypatch, controller_enabled, surface,
):
    """A slash control waits for a voice mode update, then becomes the latest state."""
    from ciel import runtime

    voice_paused = threading.Event()
    release_voice = threading.Event()
    slash_settled = threading.Event()
    failures = []

    class ObservedLock:
        """Expose lock contention so the test needs no timing assumptions."""

        def __init__(self):
            self.lock = threading.RLock()

        def __enter__(self):
            if threading.current_thread().name == "slash-mode":
                if not self.lock.acquire(blocking=False):
                    slash_settled.set()
                    self.lock.acquire()
            else:
                self.lock.acquire()
            return self

        def __exit__(self, *_args):
            self.lock.release()

    monkeypatch.setattr(runtime, "threading", SimpleNamespace(
        Event=threading.Event, RLock=ObservedLock,
    ))
    original_set = SessionLoop._set_ambient

    def pause_voice_after_capture_change(loop, enabled):
        original_set(loop, enabled)
        if threading.current_thread().name == "voice-mode":
            voice_paused.set()
            assert release_voice.wait(timeout=2)

    monkeypatch.setattr(SessionLoop, "_set_ambient", pause_voice_after_capture_change)

    def overlap_modes(submit, loop):
        canceled = threading.Event()
        loop.cancel_response.side_effect = canceled.set

        def voice():
            try:
                loop.on_transcription("ambient mode on", {
                    "wake_verified": True, "cancel_event": canceled,
                }, None)
            except BaseException as error:
                failures.append(error)

        def slash():
            try:
                submit("/mode wake")
            except BaseException as error:
                failures.append(error)
            finally:
                slash_settled.set()

        voice_worker = threading.Thread(target=voice, name="voice-mode")
        slash_worker = threading.Thread(target=slash, name="slash-mode")
        voice_worker.start()
        try:
            assert voice_paused.wait(timeout=2)
            slash_worker.start()
            assert slash_settled.wait(timeout=2)
        finally:
            release_voice.set()
            voice_worker.join(timeout=2)
            if slash_worker.ident is not None:
                slash_worker.join(timeout=2)
        assert not voice_worker.is_alive()
        assert not slash_worker.is_alive()
        assert not failures

    result = launch_session(
        surface, ["/status", "/exit"], before_commands=overlap_modes,
        controller_enabled=controller_enabled,
    )
    assert result.loop.ambient is False
    status = next(message for message in result.feedback if "Recognition:" in message)
    assert "Listening mode: wake-word" in status


@pytest.mark.parametrize("controller_enabled", [False, True])
@pytest.mark.parametrize("surface", ["terminal"])
def test_same_state_slash_mode_cancels_a_pending_voice_mode_change(
    launch_session, monkeypatch, controller_enabled, surface,
):
    """An explicit mode choice wins even when it repeats the current setting."""
    from ciel.actions import ActionRegistry

    voice_paused = threading.Event()
    release_voice = threading.Event()
    failures = []
    original_cancel = ActionRegistry.cancel_pending

    def pause_voice_before_mode_change(actions):
        original_cancel(actions)
        if threading.current_thread().name == "voice-mode":
            voice_paused.set()
            assert release_voice.wait(timeout=2)

    monkeypatch.setattr(ActionRegistry, "cancel_pending", pause_voice_before_mode_change)

    def overlap_modes(submit, loop):
        canceled = threading.Event()
        loop.cancel_response.side_effect = canceled.set

        def voice():
            try:
                loop.on_transcription("wake-word mode", {
                    "wake_verified": True, "cancel_event": canceled,
                }, None)
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=voice, name="voice-mode")
        worker.start()
        try:
            assert voice_paused.wait(timeout=2)
            submit("/mode ambient")
        finally:
            release_voice.set()
            worker.join(timeout=2)
        assert not worker.is_alive()
        assert not failures

    result = launch_session(
        surface, ["/status", "/exit"], before_commands=overlap_modes,
        controller_enabled=controller_enabled,
    )
    assert result.loop.ambient is True
    result.loop.set_ambient.assert_not_called()
    status = next(message for message in result.feedback if "Recognition:" in message)
    assert "Listening mode: ambient conversation" in status


@pytest.mark.parametrize("operation", ["pause", "resume"])
@pytest.mark.parametrize("controller_enabled", [False, True])
@pytest.mark.parametrize("surface", ["terminal", "web"])
def test_same_state_memory_control_cancels_a_pending_voice_change(
    launch_session, monkeypatch, operation, controller_enabled, surface,
):
    """A repeated explicit choice invalidates voice work before its memory mutation."""
    from ciel import persona

    voice_paused = threading.Event()
    release_voice = threading.Event()
    canceled = threading.Event()
    failures = []
    original_parse = persona.parse_persona_request
    voice_text = "resume memory recall" if operation == "pause" else "pause memory recall"

    def pause_before_memory_change(text, **kwargs):
        if threading.current_thread().name == "voice-memory" and text == voice_text:
            voice_paused.set()
            assert release_voice.wait(timeout=2)
        return original_parse(text, **kwargs)

    monkeypatch.setattr(persona, "parse_persona_request", pause_before_memory_change)

    def overlap_memory(submit, loop):
        submit(f"/memory {operation}")
        loop.cancel_response.side_effect = canceled.set

        def voice():
            try:
                loop.on_transcription(voice_text, {
                    "wake_verified": True, "cancel_event": canceled,
                }, None)
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=voice, name="voice-memory")
        worker.start()
        try:
            assert voice_paused.wait(timeout=2)
            submit(f"/memory {operation}")
            assert canceled.is_set()
        finally:
            release_voice.set()
            worker.join(timeout=2)
        assert not worker.is_alive()
        assert not failures

    result = launch_session(
        surface, ["/memory status", "/exit"], before_commands=overlap_memory,
        controller_enabled=controller_enabled,
    )
    assert result.states[0]["recall"] is (operation == "resume")


@pytest.mark.parametrize("operation", ["pause", "resume"])
@pytest.mark.parametrize("controller_enabled", [False, True])
@pytest.mark.parametrize("surface", ["terminal", "web"])
def test_memory_control_serializes_after_an_existing_voice_mutation(
    launch_session, monkeypatch, operation, controller_enabled, surface,
):
    """A voice mutation already in progress finishes before the latest explicit choice."""
    from ciel import runtime
    from ciel.memory.service import MemoryService

    voice_paused = threading.Event()
    release_voice = threading.Event()
    slash_settled = threading.Event()
    failures = []
    original_handle = MemoryService.handle

    class ObservedLock:
        """Expose command contention without waiting inside the pending voice write."""

        def __init__(self):
            self.lock = threading.RLock()

        def __enter__(self):
            if threading.current_thread().name == "slash-memory":
                if not self.lock.acquire(blocking=False):
                    slash_settled.set()
                    self.lock.acquire()
            else:
                self.lock.acquire()
            return self

        def __exit__(self, *_args):
            self.lock.release()

    monkeypatch.setattr(runtime, "threading", SimpleNamespace(
        Event=threading.Event, RLock=ObservedLock,
    ))

    def pause_memory_mutation(service, text, *args, **kwargs):
        if threading.current_thread().name == "voice-memory":
            voice_paused.set()
            assert release_voice.wait(timeout=2)
        return original_handle(service, text, *args, **kwargs)

    monkeypatch.setattr(MemoryService, "handle", pause_memory_mutation)

    def overlap_memory(submit, loop):
        submit(f"/memory {operation}")
        canceled = threading.Event()
        loop.cancel_response.side_effect = canceled.set

        def voice():
            try:
                loop.on_transcription(
                    "resume memory recall" if operation == "pause" else "pause memory recall",
                    {"wake_verified": True, "cancel_event": canceled}, None,
                )
            except BaseException as error:
                failures.append(error)

        def slash():
            try:
                submit(f"/memory {operation}")
            except BaseException as error:
                failures.append(error)
            finally:
                slash_settled.set()

        voice_worker = threading.Thread(target=voice, name="voice-memory")
        slash_worker = threading.Thread(target=slash, name="slash-memory")
        voice_worker.start()
        try:
            assert voice_paused.wait(timeout=2)
            slash_worker.start()
            assert slash_settled.wait(timeout=2)
        finally:
            release_voice.set()
            voice_worker.join(timeout=2)
            if slash_worker.ident is not None:
                slash_worker.join(timeout=2)
        assert not voice_worker.is_alive()
        assert not slash_worker.is_alive()
        assert not failures

    result = launch_session(
        surface, ["/memory status", "/exit"], before_commands=overlap_memory,
        controller_enabled=controller_enabled,
    )
    assert result.states[0]["recall"] is (operation == "resume")


@pytest.mark.parametrize('kind', ['text', 'voice'])
def test_browser_reply_never_plays_through_pc_speaker(launch_session, kind):
    from ciel.webapp.requests import WebRequest

    def drive(_submit, loop):
        loop.tts.speak.reset_mock()
        event = threading.Event()
        request = WebRequest('browser-request', 'phone', 'chat', 1, kind, 'What time is it?')
        loop.on_transcription('What time is it?', {
            'input_source': 'web_text' if kind == 'text' else 'web_voice',
            'web_request': request, 'web_current': lambda: True,
            'wake_verified': True, 'cancel_event': event,
        }, None)
        loop.tts.speak.assert_not_called()

    result = launch_session('web', ['/exit'], before_commands=drive)
    replies = [m['text'] for m in result.ui.snapshot()['messages'] if m['role'] == 'assistant']
    assert len(replies) == 1 and replies[0].startswith("It's ")


def test_web_starts_request_worker_without_opening_pc_microphone(launch_session):
    result = launch_session('web', ['/exit'])
    result.loop.start.assert_not_called()
    result.loop.start_text_only.assert_called_once()
    assert result.ui.snapshot()['pc_capture'] is False


@pytest.fixture
def interactive_terminal_displays(monkeypatch):
    """Exercise the real interactive display while keeping keyboard IO simulated."""
    import io
    import sys

    from ciel import terminal_output

    class TTY(io.StringIO):
        encoding = "utf-8"

        def isatty(self):
            return True

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    original = terminal_output.TerminalOutput
    displays = []

    def make_output(_stream, **kwargs):
        display = original(TTY(), **kwargs)
        displays.append(display)
        return display

    monkeypatch.setattr(terminal_output, "TerminalOutput", make_output)
    return displays


@pytest.mark.parametrize("source", ["keyboard", "speech"])
@pytest.mark.parametrize("controller_enabled", [True, False])
def test_terminal_speech_and_typing_share_written_reply_depth(
    launch_session, interactive_terminal_displays, source, controller_enabled,
):
    from ciel import providers
    from ciel.providers.base import LLMStreamChunk

    displays, calls = interactive_terminal_displays, []
    answer = "First explain the design. " * 55 + "\n\n```python\ndef example():\n    return 42\n```"
    query = "Explain the design with an example."

    def stream(messages, **kwargs):
        calls.append((messages, kwargs))
        yield LLMStreamChunk(answer, "fixture-model", "fixture", is_final=True)

    def drive(_submit, loop):
        providers.get_model_router().stream.side_effect = stream
        if source == "keyboard":
            displays[0].submit_input(query)
        loop.tts.speak.return_value = True
        loop.tts.clean_text_for_speech.side_effect = lambda text: text
        info = {"wake_verified": True, "cancel_event": threading.Event()}
        if source == "keyboard":
            info["input_source"] = "keyboard"
        loop.on_transcription(query, info, None)
        audible = loop.tts.speak.call_args.args[0]
        assert len(audible.split()) <= 80 and "```" not in audible

    result = launch_session("terminal", ["/exit"], before_commands=drive,
                            controller_enabled=controller_enabled)
    assert len(calls) == 1
    assert calls[0][1]["max_tokens"] == 4096
    prompt = calls[0][0][0].content
    assert "Generic short-reply or spoken-delivery preferences do not cap written answers" in prompt
    assert ("transcribed from speech" in prompt) == (source == "speech")
    assert displays[0]._transcript.count("❯ " + query) == 1
    user_index = displays[0]._transcript.index("❯ " + query)
    assert displays[0]._transcript[user_index + 1].startswith("CIEL: First explain the design.")
    assert not any("fixture-model" in line for line in displays[0]._transcript)
    assert "def example():" in displays[0]._transcript
    store = MemoryStore(result.settings.memory.db_path)
    try:
        session = store.get_setting(f"active_desktop_conversation:{PERSONA_CONTEXT_VERSION}")
        assert [turn.content for turn in store.get_recent_turns(session)] == [query, answer]
    finally:
        store.close()


@pytest.mark.parametrize("controller_enabled", [True, False])
@pytest.mark.parametrize("outcome", ["unclear", "canceled"])
def test_unaccepted_speech_does_not_become_a_normal_chat_message(
    launch_session, interactive_terminal_displays, controller_enabled, outcome,
):
    from ciel import providers

    query = "An uncertain transcription must not become an instruction."

    def drive(_submit, loop):
        canceled = threading.Event()
        info = {"wake_verified": True, "cancel_event": canceled}
        if outcome == "unclear":
            info.update(stt_needs_repeat=True, stt_confidence=0.1)
        else:
            canceled.set()
        loop.on_transcription(query, info, None)
        providers.get_model_router().stream.assert_not_called()

    result = launch_session("terminal", ["/exit"], before_commands=drive,
                            controller_enabled=controller_enabled)
    assert not any(query in line for line in interactive_terminal_displays[0]._transcript)
    store = MemoryStore(result.settings.memory.db_path)
    try:
        session = store.get_setting(f"active_desktop_conversation:{PERSONA_CONTEXT_VERSION}")
        assert not store.get_recent_turns(session)
    finally:
        store.close()


@pytest.mark.parametrize("kind", ["text", "voice"])
@pytest.mark.parametrize("outcome", ["complete", "error", "cancel"])
def test_browser_displays_answer_before_generation_finishes(
    launch_session, monkeypatch, kind, outcome,
):
    """A slow answer tail must not hide already available browser text."""
    from ciel.agent import AgentResult, MainAgent
    from ciel.providers.base import LLMStreamChunk
    from ciel.web import DesktopWebUI
    from ciel.webapp.requests import WebRequest

    visible = threading.Event()
    canceled = threading.Event()
    original_progress = DesktopWebUI.progress

    def progress(self, text, **kwargs):
        original_progress(self, text, **kwargs)
        if text == "First sentence. ":
            visible.set()

    def stream(self, *_args, **_kwargs):
        self.last_result = AgentResult(provider="fixture", model="fixture")
        yield LLMStreamChunk("First sentence. ", "fixture", "fixture")
        assert visible.wait(2), "Browser text waited for the rest of the answer"
        if outcome == "error":
            raise RuntimeError("Fixture connection failed after visible output")
        if outcome == "cancel":
            canceled.set()
        yield LLMStreamChunk("Second sentence.", "fixture", "fixture", is_final=True)

    def send(*_args, **_kwargs):
        raise AssertionError("Browser answers must use the existing validated stream")

    monkeypatch.setattr(DesktopWebUI, "progress", progress)
    monkeypatch.setattr(MainAgent, "stream", stream)
    monkeypatch.setattr(MainAgent, "send", send)

    def drive(_submit, loop):
        request = WebRequest("request", "phone", "chat", 1, kind, "Tell me a short story.")
        loop.on_transcription(request.text, {
            "input_source": "web_text" if kind == "text" else "web_voice",
            "web_request": request, "web_current": lambda: True,
            "wake_verified": True, "cancel_event": canceled,
        }, None)

    result = launch_session("web", ["/exit"], before_commands=drive)
    replies = [m for m in result.ui.snapshot()["messages"] if m["role"] == "assistant"]
    assert visible.is_set()
    assert len(replies) == 1
    expected = {
        "complete": "First sentence. Second sentence.",
        "error": "First sentence. \n\nMy connection cut out before I finished.",
        "cancel": "First sentence. ",
    }[outcome]
    assert replies[0]["text"] == expected
    assert not replies[0]["live"]
    assert bool(replies[0].get("interrupted")) == (outcome == "cancel")
    store = MemoryStore(result.settings.memory.db_path)
    try:
        session = store.get_setting(f"active_desktop_conversation:{PERSONA_CONTEXT_VERSION}")
        turns = [turn for turn in store.get_recent_turns(session) if turn.role == "assistant"]
        assert len(turns) == 1
        expected_history = (
            expected + "\n[Reply was interrupted.]" if outcome == "cancel" else expected
        )
        assert turns[0].content == expected_history
    finally:
        store.close()


@pytest.mark.parametrize("outcome", ["complete", "error", "cancel"])
def test_browser_stream_never_exposes_split_secret(launch_session, monkeypatch, outcome):
    """Intermediate snapshots and interrupted archives must omit withheld key prefixes."""
    from ciel.agent import AgentResult, MainAgent
    from ciel.logging import SecretMaskingFilter
    from ciel.providers.base import LLMStreamChunk
    from ciel.web import DesktopWebUI
    from ciel.webapp.requests import WebRequest

    secret = "fixture_" + "x" * 24 + "z"
    monkeypatch.setattr(SecretMaskingFilter, "_configured_secrets", ())
    SecretMaskingFilter.register_secrets(secret)
    progressed, canceled = threading.Event(), threading.Event()
    snapshots = []
    original_progress = DesktopWebUI.progress

    def progress(self, text, **kwargs):
        original_progress(self, text, **kwargs)
        snapshots.append([m["text"] for m in self.snapshot()["messages"]])
        progressed.set()

    def stream(self, *_args, **_kwargs):
        self.last_result = AgentResult(provider="fixture", model="fixture")
        yield LLMStreamChunk("Key: " + secret[:-1], "fixture", "fixture")
        assert progressed.wait(2)
        if outcome == "cancel":
            canceled.set()
            return
        if outcome == "error":
            raise RuntimeError("Fixture connection cut out")
        yield LLMStreamChunk(secret[-1], "fixture", "fixture", is_final=True)

    monkeypatch.setattr(DesktopWebUI, "progress", progress)
    monkeypatch.setattr(MainAgent, "stream", stream)

    def drive(_submit, loop):
        request = WebRequest("request", "phone", "chat", 1, "text", "Tell me a short story.")
        loop.on_transcription(request.text, {
            "input_source": "web_text", "web_request": request, "web_current": lambda: True,
            "wake_verified": True, "cancel_event": canceled,
        }, None)

    result = launch_session("web", ["/exit"], before_commands=drive)
    assert progressed.is_set()
    assert all(secret[:8] not in text for snapshot in snapshots for text in snapshot)
    assert all(secret[:8] not in m["text"] for m in result.ui.snapshot()["messages"])
    store = MemoryStore(result.settings.memory.db_path)
    try:
        session = store.get_setting(f"active_desktop_conversation:{PERSONA_CONTEXT_VERSION}")
        if outcome != "complete":
            assert all(secret[:8] not in turn.content for turn in store.get_recent_turns(session))
    finally:
        store.close()


def test_browser_stream_flushes_benign_trailing_digits(launch_session, monkeypatch):
    """A possible token prefix is only withheld until successful completion."""
    from ciel.agent import AgentResult, MainAgent
    from ciel.providers.base import LLMStreamChunk
    from ciel.webapp.requests import WebRequest

    def stream(self, *_args, **_kwargs):
        self.last_result = AgentResult(provider="fixture", model="fixture")
        yield LLMStreamChunk("The answer is 42", "fixture", "fixture", is_final=True)

    monkeypatch.setattr(MainAgent, "stream", stream)

    def drive(_submit, loop):
        request = WebRequest("request", "phone", "chat", 1, "text", "Tell me a short story.")
        loop.on_transcription(request.text, {
            "input_source": "web_text", "web_request": request, "web_current": lambda: True,
            "wake_verified": True, "cancel_event": threading.Event(),
        }, None)

    result = launch_session("web", ["/exit"], before_commands=drive)
    replies = [m["text"] for m in result.ui.snapshot()["messages"] if m["role"] == "assistant"]
    assert replies == ["The answer is 42"]


def test_web_controls_do_not_enable_host_capture_modes(launch_session):
    result = launch_session('web', ['/mode wake','/mode ambient','/mute on','/exit'])
    result.loop.set_ambient.assert_not_called()
    result.loop.set_microphone_mute.assert_not_called()
    result.loop.start.assert_not_called()
    result.loop.start_text_only.assert_called_once()
