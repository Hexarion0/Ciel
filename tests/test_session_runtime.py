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
        self.microphone_muted = False
        self.is_running = True
        self.state = "listening_wake"
        self.mute_transitions = []
        self.start = MagicMock()
        self.stop = MagicMock()
        self.submit_text = MagicMock(return_value=True)
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
        is_ready=lambda: True, model_size="small.en", device="cpu",
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
            loop=loops[0], feedback=feedback, states=states, ui=ui,
        )

    return launch


@pytest.mark.parametrize("surface", ["terminal", "web"])
def test_session_controls_share_live_state_and_preserve_saved_data(launch_session, surface):
    commands = [
        "/status", "/mute on", "/mute on", "/status", "/unmute", "/mute off",
        "/mode", "/mode wake", "/mode", "/status", "/mode ambient",
        "/memory", "/memory pause", "/memory status", "/memory resume",
        "/help memory", "/quit",
    ]
    result = launch_session(surface, commands)
    assert [state["muted"] for state in result.states[:6]] == [
        False, True, True, True, False, False,
    ]
    assert result.loop.mute_transitions == [True, False]
    assert result.states[7]["ambient"] is False
    assert result.states[8]["ambient"] is False
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
    assert "unavailable" not in output
    assert "could not run" not in output
    assert "microphone" in output
    assert "paused" in output
    assert "automatic" in output
    assert "/memory" in output


@pytest.mark.parametrize("surface", ["terminal", "web"])
def test_memory_pause_persists_but_microphone_and_mode_reset(launch_session, surface):
    first = launch_session(surface, ["/mute on", "/mode wake", "/memory pause", "/exit"])
    assert first.states[-1]["muted"] is True
    assert first.states[-1]["ambient"] is False
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
@pytest.mark.parametrize("surface", ["terminal", "web"])
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
@pytest.mark.parametrize("surface", ["terminal", "web"])
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
