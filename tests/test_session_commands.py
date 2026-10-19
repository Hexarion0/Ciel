"""Session controls stay local, validate input, and preserve message routing."""

import os
import threading
from unittest.mock import MagicMock

import pytest

from ciel.terminal import TerminalInput


def make_terminal(**overrides):
    """Build controls with visible local effects and no running voice engines."""
    callbacks = {
        "submit": MagicMock(return_value=True),
        "stop_reply": MagicMock(),
        "toggle_mute": MagicMock(return_value=True),
        "exit_app": MagicMock(),
        "write": MagicMock(),
        "status": MagicMock(return_value="Session ready."),
        "set_mute": MagicMock(side_effect=lambda muted: muted),
        "mode": MagicMock(return_value="Listening mode updated."),
        "memory": MagicMock(return_value="Saved-memory recall updated."),
    }
    callbacks.update(overrides)
    return TerminalInput(**callbacks), callbacks


@pytest.mark.parametrize(
    "command,callback,arguments,message",
    [
        (" /StAtUs ", "status", (), "Session ready."),
        ("/MODE", "mode", (None,), "Listening mode updated."),
        ("/mode AMBIENT", "mode", ("ambient",), "Listening mode updated."),
        ("/mode wake", "mode", ("wake",), "Listening mode updated."),
        ("/memory", "memory", ("status",), "Saved-memory recall updated."),
        ("/memory STATUS", "memory", ("status",), "Saved-memory recall updated."),
        ("/memory PAUSE", "memory", ("pause",), "Saved-memory recall updated."),
        ("/memory resume", "memory", ("resume",), "Saved-memory recall updated."),
    ],
)
def test_session_controls_use_local_callbacks(command, callback, arguments, message):
    terminal, callbacks = make_terminal()
    terminal.handle_line(command)
    callbacks[callback].assert_called_once_with(*arguments)
    callbacks["write"].assert_called_once_with(message)
    callbacks["submit"].assert_not_called()


@pytest.mark.parametrize(
    "command,muted", [("/mute on", True), ("/mute OFF", False), ("/UNMUTE", False)],
)
def test_explicit_mute_requests_never_toggle(command, muted):
    terminal, callbacks = make_terminal()
    terminal.handle_line(command)
    terminal.handle_line(command)
    assert [call.args for call in callbacks["set_mute"].call_args_list] == [(muted,), (muted,)]
    callbacks["toggle_mute"].assert_not_called()
    callbacks["submit"].assert_not_called()
    assert callbacks["write"].call_args.args[0] == (
        "Microphone muted; typed messages still work." if muted else "Microphone unmuted."
    )


@pytest.mark.parametrize(
    "command,usage",
    [
        ("/status now", "/status"),
        ("/mute yes", "/mute [on|off]"),
        ("/mute on please", "/mute [on|off]"),
        ("/unmute off", "/unmute"),
        ("/mode private", "/mode [ambient|wake]"),
        ("/mode ambient wake", "/mode [ambient|wake]"),
        ("/memory clear", "/memory [status|pause|resume]"),
        ("/memory pause resume", "/memory [status|pause|resume]"),
        ("/stop now", "/stop"),
        ("/exit now", "/exit"),
        ("/quit now", "/quit"),
        ("/help mode extra", "/help [command]"),
    ],
)
def test_malformed_controls_only_show_usage(command, usage):
    terminal, callbacks = make_terminal()
    terminal.handle_line(command)
    callbacks["write"].assert_called_once_with(f"Usage: {usage}")
    for name, callback in callbacks.items():
        if name != "write":
            callback.assert_not_called()


@pytest.mark.parametrize("topic", ["status", "MUTE", "/mode", "memory", "quit", "deep"])
def test_help_for_each_command_is_local(topic):
    terminal, callbacks = make_terminal()
    terminal.handle_line(f"/help {topic}")
    assert callbacks["write"].call_args.args[0].startswith("/" + topic.lstrip("/").lower())
    for name, callback in callbacks.items():
        if name != "write":
            callback.assert_not_called()


def test_help_lists_new_controls_and_all_provider_overrides():
    terminal, callbacks = make_terminal()
    terminal.handle_line("/help")
    help_text = callbacks["write"].call_args.args[0]
    for command in ("status", "unmute", "mode", "memory", "quit", "fast", "strong", "deep",
                    "local"):
        assert f"/{command}" in help_text
    terminal.handle_line("/help missing")
    assert callbacks["write"].call_args.args[0] == "Unknown command. Use /help for controls."
    callbacks["submit"].assert_not_called()


@pytest.mark.parametrize(
    "text", ["/FAST Explain This", "/strong  keep  spacing", "/deep hello", "/local Status"],
)
def test_provider_override_text_keeps_its_case_and_spacing(text):
    terminal, callbacks = make_terminal()
    terminal.handle_line(text)
    callbacks["submit"].assert_called_once_with(text)
    callbacks["write"].assert_not_called()


@pytest.mark.parametrize("command", ["/status", "/mute on", "/unmute", "/mode", "/memory"])
def test_optional_controls_report_unavailable_without_sending_a_message(command):
    terminal, callbacks = make_terminal(status=None, set_mute=None, mode=None, memory=None)
    terminal.handle_line(command)
    assert "unavailable in this session" in callbacks["write"].call_args.args[0]
    callbacks["submit"].assert_not_called()
    callbacks["toggle_mute"].assert_not_called()


@pytest.mark.parametrize(
    "command,callback",
    [("/status", "status"), ("/mute", "toggle_mute"), ("/unmute", "set_mute"),
     ("/mode wake", "mode"), ("/memory pause", "memory"), ("/stop", "stop_reply"),
     ("/quit", "exit_app")],
)
def test_callback_errors_stay_local_and_do_not_expose_exception_values(command, callback):
    terminal, callbacks = make_terminal(**{
        callback: MagicMock(side_effect=RuntimeError("secret-token internal/path")),
    })
    terminal.handle_line(command)
    message = callbacks["write"].call_args.args[0]
    assert message == f"Could not run {command.split()[0]}. Try again or use /help."
    assert "secret-token" not in message
    assert "internal/path" not in message
    callbacks["submit"].assert_not_called()
    assert not terminal._closed.is_set()


def test_quit_is_an_exit_alias():
    terminal, callbacks = make_terminal()
    terminal.handle_line("/QUIT")
    callbacks["exit_app"].assert_called_once_with()
    assert terminal._closed.is_set()
    assert callbacks["write"].call_args.args[0].startswith("Exiting CIEL")
    callbacks["submit"].assert_not_called()


def test_reader_continues_after_local_control_fails():
    read_fd, write_fd = os.pipe()
    exited = threading.Event()
    with os.fdopen(read_fd, encoding="utf-8") as stream:
        terminal, callbacks = make_terminal(
            stream=stream, status=MagicMock(side_effect=OSError("private device detail")),
            exit_app=exited.set,
        )
        terminal.start()
        try:
            os.write(write_fd, b"/status\nhello\n/quit\n")
            assert exited.wait(timeout=1)
        finally:
            terminal.close()
            os.close(write_fd)
    callbacks["submit"].assert_called_once_with("hello")
    assert not terminal._thread.is_alive()


def test_local_controls_serialize_without_blocking_message_submission():
    mode_entered = threading.Event()
    release_mode = threading.Event()
    memory_entered = threading.Event()

    def change_mode(_mode):
        mode_entered.set()
        assert release_mode.wait(timeout=2)
        return "Ambient listening is on."

    def change_memory(_operation):
        memory_entered.set()
        return "Saved-memory recall is paused."

    terminal, callbacks = make_terminal(mode=change_mode, memory=change_memory)
    first = threading.Thread(target=terminal.handle_line, args=("/mode ambient",))
    second = threading.Thread(target=terminal.handle_line, args=("/memory pause",))
    first.start()
    try:
        assert mode_entered.wait(timeout=1)
        second.start()
        assert not memory_entered.wait(timeout=0.02)
        terminal.handle_line("hello")
        callbacks["submit"].assert_called_once_with("hello")
    finally:
        release_mode.set()
        first.join(timeout=1)
        if second.ident is not None:
            second.join(timeout=1)
    assert not first.is_alive()
    assert not second.is_alive()
    assert memory_entered.is_set()
