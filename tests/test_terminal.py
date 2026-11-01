"""Keyboard controls remain available while a reply is running."""

import io
import os
import pty
import termios
import threading
import time
from unittest.mock import MagicMock

import pytest

from ciel.terminal import TerminalInput


def make_terminal(**overrides):
    callbacks = {
        "submit": MagicMock(return_value=True), "stop_reply": MagicMock(),
        "toggle_mute": MagicMock(side_effect=[True, False]),
        "exit_app": MagicMock(), "write": MagicMock(),
    }
    callbacks.update(overrides)
    return TerminalInput(**callbacks), callbacks


def test_commands_are_local_and_mute_is_a_toggle():
    terminal, callbacks = make_terminal()
    for text in ["", "/mute", "/MUTE", "/stop", "/help", "/unknown", "/exit"]:
        terminal.handle_line(text)
    callbacks["submit"].assert_not_called()
    assert callbacks["toggle_mute"].call_count == 2
    callbacks["stop_reply"].assert_called_once()
    callbacks["exit_app"].assert_called_once()
    messages = [call.args[0] for call in callbacks["write"].call_args_list]
    assert messages[0].startswith("Microphone muted")
    assert messages[1] == "Microphone unmuted."
    assert messages[-1].startswith("Exiting CIEL")


def test_text_and_provider_overrides_are_forwarded_intact():
    terminal, callbacks = make_terminal()
    for text in [
        "What is Ciel?", "/fast explain this", "/strong explain this", "/deep explain this",
        "  Hello.  ",
    ]:
        terminal.handle_line(text)
    assert [call.args[0] for call in callbacks["submit"].call_args_list] == [
        "What is Ciel?", "/fast explain this", "/strong explain this", "/deep explain this",
        "Hello.",
    ]


@pytest.mark.parametrize("text", [
    "/local", "/LOCAL", "/local tell me a story", " /LoCaL\thi ",
    "Hello\n/local Tell me a story", "Hello\n  /LOCAL tell me a story",
])
def test_removed_local_command_is_not_submitted(text):
    terminal, callbacks = make_terminal()
    terminal.handle_line(text)
    callbacks["submit"].assert_not_called()
    callbacks["stop_reply"].assert_not_called()
    callbacks["toggle_mute"].assert_not_called()
    callbacks["exit_app"].assert_not_called()
    assert "no longer available" in callbacks["write"].call_args.args[0]


def test_help_does_not_offer_removed_local_command():
    terminal, callbacks = make_terminal()
    terminal.handle_line("/help")
    assert "/local" not in callbacks["write"].call_args.args[0]
    terminal.handle_line("/help local")
    assert callbacks["write"].call_args.args[0].startswith("Unknown command")


def test_observation_controls_validate_arguments_and_never_forward_to_model():
    observe = MagicMock(return_value="Screen observation: on.")
    terminal, callbacks = make_terminal(observe=observe)
    for command in ("/observe", "/observe on", "/observe pause", "/observe off"):
        terminal.handle_line(command)
    assert [call.args[0] for call in observe.call_args_list] == ["status", "on", "pause", "off"]
    terminal.handle_line("/observe always")
    terminal.handle_line("/observe on please")
    assert observe.call_count == 4
    callbacks["submit"].assert_not_called()
    assert callbacks["write"].call_args.args[0].startswith("Usage: /observe")


def test_reader_handles_multiple_lines_and_exit_without_waiting_for_a_reply():
    read_fd, write_fd = os.pipe()
    exited = threading.Event()
    with os.fdopen(read_fd, encoding="utf-8") as stream:
        terminal, callbacks = make_terminal(stream=stream, exit_app=exited.set)
        terminal.start()
        try:
            os.write(write_fd, "héllo\n/stop\n/exit\nignored\n".encode())
            assert exited.wait(timeout=1)
        finally:
            terminal.close()
            os.close(write_fd)
        callbacks["submit"].assert_called_once_with("héllo")
        callbacks["stop_reply"].assert_called_once()
        assert not terminal._thread.is_alive()


def test_reader_eof_leaves_voice_running():
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    with os.fdopen(read_fd, encoding="utf-8") as stream:
        terminal, callbacks = make_terminal(stream=stream)
        terminal.start()
        terminal._thread.join(timeout=1)
        terminal.close()
        callbacks["exit_app"].assert_not_called()
        assert not terminal._thread.is_alive()


def test_interactive_prompt_returns_after_command_and_not_after_exit():
    class TtyOutput(io.StringIO):
        def isatty(self):
            return True

    class TtyInput:
        encoding = "utf-8"

        def __init__(self, descriptor):
            self.descriptor = descriptor

        def fileno(self):
            return self.descriptor

        def isatty(self):
            return True

    read_fd, write_fd = os.pipe()
    output = TtyOutput()
    terminal, _callbacks = make_terminal(
        stream=TtyInput(read_fd), output=output,
        owner_name="hexarion",
        write=lambda text: output.write(f"[system]: {text}\n"),
    )
    terminal.start()
    try:
        os.write(write_fd, b"/mute\n/exit\n")
        terminal._thread.join(timeout=1)
        assert not terminal._thread.is_alive()
        assert output.getvalue() == (
            "[hexarion]: [system]: Microphone muted; typed messages still work.\n"
            "[hexarion]: [system]: Exiting CIEL; shutting down the microphone "
            "and voice worker.\n"
        )
    finally:
        terminal.close()
        os.close(read_fd)
        os.close(write_fd)


def test_typing_during_speech_preserves_draft_and_restores_terminal_mode():
    from ciel.audio.captions import TerminalCaptions
    from ciel.terminal_output import TerminalOutput

    class TtyBuffer(io.StringIO):
        def isatty(self):
            return True

    master, slave = pty.openpty()
    original_settings = termios.tcgetattr(slave)
    buffer = TtyBuffer()
    output = TerminalOutput(buffer, width=80)
    speech = TerminalCaptions(stream=output)
    submitted = threading.Event()
    received = []

    def submit(text):
        received.append(text)
        submitted.set()
        return True

    with os.fdopen(os.dup(slave), encoding="utf-8") as stream:
        terminal, _callbacks = make_terminal(
            stream=stream, output=output, owner_name="hexarion", submit=submit,
            write=lambda text: output.write(f"[system]: {text}\n"),
        )
        terminal.start()
        try:
            deadline = time.monotonic() + 1
            while termios.tcgetattr(slave)[3] & termios.ECHO:
                assert time.monotonic() < deadline
                time.sleep(0.005)
            speech.start("Hello world.")
            speech.update("Hello")
            os.write(master, b"draft")
            deadline = time.monotonic() + 1
            while output._draft != "draft":
                assert time.monotonic() < deadline
                time.sleep(0.005)
            speech.write_log("INFO: background work")
            speech.update("Hello world.", finished=True)
            speech.close()
            assert buffer.getvalue().endswith("CIEL: Hello world.\n[hexarion]: draft")
            os.write(master, b"\x7f!\n")
            assert submitted.wait(timeout=1)
            assert received == ["draf!"]
            os.write(master, b"/exit\n")
            terminal._thread.join(timeout=1)
            assert not terminal._thread.is_alive()
            assert termios.tcgetattr(slave) == original_settings
        finally:
            terminal.close()
            os.close(master)
            os.close(slave)
