"""Slash selection and editing work through a real keyboard without changing terminal mode."""

import io
import os
import termios
import time
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from ciel.terminal import TerminalInput, command_suggestions
from ciel.terminal_output import TerminalOutput


class TerminalBuffer(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return True


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline, "Keyboard did not reach the expected state"
        time.sleep(0.005)


@contextmanager
def keyboard(monkeypatch, *, width=80):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr("shutil.get_terminal_size", lambda **_: os.terminal_size((width, 24)))
    master, slave = os.openpty()
    saved = termios.tcgetattr(slave)
    output = TerminalOutput(TerminalBuffer(), width=width, chat=True)
    submitted, messages = [], []
    reasoning = MagicMock(return_value="Reasoning changed")
    with os.fdopen(os.dup(slave), encoding="utf-8") as stream:
        control = TerminalInput(
            submit=lambda text: submitted.append(text) or True, stop_reply=MagicMock(),
            toggle_mute=MagicMock(), exit_app=MagicMock(), write=messages.append,
            status=lambda: "Current session", reasoning=reasoning, stream=stream, output=output,
        )
        control.start()
        try:
            wait_until(lambda: not termios.tcgetattr(slave)[3] & termios.ECHO)
            yield master, control, output, submitted, messages, reasoning
            os.write(master, b"\x15/exit\n")
            wait_until(lambda: not control._thread.is_alive())
            assert termios.tcgetattr(slave) == saved
        finally:
            control.close()
            os.close(master)
            os.close(slave)


@pytest.mark.parametrize("keys", [b"/sta\t\n", b"/\x1b[B\r\r"])
def test_picker_completes_then_runs_local_command(monkeypatch, keys):
    with keyboard(monkeypatch) as (fd, _control, _output, submitted, messages, _reasoning):
        os.write(fd, keys)
        wait_until(lambda: "Current session" in messages)
        assert not submitted


def test_picker_enter_inserts_without_executing(monkeypatch):
    with keyboard(monkeypatch) as (fd, _control, output, submitted, messages, _reasoning):
        os.write(fd, b"/sta\r")
        wait_until(lambda: output._draft == "/status")
        assert messages == submitted == []


def test_argument_picker_uses_arrows_and_does_not_submit_to_model(monkeypatch):
    with keyboard(monkeypatch) as (fd, _control, _output, submitted, _messages, reasoning):
        os.write(fd, b"/reasoning\t\x1b[B\r\r")
        wait_until(lambda: reasoning.called)
        reasoning.assert_called_once_with("on")
        assert not submitted


def test_escape_closes_picker_without_losing_draft_or_next_character(monkeypatch):
    with keyboard(monkeypatch) as (fd, control, output, _submitted, _messages, _reasoning):
        os.write(fd, b"/sta\x1b")
        wait_until(lambda: control._completion_dismissed == "/sta")
        assert output._draft == "/sta" and not output._suggestions
        os.write(fd, b"t")
        wait_until(lambda: output._draft == "/stat")
        assert output._suggestions[0][0] == "/status"


@pytest.mark.parametrize("keys,expected", [
    (b"helo\x1b[D\x1b[Dl\x1b[F!\n", "hello!"),
    (b"abc\x1b[H\x1b[3~z\x1b[F\n", "zbc"),
    ("a音b".encode() + b"\x1b[D!\n", "a音!b"),
])
def test_arrows_edit_at_cursor_and_submit_complete_draft(monkeypatch, keys, expected):
    with keyboard(monkeypatch, width=20) as (fd, _control, _output, submitted, _m, _r):
        os.write(fd, keys)
        wait_until(lambda: submitted)
        assert submitted == [expected]


def test_history_up_down_preserves_unsent_draft(monkeypatch):
    with keyboard(monkeypatch) as (fd, _control, _output, submitted, _messages, _reasoning):
        os.write(fd, b"one\ntwo\nscratch\x1b[A\x1b[A\x1b[B\x1b[B!\n")
        wait_until(lambda: len(submitted) == 3)
        assert submitted == ["one", "two", "scratch!"]


def test_picker_survives_background_output_and_scrolls_selection(monkeypatch):
    with keyboard(monkeypatch) as (fd, _control, output, _submitted, _messages, _reasoning):
        os.write(fd, b"/" + b"\x1b[B" * 8)
        wait_until(lambda: output._selected_suggestion == 8)
        choices = list(output._suggestions)
        output.write("CIEL: A background reply.\n")
        assert output._draft == "/"
        assert output._suggestions == choices and output._selected_suggestion == 8
        assert "A background reply." in output.stream.getvalue()
        assert output._live_rows <= 11


def test_suggestions_only_match_local_leading_commands():
    assert command_suggestions("/")[0][0] == "/help"
    assert [key for key, _ in command_suggestions("/sta")] == ["/status"]
    assert [key for key, _ in command_suggestions("/MODE w")] == ["/mode wake"]
    assert command_suggestions("text /status") == []
    assert command_suggestions("/fast explain this") == []
    assert command_suggestions("/missing") == []


def test_picker_stays_disabled_for_plain_or_tiny_terminals(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    assert not TerminalOutput(TerminalBuffer(), chat=True).supports_command_picker()
    monkeypatch.setenv("TERM", "xterm-256color")
    assert not TerminalOutput(io.StringIO(), chat=True).supports_command_picker()
    assert not TerminalOutput(TerminalBuffer(), width=8, chat=True).supports_command_picker()
