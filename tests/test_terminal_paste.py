"""Pasting into the terminal draft sends one message only when the owner presses Enter."""

import io
import os
import termios
import time
from contextlib import contextmanager
from unittest.mock import MagicMock

from ciel.terminal import TerminalInput
from ciel.terminal_output import TerminalOutput
from ciel.terminal_paste import DISABLE, ENABLE, Pastes, looks_pasted

START, END = b"\x1b[200~", b"\x1b[201~"
ARTICLE = "First paragraph.\n\nSecond paragraph,\r\nwith a Windows line ending.\nLast line."


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
def keyboard(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr("shutil.get_terminal_size", lambda **_: os.terminal_size((80, 24)))
    master, slave = os.openpty()
    saved = termios.tcgetattr(slave)
    buffer = TerminalBuffer()
    output = TerminalOutput(buffer, width=80, chat=True)
    submitted, messages = [], []
    exit_app = MagicMock()
    with os.fdopen(os.dup(slave), encoding="utf-8") as stream:
        control = TerminalInput(
            submit=lambda text: submitted.append(text) or True, stop_reply=MagicMock(),
            toggle_mute=MagicMock(), exit_app=exit_app, write=messages.append,
            stream=stream, output=output,
        )
        control.start()
        try:
            wait_until(lambda: not termios.tcgetattr(slave)[3] & termios.ECHO)
            yield master, output, buffer, submitted, exit_app
            os.write(master, b"\x15/exit\n")
            wait_until(lambda: not control._thread.is_alive())
            assert termios.tcgetattr(slave) == saved
        finally:
            control.close()
            os.close(master)
            os.close(slave)


def test_bracketed_paste_waits_for_enter_and_sends_the_whole_text(monkeypatch):
    with keyboard(monkeypatch) as (master, output, buffer, submitted, _exit):
        os.write(master, START + ARTICLE.encode() + END)
        wait_until(lambda: output._draft.startswith("[Pasted text #1 +5 lines]"))
        time.sleep(0.1)
        assert submitted == []  # The paste's line breaks never submit.

        os.write(master, b" summarize this\r")
        wait_until(lambda: submitted)

        assert submitted == [
            "First paragraph.\n\nSecond paragraph,\nwith a Windows line ending.\n"
            "Last line. summarize this",
        ]
        assert "[Pasted text #1 +5 lines] summarize this" in buffer.getvalue()
        assert ENABLE in buffer.getvalue()
    assert DISABLE in buffer.getvalue()


def test_pasted_text_starting_with_a_slash_is_a_message(monkeypatch):
    with keyboard(monkeypatch) as (master, output, _buffer, submitted, exit_app):
        os.write(master, START + b"/exit\nis a command in the docs" + END + b"\r")
        wait_until(lambda: submitted)

        assert submitted == ["/exit\nis a command in the docs"]
        exit_app.assert_not_called()


def test_backspace_removes_a_whole_placeholder(monkeypatch):
    with keyboard(monkeypatch) as (master, output, _buffer, submitted, _exit):
        os.write(master, b"see: " + START + ARTICLE.encode() + END)
        wait_until(lambda: "[Pasted text #1" in output._draft)
        os.write(master, b"\x7f")
        wait_until(lambda: output._draft == "see: ")

        os.write(master, b"done\r")
        wait_until(lambda: submitted)
        assert submitted == ["see: done"]


def test_short_single_line_paste_is_ordinary_text(monkeypatch):
    with keyboard(monkeypatch) as (master, output, _buffer, submitted, _exit):
        os.write(master, START + b"rtx 5090 price" + END)
        wait_until(lambda: output._draft == "rtx 5090 price")
        os.write(master, b"\r")
        wait_until(lambda: submitted)
        assert submitted == ["rtx 5090 price"]


def test_unmarked_multi_line_paste_is_one_message(monkeypatch):
    with keyboard(monkeypatch) as (master, output, _buffer, submitted, _exit):
        # A terminal without bracketed paste delivers the block in one read.
        os.write(master, b"line one\rline two\rline three")  # Terminals paste CR.
        wait_until(lambda: output._draft.startswith("[Pasted text #1 +3 lines]"))
        time.sleep(0.1)
        os.write(master, b"\r")
        wait_until(lambda: submitted)
        assert submitted == ["line one\nline two\nline three"]


def test_paste_placeholders_expand_and_merge():
    pastes = Pastes()
    draft, cursor = pastes.insert("ask ", 4, "a\nb")
    assert draft == "ask [Pasted text #1 +2 lines]" and cursor == len(draft)

    pastes.continuing(True)  # The same unmarked paste arriving in a second read.
    draft, cursor = pastes.insert(draft, cursor, "\nc")
    assert draft == "ask [Pasted text #1 +3 lines]"
    assert pastes.expand(draft) == "ask a\nb\nc"
    assert pastes.span_at(draft, cursor) == (4, len(draft))

    assert looks_pasted("one\ntwo") and looks_pasted("one\r\ttwo") and not looks_pasted("typed\r")
    assert not looks_pasted("\x1b[A\n") and not looks_pasted("\x7f!\n/stop\n")


def test_short_pasted_slash_command_is_a_message(monkeypatch):
    with keyboard(monkeypatch) as (master, output, _buffer, submitted, exit_app):
        os.write(master, START + b"/exit" + END)
        wait_until(lambda: output._draft == "/exit")
        os.write(master, b"\r")
        wait_until(lambda: submitted)

        assert submitted == ["/exit"]
        exit_app.assert_not_called()


def test_placeholders_expand_in_one_pass():
    pastes = Pastes()
    draft, cursor = pastes.insert("", 0, "REAL\nTWO")
    label = draft
    # The second paste quotes the first placeholder's label literally.
    draft, cursor = pastes.insert(draft, cursor, f"see {label}\nabove")

    assert pastes.expand(draft) == f"REAL\nTWOsee {label}\nabove"


def test_typed_slash_command_after_clear_is_still_a_command():
    pastes = Pastes()
    pastes.insert("", 0, "/stop")
    assert pastes.contains("/stop")
    pastes.clear()
    assert not pastes.contains("/stop")
