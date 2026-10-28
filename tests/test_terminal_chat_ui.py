"""Chat presentation stays quiet, safe, and usable while voice is streaming."""

import io
import os
import re
import unicodedata
from unittest.mock import MagicMock

import pytest

from ciel.terminal import TerminalInput
from ciel.terminal_output import TerminalOutput
from ciel.terminal_ui import ConsoleUI


class TTYBuffer(io.StringIO):
    """Capture a terminal without requiring a desktop or audio device."""

    encoding = "utf-8"

    def isatty(self):
        return True


@pytest.fixture
def terminal(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("NO_COLOR", "1")
    return TTYBuffer()


def make_input(**extra):
    callbacks = {name: MagicMock() for name in
                 ("submit", "stop_reply", "toggle_mute", "exit_app", "write")}
    return TerminalInput(**callbacks, **extra), callbacks


def test_reasoning_commands_validate_before_calling_runtime():
    reasoning = MagicMock(return_value="Reasoning: auto")
    keyboard, callbacks = make_input(reasoning=reasoning)
    for command in ("/reasoning", "/REASONING ON", "/reasoning off", "/reasoning auto"):
        keyboard.handle_line(command)
    assert [call.args[0] for call in reasoning.call_args_list] == [None, "on", "off", "auto"]
    reasoning.reset_mock()
    for command in ("/reasoning yes", "/reasoning on off", "/reasoning /fast prompt"):
        keyboard.handle_line(command)
    reasoning.assert_not_called()
    callbacks["submit"].assert_not_called()
    assert callbacks["write"].call_args.args == ("Usage: /reasoning [auto|on|off]",)


def test_reasoning_help_and_unavailable_session_are_local():
    keyboard, callbacks = make_input()
    keyboard.handle_line("/help")
    assert "/reasoning [auto|on|off]" in callbacks["write"].call_args.args[0]
    keyboard.handle_line("/help reasoning")
    assert "session" in callbacks["write"].call_args.args[0]
    keyboard.handle_line("/reasoning")
    assert "unavailable" in callbacks["write"].call_args.args[0]
    callbacks["submit"].assert_not_called()


@pytest.mark.parametrize("tty,term", [(False, "xterm"), (True, "dumb")])
def test_plain_stream_has_no_redraw_sequences_or_composer(monkeypatch, tty, term):
    monkeypatch.setenv("TERM", term)
    buffer = TTYBuffer() if tty else io.StringIO()
    output = TerminalOutput(buffer, chat=True)
    ui = ConsoleUI(output)
    ui.chat_header("120B · complex 550B", status="Reasoning auto")
    output.set_input("owner", "draft")
    output.write("CIEL: Hello\x1b[2J world.\n")
    output.hide_input()
    rendered = buffer.getvalue()
    assert "120B" in rendered and "550B" in rendered and "Hello world." in rendered
    assert "\x1b" not in rendered and "draft" not in rendered
    assert "/help" not in rendered


def test_chat_composer_does_not_replace_or_truncate_streamed_speech(terminal):
    output = TerminalOutput(terminal, width=24, chat=True)
    output.set_input("owner", "an editable draft")
    output.write("CIEL: Hello")
    output.set_input("owner", "an editable draft!")
    output.write(" world.\n")
    output.submit_input("an editable draft!")
    assert "CIEL: Hello world.\n" in terminal.getvalue()
    assert "❯ an editable draft!\n" in terminal.getvalue()
    assert output._draft == ""
    assert output._pending == ""
    output.hide_input()
    assert output._live_rows == 0


def test_untrusted_output_cannot_move_cursor_change_title_or_hide_text(terminal):
    output = TerminalOutput(terminal, chat=True)
    output.write("safe\x1b[2J\x1b]0;forged\x07\x1b[8mtext\x1b[0m\u202e\x9b31m")
    assert "\x1b" not in terminal.getvalue()
    assert "forged" not in terminal.getvalue()
    assert "\u202e" not in terminal.getvalue()
    assert "safetext" in terminal.getvalue()


def test_narrow_header_and_multiline_messages_keep_paragraphs(monkeypatch):
    monkeypatch.setattr("shutil.get_terminal_size", lambda **_: os.terminal_size((20, 24)))
    buffer = io.StringIO()
    ui = ConsoleUI(buffer)
    ui.chat_header("120B / 550B", version="0.3.6", status="Reasoning auto")
    ui.message("assistant", "First paragraph with 音频.\n\nSecond paragraph.")
    rendered = buffer.getvalue()
    assert "\n\n  Second paragraph.\n" in rendered
    assert "CIEL" in rendered and "120B / 550B" in rendered
    for line in rendered.splitlines():
        cells = sum(0 if unicodedata.combining(char) else
                    2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
                    for char in line)
        assert cells < 20


def test_fenced_python_keeps_copyable_indentation_blank_lines_and_tabs():
    buffer = io.StringIO()
    ui = ConsoleUI(buffer)
    ui.message("assistant", "Try this:\n\n```python\n"
               "def greet(names):\n\tfor name in names:\n\t\tprint(name)\n\n"
               "greet(['CIEL'])\n```\n\nIt keeps each greeting on its own line.")
    rendered = buffer.getvalue()
    code = rendered.split("```python\n", 1)[1].split("\n```", 1)[0]
    assert code == ("def greet(names):\n    for name in names:\n        print(name)\n\n"
                    "greet(['CIEL'])")
    compile(code, "<terminal-example>", "exec")
    assert "  Try this:" in rendered
    assert "  It keeps each greeting on its own line." in rendered


def test_fenced_code_stays_safe_and_narrow_without_collapsing_spaces(monkeypatch):
    monkeypatch.setattr("shutil.get_terminal_size", lambda **_: os.terminal_size((20, 24)))
    buffer = io.StringIO()
    ConsoleUI(buffer).message("assistant", "~~~python\n    x  =  'ok'\x1b[2J\n\n"
                              "        print(x)\u202e\n~~~")
    rendered = buffer.getvalue()
    assert "\n    x  =  'ok'\n\n        print(x)\n" in rendered
    assert "\x1b" not in rendered and "\u202e" not in rendered
    assert all(len(line) < 20 for line in rendered.splitlines())


def test_long_code_lines_keep_source_newlines_for_copying(monkeypatch):
    monkeypatch.setattr("shutil.get_terminal_size", lambda **_: os.terminal_size((20, 24)))
    buffer = io.StringIO()
    code = "def greet():\n    print('A long message that should remain a valid string literal.')"
    ConsoleUI(buffer).message("assistant", f"```python\n{code}\n```")
    copied = buffer.getvalue().split("```python\n", 1)[1].split("\n```", 1)[0]
    assert copied == code
    compile(copied, "<terminal-example>", "exec")


def test_streamed_code_expands_tabs_instead_of_dropping_indentation():
    buffer = io.StringIO()
    output = TerminalOutput(buffer, chat=True)
    output.write("def greet():\n\tif True:\n\t\tprint('CIEL')\n")
    assert buffer.getvalue() == "def greet():\n    if True:\n        print('CIEL')\n"


def test_resize_and_long_unicode_draft_preserve_complete_submission(terminal):
    output = TerminalOutput(terminal, width=80, chat=True)
    draft = "Explain 音频 and cafe\u0301 in plain language."
    output.set_input("owner", draft)
    output.write("CIEL: A reply.")
    output.width = 20
    output.set_input("owner", draft)
    assert output._draft == draft
    output.submit_input(draft)
    plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", terminal.getvalue())
    assert "Explain 音频" in plain and "plain language." in plain
    assert output._pending == "CIEL: A reply."


def test_long_live_caption_releases_old_rows_into_scrollback(terminal, monkeypatch):
    monkeypatch.setattr("shutil.get_terminal_size", lambda **_: os.terminal_size((20, 12)))
    output = TerminalOutput(terminal, width=20, chat=True)
    output.set_input("owner", "draft")
    output.write("CIEL: " + "a" * 500)
    assert output._live_rows <= 10
    assert len(output._pending) < 500
    assert output._draft == "draft"
    output.write(" finished.\n")
    assert "finished." in terminal.getvalue()


def test_ascii_terminal_uses_portable_composer(monkeypatch, terminal):
    terminal.encoding = "ascii"
    output = TerminalOutput(terminal, width=20, chat=True)
    output.set_input("owner", "café 音频")
    output.write("CIEL: Très bien.\n")
    assert terminal.getvalue().isascii()
    assert "> cafe ??" in terminal.getvalue()
    assert "Tres bien." in terminal.getvalue()


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="POSIX terminal required")
def test_real_keyboard_remains_editable_during_live_chat_and_restores_mode(terminal):
    import termios
    import threading
    import time

    master, slave = os.openpty()
    original = termios.tcgetattr(slave)
    output = TerminalOutput(terminal, width=40, chat=True)
    submitted = threading.Event()
    received = []

    def submit(text):
        received.append(text)
        submitted.set()
        return True

    def wait_for(predicate):
        deadline = time.monotonic() + 2
        while not predicate():
            assert time.monotonic() < deadline
            time.sleep(0.005)

    with os.fdopen(os.dup(slave), encoding="utf-8") as stream:
        keyboard, callbacks = make_input(stream=stream, output=output)
        keyboard.submit = submit
        keyboard.start()
        try:
            wait_for(lambda: not termios.tcgetattr(slave)[3] & termios.ECHO)
            wait_for(lambda: output._input_active)
            output.write("CIEL: A streaming")
            os.write(master, "音频 draft".encode())
            wait_for(lambda: output._draft == "音频 draft")
            output.width = 20
            output.write(" reply.")
            os.write(master, b"\x7f!\n/stop\n")
            assert submitted.wait(timeout=2)
            wait_for(lambda: callbacks["stop_reply"].call_count == 1)
            assert received == ["音频 draf!"]
            assert output._pending == "CIEL: A streaming reply."
            os.write(master, b"/exit\n")
            keyboard._thread.join(timeout=2)
            assert not keyboard._thread.is_alive()
            assert termios.tcgetattr(slave) == original
        finally:
            keyboard.close()
            os.close(master)
            os.close(slave)
