"""Terminal presentation stays readable and restores interactive input state."""

import io
import os
import unicodedata
from unittest.mock import MagicMock

import pytest

from ciel.terminal_ui import ConsoleUI


class TTYBuffer(io.StringIO):
    def __init__(self, text="", encoding="utf-8"):
        super().__init__(text)
        self._encoding = encoding
        self.writes = []

    @property
    def encoding(self):
        return self._encoding

    def isatty(self):
        return True

    def fileno(self):
        return 42

    def write(self, text):
        self.writes.append(text)
        return super().write(text)


@pytest.mark.parametrize("no_color", [False, True])
def test_clear_supported_terminal_once_and_flushes_without_erasing_scrollback(
    monkeypatch, no_color,
):
    monkeypatch.setenv("TERM", "xterm-256color")
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    output = TTYBuffer()
    output.flush = MagicMock()

    ConsoleUI(stream=output).clear()

    assert output.writes == ["\x1b[2J\x1b[H"]
    output.flush.assert_called_once()


@pytest.mark.parametrize("term", ["xterm-256color", "dumb", "unknown"])
def test_clear_redirected_output_emits_no_terminal_controls(monkeypatch, term):
    monkeypatch.setenv("TERM", term)
    output = io.StringIO()
    output.write("Existing log\n")

    ConsoleUI(stream=output).clear()

    assert output.getvalue() == "Existing log\n"


@pytest.mark.parametrize("term", ["dumb", "unknown"])
def test_clear_unsupported_terminal_preserves_output(monkeypatch, term):
    monkeypatch.setenv("TERM", term)
    output = TTYBuffer()

    ConsoleUI(stream=output).clear()

    assert output.writes == []


@pytest.fixture
def interactive(monkeypatch):
    import select
    import termios
    import tty

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    saved = ["original terminal state"]
    monkeypatch.setattr(termios, "tcgetattr", lambda _fd: saved)
    restore = MagicMock()
    monkeypatch.setattr(termios, "tcsetattr", restore)
    monkeypatch.setattr(tty, "setcbreak", MagicMock())
    monkeypatch.setattr(select, "select", lambda *_args: ([42], [], []))
    return saved, restore


@pytest.mark.parametrize("keys,default,expected", [
    ("\x1b[B\r", "a", "b"),
    ("\x1b[A\r", "a", "c"),
    ("\x1b[B\x1b[B\r", "b", "a"),
    ("\r", "b", "b"),
])
def test_arrow_choices_return_values_and_restore_terminal(interactive, keys, default, expected):
    saved, restore = interactive
    output = TTYBuffer()
    ui = ConsoleUI(stream=output, input_stream=TTYBuffer(keys))

    assert ui.choose("Voice", [("a", "Alpha"), ("b", "Beta"), ("c", "Gamma")], default) == expected

    restore.assert_called_once()
    assert restore.call_args.args[0] == 42 and restore.call_args.args[2] is saved
    assert "\x1b[?25h" in output.getvalue()


@pytest.mark.parametrize("keys,error", [("\x03", KeyboardInterrupt), ("", EOFError)])
def test_interrupted_choice_always_restores_terminal(interactive, keys, error):
    saved, restore = interactive
    output = TTYBuffer()
    ui = ConsoleUI(stream=output, input_stream=TTYBuffer(keys))

    with pytest.raises(error):
        ui.choose("Voice", [("pocket", "Pocket")])

    assert restore.call_args.args[2] is saved
    assert output.getvalue().endswith("\x1b[?25h")


def test_numbered_fallback_validates_input_and_honors_default():
    output = io.StringIO()
    ui = ConsoleUI(stream=output, input_stream=io.StringIO("wrong\n8\n\n"))
    assert ui.choose("Voice", [("pocket", "Pocket"), ("turbo", "Turbo")], "turbo") == "turbo"
    assert "1" in output.getvalue() and "2" in output.getvalue()
    assert "\x1b" not in output.getvalue()


def test_numbered_input_returns_semantic_value():
    ui = ConsoleUI(stream=io.StringIO(), input_stream=io.StringIO("2\n"))
    assert ui.choose("Voice", [("pocket", "Pocket"), ("turbo", "Turbo")]) == "turbo"


def test_dumb_terminal_uses_plain_numbered_choice(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    output = TTYBuffer()
    ui = ConsoleUI(stream=output, input_stream=TTYBuffer("1\n"))
    assert ui.choose("Voice", [("pocket", "Pocket")]) == "pocket"
    assert "\x1b" not in output.getvalue()


def test_no_color_disables_header_and_panel_colors(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("NO_COLOR", "")
    output = TTYBuffer()
    ui = ConsoleUI(stream=output)
    ui.banner("Desktop companion", version="0.3.6")
    ui.panel([("Voice", "Pocket")])
    assert "\x1b" not in output.getvalue()
    assert "0.3.6" in output.getvalue() and "Pocket" in output.getvalue()


def test_ascii_encoding_covers_decorations_and_labels(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    output = TTYBuffer(encoding="ascii")
    wrapper = type("OutputWrapper", (), {
        "stream": output, "write": output.write, "flush": output.flush,
        "isatty": output.isatty, "fileno": output.fileno,
    })()
    ui = ConsoleUI(stream=wrapper)
    ui.banner("Café · 音频", version="0.3.6")
    ui.step("Microphone ✦")
    ui.note("USB 音频")
    ui.summary([("Audio", "Turbo ✓")])
    ui.finish("Ready ✓")
    assert output.getvalue().isascii()
    assert "Cafe" in output.getvalue()


def test_narrow_panel_wraps_using_display_cells(monkeypatch):
    monkeypatch.setattr("shutil.get_terminal_size", lambda **_: os.terminal_size((18, 24)))
    output = io.StringIO()
    ui = ConsoleUI(stream=output)
    ui.panel([("麦克风", "USB 音频 interface with a long label"), ("Mode", "Ambient")])
    ui.note("Long status with 音频 and e\u0301 characters")
    for line in output.getvalue().splitlines():
        cells = sum(0 if unicodedata.combining(char) else
                    2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
                    for char in line)
        assert cells <= 18, repr(line)
    assert "Ambient" in output.getvalue()


def test_labels_cannot_inject_terminal_controls():
    output = io.StringIO()
    ui = ConsoleUI(stream=output)
    ui.panel([("INPUT\x1b[2J\nFORGED", "USB\x1b]0;pwn\x07Mic\r\n\x1b[31mRED")])
    ui.note("Safe\x00\x08\u202eevil")
    rendered = output.getvalue()
    assert "\x1b" not in rendered and "\x07" not in rendered and "\u202e" not in rendered
    assert "pwn" not in rendered
    assert "USBMic" in rendered


def test_panel_is_one_atomic_write(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    output = TTYBuffer()
    ConsoleUI(stream=output).panel([("Voice", "Turbo"), ("Input", "USB microphone")])
    assert len(output.writes) == 1
    assert "Turbo" in output.writes[0] and "USB microphone" in output.writes[0]


def test_prompt_returns_input_and_raises_on_eof():
    output = io.StringIO()
    ui = ConsoleUI(stream=output, input_stream=io.StringIO("custom value\n"))
    assert ui.prompt("Microphone") == "custom value"
    with pytest.raises(EOFError):
        ui.prompt("Microphone")


@pytest.mark.parametrize("choices,default", [([], None), ([("a", "Alpha")], "missing")])
def test_invalid_choice_configuration_fails_clearly(choices, default):
    with pytest.raises(ValueError):
        ConsoleUI(stream=io.StringIO()).choose("Voice", choices, default)


def test_empty_string_default_is_honored_in_arrow_mode(interactive):
    ui = ConsoleUI(stream=TTYBuffer(), input_stream=TTYBuffer("\r"))
    assert ui.choose("Input", [("usb", "USB"), ("", "System default")], "") == ""


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="POSIX terminal required")
def test_real_pty_arrow_sequence_does_not_stall_in_text_buffer():
    import select
    import subprocess
    import sys
    import termios
    import time
    from pathlib import Path

    master, slave = os.openpty()
    saved = termios.tcgetattr(slave)
    source = Path(__file__).resolve().parents[1] / "src"
    script = (
        "from ciel.terminal_ui import ConsoleUI; "
        "value = ConsoleUI().choose('Voice', [('a', 'Alpha'), ('b', 'Beta')], 'a'); "
        "print('PICKED=' + value)"
    )
    process = subprocess.Popen(
        [sys.executable, "-S", "-u", "-c", script], stdin=slave, stdout=slave, stderr=slave,
        env={"PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1", "TERM": "xterm"},
    )
    output = bytearray()
    try:
        deadline = time.monotonic() + 3
        while b"Enter to select" not in output and time.monotonic() < deadline:
            if select.select([master], [], [], 0.05)[0]:
                output.extend(os.read(master, 4096))
        assert b"Enter to select" in output
        os.write(master, b"\x1b[B\r")
        assert process.wait(timeout=3) == 0
        while select.select([master], [], [], 0.05)[0]:
            output.extend(os.read(master, 4096))
        assert b"PICKED=b" in output
        assert termios.tcgetattr(slave) == saved
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        os.close(master)
        os.close(slave)


def test_non_tty_prompt_separates_following_output_from_input_row():
    output = io.StringIO()
    ui = ConsoleUI(stream=output, input_stream=io.StringIO("1\n"))

    assert ui.prompt("Choose 1-2 [1]") == "1"
    ui.note("Next step")

    assert "› \n" in output.getvalue() or "> \n" in output.getvalue()
    assert output.getvalue().splitlines()[-1].endswith("Next step")
