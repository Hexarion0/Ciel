"""Check visible terminal positions and scrollback rather than redraw implementation details."""

import io
import os
import re

import pytest

from ciel.terminal import command_suggestions
from ciel.terminal_output import TerminalOutput


class Screen(io.StringIO):
    """Small VT screen for cursor addressing, erasure, wrapping, and normal scrolling."""

    def __init__(self, columns=80, lines=24):
        super().__init__()
        self.columns, self.lines = columns, lines
        self.rows = [""] * lines
        self.history = []
        self.x = self.y = 0

    def isatty(self):
        return True

    def newline(self):
        self.x = 0
        self.y += 1
        if self.y >= self.lines:
            self.history.append(self.rows.pop(0))
            self.rows.append("")
            self.y = self.lines - 1

    def resize(self, lines, columns=None):
        if columns is not None:
            reflowed = [part for row in self.rows
                        for part in ([row[i:i + columns] for i in range(0, len(row), columns)]
                                     or [""])]
            excess = max(0, len(reflowed) - self.lines)
            self.history.extend(reflowed[:excess])
            self.rows = reflowed[excess:]
            self.columns = columns
        if lines < self.lines:
            removed = self.lines - lines
            self.history.extend(self.rows[:removed])
            self.rows = self.rows[removed:]
            self.y = max(0, self.y - removed)
        else:
            self.rows.extend([""] * (lines - self.lines))
        self.lines = lines

    def write(self, text):
        super().write(text)
        for token in re.findall(r"\x1b\[[0-9;]*[A-Za-z]|[^\x1b]", text):
            if token.startswith("\x1b"):
                arguments, action = token[2:-1], token[-1]
                values = [int(v or "0") for v in arguments.split(";")]
                value = values[0] or 1
                if action == "H":
                    self.y = min(self.lines - 1, value - 1)
                    self.x = (values[1] or 1) - 1 if len(values) > 1 else 0
                elif action == "J":
                    assert values[0] == 0
                    self.rows[self.y] = self.rows[self.y][:self.x]
                    self.rows[self.y + 1:] = [""] * (self.lines - self.y - 1)
                elif action == "K":
                    assert values[0] == 2
                    self.rows[self.y] = ""
                elif action in {"A", "B", "C", "D"}:
                    if action == "A":
                        self.y = max(0, self.y - value)
                    elif action == "B":
                        self.y = min(self.lines - 1, self.y + value)
                    elif action == "C":
                        self.x = min(self.columns - 1, self.x + value)
                    else:
                        self.x = max(0, self.x - value)
                else:
                    assert action == "m"
            elif token == "\n":
                self.newline()
            elif token == "\r":
                self.x = 0
            else:
                if self.x >= self.columns:
                    self.newline()
                row = self.rows[self.y].ljust(self.x)
                self.rows[self.y] = row[:self.x] + token + row[self.x + 1:]
                self.x += 1
        return len(text)


@pytest.fixture
def screen(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("NO_COLOR", "1")
    terminal = Screen()
    monkeypatch.setattr("shutil.get_terminal_size", lambda **_: os.terminal_size(
        (terminal.columns, terminal.lines),
    ))
    return terminal


def assert_bottom_bar(screen, draft):
    assert screen.rows[-3] == "❯ " + draft
    assert "Enter" in screen.rows[-1]
    assert screen.y == screen.lines - 3


def test_bar_stays_at_bottom_and_output_enters_scrollback(screen):
    output = TerminalOutput(screen, chat=True)
    output.write("Earlier output\n")
    output.set_input("owner", "draft")
    assert_bottom_bar(screen, "draft")
    for index in range(50):
        output.write(f"Reply {index}\n")
        assert_bottom_bar(screen, "draft")
    transcript = [*screen.history, *screen.rows]
    assert "Earlier output" in transcript
    assert all(transcript.count(f"Reply {index}") == 1 for index in range(50))


def test_live_reply_and_picker_stay_above_bottom_input(screen):
    output = TerminalOutput(screen, chat=True)
    output.set_input("owner", "/", suggestions=command_suggestions("/"), selected=2)
    output.write("A streaming")
    output.write(" answer")
    assert_bottom_bar(screen, "/")
    assert output._pending == "A streaming answer"
    assert any("/help" in row for row in screen.rows[:-4])
    output.set_input("owner", "draft")
    assert_bottom_bar(screen, "draft")
    assert "A streaming answer" in screen.rows[:-4]
    assert not any("commands" in row or "/help" in row for row in screen.rows[:-4])
    output.submit_input("draft")
    output.write(" finished.\n")
    assert_bottom_bar(screen, "")
    assert "A streaming answer finished." in [*screen.history, *screen.rows]


def test_idle_bar_reanchors_after_growing_and_shrinking(screen):
    output = TerminalOutput(screen, chat=True)
    output.set_input("owner", "draft")
    for height in (40, 15, 30, 10):
        screen.resize(height)
        output.refresh_layout()
        assert_bottom_bar(screen, "draft")
        assert sum(row.startswith("❯ ") for row in screen.rows) == 1
    output.hide_input()
    assert not any(row.startswith("❯ ") for row in screen.rows)
    output.write("Normal shell output\n")
    assert "Normal shell output" in screen.rows


def test_tiny_terminal_keeps_input_on_its_last_row(screen):
    screen.resize(3)
    output = TerminalOutput(screen, chat=True)
    output.set_input("owner", "draft")
    assert screen.rows[-1] == "❯ draft"
    assert screen.y == 2


def test_width_resize_clears_reflowed_bar_and_preserves_draft(screen):
    output = TerminalOutput(screen, chat=True)
    output.set_input("owner", "draft")
    screen.resize(24, columns=30)
    output.refresh_layout()
    assert_bottom_bar(screen, "draft")
    assert sum(row.startswith("❯ ") for row in screen.rows) == 1
    assert not any(row.startswith("─") for row in screen.rows[:-4])


def test_opening_and_closing_picker_does_not_scroll_startup(screen):
    output = TerminalOutput(screen, chat=True)
    startup = ["CIEL v0.3.6", "Your voice. Your companion.",
               "Voice: Ciel Turbo", "Recognition: small.en", "Ready to chat."]
    output.write("\n".join(startup) + "\n")
    output.set_input("owner", "")
    history = list(screen.history)
    position = screen.rows[:len(startup)]
    for _ in range(10):
        output.set_input("owner", "/", suggestions=command_suggestions("/"))
        assert all(line in screen.rows for line in startup)
        assert screen.rows[:len(startup)] == position
        assert_bottom_bar(screen, "/")
        output.set_input("owner", "")
        assert all(line in screen.rows for line in startup)
    assert screen.history == history


def test_picker_restores_covered_reply_in_a_full_viewport(screen):
    output = TerminalOutput(screen, chat=True)
    output.set_input("owner", "")
    for index in range(30):
        output.write(f"Reply {index}\n")
    original, history = list(screen.rows[:-4]), list(screen.history)
    for selected in (0, 8, 12):
        output.set_input("owner", "/", suggestions=command_suggestions("/"), selected=selected)
        count = len(command_suggestions("/"))
        assert screen.rows[:-(4 + count)] == original[:-count]
        assert_bottom_bar(screen, "/")
        output.set_input("owner", "")
        assert screen.rows[:-4] == original
        assert screen.history == history


@pytest.mark.parametrize("streaming", [False, True])
def test_picker_shows_every_command_even_with_text_above(screen, streaming):
    output = TerminalOutput(screen, chat=True)
    output.set_input("owner", "")
    if streaming:
        output.write("A" * 1000)
    else:
        output.write("\n".join(f"Reply {index}" for index in range(30)) + "\n")
    original, history, pending = list(screen.rows), list(screen.history), output._pending
    choices = command_suggestions("/")
    output.set_input("owner", "/", suggestions=choices)
    for command, _description in choices:
        assert any(command in row.split() for row in screen.rows[:-4]), command
    assert screen.history == history
    assert output._pending == pending
    assert_bottom_bar(screen, "/")
    output.set_input("owner", "")
    assert screen.rows == original
    assert screen.history == history


def test_large_reply_batch_reaches_scrollback_without_losing_lines(screen):
    output = TerminalOutput(screen, chat=True)
    output.set_input("owner", "draft")
    lines = [f"Batch line {index}" for index in range(80)]
    output.write("\n".join(lines) + "\n")
    assert_bottom_bar(screen, "draft")
    transcript = [*screen.history, *screen.rows]
    assert all(transcript.count(line) == 1 for line in lines)


def test_seeded_startup_survives_first_draw_and_picker(screen):
    startup = "CIEL v0.3.6\nYour voice. Your companion.\n"
    screen.write(startup)
    output = TerminalOutput(screen, chat=True)
    output.seed_transcript(startup)
    assert screen.getvalue() == startup
    output.write("Voice: Ciel Turbo\nReady.\n")
    output.set_input("owner", "/", suggestions=command_suggestions("/"))
    assert screen.rows[:4] == ["CIEL v0.3.6", "Your voice. Your companion.",
                               "Voice: Ciel Turbo", "Ready."]
    assert_bottom_bar(screen, "/")


def test_short_terminal_scrolls_entire_picker_without_moving_text(screen):
    screen.resize(10)
    output = TerminalOutput(screen, chat=True)
    output.write("Earlier reply\n")
    output.set_input("owner", "")
    original, history = list(screen.rows), list(screen.history)
    choices = command_suggestions("/")
    for selected, (command, _description) in enumerate(choices):
        output.set_input("owner", "/", suggestions=choices, selected=selected)
        assert any(row.startswith("› " + command + " ") for row in screen.rows[:-4])
        assert any(f"{selected + 1} / {len(choices)} commands" in row
                   for row in screen.rows[:-4])
        assert_bottom_bar(screen, "/")
        assert screen.history == history
    output.set_input("owner", "")
    assert screen.rows == original


def test_received_speech_uses_typed_message_style_and_preserves_keyboard_draft(screen):
    output = TerminalOutput(screen, chat=True)
    output.set_input("owner", "/rea", suggestions=command_suggestions("/rea"), cursor=2)
    output.received_input("A spoken question.")
    assert "❯ A spoken question." in output._transcript
    assert output._draft == "/rea" and output._draft_cursor == 2
    assert output._suggestions == command_suggestions("/rea")
    output.submit_input("A spoken question.")
    assert output._transcript.count("❯ A spoken question.") == 2
