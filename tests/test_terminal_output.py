"""Output and editable drafts use separate rows, including wrapping speech."""

import io

from ciel.terminal_output import TerminalOutput


def test_speech_and_logs_preserve_owner_name_and_draft():
    buffer = io.StringIO()
    output = TerminalOutput(buffer, width=40)
    output.set_input("hexarion", "my draft")
    output.write("CIEL: Hel")
    assert buffer.getvalue().endswith("CIEL: Hel\n[hexarion]: my draft")
    output.set_input("hexarion", "my draft!")
    output.write("lo.\nINFO: background work\n")
    assert buffer.getvalue().endswith(
        "CIEL: Hello.\nINFO: background work\n[hexarion]: my draft!"
    )


def test_long_speech_wraps_without_truncation_and_long_draft_submits_intact():
    buffer = io.StringIO()
    output = TerminalOutput(buffer, width=18)
    draft = "This draft is longer than the terminal width."
    speech = "CIEL: This reply is longer than one row."
    output.set_input("hexarion", draft)
    output.write(speech)
    assert speech in buffer.getvalue()
    output.submit_input(draft)
    assert f"[hexarion]: {draft}\n" in buffer.getvalue()
    output.write("\n")
    assert buffer.getvalue().endswith(speech + "\n[hexarion]: ")


def test_owner_name_cannot_inject_terminal_control_sequences():
    buffer = io.StringIO()
    output = TerminalOutput(buffer)
    output.set_input("owner\x1b\nname", "hello")
    assert buffer.getvalue() == "[ownername]: hello"
