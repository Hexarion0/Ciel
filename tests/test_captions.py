"""Playback-driven terminal captions and coordinated logging."""

import io
import logging
import threading

from ciel.audio.captions import TerminalCaptions, install_caption_logging
from ciel.logging import SecretMaskingFilter


class TtyBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_redirected_output_contains_only_final_snapshot():
    output = io.StringIO()
    captions = TerminalCaptions(output)
    captions.start("What's on your mind?")
    for prefix in ["W", "Wh", "What's", "What's on your"]:
        captions.update(prefix)
    assert output.getvalue() == ""

    captions.update("What's on your mind?", finished=True)
    captions.close()

    assert output.getvalue() == "CIEL: What's on your mind?\n"


def test_interruption_saves_visible_words_without_unheard_tail():
    output = io.StringIO()
    captions = TerminalCaptions(output)
    captions.start("Take your time and tell me.")
    captions.update("Take your")
    captions.update("Take your", interrupted=True)
    captions.update("Take your time and tell me.", finished=True)
    captions.close()

    assert output.getvalue() == "CIEL: Take your [interrupted]\n"


def test_multiple_sentences_share_one_reply_line():
    output = io.StringIO()
    captions = TerminalCaptions(output)
    for sentence in ["Hello, Hexarion.", "What's on your mind?"]:
        captions.start(sentence)
        captions.update(sentence[:4])
        captions.update(sentence, finished=True)
    captions.close()

    assert output.getvalue() == "CIEL: Hello, Hexarion. What's on your mind?\n"


def test_close_preserves_partial_sentence_and_can_restart():
    output = io.StringIO()
    captions = TerminalCaptions(output)
    captions.start("Hello, Hexarion.")
    captions.update("Hello")
    captions.close()
    captions.close()
    captions.start("Welcome back.")
    captions.update("Welcome back.", finished=True)
    captions.close()

    assert output.getvalue() == (
        "CIEL: Hello [interrupted]\nCIEL: Welcome back.\n"
    )


def test_mismatched_or_late_updates_do_not_rewind_visible_text():
    output = io.StringIO()
    captions = TerminalCaptions(output)
    captions.start("Hello, Hexarion.")
    captions.update("Hello,")
    captions.update("wrong sentence", finished=True)
    captions.update("He")
    captions.update("He", interrupted=True)

    assert output.getvalue() == "CIEL: Hello, [interrupted]\n"


def test_tty_updates_only_when_supplied_and_never_repeats_same_prefix():
    output = TtyBuffer()
    captions = TerminalCaptions(output, width=80)
    captions.start("Hello.")
    captions.update("H")
    written = output.getvalue()
    captions.update("H")
    assert output.getvalue() == written
    assert "Hello." not in written

    captions.update("Hello.", finished=True)
    captions.close()

    assert output.getvalue() == "CIEL: Hello.\n"


def test_empty_interruption_does_not_save_unplayed_words():
    output = io.StringIO()
    captions = TerminalCaptions(output)
    captions.start("Unplayed speech.")
    captions.update("", interrupted=True)
    captions.close()

    assert output.getvalue() == ""


def test_caption_control_sequences_cannot_inject_terminal_commands():
    output = io.StringIO()
    captions = TerminalCaptions(output)
    sentence = "Hello\x1b[31m,\x1b[0m\nHexarion."
    captions.start(sentence)
    captions.update(sentence, finished=True)
    captions.close()

    assert output.getvalue() == "CIEL: Hello, Hexarion.\n"


def test_progressive_secret_prefixes_never_leak_token_characters():
    output = TtyBuffer()
    captions = TerminalCaptions(output, width=80)
    sentence = "The token is nvapi-0123456789secret. Keep it private."
    captions.start(sentence)
    for end in range(1, len(sentence)):
        captions.update(sentence[:end])
    captions.update(sentence, finished=True)
    captions.close()

    text = output.getvalue()
    assert "nvapi" not in text
    assert "0123456789" not in text
    assert "[REDACTED_SECRET]" in text
    assert text.endswith("CIEL: The token is [REDACTED_SECRET]. Keep it private.\n")


def test_interrupting_during_a_secret_saves_only_safe_prefix():
    output = io.StringIO()
    captions = TerminalCaptions(output)
    captions.start("The token is nvapi-0123456789secret.")
    captions.update("The token is")
    captions.update("The token is nvapi-01", interrupted=True)
    captions.close()

    assert output.getvalue() == "CIEL: The token is [interrupted]\n"


def test_tty_preserves_long_reply_for_natural_terminal_wrapping():
    output = TtyBuffer()
    captions = TerminalCaptions(output, width=18)
    for sentence in ["This is a long sentence.", "Here is another."]:
        captions.start(sentence)
        for end in range(1, len(sentence) + 1):
            captions.update(sentence[:end])
        captions.update(sentence, finished=True)
    captions.close()

    assert output.getvalue() == "CIEL: This is a long sentence. Here is another.\n"


def test_logging_separates_then_restores_current_caption():
    output = TtyBuffer()
    captions = TerminalCaptions(output, width=80)
    captions.start("Hello.")
    captions.update("Hel")
    captions.write_log("INFO: background work finished")

    assert output.getvalue().endswith(
        "CIEL: Hel\nINFO: background work finished\nCIEL: Hel"
    )
    captions.update("Hello.", finished=True)
    captions.close()
    assert output.getvalue().endswith("CIEL: Hello.\n")


def test_installed_handler_preserves_redaction_and_restores_original():
    output = io.StringIO()
    other_output = io.StringIO()
    captions = TerminalCaptions(output)
    logger = logging.Logger("caption-test", level=logging.DEBUG)
    original = logging.StreamHandler(output)
    original.setLevel(logging.INFO)
    original.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    original.addFilter(SecretMaskingFilter())
    other = logging.StreamHandler(other_output)
    logger.addHandler(original)
    logger.addHandler(other)
    restore = install_caption_logging(captions, logger)
    try:
        captions.start("Hello.")
        captions.update("He")
        logger.debug("not shown on caption output")
        logger.info("token=%s", "nvapi-0123456789secret")
        captions.update("Hello.", finished=True)
    finally:
        captions.close()
        restore()
        restore()

    assert output.getvalue() == "INFO token=[REDACTED_SECRET]\nCIEL: Hello.\n"
    assert "nvapi-" not in other_output.getvalue()
    assert "[REDACTED_SECRET]" in other_output.getvalue()
    assert original in logger.handlers
    assert other in logger.handlers
    assert logger.handlers == [original, other]


def test_concurrent_logs_cannot_split_caption_control_sequences():
    output = TtyBuffer()
    captions = TerminalCaptions(output, width=80)
    captions.start("Hello.")
    barrier = threading.Barrier(3)

    def log():
        barrier.wait()
        for index in range(20):
            captions.write_log(f"log-{index}")

    def progress():
        barrier.wait()
        for prefix in ["H", "He", "Hel", "Hell", "Hello."]:
            captions.update(prefix)

    workers = [threading.Thread(target=log), threading.Thread(target=progress)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=1)
        assert not worker.is_alive()
    captions.update("Hello.", finished=True)
    captions.close()

    text = output.getvalue()
    for index in range(20):
        assert f"log-{index}\n" in text
    assert "\x1b" not in text
    assert text.endswith("CIEL: Hello.\n")
