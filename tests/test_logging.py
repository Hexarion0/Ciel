"""Tests for structured logging and secret masking."""

import logging

from ciel.logging import (
    CielFormatter,
    ConciseLogFilter,
    ConversationFormatter,
    SecretMaskingFilter,
    get_logger,
    setup_logging,
)


def test_secret_masking_filter():
    """Verify that known sensitive API key patterns are masked."""
    redactor = SecretMaskingFilter()

    # Test raw message masking
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Using key nvapi-abcdef1234567890 for NIM API",
        args=(),
        exc_info=None,
    )
    assert redactor.filter(record) is True
    assert "nvapi-abcdef1234567890" not in record.msg
    assert "[REDACTED_SECRET]" in record.msg

    # Test args masking (tuple/list)
    record_args = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Calling Groq with key: %s",
        args=("gsk_1234567890abcdefgh",),
        exc_info=None,
    )
    assert redactor.filter(record_args) is True
    assert "gsk_1234567890abcdefgh" not in record_args.args[0]
    assert "[REDACTED_SECRET]" in record_args.args[0]


def test_setup_logging():
    """Verify logger creation and namespacing."""
    logger = setup_logging("DEBUG")
    assert logger.name == "ciel"

    sub_logger = get_logger("providers.nim")
    assert sub_logger.name == "ciel.providers.nim"


def test_concise_log_filter_keeps_conversation_and_problems_only():
    concise = ConciseLogFilter()

    def record(level, message):
        return logging.LogRecord(
            name="ciel.audio.listener", level=level, pathname="test.py", lineno=1,
            msg=message, args=(), exc_info=None,
        )

    assert concise.filter(record(logging.INFO, '🗣️ You: "hello"'))
    assert concise.filter(record(logging.INFO, '⌨️ You: "hello"'))
    assert not concise.filter(record(logging.INFO, "Keyboard ready — type a message"))
    assert concise.filter(record(logging.INFO, "[system]: Microphone muted."))
    assert concise.filter(record(logging.INFO, '🤖 CIEL: "hi"'))
    assert not concise.filter(record(logging.INFO, "Microphone active — ambient listening"))
    assert not concise.filter(record(logging.INFO, "Chatterbox Turbo ready (pid=1)"))
    assert concise.filter(record(logging.INFO, "STT ready: small.en (cuda/int8_float16)"))
    assert not concise.filter(record(logging.INFO, "[system]: AI chose local (intent=request)"))
    assert not concise.filter(record(logging.INFO, "Utterance recording started"))
    assert concise.filter(record(logging.INFO, "STT candidate (diagnostic): 'hello'"))
    assert concise.filter(record(
        logging.INFO, '🗣️ Heard before a newer utterance superseded this turn: "and this"',
    ))
    assert not concise.filter(record(logging.INFO, "Transcribing 4.8s of audio"))
    assert not concise.filter(record(logging.DEBUG, "provider payload details"))
    assert concise.filter(record(logging.WARNING, "provider request failed"))


def test_concise_format_keeps_problems_and_redacts_secrets(capsys):
    logger = setup_logging("INFO", concise=True)
    logger.info('⌨️ You: "hello"')
    logger.warning("Provider rejected Bearer secret1234567890")
    logger.info("STT ready: small.en (cpu/int8)")
    output = capsys.readouterr().out
    assert output.startswith('⌨️ You: "hello"\n')
    assert "Warning: Provider rejected [REDACTED_SECRET]" in output
    assert "secret1234567890" not in output
    assert "test_logging" not in output
    assert "Recognition loaded: small.en (cpu/int8)" in output
    assert isinstance(logging.getLogger().handlers[0].formatter, ConversationFormatter)
    setup_logging("DEBUG", concise=False)
    assert isinstance(logging.getLogger().handlers[0].formatter, CielFormatter)
