"""Tests for structured logging and secret masking."""

import logging

import pytest

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


@pytest.mark.parametrize("key", [
    "tvly-synthetic_search_secret_123456789", "TVLY-synthetic_search_secret_123456789",
])
def test_tavily_key_is_masked_without_loading_settings(key):
    assert SecretMaskingFilter._redact(f"Search token {key}") == "Search token [REDACTED_SECRET]"


def test_configured_search_key_without_a_known_prefix_is_masked_everywhere(monkeypatch):
    from ciel.config import InternetConfig, Settings
    from ciel.internet import _clean
    from ciel.memory.web_evidence import normalize_web_evidence

    monkeypatch.setattr(SecretMaskingFilter, "_configured_secrets", ())
    key = "synthetic-arbitrary-search-credential-54321"
    # Register flat settings early, without loading a real .env or constructing .internet.
    Settings(_env_file=None, brave_search_api_key=key)
    assert _clean(f"search docs {key}") == "search docs [REDACTED_SECRET]"
    assert SecretMaskingFilter._redact(f"token {key}") == "token [REDACTED_SECRET]"
    payload = {
        "kind": "web", "query": f"search {key}", "retrieved_at": "2026-10-21",
        "sources": [{"url": "https://example.com/", "content": key}],
    }
    normalized = normalize_web_evidence(payload)
    assert normalized["query"] == "search [REDACTED_SECRET]"
    assert normalized["sources"][0]["content"] == "[REDACTED_SECRET]"
    payload["sources"][0]["url"] += key
    assert normalize_web_evidence(payload) == {}
    # Programmatic structured configs use the same registry.
    other_key = "second-arbitrary-search-credential-12345"
    InternetConfig(provider="tavily", tavily_api_key=other_key)
    assert SecretMaskingFilter._redact(other_key) == "[REDACTED_SECRET]"


def test_configured_secret_registry_is_bounded_and_refreshes_active_keys(monkeypatch):
    monkeypatch.setattr(SecretMaskingFilter, "_configured_secrets", ())
    SecretMaskingFilter.register_secrets("", "key", "x" * 4097)
    assert SecretMaskingFilter._configured_secrets == ()
    for index in range(100):
        SecretMaskingFilter.register_secrets(f"synthetic-secret-{index:04}")
    assert len(SecretMaskingFilter._configured_secrets) == 64
    active_key = SecretMaskingFilter._configured_secrets[0]
    SecretMaskingFilter.register_secrets(active_key, "new-synthetic-secret-12345")
    assert active_key in SecretMaskingFilter._configured_secrets
    assert SecretMaskingFilter._redact(active_key) == "[REDACTED_SECRET]"


def test_partial_redaction_withholds_every_configured_secret_prefix(monkeypatch):
    monkeypatch.setattr(SecretMaskingFilter, "_configured_secrets", ())
    key = "fixture_" + "x" * 24 + "z"
    SecretMaskingFilter.register_secrets(key)
    for end in range(1, len(key)):
        assert SecretMaskingFilter._redact_partial("Answer: " + key[:end]) == "Answer: "
    assert SecretMaskingFilter._redact_partial("Answer: " + key) == (
        "Answer: [REDACTED_SECRET]"
    )


@pytest.mark.parametrize("key", ["fixture" * 4, "]fixture_private_credential"])
def test_partial_redaction_keeps_complete_overlapping_secrets_masked(monkeypatch, key):
    monkeypatch.setattr(SecretMaskingFilter, "_configured_secrets", ())
    SecretMaskingFilter.register_secrets(key)
    assert SecretMaskingFilter._redact_partial(key) == "[REDACTED_SECRET]"


@pytest.mark.parametrize("key", [
    "nvapi-" + "x" * 24,
    "SK-" + "x" * 24,
    "gsk_" + "x" * 24,
    "TVLY-" + "x" * 24,
    "Bearer " + "x" * 24,
    "bEaReR\t" + "x" * 24,
    "123456789:" + "x" * 24,
])
def test_partial_redaction_protects_unconfigured_token_patterns_at_every_split(monkeypatch, key):
    monkeypatch.setattr(SecretMaskingFilter, "_configured_secrets", ())
    for end in range(1, len(key) + 1):
        visible = SecretMaskingFilter._redact_partial("Answer: " + key[:end])
        assert visible in {"Answer: ", "Answer: [REDACTED_SECRET]"}
    assert visible == "Answer: [REDACTED_SECRET]"


@pytest.mark.parametrize("text,expected", [
    ("Ordinary reply.", "Ordinary reply."),
    ("An unfinished nva", "An unfinished "),
    ("An unfinished BeAr", "An unfinished "),
    ("The count is 12345", "The count is "),
    ("The count is 1234567890123456", "The count is 1234567890123456"),
    ("Boundary x12345:abcdefQ", "Boundary x12345:abcdefQ"),
    ("Short nvapi-abc!", "Short nvapi-abc!"),
    ("Complete nvapi-" + "x" * 12 + ".", "Complete [REDACTED_SECRET]."),
])
def test_partial_redaction_only_holds_ambiguous_trailing_text(monkeypatch, text, expected):
    monkeypatch.setattr(SecretMaskingFilter, "_configured_secrets", ())
    assert SecretMaskingFilter._redact_partial(text) == expected


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
