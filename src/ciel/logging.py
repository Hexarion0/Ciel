"""Structured logging system with secret redaction for CIEL."""

import logging
import re
import sys
import threading

# Known token/secret patterns to redact automatically
SENSITIVE_PATTERNS = [
    re.compile(r"(nvapi-[A-Za-z0-9_-]{10,})", re.IGNORECASE),
    re.compile(r"(sk-[A-Za-z0-9_-]{10,})", re.IGNORECASE),
    re.compile(r"(gsk_[A-Za-z0-9_-]{10,})", re.IGNORECASE),
    re.compile(r"(tvly-[A-Za-z0-9_-]{10,})", re.IGNORECASE),
    re.compile(r"(?<!\w)[0-9]{5,15}:[A-Za-z0-9_-]{20,}(?!\w)"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/-]+", re.IGNORECASE),
]
_SECRET_MARKERS = ("nvapi-", "sk-", "gsk_", "tvly-", "bearer")
_PARTIAL_TOKEN_SUFFIX = re.compile(
    r"(?:nvapi-|sk-|gsk_|tvly-)[A-Za-z0-9_-]*\Z"
    r"|Bearer\s*\Z"
    r"|(?<!\w)(?:[0-9]{1,15}|[0-9]{5,15}:[A-Za-z0-9_-]*)\Z",
    re.IGNORECASE,
)


class SecretMaskingFilter(logging.Filter):
    """Logging filter that masks sensitive API keys and tokens from log records."""

    _configured_secrets: tuple[str, ...] = ()
    _secret_lock = threading.Lock()

    @classmethod
    def register_secrets(cls, *values: str) -> None:
        """Keep a bounded process-local set of configured keys without logging them.

        Search keys have no universal prefix. Retaining recent keys also masks a key
        after a setup reload rotates it. Ignore short placeholders that could censor
        ordinary conversation, and cap memory used by malformed configuration values.
        """
        accepted = [value.strip() for value in values if isinstance(value, str)
                    and 8 <= len(value.strip()) <= 4096]
        if not accepted:
            return
        with cls._secret_lock:
            cls._configured_secrets = tuple(dict.fromkeys(
                (*(value for value in cls._configured_secrets if value not in accepted), *accepted),
            ))[-64:]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._redact(v) if isinstance(v, str) else v for k, v in record.args.items()
                }
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(
                    self._redact(v) if isinstance(v, str) else v for v in record.args
                )
        return True

    @classmethod
    def _redact(cls, text: str) -> str:
        with cls._secret_lock:
            secrets = cls._configured_secrets
        for secret in sorted(secrets, key=len, reverse=True):
            text = text.replace(secret, "[REDACTED_SECRET]")
        for pattern in SENSITIVE_PATTERNS:
            text = pattern.sub("[REDACTED_SECRET]", text)
        return text

    @classmethod
    def _redact_partial(cls, text: str) -> str:
        """Redact cumulative streamed text and hold possible unfinished secret suffixes.

        Callers retain the raw cumulative text and call again as it grows. Only a
        successful completion may release benign held text with ``_redact``;
        interrupted streams must discard the held suffix.
        """
        text = cls._redact(text)
        with cls._secret_lock:
            secrets = cls._configured_secrets
        # Complete masks are atomic: a key beginning with ']' must not trim one.
        mask = "[REDACTED_SECRET]"
        last_mask = text.rfind(mask)
        tail_start = last_mask + len(mask) if last_mask >= 0 else 0
        tail = text[tail_start:]
        end = len(tail)
        for secret in secrets:
            start = max(0, len(tail) - len(secret) + 1)
            candidate = tail.find(secret[0], start, end)
            while candidate >= 0:
                if secret.startswith(tail[candidate:]):
                    end = candidate
                    break
                candidate = tail.find(secret[0], candidate + 1, end)
        lowered = tail.lower()
        for marker in _SECRET_MARKERS:
            for length in range(1, len(marker) + 1):
                if lowered.endswith(marker[:length]):
                    end = min(end, len(tail) - length)
        token = _PARTIAL_TOKEN_SUFFIX.search(tail)
        if token is not None:
            end = min(end, token.start())
        return text[:tail_start + end]


class CielFormatter(logging.Formatter):
    """Clean, structured formatter for CIEL console output."""

    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FORMAT, datefmt=self.DATE_FORMAT)


class ConversationFormatter(logging.Formatter):
    """Show everyday messages without developer timestamps and source locations."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.levelno >= logging.WARNING:
            return f"{record.levelname.title()}: {message}"
        if message.startswith("STT ready:"):
            return message.replace("STT ready:", "Recognition loaded:", 1)
        return message


class ConciseLogFilter(logging.Filter):
    """Keep conversational and operational INFO messages on normal startup."""

    _VISIBLE_INFO = (
        "🗣️ You:",
        "⌨️ You:",
        "[system]:",
        "Desktop web interface:",
        "🤖 CIEL",
        "Ready.",
        "Heard during CIEL reply",
        "Heard before a newer utterance superseded this turn",
        "Merged ",
        "Interrupted speech was unclear to STT",
        "Interrupted speech wasn't merged",
        "STT candidate (diagnostic):",
        "STT ready:",
        "Speech was unclear; asking for a repeat",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        if record.levelno != logging.INFO:
            return False
        message = record.getMessage()
        if message.startswith("[system]: AI chose "):
            return False
        return any(visible in message for visible in self._VISIBLE_INFO)


def setup_logging(log_level: str | None = None, *, concise: bool = False) -> logging.Logger:
    """Initialize structured logging for CIEL with secret masking."""
    if log_level is None:
        from ciel.config import get_settings

        log_level = get_settings().app.log_level

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(ConversationFormatter() if concise else CielFormatter())
    console_handler.addFilter(SecretMaskingFilter())
    if concise:
        console_handler.addFilter(ConciseLogFilter())

    root_logger.addHandler(console_handler)

    # Keep normal output focused; development mode exposes transport and decoder logs.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("ctranslate2").setLevel(logging.WARNING)
    if not concise:
        for name in ("urllib3", "httpcore", "httpx", "faster_whisper", "ctranslate2"):
            logging.getLogger(name).setLevel(logging.NOTSET)

    logger = logging.getLogger("ciel")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Obtain a namespaced logger for CIEL components."""
    if not name.startswith("ciel"):
        name = f"ciel.{name}"
    return logging.getLogger(name)
