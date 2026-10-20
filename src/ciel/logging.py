"""Structured logging system with secret redaction for CIEL."""

import logging
import re
import sys

# Known token/secret patterns to redact automatically
SENSITIVE_PATTERNS = [
    re.compile(r"(nvapi-[A-Za-z0-9_-]{10,})", re.IGNORECASE),
    re.compile(r"(sk-[A-Za-z0-9_-]{10,})", re.IGNORECASE),
    re.compile(r"(gsk_[A-Za-z0-9_-]{10,})", re.IGNORECASE),
    re.compile(r"(?<!\w)[0-9]{5,15}:[A-Za-z0-9_-]{20,}(?!\w)"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/-]+", re.IGNORECASE),
]


class SecretMaskingFilter(logging.Filter):
    """Logging filter that masks sensitive API keys and tokens from log records."""

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

    @staticmethod
    def _redact(text: str) -> str:
        for pattern in SENSITIVE_PATTERNS:
            text = pattern.sub("[REDACTED_SECRET]", text)
        return text


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
