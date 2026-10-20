"""Select a private owner's cross-channel history without rewriting the archives."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ciel.persona import PERSONA_CONTEXT_VERSION

if TYPE_CHECKING:
    from ciel.config import Settings


@dataclass(frozen=True)
class ConversationContext:
    """An exact set of archives and the identity used for its running summary."""

    summary_session_id: str
    session_ids: tuple[str, ...]


def desktop_conversation_id() -> str:
    """Return the current desktop archive, retaining existing voice history."""
    return f"desktop_session:{PERSONA_CONTEXT_VERSION}"


def telegram_conversation_id(bot_id: int, owner_id: int) -> str:
    """Name only an explicitly authenticated bot/owner conversation."""
    if (isinstance(bot_id, bool) or isinstance(owner_id, bool)
            or not isinstance(bot_id, int) or not isinstance(owner_id, int)
            or bot_id <= 0 or owner_id <= 0):
        raise ValueError("Telegram history requires a verified positive bot and owner ID")
    return f"telegram:{bot_id}:{owner_id}:{PERSONA_CONTEXT_VERSION}"


def _paired_telegram_id(settings: Settings) -> str | None:
    """Read the current exact binding without creating state or initializing a transport."""
    configuration = settings.chat
    if "telegram" not in configuration.services or not configuration.telegram_bot_token:
        return None
    # A Bot API token starts with its bot's numeric identity. The binding was
    # established using getMe and an expiring local pairing code, not this prefix.
    token = configuration.telegram_bot_token.get_secret_value()
    match = re.match(r"^([1-9][0-9]*):", token)
    if not match:
        return None
    bot_id = int(match[1])
    path = Path(configuration.state_path).resolve()
    if not path.is_file():
        return None
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1.0) as connection:
            row = connection.execute(
                "SELECT owner_id FROM telegram_bots WHERE bot_id=?", (bot_id,),
            ).fetchone()
        if row and isinstance(row[0], int) and row[0] > 0:
            return telegram_conversation_id(bot_id, row[0])
    except (OSError, sqlite3.Error, ValueError):
        pass
    return None


def conversation_context(settings: Settings, write_session_id: str) -> ConversationContext:
    """Bridge only desktop and the currently paired owner; everything else stays isolated.

    Resolve on every read so a running desktop immediately respects unpairing or
    re-pairing. Old bot/owner archives remain intact and are never selected by a
    different binding. Failed or ambiguous state reads fall back to isolation.
    """
    desktop = desktop_conversation_id()
    if write_session_id != desktop and not write_session_id.startswith("telegram:"):
        return ConversationContext(write_session_id, (write_session_id,))
    paired = _paired_telegram_id(settings)
    if paired is None or write_session_id not in {desktop, paired}:
        return ConversationContext(write_session_id, (write_session_id,))
    sessions = (desktop, paired)
    digest = hashlib.sha256("\0".join(sessions).encode()).hexdigest()[:24]
    return ConversationContext(f"owner_bridge:{digest}:{PERSONA_CONTEXT_VERSION}", sessions)


def conversation_channel(session_id: str) -> str:
    """Give model context a bounded origin label without exposing account identifiers."""
    if session_id.startswith("desktop_session:"):
        return "desktop voice/text"
    if session_id.startswith("telegram:"):
        return "Telegram text"
    return "local conversation"
