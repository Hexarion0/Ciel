"""Owner-paired Telegram Bot API transport; independent of audio libraries."""

import logging
import re
import secrets
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from ciel.chat.state import ChatState

MAX_MESSAGE_AGE_SECONDS = 300
PAIR_TTL_SECONDS = 300


class TelegramError(RuntimeError):
    """Safe actionable error containing neither tokens nor private message text."""

    def __init__(self, message: str, *, transient: bool = False, retry_after: int = 0) -> None:
        super().__init__(message)
        self.transient = transient
        self.retry_after = retry_after


class _HideTelegramURLs(logging.Filter):
    """httpx/httpcore normally log full token-bearing Telegram request URLs."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "api.telegram.org" not in message and not re.search(
            r"/bot[^\s/'\"]+", message,
        )


def protect_telegram_logs() -> None:
    """Install a token-URL guard even when an embedding app enables HTTP debug logs."""
    for name in ("httpx", "httpcore.http11", "httpcore.connection", "httpcore.proxy"):
        logger = logging.getLogger(name)
        if not any(isinstance(item, _HideTelegramURLs) for item in logger.filters):
            logger.addFilter(_HideTelegramURLs())


class TelegramClient:
    """Small synchronous Bot API client with sanitized errors and explicit timeouts."""

    def __init__(self, token: str, *, client: httpx.Client | None = None) -> None:
        protect_telegram_logs()
        if not token.strip() or any(character.isspace() for character in token):
            raise TelegramError("Set TELEGRAM_BOT_TOKEN using ciel chat setup first.")
        self._base_url = f"https://api.telegram.org/bot{token}/"
        self._client = client or httpx.Client(timeout=httpx.Timeout(35.0, connect=10.0))
        self._owns_client = client is None

    def request(self, method: str, **parameters: Any) -> Any:
        """Call a fixed Bot API method without exposing httpx's token-containing errors."""
        try:
            response = self._client.post(self._base_url + method, json=parameters)
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            raise TelegramError(
                "Telegram is temporarily unreachable. Check your internet connection.",
                transient=True,
            ) from None
        if not isinstance(payload, dict):
            raise TelegramError("Telegram returned an invalid response.", transient=True)
        if payload.get("ok") is True:
            return payload.get("result")
        code = payload.get("error_code", response.status_code)
        if code in (401, 404):
            raise TelegramError("Telegram rejected the bot token. Update it with ciel chat setup.")
        if code == 409:
            raise TelegramError(
                "Telegram polling conflicts with a webhook or another bot process. "
                "Stop the other process or remove its webhook before starting Ciel."
            )
        if code == 429:
            parameters = payload.get("parameters")
            retry = parameters.get("retry_after", 1) if isinstance(parameters, dict) else 1
            retry = retry if isinstance(retry, int) and 0 < retry <= 300 else 30
            raise TelegramError("Telegram rate limit reached.", transient=True, retry_after=retry)
        if isinstance(code, int) and code >= 500:
            raise TelegramError("Telegram is temporarily unavailable.", transient=True)
        raise TelegramError("Telegram rejected the request. Check the bot's private chat access.")

    def identity(self) -> dict[str, Any]:
        """Validate the token and return this bot's stable numeric identity."""
        identity = self.request("getMe")
        if not isinstance(identity, dict) or type(identity.get("id")) is not int:
            raise TelegramError("Telegram returned an invalid bot identity.")
        return identity

    def check_polling(self) -> None:
        """Leave existing webhooks untouched and explain incompatible configuration."""
        webhook = self.request("getWebhookInfo")
        if not isinstance(webhook, dict):
            raise TelegramError("Telegram returned invalid webhook information.")
        if webhook.get("url"):
            raise TelegramError(
                "This bot has a webhook configured. Remove it in the bot's existing integration "
                "before using Ciel's polling service."
            )

    def updates(self, offset: int, *, timeout: int = 20) -> list[dict[str, Any]]:
        """Long poll for new message updates; offsets are committed by ChatState first."""
        result = self.request(
            "getUpdates", offset=offset, timeout=timeout, limit=100, allowed_updates=["message"],
        )
        if not isinstance(result, list):
            raise TelegramError("Telegram returned an invalid update list.", transient=True)
        return [item for item in result if isinstance(item, dict)]

    def send(self, owner_id: int, text: str, stop: threading.Event) -> None:
        """Send plain text in legal chunks; retry only explicit rate-limit rejection."""
        for index, chunk in enumerate(message_chunks(text)):
            if index and stop.wait(1.05):
                return
            for attempt in range(3):
                if stop.is_set():
                    return
                try:
                    self.request("sendMessage", chat_id=owner_id, text=chunk)
                    break
                except TelegramError as error:
                    if not error.retry_after or attempt == 2:
                        raise
                    if stop.wait(error.retry_after):
                        return

    def close(self) -> None:
        """Release only the HTTP client created by this transport."""
        if self._owns_client:
            self._client.close()


def message_chunks(text: str) -> list[str]:
    """Respect Telegram's 4096-character ceiling, including UTF-16 surrogate pairs."""
    chunks: list[str] = []
    start = 0
    size = 0
    for index, character in enumerate(text):
        width = 2 if ord(character) > 0xFFFF else 1
        if size + width > 4096:
            chunks.append(text[start:index])
            start, size = index, 0
        size += width
    if text[start:]:
        chunks.append(text[start:])
    return chunks or ["I couldn't prepare a reply. Please try again."]


def private_message(update: dict[str, Any], now: float) -> tuple[int, str] | None:
    """Validate source and freshness before any conversation or memory access."""
    message = update.get("message")
    if not isinstance(message, dict):
        return None
    sender, chat = message.get("from"), message.get("chat")
    if not isinstance(sender, dict) or not isinstance(chat, dict):
        return None
    user_id = sender.get("id")
    if (type(user_id) is not int or sender.get("is_bot") is not False
            or chat.get("type") != "private" or chat.get("id") != user_id):
        return None
    if any(key in message for key in ("forward_origin", "forward_from", "forward_sender_name")):
        return None
    text, sent_at = message.get("text"), message.get("date")
    if (not isinstance(text, str) or not text.strip() or type(sent_at) is not int
            or not -30 <= now - sent_at <= MAX_MESSAGE_AGE_SECONDS):
        return None
    return user_id, text.strip()


class TelegramService:
    """Sequential owner-only polling, with durable claims before processing a turn."""

    def __init__(
        self, client: TelegramClient, state: ChatState, bot_id: int,
        reply: Callable[[str, threading.Event], str], *,
        stop: threading.Event | None = None, clock: Callable[[], float] = time.time,
        report: Callable[[str], None] = print,
    ) -> None:
        self.client, self.state, self.bot_id = client, state, bot_id
        self.reply, self.clock, self.report = reply, clock, report
        self.stop = stop or threading.Event()

    def process(self, update: dict[str, Any]) -> None:
        """Consume once, then authorize the owner before invoking the conversation."""
        update_id = update.get("update_id")
        if type(update_id) is not int or not self.state.claim(
            self.bot_id, update_id, now=self.clock(),
        ):
            return
        incoming = private_message(update, self.clock())
        owner = self.state.owner(self.bot_id)
        if incoming is None or owner is None or incoming[0] != owner:
            self.state.finish(self.bot_id, update_id, "ignored")
            return
        try:
            answer = self.reply(incoming[1], self.stop)
        except Exception:
            self.state.finish(self.bot_id, update_id, "failed")
            answer = (
                "I couldn't complete that request. It may have been partly processed; "
                "ask what I remember before repeating a memory update."
            )
        if self.stop.is_set() or not answer:
            self.state.finish(self.bot_id, update_id, "cancelled")
            return
        try:
            self.client.send(owner, answer, self.stop)
        except TelegramError as error:
            self.state.finish(self.bot_id, update_id, "reply_failed")
            self.report(f"Reply delivery failed: {error} The request will not be run again.")
            return
        self.state.finish(self.bot_id, update_id, "done")

    def poll(self, *, timeout: int = 20) -> list[dict[str, Any]]:
        """Retry read-only polls with bounded, interruptible delays."""
        self.state.recover_idle_offset(self.bot_id, self.clock())
        for attempt in range(6):
            if self.stop.is_set():
                return []
            try:
                return self.client.updates(self.state.offset(self.bot_id), timeout=timeout)
            except TelegramError as error:
                if not error.transient or attempt == 5:
                    raise
                self.report(f"{error} Retrying the connection.")
                if self.stop.wait(error.retry_after or min(2 ** attempt, 30)):
                    return []
        return []

    def run(self) -> None:
        """Poll until stopped; processing stays sequential to preserve confirmations."""
        if self.state.owner(self.bot_id) is None:
            raise TelegramError("Pair your account locally first: ciel chat pair telegram")
        while not self.stop.is_set():
            for update in self.poll():
                if self.stop.is_set():
                    break
                self.process(update)

    def pair(self, show_code: Callable[[str], None]) -> int | None:
        """Discard queued history, then require a fresh expiring code shown only locally."""
        if self.state.owner(self.bot_id) is not None:
            raise TelegramError(
                "This bot is already paired. Run ciel chat unpair to change owners."
            )
        # Drain Telegram's bounded backlog before issuing the unpredictable local code.
        for _ in range(100):
            backlog = self.poll(timeout=0)
            if not backlog:
                break
            for update in backlog:
                update_id = update.get("update_id")
                if type(update_id) is int:
                    self.state.claim(self.bot_id, update_id, now=self.clock())
        else:
            raise TelegramError(
                "Too many queued updates. Try pairing again after clearing the bot."
            )
        code = secrets.token_urlsafe(18)
        issued_at = self.clock()
        show_code(code)
        while not self.stop.is_set() and self.clock() - issued_at < PAIR_TTL_SECONDS:
            for update in self.poll(timeout=10):
                update_id = update.get("update_id")
                if type(update_id) is not int or not self.state.claim(
                    self.bot_id, update_id, now=self.clock(),
                ):
                    continue
                incoming = private_message(update, self.clock())
                if incoming is None:
                    continue
                user_id, text = incoming
                if text != f"/start {code}" or self.clock() - issued_at >= PAIR_TTL_SECONDS:
                    continue
                self.state.bind(self.bot_id, user_id)
                self.state.finish(self.bot_id, update_id, "paired")
                try:
                    self.client.send(
                        user_id,
                        "Paired with Ciel. Start the chat service on your PC to talk here.",
                        self.stop,
                    )
                except TelegramError:
                    self.report(
                        "Pairing saved, but the Telegram confirmation could not be delivered.",
                    )
                return user_id
        return None
