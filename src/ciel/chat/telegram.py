"""Owner-paired Telegram Bot API transport; independent of audio libraries."""

import asyncio
import logging
import re
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Any

import httpx

from ciel.chat.state import ChatState

MAX_MESSAGE_AGE_SECONDS = 300
PAIR_TTL_SECONDS = 300
MAX_PENDING_REQUESTS = 16
REPLY_SHUTDOWN_SECONDS = 2.0


class _ReplyCancellation(threading.Event):
    """Combine direct cancellation with shutdown without cancelling later requests."""

    def __init__(self, shutdown: threading.Event) -> None:
        super().__init__()
        self._shutdown = shutdown

    def is_set(self) -> bool:
        return super().is_set() or self._shutdown.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                break
            super().wait(0.05 if remaining is None else min(0.05, remaining))
        return self.is_set()


@dataclass
class _ReplyTask:
    """One already claimed, authorized request; only its reply runs off-thread."""

    update_id: int
    owner: int
    text: str
    sent_at: int
    cancel: threading.Event
    done: threading.Event = field(default_factory=threading.Event)
    answer: str = ""
    progress_sent: bool = False


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

    def __init__(
        self, token: str, *, client: httpx.Client | None = None,
        stop: threading.Event | None = None,
    ) -> None:
        protect_telegram_logs()
        if not token.strip() or any(character.isspace() for character in token):
            raise TelegramError("Set TELEGRAM_BOT_TOKEN using ciel chat setup first.")
        self._base_url = f"https://api.telegram.org/bot{token}/"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(35.0, connect=10.0))
        self._stop = stop
        # Keep the async connection pool and its cleanup on one loop across synchronous calls.
        self._loop = asyncio.new_event_loop() if self._owns_client else None

    def set_stop_event(self, stop: threading.Event) -> None:
        """Attach service/pairing cancellation without changing injected client constructors."""
        self._stop = stop

    async def _post(self, method: str, parameters: dict[str, Any]) -> httpx.Response:
        """Cancel the actual in-flight HTTP task before releasing pairing or polling locks."""
        task = asyncio.create_task(self._client.post(self._base_url + method, json=parameters))
        try:
            while not task.done():
                if self._stop is not None and self._stop.is_set():
                    raise TelegramError("Telegram request cancelled.")
                await asyncio.wait({task}, timeout=0.05)
            if self._stop is not None and self._stop.is_set():
                raise TelegramError("Telegram request cancelled.")
            return await task
        finally:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def request(self, method: str, **parameters: Any) -> Any:
        """Call a fixed Bot API method without exposing httpx's token-containing errors."""
        try:
            if self._stop is not None and self._stop.is_set():
                raise TelegramError("Telegram request cancelled.")
            response = self._loop.run_until_complete(self._post(method, parameters)) if (
                self._loop is not None
            ) else self._client.post(self._base_url + method, json=parameters)
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
        if self._owns_client and self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.run_until_complete(self._client.aclose())
            finally:
                self._loop.close()


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
        progress: Queue[tuple[threading.Event, str]] | None = None,
    ) -> None:
        self.client, self.state, self.bot_id = client, state, bot_id
        self.reply, self.clock, self.report = reply, clock, report
        self.stop = stop or threading.Event()
        self._reply_worker: threading.Thread | None = None
        self.progress = progress

    def _claim_request(self, update: dict[str, Any]) -> _ReplyTask | None:
        """Claim before any reply work, retaining owner and freshness boundaries."""
        update_id = update.get("update_id")
        if type(update_id) is not int or not self.state.claim(
            self.bot_id, update_id, now=self.clock(),
        ):
            return None
        incoming = private_message(update, self.clock())
        owner = self.state.owner(self.bot_id)
        if incoming is None or owner is None or incoming[0] != owner:
            self.state.finish(self.bot_id, update_id, "ignored")
            return None
        return _ReplyTask(
            update_id, owner, incoming[1], update["message"]["date"],
            _ReplyCancellation(self.stop),
        )

    def process(self, update: dict[str, Any]) -> None:
        """Consume once, then authorize the owner before invoking the conversation."""
        request = self._claim_request(update)
        if request is None:
            return
        try:
            answer = self.reply(request.text, self.stop)
        except Exception:
            self.state.finish(self.bot_id, request.update_id, "failed")
            answer = (
                "I couldn't complete that request. It may have been partly processed; "
                "ask what I remember before repeating a memory update."
            )
        if self.stop.is_set() or not answer or self.state.owner(self.bot_id) != request.owner:
            self.state.finish(self.bot_id, request.update_id, "cancelled")
            return
        try:
            self.client.send(request.owner, answer, self.stop)
        except TelegramError as error:
            self.state.finish(self.bot_id, request.update_id, "reply_failed")
            self.report(f"Reply delivery failed: {error} The request will not be run again.")
            return
        self.state.finish(self.bot_id, request.update_id, "done")

    def poll(self, *, timeout: int = 20) -> list[dict[str, Any]]:
        """Retry read-only polls with bounded, interruptible delays."""
        self.state.recover_idle_offset(self.bot_id, self.clock())
        for attempt in range(6):
            if self.stop.is_set():
                return []
            try:
                return self.client.updates(self.state.offset(self.bot_id), timeout=timeout)
            except TelegramError as error:
                if self.stop.is_set():
                    return []
                if not error.transient or attempt == 5:
                    raise
                self.report(f"{error} Retrying the connection.")
                if self.stop.wait(error.retry_after or min(2 ** attempt, 30)):
                    return []
        return []

    def run(self) -> None:
        """Keep polling for controls while exactly one worker generates replies.

        The polling thread alone owns Telegram's HTTP event loop and ChatState's
        SQLite handle. A single daemon worker serializes all conversation access.
        """
        if self._reply_worker is not None and self._reply_worker.is_alive():
            raise TelegramError("A previous chat reply is still stopping.")
        owner = self.state.owner(self.bot_id)
        if owner is None:
            raise TelegramError("Pair your account locally first: ciel chat pair telegram")
        pending: deque[_ReplyTask] = deque()
        controls: deque[_ReplyTask] = deque()
        work: Queue[_ReplyTask | None] = Queue(maxsize=1)
        worker_stop = threading.Event()
        active: _ReplyTask | None = None

        def generate() -> None:
            while not worker_stop.is_set():
                request = work.get()
                if request is None:
                    return
                try:
                    if not request.cancel.is_set():
                        request.answer = self.reply(request.text, request.cancel)
                except Exception:
                    request.answer = (
                        "I couldn't complete that request. It may have been partly processed; "
                        "ask what I remember before repeating a memory update."
                    )
                finally:
                    request.done.set()

        def start_next() -> None:
            nonlocal active
            while active is None and (controls or pending) and not self.stop.is_set():
                request = controls.popleft() if controls else pending.popleft()
                if (
                    self.state.owner(self.bot_id) != request.owner or request.owner != owner
                    or not -30 <= self.clock() - request.sent_at <= MAX_MESSAGE_AGE_SECONDS
                ):
                    self.state.finish(self.bot_id, request.update_id, "ignored")
                    continue
                active = request
                work.put_nowait(request)

        def enqueue(update: dict[str, Any]) -> None:
            request = self._claim_request(update)
            if request is None:
                return
            if request.owner != owner:
                self.state.finish(self.bot_id, request.update_id, "ignored")
                return
            # Keep setup/pairing imports light; classify only authenticated text.
            from ciel.controller import conversation_control

            text = re.sub(r"^/(?:local|fast|strong|deep)\s+", "", request.text, flags=re.I)
            if conversation_control(text) is not None:
                if active is not None:
                    active.cancel.set()
                while pending:
                    dropped = pending.popleft()
                    dropped.cancel.set()
                    self.state.finish(self.bot_id, dropped.update_id, "cancelled")
                if len(pending) + len(controls) >= MAX_PENDING_REQUESTS:
                    dropped = pending.pop() if pending else controls.popleft()
                    self.state.finish(self.bot_id, dropped.update_id, "cancelled")
                controls.append(request)
            elif len(pending) + len(controls) < MAX_PENDING_REQUESTS:
                pending.append(request)
            else:
                self.state.finish(self.bot_id, request.update_id, "queue_full")
                if not self.stop.is_set() and self.state.owner(self.bot_id) == request.owner:
                    try:
                        self.client.send(
                            request.owner,
                            "I'm still working and the waiting queue is full. "
                            "Please send that again after I reply.",
                            request.cancel,
                        )
                    except TelegramError as error:
                        self.report(f"Reply delivery failed: {error}")

        def send_progress() -> None:
            if self.progress is None:
                return
            while True:
                try:
                    event, message = self.progress.get_nowait()
                except Empty:
                    return
                if (
                    active is None or event is not active.cancel or active.progress_sent
                    or active.done.is_set() or active.cancel.is_set()
                    or self.state.owner(self.bot_id) != active.owner
                    or not isinstance(message, str) or not message.strip()
                ):
                    continue
                active.progress_sent = True
                try:
                    self.client.send(active.owner, message.strip()[:300], active.cancel)
                except TelegramError as error:
                    self.report(f"Progress delivery failed: {error}")

        def finish(request: _ReplyTask) -> None:
            if (
                request.cancel.is_set() or not request.answer
                or self.state.owner(self.bot_id) != request.owner
            ):
                self.state.finish(self.bot_id, request.update_id, "cancelled")
                return
            try:
                self.client.send(request.owner, request.answer, request.cancel)
            except TelegramError as error:
                self.state.finish(self.bot_id, request.update_id, "reply_failed")
                self.report(f"Reply delivery failed: {error} The request will not be run again.")
                return
            self.state.finish(self.bot_id, request.update_id, "done")

        worker = threading.Thread(target=generate, name="ciel-telegram-reply", daemon=True)
        self._reply_worker = worker
        worker.start()
        try:
            while not self.stop.is_set():
                if self.state.owner(self.bot_id) != owner:
                    raise TelegramError(
                        "Telegram pairing changed. Restarting chat.", transient=True,
                    )
                if active is not None and active.done.is_set():
                    finish(active)
                    active = None
                start_next()
                for update in self.poll(timeout=1 if active is not None else 20):
                    if self.stop.is_set():
                        break
                    enqueue(update)
                    start_next()
                send_progress()
        finally:
            worker_stop.set()
            if active is not None:
                active.cancel.set()
                self.state.finish(self.bot_id, active.update_id, "cancelled")
            for request in (*controls, *pending):
                request.cancel.set()
                self.state.finish(self.bot_id, request.update_id, "cancelled")
            with suppress(Full):
                work.put_nowait(None)
            worker.join(REPLY_SHUTDOWN_SECONDS)
            if worker.is_alive() and not self.stop.is_set():
                # Starting another reply worker in this process could race memory writes.
                raise TelegramError(
                    "The current reply did not stop. The chat service must restart.",
                )

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
                        "Paired with Ciel. Chat resumes automatically on your PC; "
                        "you can talk here shortly.",
                        self.stop,
                    )
                except TelegramError:
                    self.report(
                        "Pairing saved, but the Telegram confirmation could not be delivered.",
                    )
                return user_id
        return None
