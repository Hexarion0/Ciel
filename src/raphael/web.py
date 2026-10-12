"""PC-local web interface backed by the existing desktop conversation runtime."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from raphael.conversation import strip_internal_reply_notes
from raphael.logging import SecretMaskingFilter
from raphael.memory.models import normalize_sources


class DesktopWebUI:
    """Bounded live chat/status view; requests are submitted to the voice worker."""

    def __init__(self, owner: str, wake_phrase: str, *, voice_enabled: bool = True) -> None:
        self.owner = owner
        self.wake_phrase = wake_phrase
        self.voice_enabled = voice_enabled
        self._lock = threading.RLock()
        self._messages: deque[dict[str, Any]] = deque(maxlen=200)
        self._activity: deque[dict[str, Any]] = deque(maxlen=30)
        self._next_id = 0
        self._reply: dict[str, Any] | None = None
        self._sentence_base = ""
        self._muted = False
        self._state = "listening_wake"
        self._speaking = False
        self._provider = ""
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _message(self, role: str, text: str, **extra: Any) -> dict[str, Any]:
        self._next_id += 1
        message = {"id": self._next_id, "role": role,
                   "text": SecretMaskingFilter._redact(text), "time": time.time(), **extra}
        self._messages.append(message)
        return message

    def load_history(self, turns: Iterable[Any]) -> None:
        """Load the current conversation without altering its persistent history."""
        with self._lock:
            for turn in turns:
                if turn.role in {"user", "assistant"}:
                    text = (
                        strip_internal_reply_notes(turn.content) if turn.role == "assistant"
                        else turn.content
                    )
                    timestamp = getattr(turn, "timestamp", None)
                    extra = {"time": timestamp.timestamp()} if timestamp is not None else {}
                    if turn.role == "assistant":
                        extra["sources"] = normalize_sources(getattr(turn, "sources", []))
                    self._message(turn.role, text, **extra)

    def user_message(self, text: str) -> int:
        """Record a user message and return its identity for reply updates."""
        with self._lock:
            self.finish_reply()
            return self._message("user", text)["id"]

    def start_sentence(self) -> None:
        with self._lock:
            if self._reply is None:
                self._reply = self._message("assistant", "", live=True)
            self._sentence_base = self._reply["text"]
            self._speaking = True

    def progress(self, visible: str, *, interrupted: bool = False) -> None:
        with self._lock:
            if self._reply is None:
                return
            base = self._sentence_base
            self._reply["text"] = SecretMaskingFilter._redact(
                base + (" " if base and visible else "") + visible
            )
            if interrupted:
                self._reply["interrupted"] = True
                self.finish_reply()

    def finish_reply(self) -> None:
        with self._lock:
            if self._reply is not None:
                self._reply["live"] = False
                if not self._reply["text"]:
                    self._messages.remove(self._reply)
            self._reply = None
            self._sentence_base = ""
            self._speaking = False

    def text_reply(self, text: str) -> None:
        with self._lock:
            self.finish_reply()
            self._message("assistant", text)

    def set_reply_sources(
        self, sources: list[dict[str, str]], *, user_message_id: int | None = None,
    ) -> None:
        """Attach safe citations only while the newest message is an assistant reply."""
        with self._lock:
            if user_message_id is not None:
                current_user_id = next(
                    (message["id"] for message in reversed(self._messages)
                     if message["role"] == "user"), None,
                )
                if current_user_id != user_message_id:
                    return
            if self._messages and self._messages[-1]["role"] == "assistant":
                self._messages[-1]["sources"] = normalize_sources(sources)

    def feedback(self, text: str, *, error: bool = False) -> None:
        with self._lock:
            self._activity.append({"text": SecretMaskingFilter._redact(text),
                                   "error": error, "time": time.time()})

    def set_muted(self, muted: bool) -> None:
        with self._lock:
            self._muted = muted

    def set_state(self, state: Any) -> None:
        with self._lock:
            self._state = state.value if hasattr(state, "value") else str(state)

    def set_provider(self, provider: str, model: str) -> None:
        with self._lock:
            self._provider = f"{provider} / {model}"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"owner": self.owner, "wake_phrase": self.wake_phrase,
                    "voice_enabled": self.voice_enabled, "muted": self._muted,
                    "state": "speaking" if self._speaking else self._state,
                    "provider": self._provider,
                    "messages": [
                        {**message, **({"sources": normalize_sources(message["sources"])}
                                      if "sources" in message else {})}
                        for message in self._messages
                    ],
                    "activity": [dict(event) for event in self._activity]}

    def start(self, submit: Callable[[str], bool], *, port: int = 8765) -> str:
        """Serve on loopback only; phone/LAN access is intentionally a later stage."""
        ui = self
        document = (Path(__file__).parent / "web_assets/index.html").read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(3)

            def log_message(self, _format: str, *args: Any) -> None:
                pass

            def _send(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Content-Security-Policy", "default-src 'self'; "
                                 "script-src 'self' 'unsafe-inline'; style-src 'self' "
                                 "'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, status: int, data: Any) -> None:
                self._send(status, json.dumps(data).encode(), "application/json; charset=utf-8")

            def _local_request(self) -> bool:
                port = self.server.server_address[1]
                allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
                host = self.headers.get("Host", "")
                origin = self.headers.get("Origin")
                if (
                    host not in allowed
                    or origin is not None and origin != f"http://{host}"
                    or self.headers.get("Sec-Fetch-Site") == "cross-site"
                ):
                    self._json(403, {"error": "This interface accepts PC-local requests only."})
                    return False
                return True

            def do_GET(self) -> None:
                if not self._local_request():
                    return
                if self.path == "/":
                    self._send(200, document, "text/html; charset=utf-8")
                elif self.path == "/api/state":
                    self._json(200, ui.snapshot())
                else:
                    self._json(404, {"error": "Not found."})

            def do_POST(self) -> None:
                if not self._local_request():
                    return
                if self.path != "/api/message":
                    self._json(404, {"error": "Not found."})
                    return
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    self._json(415, {"error": "Send a JSON message."})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 16384:
                        self._json(413, {"error": "Message is too large or empty."})
                        return
                    data = json.loads(self.rfile.read(length))
                    text = data.get("text") if isinstance(data, dict) else None
                    if not isinstance(text, str) or not text.strip() or len(text) > 4000:
                        self._json(400, {"error": "Enter a message of 1–4000 characters."})
                        return
                except (ValueError, UnicodeError):
                    self._json(400, {"error": "Invalid message."})
                    return
                try:
                    accepted = submit(text.strip())
                except Exception:
                    ui.feedback("The request could not be submitted.", error=True)
                    self._json(503, {"error": "The assistant could not accept this request."})
                    return
                self._json(200 if accepted else 503, {"accepted": bool(accepted)})

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="raphael-desktop-web", daemon=True,
        )
        self._thread.start()
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1)


class WebErrorHandler(logging.Handler):
    """Expose masked runtime warnings and errors without collecting provider payloads."""

    def __init__(self, ui: DesktopWebUI) -> None:
        super().__init__(level=logging.WARNING)
        self.ui = ui
        self.addFilter(SecretMaskingFilter())

    def emit(self, record: logging.LogRecord) -> None:
        self.ui.feedback(record.getMessage(), error=True)
