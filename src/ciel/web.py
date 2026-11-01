"""PC-local web interface backed by the existing desktop conversation runtime."""

from __future__ import annotations

import ipaddress
import json
import logging
import ssl
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ciel.conversation import strip_internal_reply_notes
from ciel.logging import SecretMaskingFilter
from ciel.memory.models import normalize_sources
from ciel.webapp.channels import WebChannels
from ciel.webapp.http import load_web_asset
from ciel.webapp.server import BrowserHTTPServer
from ciel.webapp.tls import server_tls


class DesktopWebUI:
    """Bounded live chat/status view; requests are submitted to the voice worker."""

    def __init__(self, owner: str, wake_phrase: str, *, voice_enabled: bool = True) -> None:
        self.owner = owner
        self.wake_phrase = wake_phrase
        self.voice_enabled = voice_enabled
        self.pc_capture = True
        self.insecure_lan = False
        self._lock = threading.RLock()
        self._messages: deque[dict[str, Any]] = deque(maxlen=200)
        self._activity: deque[dict[str, Any]] = deque(maxlen=30)
        self._next_id = 0
        self._cursor = 0
        self._reset_cursor = 0
        self._reply: dict[str, Any] | None = None
        self._sentence_base = ""
        self._muted = False
        self._state = "listening_wake"
        self._speaking = False
        self._provider = ""
        self._observation_status: Callable[[], str] = lambda: "Screen observation: off."
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._local_server: ThreadingHTTPServer | None = None
        self._local_thread: threading.Thread | None = None
        self.catalog = None
        self.selection = None
        self.channels = WebChannels(self)
        self.admin = None
        self.phone_url = ''
        self._allowed_hosts = {'127.0.0.1', 'localhost'}
        self._lan_server = None
        self._lan_thread = None
        self._handler = None

    def configure_history(self, catalog: Any, selection: Any) -> None:
        """Attach owner-scoped history without creating another inference runtime."""
        self.catalog, self.selection = catalog, selection

    def replace_history(self, turns: Iterable[Any]) -> None:
        """Replace transient browser state after explicit runtime activation."""
        with self._lock:
            self.finish_reply()
            self._messages.clear()
            self.load_history(turns)
            self._cursor += 1
            self._reset_cursor = self._cursor

    def _changed(self, message: dict[str, Any]) -> None:
        self._cursor += 1
        message['version'] = self._cursor

    def _message(self, role: str, text: str, **extra: Any) -> dict[str, Any]:
        self._next_id += 1
        message = {"id": self._next_id, "role": role,
                   "text": SecretMaskingFilter._redact(text), "time": time.time(), **extra}
        self._messages.append(message)
        self._changed(message)
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
            self._changed(self._reply)
            if interrupted:
                self._reply["interrupted"] = True
                self.finish_reply()

    def finish_reply(self) -> None:
        with self._lock:
            if self._reply is not None:
                self._reply["live"] = False
                self._changed(self._reply)
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
                self._changed(self._messages[-1])

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

    def set_observation_status(self, status: Callable[[], str]) -> None:
        """Show the actual session-only observation state, including automatic suppression."""
        with self._lock:
            self._observation_status = status

    def snapshot(self, *, after: int | None = None) -> dict[str, Any]:
        with self._lock:
            reset = after is None or after < self._reset_cursor or after > self._cursor
            return {"owner": self.owner, "wake_phrase": self.wake_phrase,
                    'cursor': self._cursor, 'reset': reset,
                    'retained_ids': [message['id'] for message in self._messages],
                    "active": asdict(self.selection.current()) if self.selection else None,
                    "voice_enabled": self.voice_enabled, "pc_capture": self.pc_capture,
                    "muted": self._muted,
                    "state": "speaking" if self._speaking else self._state,
                    "provider": self._provider,
                    "observation": self._observation_status(),
                    "messages": [
                        {**message, **({"sources": normalize_sources(message["sources"])}
                                      if "sources" in message else {})}
                        for message in self._messages
                        if reset or message.get('version', 0) > after
                    ],
                    "activity": [dict(event) for event in self._activity]}

    def start(self, submit: Callable[[str], bool], *, port: int = 8765,
              host: str = '127.0.0.1', certificate: str | None = None,
              key: str | None = None) -> str:
        """Serve loopback by default, with optional private-network access."""
        ui = self
        tls = (server_tls(host, certificate, key, insecure_lan=True) if self.insecure_lan
               else server_tls(host, certificate, key))
        scheme = 'https' if tls else 'http'
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
                for name, value in getattr(self, 'response_headers', {}).items():
                    self.send_header(name, value)
                self.response_headers = {}
                self.send_header("Content-Security-Policy", "default-src 'self'; "
                                 "script-src 'self' 'unsafe-inline'; style-src 'self' "
                                 "'unsafe-inline'; img-src 'self' blob: data:; "
                                 "frame-ancestors 'none'; "
                                 "base-uri 'none'")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, status: int, data: Any) -> None:
                self._send(status, json.dumps(data).encode(), "application/json; charset=utf-8")

            def _local_request(self) -> bool:
                port = self.server.server_address[1]
                allowed = {f"{name}:{port}" for name in ui._allowed_hosts}
                request_host = self.headers.get("Host", "")
                origin = self.headers.get("Origin")
                request_scheme = 'https' if isinstance(self.connection, ssl.SSLSocket) else 'http'
                if (
                    request_host not in allowed
                    or origin is not None and origin != f"{request_scheme}://{request_host}"
                    or request_scheme == 'https' and self.command == 'POST' and origin is None
                    or self.headers.get("Sec-Fetch-Site") == "cross-site"
                ):
                    self._json(403, {"error": "This interface accepts PC-local requests only."})
                    return False
                if self.server.tls_context is not None and request_scheme != 'https':
                    if self.command in {'GET', 'HEAD'}:
                        self.response_headers = {'Location': f'https://{request_host}/'}
                        self._send(307, b'', 'text/plain; charset=utf-8')
                    else:
                        self._json(400, {'error': 'Open this connection with https://.'})
                    return False
                return ui.channels.authorize(self)

            def do_HEAD(self) -> None:
                if self._local_request():
                    self._send(405, b'', 'text/plain; charset=utf-8')

            def do_GET(self) -> None:
                if not self._local_request():
                    return
                parsed = urlsplit(self.path)
                if ui.channels.get(self):
                    return
                if parsed.path.startswith('/api/conversations') and ui.catalog:
                    try:
                        query = parse_qs(parsed.query)
                        paging = {key: int(query[key][0]) for key in ('before', 'limit')
                                  if key in query}
                        result = (ui.catalog.list(**paging) if parsed.path == '/api/conversations'
                                  else ui.catalog.read(parsed.path.rsplit('/', 1)[-1], **paging))
                        self._json(200, result)
                    except (ValueError, TypeError):
                        self._json(400, {'error': 'This history page is unavailable.'})
                    return
                if self.path == "/":
                    self._send(200, document, "text/html; charset=utf-8")
                elif parsed.path == "/api/state":
                    try:
                        after = parse_qs(parsed.query).get('after', [None])[0]
                        cursor = int(after) if after is not None else None
                        if cursor is not None and cursor < 0:
                            raise ValueError()
                        self._json(200, ui.snapshot(after=cursor))
                    except ValueError:
                        self._json(400, {'error': 'Invalid message cursor.'})
                elif (asset := load_web_asset(self.path)) is not None:
                    self._send(200, *asset)
                else:
                    self._json(404, {"error": "Not found."})

            def do_POST(self) -> None:
                if not self._local_request():
                    return
                if ui.channels.post(self):
                    return
                if self.path == '/api/conversations/activate' and ui.selection:
                    try:
                        length = int(self.headers.get('Content-Length', '0'))
                        if not 0 < length <= 1024:
                            raise ValueError()
                        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                            raise ValueError()
                        data = json.loads(self.rfile.read(length))
                        handle = data.get('handle')
                        if handle is not None and (not isinstance(handle, str) or len(handle) > 64):
                            raise ValueError()
                        active = ui.selection.activate(handle, data.get('expected_revision'))
                        self._json(200, asdict(active))
                    except (ValueError, TypeError, AttributeError):
                        self._json(409, {'error': 'The active chat changed. Refresh and retry.'})
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
                    routing_prefix = text.strip().split()[0].casefold() in {
                        '/fast', '/strong', '/deep',
                    }
                    if ui.channels.submit and (not text.strip().startswith('/') or routing_prefix):
                        request = ui.channels.request(self, data)
                        accepted = ui.channels.submit(
                            request, None,
                            lambda: ui.channels.valid(request, self.device_token),
                        )
                    else:
                        accepted = submit(text.strip())
                except Exception:
                    ui.feedback("The request could not be submitted.", error=True)
                    self._json(503, {"error": "The assistant could not accept this request."})
                    return
                self._json(200 if accepted else 503, {"accepted": bool(accepted)})

        self._allowed_hosts.add(host)
        self._handler = Handler
        self._server = BrowserHTTPServer((host, port), Handler, tls_context=tls)
        if tls:
            logger = logging.getLogger('ciel.web')
            logger.info('Phone pairing code (expires in 5 minutes): %s',
                        ui.channels.auth.issue_code())
        if host != '127.0.0.1':
            try:
                self._local_server = BrowserHTTPServer(
                    ('127.0.0.1', self._server.server_address[1]), Handler,
                )
                self._local_server.daemon_threads = True
            except OSError:
                self._server.server_close()
                raise
            self._local_thread = threading.Thread(
                target=self._local_server.serve_forever, name='ciel-web-pc', daemon=True,
            )
            self._local_thread.start()
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="ciel-desktop-web", daemon=True,
        )
        self._thread.start()
        if (tls or self.insecure_lan) and not ipaddress.ip_address(host).is_loopback:
            self.phone_url = f'{scheme}://{host}:{self._server.server_address[1]}'
        return f"{scheme}://{host}:{self._server.server_address[1]}"

    def enable_lan(self, host: str, certificate: Path, key: Path) -> str:
        """Add a private HTTPS listener while keeping the PC-local app available."""
        with self._lock:
            if self.phone_url:
                if self.phone_url.startswith(f'https://{host}:'):
                    return self.phone_url
                raise ValueError('Restart Ciel before changing the active network address.')
            if self._server is None or self._handler is None:
                raise ValueError('Start the local web server first.')
            tls = server_tls(host, str(certificate), str(key))
            port = self._server.server_address[1]
            server = BrowserHTTPServer((host, port), self._handler, tls_context=tls)
            self._allowed_hosts.add(host)
            self._lan_server = server
            self._lan_thread = threading.Thread(target=server.serve_forever,
                                                name='ciel-web-wifi', daemon=True)
            self._lan_thread.start()
            self.phone_url = f'https://{host}:{port}'
            return self.phone_url

    def close(self) -> None:
        self.channels.cancel()
        self.channels.auth.clear()
        if self.admin is not None:
            self.admin.network.close()
        if self._lan_server is not None:
            self._lan_server.shutdown()
            self._lan_server.server_close()
        if self._lan_thread is not None:
            self._lan_thread.join(timeout=1)
        if self._local_server is not None:
            self._local_server.shutdown()
            self._local_server.server_close()
        if self._local_thread is not None:
            self._local_thread.join(timeout=1)
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
