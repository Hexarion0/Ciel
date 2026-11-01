"""Authenticated browser request, upload, and playback routes for the web facade."""

from __future__ import annotations

import ipaddress
import json
import re
import secrets
import threading
from http.cookies import SimpleCookie
from urllib.parse import parse_qs, urlsplit

from ciel.webapp.audio import decode_voice_wav
from ciel.webapp.auth import (
    BROWSER_SESSION_LIFETIME,
    PAIRED_DEVICE_LIFETIME,
    DeviceAuth,
    DeviceSession,
)
from ciel.webapp.images import normalize_upload
from ciel.webapp.media import MediaStore
from ciel.webapp.requests import WebRequest
from ciel.webapp.speech import BrowserSpeech


class WebChannels:
    """Assign identities on the server and recheck them at asynchronous boundaries."""

    def __init__(self, ui) -> None:
        self.ui = ui
        self.owner_key = secrets.token_hex(16)
        self.auth = DeviceAuth(self.owner_key)
        self.media = MediaStore()
        self.speech = None
        self.submit = None
        self.local = DeviceSession("pc", self.owner_key, float("inf"))
        self._upload_lock = threading.RLock()
        self._attachments: dict[tuple[str, str, int], list[str]] = {}
        self._cookie_renewed: dict[str, float] = {}

    def _cookie_renewal_due(self, device_id: str, path: str) -> bool:
        """Renew on each app launch, and about daily while an open app keeps polling."""
        if path not in {"/api/session", "/api/state"}:
            return False
        now = self.auth.clock()
        with self._upload_lock:
            last = self._cookie_renewed.get(device_id)
            if path == "/api/state" and last is not None and now - last < 24 * 3600:
                return False
            self._cookie_renewed[device_id] = now
            return True

    @staticmethod
    def paired_cookie(token: str) -> str:
        """Paired phones use HTTPS only; the cookie lasts as long as the idle window."""
        return (f"ciel_device={token}; Path=/; Secure; HttpOnly; SameSite=Strict; "
                f"Max-Age={PAIRED_DEVICE_LIFETIME}")

    def configure(self, submit, tts) -> None:
        """Connect existing listener and TTS after the desktop runtime starts."""
        self.submit = submit
        self.speech = BrowserSpeech(tts)

    def authorize(self, handler) -> bool:
        """Allow public shell/pairing, but require a current device for private APIs."""
        path = urlsplit(handler.path).path
        if ipaddress.ip_address(handler.client_address[0]).is_loopback:
            handler.device = self.local
            handler.device_token = ""
            return True
        cookie = SimpleCookie()
        try:
            cookie.load(handler.headers.get("Cookie", ""))
            token = cookie["ciel_device"].value if "ciel_device" in cookie else ""
        except Exception:
            token = ""
        handler.device_token = token
        handler.device = self.auth.authenticate(token)
        if (
            handler.device is None and self.ui.insecure_lan
            and ipaddress.ip_address(handler.client_address[0]).is_private
        ):
            device = self.auth.create_session()
            handler.device, handler.device_token = device, device.token
            handler.response_headers = {
                "Set-Cookie": f"ciel_device={device.token}; Path=/; HttpOnly; "
                f"SameSite=Strict; Max-Age={BROWSER_SESSION_LIFETIME}"
            }
        elif (handler.device is not None and handler.device.paired
              and self._cookie_renewal_due(handler.device.device_id, path)):
            # Keep the browser cookie in step with the sliding server-side idle window.
            handler.response_headers = {"Set-Cookie": self.paired_cookie(token)}
        public = (path in {"/", "/sw.js", "/api/session", "/api/pair"}
                  or path.startswith("/assets/"))
        if handler.device is None and not public:
            handler._json(401, {"error": "Pair this device with the code on your PC."})
            return False
        if (
            handler.command == "POST"
            and path != "/api/pair"
            and handler.device
            and not secrets.compare_digest(
                handler.headers.get("X-CSRF-Token", ""), handler.device.csrf
            )
        ):
            handler._json(403, {"error": "Refresh this device session and try again."})
            return False
        return True

    def revision(self) -> int:
        """Return the current selection revision, including new-chat activations."""
        return self.ui.selection.current().revision if self.ui.selection else 0

    def valid(self, request: WebRequest, token: str) -> bool:
        """Recheck revocation and conversation selection during STT, inference and synthesis."""
        device = self.local if request.device_id == "pc" else self.auth.authenticate(token)
        return bool(
            device
            and device.device_id == request.device_id
            and self.revision() == request.revision
            and (
                not self.ui.selection
                or self.ui.selection.current().handle == request.conversation_handle
            )
        )

    @staticmethod
    def read(handler, maximum: int) -> bytes:
        """Bound all request bodies before allocation, rejecting transfer encodings."""
        length = int(handler.headers.get("Content-Length", "0"))
        if handler.headers.get("Transfer-Encoding") or not 0 < length <= maximum:
            raise ValueError("This upload is too large or empty.")
        data = handler.rfile.read(length)
        if len(data) != length:
            raise ValueError("The upload was incomplete. Try again.")
        return data

    @classmethod
    def json(cls, handler, maximum: int = 16384) -> dict:
        """Read an explicit JSON object, with a route-specific size bound."""
        if handler.headers.get("Content-Type", "").split(";")[0] != "application/json":
            raise ValueError("Send a JSON request.")
        value = json.loads(cls.read(handler, maximum))
        if not isinstance(value, dict):
            raise ValueError("Invalid request.")
        return value

    def request(self, handler, data: dict, *, kind: str = "text") -> WebRequest:
        """Bind client input to authenticated identity and the expected active selection."""
        active = self.ui.selection.current() if self.ui.selection else None
        if active and (
            data.get("revision") != active.revision
            or data.get("conversation_handle") != active.handle
        ):
            raise ValueError("The active conversation changed. Review your draft and try again.")
        text = data.get("text", "")
        handles = data.get("attachment_ids", [])
        request_id = data.get("request_id", secrets.token_hex(16))
        if (
            not isinstance(text, str)
            or len(text) > 4000
            or not isinstance(handles, list)
            or len(handles) > 2
            or not all(isinstance(h, str) and len(h) < 64 for h in handles)
            or not isinstance(request_id, str)
            or not re.fullmatch("[a-zA-Z0-9_-]{16,64}", request_id)
        ):
            raise ValueError("Invalid message or attachments.")
        request = WebRequest(
            request_id,
            handler.device.device_id,
            active.handle if active else "",
            active.revision if active else 0,
            kind,
            text.strip(),
            tuple(handles),
            handler.device.owner_key,
        )
        for handle in request.attachment_ids:
            self.media.get(
                handle,
                owner_key=request.owner_key,
                device_id=request.device_id,
                request_id=request.request_id,
                revision=request.revision,
            )
        return request

    def get(self, handler) -> bool:
        """Handle private session and one-device playback reads."""
        if self.ui.admin is not None and self.ui.admin.get(handler):
            return True
        parsed = urlsplit(handler.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/api/session":
            device = handler.device
            handler._json(
                200,
                {
                    "paired": device is not None,
                    "csrf": device.csrf if device else "",
                    "device_id": device.device_id if device else "",
                    "browser_audio": self.submit is not None,
                    "pc_capture": self.ui.pc_capture,
                    "active": self.ui.snapshot().get("active") if device else None,
                },
            )
            return True
        if parsed.path == "/api/playback":
            jobs = (
                self.speech.list(handler.device.device_id, self.revision()) if self.speech else []
            )
            handler._json(200, {"jobs": jobs})
            return True
        if parsed.path.startswith("/api/audio/") and self.speech:
            try:
                client = query.get("client", [""])[0]
                if not re.fullmatch("[a-zA-Z0-9_-]{16,64}", client):
                    raise ValueError()
                data = self.speech.chunk(
                    parsed.path.rsplit("/", 1)[-1],
                    handler.device.device_id,
                    self.revision(),
                    client,
                )
                handler._send(200 if data else 204, data or b"", "audio/wav")
            except EOFError as error:
                handler._json(410, {"error": str(error)})
            except ValueError:
                handler._json(409, {"error": "This playback ended or is playing in another tab."})
            return True
        return False

    def post(self, handler) -> bool:
        """Handle authenticated pairing, temporary uploads and structured input."""
        if self.ui.admin is not None and self.ui.admin.post(handler):
            return True
        path = urlsplit(handler.path).path
        if path not in {
            "/api/pair",
            "/api/pair-code",
            "/api/revoke",
            "/api/images",
            "/api/images/remove",
            "/api/voice",
            "/api/replay",
        }:
            return False
        try:
            if path == "/api/pair-code":
                self.json(handler, 1024)
                if handler.device.device_id != "pc":
                    raise ValueError("Pairing codes are available only on the PC.")
                handler._json(200, {"code": self.auth.issue_code(), "expires_in": 300})
            elif path == "/api/pair":
                data = self.json(handler, 1024)
                code = data.get("code", "")
                if not isinstance(code, str) or len(code) > 16:
                    raise ValueError("Invalid pairing code.")
                device = self.auth.pair(code, source=handler.client_address[0])
                handler.response_headers = {"Set-Cookie": self.paired_cookie(device.token)}
                handler._json(200, {"paired": True, "csrf": device.csrf})
            elif path == "/api/revoke":
                data = self.json(handler, 1024)
                if data.get("all") is True:
                    if handler.device.device_id != "pc":
                        raise ValueError("Only the PC can disconnect all phones.")
                    removed = self.auth.revoke_all()
                    for identity in removed:
                        self.media.invalidate(device_id=identity)
                    handler._json(200, {"revoked": True, "devices": len(removed)})
                    return True
                # A phone may disconnect itself. Only the local PC may revoke another device.
                identity = data.get("device_id", handler.device.device_id)
                if handler.device.device_id != "pc" and identity != handler.device.device_id:
                    raise ValueError("A device can disconnect only its own session.")
                self.auth.revoke(identity)
                self.media.invalidate(device_id=identity)
                handler._json(200, {"revoked": True})
            elif path == "/api/images/remove":
                request = self.request(handler, self.json(handler, 2048))
                for handle in request.attachment_ids:
                    self.media.discard(
                        handle,
                        owner_key=request.owner_key,
                        device_id=request.device_id,
                        request_id=request.request_id,
                        revision=request.revision,
                    )
                handler._json(200, {"removed": True})
            elif path == "/api/images":
                raw = self.read(handler, 10 * 1024**2)
                data = {
                    "request_id": handler.headers.get("X-Request-ID", ""),
                    "revision": int(handler.headers.get("X-Revision", "-1")),
                    "conversation_handle": handler.headers.get("X-Conversation", ""),
                }
                request = self.request(handler, data)
                normalized = normalize_upload(raw, handler.headers.get("Content-Type", ""))
                key = (request.device_id, request.request_id, request.revision)
                with self._upload_lock:
                    handles = self._attachments.setdefault(key, [])
                    live = []
                    for handle in handles:
                        try:
                            self.media.get(
                                handle,
                                owner_key=request.owner_key,
                                device_id=request.device_id,
                                request_id=request.request_id,
                                revision=request.revision,
                            )
                            live.append(handle)
                        except ValueError:
                            pass
                    if len(live) >= 2:
                        raise ValueError("Attach at most two images per message.")
                    handle = self.media.put(
                        normalized,
                        owner_key=request.owner_key,
                        device_id=request.device_id,
                        request_id=request.request_id,
                        revision=request.revision,
                        content_type="image/png",
                        ttl=900,
                    )
                    self._attachments[key] = [*live, handle]
                    if len(self._attachments) > 256:
                        self._attachments.pop(next(iter(self._attachments)))
                handler._json(200, {"handle": handle, "expires_in": 900})
            elif path == "/api/voice":
                if not self.submit:
                    raise ValueError("Browser voice is unavailable in this session.")
                raw = self.read(handler, 1_920_256)
                audio = decode_voice_wav(raw)
                data = {
                    "request_id": handler.headers.get("X-Request-ID", ""),
                    "revision": int(handler.headers.get("X-Revision", "-1")),
                    "conversation_handle": handler.headers.get("X-Conversation", ""),
                }
                request = self.request(handler, data, kind="voice")
                accepted = self.submit(
                    request, audio, lambda: self.valid(request, handler.device_token)
                )
                handler._json(200 if accepted else 503, {"accepted": bool(accepted)})
            elif path == "/api/replay":
                data = self.json(handler, 1024)
                handle = (
                    self.speech.replay(
                        data.get("job", ""), handler.device.device_id, self.revision()
                    )
                    if self.speech
                    else ""
                )
                handler._json(200, {"job": handle})
        except (ValueError, TypeError, KeyError, UnicodeError):
            handler._json(
                400,
                {
                    "error": "Could not accept this request. Check the active chat, "
                    "pairing code, or attachment and try again."
                },
            )
        return True

    def cancel(self) -> None:
        """Discard temporary media when active work is canceled or switched."""
        self.media.invalidate()
        self._attachments.clear()
        if self.speech:
            self.speech.cancel()
