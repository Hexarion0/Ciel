"""Session-only local device pairing; bearer tokens are retained only as hashes."""

import hashlib
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class DeviceSession:
    """A paired owner device and its session-only CSRF secret."""

    device_id: str
    owner_key: str
    expires_at: float
    generation: int = 0
    csrf: str = field(default="", repr=False)
    token: str = field(default="", repr=False)


class DeviceAuth:
    """Single-use codes expire after five minutes; devices expire after twelve hours."""

    def __init__(self, owner_key: str, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.owner_key, self.clock = owner_key, clock
        self._lock = threading.RLock()
        self._code = ""
        self._code_expiry = 0.0
        self._devices: dict[str, DeviceSession] = {}
        self._attempts: deque[tuple[float, str]] = deque(maxlen=500)

    def issue_code(self) -> str:
        """Issue a short-lived code for display only on the trusted PC."""
        with self._lock:
            self._code = f"{secrets.randbelow(100000000):08d}"
            self._code_expiry = self.clock() + 300
            return self._code

    def pair(self, code: str, *, source: str = "local") -> DeviceSession:
        """Exchange one code for an unguessable device token, subject to rate limits."""
        with self._lock:
            now = self.clock()
            while self._attempts and self._attempts[0][0] <= now - 60:
                self._attempts.popleft()
            if len(self._attempts) >= 50 or sum(s == source for _, s in self._attempts) >= 5:
                raise ValueError("Too many pairing attempts. Wait one minute.")
            if (
                not self._code
                or now >= self._code_expiry
                or not secrets.compare_digest(self._code, code)
            ):
                self._attempts.append((now, source))
                raise ValueError("The pairing code is incorrect or expired.")
            self._code = ""
            return self.create_session()

    def create_session(self) -> DeviceSession:
        """Allocate a browser identity, also used by explicit open home-network mode."""
        with self._lock:
            token = secrets.token_urlsafe(32)
            device = DeviceSession(
                secrets.token_hex(16), self.owner_key, self.clock() + 43200,
                csrf=secrets.token_urlsafe(32)
            )
            self._devices[hashlib.sha256(token.encode()).hexdigest()] = device
            return replace(device, token=token)

    def authenticate(self, cookie: str) -> DeviceSession | None:
        """Authenticate by token digest, never by browser-supplied owner identity."""
        if not isinstance(cookie, str) or len(cookie) > 128:
            return None
        with self._lock:
            device = self._devices.get(hashlib.sha256(cookie.encode()).hexdigest())
            return device if device and device.expires_at > self.clock() else None

    def revoke(self, device_id: str) -> None:
        """Immediately revoke a paired device; in-flight guards recheck authentication."""
        with self._lock:
            self._devices = {h: d for h, d in self._devices.items() if d.device_id != device_id}

    def clear(self) -> None:
        """End every device session on shutdown."""
        with self._lock:
            self._devices.clear()
            self._code = ""
