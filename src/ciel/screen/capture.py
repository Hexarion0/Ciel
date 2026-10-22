"""Bounded Hyprland capture with an explicit logical desktop coordinate frame."""

from __future__ import annotations

import io
import json
import math
import os
import selectors
import signal
import stat
import subprocess
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_CAPTURE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000
MAX_IMAGE_SIDE = 32_768
MAX_MONITORS = 16


class ScreenCaptureError(RuntimeError):
    """The current desktop could not safely produce a new capture."""


@dataclass(frozen=True)
class Monitor:
    """Output placement in global logical coordinates; dimensions are native pixels."""

    name: str
    x: float
    y: float
    width: int
    height: int
    scale: float = 1.0
    focused: bool = False
    transform: int = 0

    @property
    def logical_width(self) -> float:
        """Return displayed width after output rotation and scaling."""
        return (self.height if self.transform % 2 else self.width) / self.scale

    @property
    def logical_height(self) -> float:
        """Return displayed height after output rotation and scaling."""
        return (self.width if self.transform % 2 else self.height) / self.scale

    def contains(self, x: float, y: float) -> bool:
        """Check a global logical point against the output's half-open rectangle."""
        return self.x <= x < self.x + self.logical_width and (
            self.y <= y < self.y + self.logical_height
        )


@dataclass(frozen=True)
class Cursor:
    """Separately sampled pointer metadata; unstable samples cannot identify a target."""

    x: float
    y: float
    sampled_at: float
    stable: bool
    monitor: str | None = None
    image_x: float | None = None
    image_y: float | None = None


@dataclass(frozen=True)
class ScreenImage:
    """PNG bytes plus the transform from global logical positions into image pixels."""

    name: str
    data: bytes = field(repr=False)
    width: int
    height: int
    mime_type: str = "image/png"
    origin_x: float = 0.0
    origin_y: float = 0.0
    logical_width: float | None = None
    logical_height: float | None = None

    def point_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Map a global logical point into this image, including a crop's offset."""
        width = self.logical_width if self.logical_width is not None else self.width
        height = self.logical_height if self.logical_height is not None else self.height
        return ((x - self.origin_x) * self.width / width,
                (y - self.origin_y) * self.height / height)


@dataclass(frozen=True)
class Capture:
    """One all-monitor observation; source images and crops share one expiry time."""

    id: str
    captured_at: float
    monitors: tuple[Monitor, ...]
    cursor: Cursor
    images: tuple[ScreenImage, ...]

    def image(self, name: str = "desktop") -> ScreenImage:
        """Get an image by its stable name without changing its capture time."""
        for image in self.images:
            if image.name == name:
                return image
        raise ScreenCaptureError("That image is no longer available.")

    def geometry_description(self) -> str:
        """Describe layout and pointer sampling without claiming exact live access."""
        outputs = "; ".join(
            f"{m.name}: logical ({m.x:g},{m.y:g}) "
            f"{m.logical_width:g}x{m.logical_height:g}, scale {m.scale:g}"
            + (", focused" if m.focused else "")
            for m in self.monitors
        )
        pointer = f"Pointer sampled at ({self.cursor.x:g},{self.cursor.y:g})"
        if self.cursor.stable:
            pointer += f" on {self.cursor.monitor or 'an output gap'}."
        else:
            pointer += "; it moved during capture, so its exact target is uncertain."
        return f"Capture time {self.captured_at:.3f}. {outputs}. {pointer}"


def decode_image(data: bytes) -> Image.Image:
    """Validate encoded byte and decoded pixel bounds before allocating RGB pixels."""
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ScreenCaptureError("The screen image exceeded the supported size.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if (image.format != "PNG" or not 0 < width <= MAX_IMAGE_SIDE
                    or not 0 < height <= MAX_IMAGE_SIDE
                    or width * height > MAX_IMAGE_PIXELS):
                raise ScreenCaptureError("The screen image exceeded the supported dimensions.")
            image.load()
            return image.convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ScreenCaptureError("The desktop returned an unreadable image.") from exc


def encode_image(image: Image.Image) -> bytes:
    """Encode a bounded image as PNG without writing personal pixels to a temp file."""
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ScreenCaptureError("The desktop layout exceeded the supported image size.")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    if len(data) > MAX_IMAGE_BYTES:
        raise ScreenCaptureError("The screen image exceeded the supported size.")
    return data


class HyprlandCaptureBackend:
    """Capture all active outputs only when the caller explicitly invokes capture()."""

    def __init__(self, *, env: Mapping[str, str] | None = None, timeout: float = 8.0) -> None:
        self.env = dict(os.environ if env is None else env)
        self.timeout = max(0.1, min(float(timeout), 30.0))

    def _run(self, args: list[str], cancel: threading.Event | None,
             *, max_bytes: int = 262_144) -> bytes:
        """Read bounded stdout, poll cancellation, and always reap a stopped process."""
        if cancel is not None and cancel.is_set():
            raise ScreenCaptureError("Screen capture was cancelled.")
        try:
            process = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=self.env, start_new_session=True,
            )
        except OSError as exc:
            raise ScreenCaptureError(
                "Desktop capture is unavailable; check the logged-in Hyprland session "
                "and install grim and hyprctl."
            ) from exc
        deadline = time.monotonic() + self.timeout
        result = bytearray()
        completed = False
        try:
            assert process.stdout is not None
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    if cancel is not None and cancel.is_set():
                        raise ScreenCaptureError("Screen capture was cancelled.")
                    if time.monotonic() >= deadline:
                        raise ScreenCaptureError("The desktop capture operation timed out.")
                    for key, _ in selector.select(timeout=0.05):
                        chunk = os.read(key.fileobj.fileno(), 65_536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        result.extend(chunk)
                        if len(result) > max_bytes:
                            raise ScreenCaptureError("The desktop returned too much image data.")
                while process.poll() is None:
                    if cancel is not None and cancel.is_set():
                        raise ScreenCaptureError("Screen capture was cancelled.")
                    if time.monotonic() >= deadline:
                        raise ScreenCaptureError("The desktop capture operation timed out.")
                    time.sleep(0.01)
            if process.returncode != 0:
                raise ScreenCaptureError(
                    "Desktop capture was denied or the session is unavailable.",
                )
            completed = True
            return bytes(result)
        finally:
            if not completed or process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            if process.stdout is not None:
                process.stdout.close()

    def _json(self, command: str, cancel: threading.Event | None) -> object:
        try:
            return json.loads(self._run(["hyprctl", "-j", command], cancel))
        except (ValueError, UnicodeError) as exc:
            raise ScreenCaptureError("The desktop session returned invalid metadata.") from exc

    def _unlocked(self, cancel: threading.Event | None) -> None:
        state = self._json("locked", cancel)
        if not isinstance(state, dict) or state.get("locked") is not False:
            raise ScreenCaptureError("Unlock the desktop before asking Ciel to look at it.")

    def _ensure_session(self, cancel: threading.Event | None) -> None:
        """Find one same-user compositor for a background service missing desktop variables."""
        if self.env.get("WAYLAND_DISPLAY") and self.env.get("HYPRLAND_INSTANCE_SIGNATURE"):
            return
        runtime = Path(self.env.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
        try:
            info = runtime.lstat()
            if (not runtime.is_absolute() or not stat.S_ISDIR(info.st_mode)
                    or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
                raise ValueError("invalid runtime directory")
        except (OSError, ValueError) as exc:
            raise ScreenCaptureError(
                "Ciel cannot reach the logged-in user's desktop runtime directory.",
            ) from exc
        self.env["XDG_RUNTIME_DIR"] = str(runtime)
        instances = self._json("instances", cancel)
        candidates = []
        if isinstance(instances, list) and len(instances) <= MAX_MONITORS:
            for value in instances:
                if not isinstance(value, dict):
                    continue
                signature, display, pid = (
                    value.get("instance"), value.get("wl_socket"), value.get("pid"),
                )
                if (not isinstance(signature, str) or not signature or len(signature) > 256
                        or "/" in signature or "\x00" in signature
                        or not isinstance(display, str) or not display or len(display) > 256
                        or "/" in display or "\x00" in display
                        or not isinstance(pid, int) or pid <= 0):
                    continue
                try:
                    socket_info = (runtime / display).lstat()
                    process_info = Path(f"/proc/{pid}").stat()
                    if (socket_info.st_uid != os.getuid()
                            or not stat.S_ISSOCK(socket_info.st_mode)
                            or process_info.st_uid != os.getuid()):
                        continue
                except OSError:
                    continue
                candidates.append((signature, display))
        known_signature = self.env.get("HYPRLAND_INSTANCE_SIGNATURE")
        known_display = self.env.get("WAYLAND_DISPLAY")
        matching = [candidate for candidate in candidates
                    if (not known_signature or candidate[0] == known_signature)
                    and (not known_display or candidate[1] == known_display)]
        if len(matching) != 1:
            raise ScreenCaptureError(
                "Ciel cannot identify one logged-in Hyprland desktop; "
                "open Ciel in the intended desktop session.",
            )
        self.env["HYPRLAND_INSTANCE_SIGNATURE"], self.env["WAYLAND_DISPLAY"] = matching[0]

    def _monitors(self, cancel: threading.Event | None) -> tuple[Monitor, ...]:
        values = self._json("monitors", cancel)
        if not isinstance(values, list) or not 0 < len(values) <= MAX_MONITORS:
            raise ScreenCaptureError("No usable active desktop outputs are available.")
        monitors = []
        try:
            for value in values:
                if not isinstance(value, dict):
                    raise ValueError("invalid output")
                if value.get("disabled") or value.get("dpmsStatus") is False:
                    raise ScreenCaptureError(
                        "Wake all desktop outputs before capturing the screen.",
                    )
                name = value["name"]
                if not isinstance(name, str) or not name or len(name) > 256 or "\x00" in name:
                    raise ValueError("invalid output name")
                monitor = Monitor(
                    name=name, x=float(value["x"]), y=float(value["y"]),
                    width=int(value["width"]), height=int(value["height"]),
                    scale=float(value.get("scale", 1)), focused=value.get("focused") is True,
                    transform=int(value.get("transform", 0)),
                )
                if (not math.isfinite(monitor.x) or not math.isfinite(monitor.y)
                        or not math.isfinite(monitor.scale) or not 0.25 <= monitor.scale <= 8
                        or not 0 < monitor.width <= MAX_IMAGE_SIDE
                        or not 0 < monitor.height <= MAX_IMAGE_SIDE
                        or not 0 <= monitor.transform <= 7):
                    raise ValueError("invalid output geometry")
                monitors.append(monitor)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ScreenCaptureError("The desktop output layout could not be read.") from exc
        if len({monitor.name for monitor in monitors}) != len(monitors):
            raise ScreenCaptureError("The desktop returned duplicate output names.")
        return tuple(monitors)

    def _cursor(self, cancel: threading.Event | None) -> tuple[float, float, float]:
        value = self._json("cursorpos", cancel)
        try:
            if not isinstance(value, dict):
                raise ValueError("invalid pointer")
            x, y = float(value["x"]), float(value["y"])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("invalid pointer")
            return x, y, time.time()
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ScreenCaptureError("The desktop pointer position could not be read.") from exc

    def capture(self, cancel: threading.Event | None = None) -> Capture:
        """Obtain fresh pixels, rejecting locked, changed, incomplete, or cancelled captures."""
        self._ensure_session(cancel)
        self._unlocked(cancel)
        monitors = self._monitors(cancel)
        before = self._cursor(cancel)
        captured_at = time.time()
        left = math.floor(min(m.x for m in monitors))
        top = math.floor(min(m.y for m in monitors))
        right = math.ceil(max(m.x + m.logical_width for m in monitors))
        bottom = math.ceil(max(m.y + m.logical_height for m in monitors))
        if (right - left > MAX_IMAGE_SIDE or bottom - top > MAX_IMAGE_SIDE
                or (right - left) * (bottom - top) > MAX_IMAGE_PIXELS):
            raise ScreenCaptureError("The full desktop layout is too large to capture safely.")
        composite = Image.new("RGB", (right - left, bottom - top), "black")
        images = []
        total = 0
        for monitor in monitors:
            data = self._run(["grim", "-c", "-t", "png", "-o", monitor.name, "-"], cancel,
                             max_bytes=MAX_IMAGE_BYTES)
            image = decode_image(data)
            expected = ((monitor.height, monitor.width) if monitor.transform % 2
                        else (monitor.width, monitor.height))
            if image.size != expected:
                raise ScreenCaptureError(
                    "An output returned incomplete or changed screen pixels; please try again.",
                )
            target_size = (max(1, round(monitor.logical_width)),
                           max(1, round(monitor.logical_height)))
            composite.paste(image.resize(target_size, Image.Resampling.LANCZOS),
                            (round(monitor.x - left), round(monitor.y - top)))
            images.append(ScreenImage(
                name=monitor.name, data=data, width=image.width, height=image.height,
                origin_x=monitor.x, origin_y=monitor.y,
                logical_width=monitor.logical_width, logical_height=monitor.logical_height,
            ))
            total += len(data)
            if total > MAX_CAPTURE_BYTES:
                raise ScreenCaptureError("The desktop images exceeded the supported size.")
        after = self._cursor(cancel)
        after_monitors = self._monitors(cancel)
        before_layout = tuple(
            (m.name, m.x, m.y, m.width, m.height, m.scale, m.transform) for m in monitors
        )
        after_layout = tuple(
            (m.name, m.x, m.y, m.width, m.height, m.scale, m.transform) for m in after_monitors
        )
        if after_layout != before_layout:
            raise ScreenCaptureError("The desktop layout changed during capture; please try again.")
        monitors = after_monitors
        self._unlocked(cancel)
        if cancel is not None and cancel.is_set():
            raise ScreenCaptureError("Screen capture was cancelled.")
        desktop = ScreenImage(
            name="desktop", data=encode_image(composite), width=composite.width,
            height=composite.height, origin_x=left, origin_y=top,
            logical_width=composite.width, logical_height=composite.height,
        )
        if total + len(desktop.data) > MAX_CAPTURE_BYTES:
            raise ScreenCaptureError("The desktop images exceeded the supported size.")
        stable = math.hypot(after[0] - before[0], after[1] - before[1]) <= 2
        target = next((m for m in monitors if m.contains(after[0], after[1])), None)
        point = desktop.point_to_pixel(after[0], after[1]) if stable else (None, None)
        cursor = Cursor(after[0], after[1], after[2], stable,
                        target.name if stable and target else None, *point)
        return Capture(uuid.uuid4().hex, captured_at, monitors, cursor, (desktop, *images))
