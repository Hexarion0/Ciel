"""Synthetic coverage of layout transforms and bounded desktop capture failures."""

import io
import json
import os
import socket
import sys
import threading
import time

import pytest
from PIL import Image

from ciel.screen.capture import (
    HyprlandCaptureBackend,
    Monitor,
    ScreenCaptureError,
    ScreenImage,
    decode_image,
)


def png(width=200, height=100, color="red"):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def backend(monkeypatch, *, cursor_before=None, cursor_after=None, locked=False):
    capture = HyprlandCaptureBackend(env={
        "WAYLAND_DISPLAY": "synthetic-0", "HYPRLAND_INSTANCE_SIGNATURE": "synthetic",
    })
    outputs = [
        {"name": "left", "x": -100, "y": -20, "width": 200, "height": 100,
         "scale": 2.0, "focused": True},
        {"name": "right", "x": 0, "y": 0, "width": 100, "height": 80,
         "scale": 1.0, "focused": False},
    ]
    cursors = iter([cursor_before or {"x": -50, "y": 0},
                    cursor_after or {"x": -50, "y": 0}])
    calls = []

    def run(args, cancel, *, max_bytes=262_144):
        calls.append(args)
        if args[0] == "grim":
            assert args[1:4] == ["-c", "-t", "png"]
            return png() if args[-2] == "left" else png(100, 80, "blue")
        command = args[-1]
        value = {"monitors": outputs, "locked": {"locked": locked}}
        return json.dumps(next(cursors) if command == "cursorpos" else value[command]).encode()

    monkeypatch.setattr(capture, "_run", run)
    return capture, outputs, calls


def test_capture_all_outputs_preserves_negative_offsets_and_mixed_scale(monkeypatch):
    capture, _, calls = backend(monkeypatch)
    result = capture.capture()
    desktop = result.image()
    image = decode_image(desktop.data)
    assert image.size == (200, 100)
    assert image.getpixel((50, 20)) == (255, 0, 0)
    assert image.getpixel((150, 40)) == (0, 0, 255)
    assert image.getpixel((150, 0)) == (0, 0, 0)
    assert desktop.origin_x == -100 and desktop.origin_y == -20
    assert result.cursor.monitor == "left"
    assert (result.cursor.image_x, result.cursor.image_y) == (50, 20)
    assert result.image("left").point_to_pixel(-50, 0) == (100, 40)
    assert [args[-2] for args in calls if args[0] == "grim"] == ["left", "right"]
    assert "focused" in result.geometry_description()


def test_pointer_movement_does_not_claim_precise_target(monkeypatch):
    capture, _, _ = backend(monkeypatch, cursor_after={"x": 75, "y": 50})
    cursor = capture.capture().cursor
    assert not cursor.stable
    assert cursor.monitor is None and cursor.image_x is None and cursor.image_y is None


def test_locked_session_never_captures(monkeypatch):
    capture, _, calls = backend(monkeypatch, locked=True)
    with pytest.raises(ScreenCaptureError, match="Unlock"):
        capture.capture()
    assert not any(args[0] == "grim" for args in calls)


def test_missing_lock_probe_fails_closed(monkeypatch):
    capture, _, calls = backend(monkeypatch)
    monkeypatch.setattr(capture, "_json", lambda *_: {})
    with pytest.raises(ScreenCaptureError, match="Unlock"):
        capture.capture()
    assert calls == []


def test_layout_change_discards_all_pixels(monkeypatch):
    capture, outputs, _ = backend(monkeypatch)
    original = capture._monitors
    count = 0

    def monitors(cancel):
        nonlocal count
        count += 1
        if count == 2:
            outputs[0]["x"] = -200
        return original(cancel)

    monkeypatch.setattr(capture, "_monitors", monitors)
    with pytest.raises(ScreenCaptureError, match="layout changed"):
        capture.capture()


def test_focus_change_is_not_a_layout_change(monkeypatch):
    capture, outputs, _ = backend(monkeypatch, cursor_after={"x": 75, "y": 50})
    original = capture._monitors
    count = 0

    def monitors(cancel):
        nonlocal count
        count += 1
        if count == 2:
            outputs[0]["focused"] = False
            outputs[1]["focused"] = True
        return original(cancel)

    monkeypatch.setattr(capture, "_monitors", monitors)
    result = capture.capture()
    assert not result.cursor.stable
    assert result.monitors[1].focused


def test_capture_rejects_incomplete_output(monkeypatch):
    capture, _, _ = backend(monkeypatch)
    original = capture._run

    def run(args, cancel, **kwargs):
        return png(1, 1) if args[0] == "grim" else original(args, cancel, **kwargs)

    monkeypatch.setattr(capture, "_run", run)
    with pytest.raises(ScreenCaptureError, match="incomplete"):
        capture.capture()


def test_large_layout_rejected_before_grim(monkeypatch):
    capture, outputs, calls = backend(monkeypatch)
    outputs[1]["x"] = 100_000
    with pytest.raises(ScreenCaptureError, match="too large"):
        capture.capture()
    assert not any(args[0] == "grim" for args in calls)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_invalid_scale_rejected(monkeypatch, value):
    capture, outputs, _ = backend(monkeypatch)
    outputs[0]["scale"] = value
    with pytest.raises(ScreenCaptureError, match="layout"):
        capture.capture()


def test_output_rotation_and_crop_coordinate_transform():
    monitor = Monitor("rotated", -50, 10, 80, 120, 2, transform=1)
    assert (monitor.logical_width, monitor.logical_height) == (60, 40)
    image = ScreenImage("crop", png(80, 40), 80, 40, origin_x=-20, origin_y=10,
                        logical_width=20, logical_height=10)
    assert image.point_to_pixel(-10, 15) == (40, 20)


def test_decode_rejects_invalid_bytes_and_oversized_dimensions(monkeypatch):
    with pytest.raises(ScreenCaptureError, match="unreadable"):
        decode_image(b"not a png")
    monkeypatch.setattr("ciel.screen.capture.MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ScreenCaptureError, match="dimensions"):
        decode_image(png(10, 10))


def test_process_timeout_is_bounded_and_stopped():
    capture = HyprlandCaptureBackend(timeout=0.1)
    started = time.monotonic()
    with pytest.raises(ScreenCaptureError, match="timed out"):
        capture._run([sys.executable, "-c", "import time; time.sleep(10)"], None)
    assert time.monotonic() - started < 2


def test_process_cancel_remains_responsive():
    capture = HyprlandCaptureBackend()
    event = threading.Event()
    timer = threading.Timer(0.1, event.set)
    timer.start()
    try:
        started = time.monotonic()
        with pytest.raises(ScreenCaptureError, match="cancelled"):
            capture._run([sys.executable, "-c", "import time; time.sleep(10)"], event)
        assert time.monotonic() - started < 2
    finally:
        timer.cancel()


def test_process_output_bound_stops_stream():
    capture = HyprlandCaptureBackend()
    with pytest.raises(ScreenCaptureError, match="too much"):
        capture._run([sys.executable, "-c", "print('x' * 10000)"], None, max_bytes=10)


def test_missing_session_fails_before_subprocess(monkeypatch, tmp_path):
    capture = HyprlandCaptureBackend(env={"XDG_RUNTIME_DIR": str(tmp_path / "missing")})
    monkeypatch.setattr(capture, "_run", lambda *_: pytest.fail("must not run"))
    with pytest.raises(ScreenCaptureError, match="logged-in"):
        capture.capture()


def test_background_service_discovers_same_user_session_without_env(monkeypatch, tmp_path):
    desktop_socket = socket.socket(socket.AF_UNIX)
    desktop_socket.bind(str(tmp_path / "wayland-test"))
    capture = HyprlandCaptureBackend(env={"XDG_RUNTIME_DIR": str(tmp_path)})
    monkeypatch.setattr(capture, "_json", lambda command, _: [
        {"instance": "test_instance", "wl_socket": "wayland-test", "pid": os.getpid()},
    ] if command == "instances" else pytest.fail("unexpected capture"))
    try:
        capture._ensure_session(None)
        assert capture.env["HYPRLAND_INSTANCE_SIGNATURE"] == "test_instance"
        assert capture.env["WAYLAND_DISPLAY"] == "wayland-test"
    finally:
        desktop_socket.close()


def test_background_service_refuses_ambiguous_desktops(monkeypatch, tmp_path):
    sockets = []
    for index in range(2):
        desktop_socket = socket.socket(socket.AF_UNIX)
        desktop_socket.bind(str(tmp_path / f"wayland-{index}"))
        sockets.append(desktop_socket)
    capture = HyprlandCaptureBackend(env={"XDG_RUNTIME_DIR": str(tmp_path)})
    monkeypatch.setattr(capture, "_json", lambda *_: [
        {"instance": f"instance_{index}", "wl_socket": f"wayland-{index}",
         "pid": os.getpid()} for index in range(2)
    ])
    try:
        with pytest.raises(ScreenCaptureError, match="identify one"):
            capture._ensure_session(None)
    finally:
        for desktop_socket in sockets:
            desktop_socket.close()


def test_image_repr_never_includes_pixels():
    image = ScreenImage("screen", b"private-pixels", 2, 2)
    assert "private-pixels" not in repr(image)
