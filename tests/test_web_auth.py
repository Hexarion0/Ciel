"""Pairing is expiring, single use, rate limited, and revocable."""

import pytest


def test_pair_code_single_use_expiry_and_revocation():
    from ciel.webapp.auth import DeviceAuth

    now = [0.0]
    auth = DeviceAuth("owner", clock=lambda: now[0])
    code = auth.issue_code()
    device = auth.pair(code, source="phone")
    assert auth.authenticate(device.token).device_id == device.device_id
    with pytest.raises(ValueError):
        auth.pair(code, source="phone")
    auth.revoke(device.device_id)
    assert auth.authenticate(device.token) is None
    code = auth.issue_code()
    now[0] = 301
    with pytest.raises(ValueError):
        auth.pair(code, source="phone")


def test_pair_rate_limit_applies_even_to_correct_code_after_failed_attempts():
    from ciel.webapp.auth import DeviceAuth

    auth = DeviceAuth("owner")
    code = auth.issue_code()
    for _ in range(5):
        with pytest.raises(ValueError):
            auth.pair("wrong", source="phone")
    with pytest.raises(ValueError):
        auth.pair(code, source="phone")


def test_remote_api_requires_session_and_csrf_but_shell_is_public():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from ciel.web import DesktopWebUI

    ui = DesktopWebUI("owner", "hey ciel")
    handler = SimpleNamespace(
        path="/api/state",
        command="GET",
        headers={},
        client_address=("192.168.1.20", 1234),
        _json=MagicMock(),
    )
    assert not ui.channels.authorize(handler)
    assert handler._json.call_args.args[0] == 401
    handler.path = "/"
    assert ui.channels.authorize(handler)
    device = ui.channels.auth.pair(ui.channels.auth.issue_code(), source="phone")
    handler.path = "/api/message"
    handler.command = "POST"
    handler.headers = {"Cookie": "ciel_device=" + device.token}
    assert not ui.channels.authorize(handler)
    handler.headers["X-CSRF-Token"] = device.csrf
    assert ui.channels.authorize(handler)
    ui.channels.auth.revoke(device.device_id)
    assert not ui.channels.authorize(handler)


def test_lan_cannot_start_without_https_and_public_bind_is_rejected():
    from ciel.webapp.tls import server_tls

    assert server_tls("127.0.0.1", None, None) is None
    for host in ["0.0.0.0", "192.168.1.10", "8.8.8.8"]:
        with pytest.raises(ValueError):
            server_tls(host, None, None)


def test_explicit_plain_lan_mode_needs_no_certificate_but_rejects_public_bind():
    from ciel.webapp.tls import server_tls

    assert server_tls("192.168.1.100", None, None, insecure_lan=True) is None
    for host in ["0.0.0.0", "8.8.8.8"]:
        with pytest.raises(ValueError):
            server_tls(host, None, None, insecure_lan=True)


def test_plain_lan_mode_assigns_separate_browser_sessions_without_pairing():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from ciel.web import DesktopWebUI

    ui = DesktopWebUI("owner", "hey ciel")
    ui.insecure_lan = True

    def browser():
        return SimpleNamespace(path="/api/session", command="GET", headers={},
                               client_address=("192.168.1.20", 1234), _json=MagicMock())

    first, second = browser(), browser()
    assert ui.channels.authorize(first)
    assert ui.channels.authorize(second)
    assert first.device and second.device
    assert first.device.device_id != second.device.device_id != "pc"
    assert "Secure;" not in first.response_headers["Set-Cookie"]
    first.headers["Cookie"] = first.response_headers["Set-Cookie"].split(";", 1)[0]
    identity = first.device.device_id
    assert ui.channels.authorize(first)
    assert first.device.device_id == identity
    first.path, first.command = "/api/message", "POST"
    assert not ui.channels.authorize(first)
    first.headers["X-CSRF-Token"] = first.device.csrf
    assert ui.channels.authorize(first)


def test_plain_lan_mode_does_not_grant_public_clients_access():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from ciel.web import DesktopWebUI

    ui = DesktopWebUI("owner", "hey ciel")
    ui.insecure_lan = True
    handler = SimpleNamespace(path="/api/state", command="GET", headers={},
                              client_address=("8.8.8.8", 1234), _json=MagicMock())
    assert not ui.channels.authorize(handler)


def test_a_non_ascii_csrf_header_is_refused_without_an_error():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from ciel.web import DesktopWebUI

    ui = DesktopWebUI("owner", "hey ciel")
    device = ui.channels.auth.pair(ui.channels.auth.issue_code(), source="phone")
    handler = SimpleNamespace(
        path="/api/message", command="POST", client_address=("192.168.1.20", 1234),
        headers={"Cookie": "ciel_device=" + device.token, "X-CSRF-Token": "jÃ©"},
        _json=MagicMock(),
    )
    assert not ui.channels.authorize(handler)
    assert handler._json.call_args.args[0] == 403


def test_open_lan_browser_sessions_stay_bounded():
    # Every cookieless request on an open home network creates a session; a client
    # that never keeps cookies must not grow server memory without limit.
    from ciel.webapp.auth import MAX_BROWSER_SESSIONS, DeviceAuth

    auth = DeviceAuth("owner")
    paired = auth.pair(auth.issue_code(), source="phone")
    sessions = [auth.create_session() for _ in range(MAX_BROWSER_SESSIONS + 50)]

    assert len(auth._devices) <= MAX_BROWSER_SESSIONS + 1
    assert auth.authenticate(sessions[-1].token) is not None
    assert auth.authenticate(paired.token) is not None  # Paired phones are never evicted.


def test_a_failed_local_listener_does_not_hang_shutdown(monkeypatch):
    import threading

    from ciel import web
    from ciel.web import DesktopWebUI

    real, created = web.BrowserHTTPServer, []

    def server(*args, **kwargs):
        created.append(args)
        if len(created) == 2:
            raise OSError("address in use")
        return real(*args, **kwargs)

    monkeypatch.setattr(web, "BrowserHTTPServer", server)
    ui = DesktopWebUI("owner", "hey ciel")
    ui.insecure_lan = True
    with pytest.raises(OSError):
        ui.start(lambda text: True, host="127.0.0.2", port=0)
    closer = threading.Thread(target=ui.close, daemon=True)
    closer.start()
    closer.join(3)
    assert not closer.is_alive(), "close() hung on a server that never started serving"
