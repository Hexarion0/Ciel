"""Paired phones survive restarts for 30 days of inactivity, stored only as token digests."""

import io
import json
import stat
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

DAY = 24 * 3600


def make_auth(path, now, owner="owner"):
    from ciel.webapp.auth import DeviceAuth

    auth = DeviceAuth(owner, clock=lambda: now[0])
    auth.attach_store(path)
    return auth


def test_paired_device_survives_restart_without_storing_the_raw_token(tmp_path):
    now = [1_000_000.0]
    store = tmp_path / "web" / "paired-devices.json"
    first = make_auth(store, now)
    device = first.pair(first.issue_code(), source="phone")

    saved = store.read_text()
    assert device.token not in saved
    assert stat.S_IMODE(store.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.parent.stat().st_mode) == 0o700

    restarted = make_auth(store, now, owner="new-process-owner")
    found = restarted.authenticate(device.token)
    assert found is not None
    assert found.device_id == device.device_id
    assert found.csrf == device.csrf
    assert found.owner_key == "new-process-owner"


def test_paired_device_expires_after_thirty_idle_days(tmp_path):
    now = [1_000_000.0]
    store = tmp_path / "paired-devices.json"
    auth = make_auth(store, now)
    device = auth.pair(auth.issue_code(), source="phone")

    now[0] += 30 * DAY + 1
    assert auth.authenticate(device.token) is None
    assert make_auth(store, now).authenticate(device.token) is None


def test_use_extends_the_thirty_day_window_across_restarts(tmp_path):
    now = [1_000_000.0]
    store = tmp_path / "paired-devices.json"
    auth = make_auth(store, now)
    device = auth.pair(auth.issue_code(), source="phone")

    now[0] += 20 * DAY
    assert auth.authenticate(device.token) is not None
    now[0] += 20 * DAY
    assert make_auth(store, now).authenticate(device.token) is not None


def test_revoking_a_device_is_saved(tmp_path):
    now = [1_000_000.0]
    store = tmp_path / "paired-devices.json"
    auth = make_auth(store, now)
    kept = auth.pair(auth.issue_code(), source="phone")
    removed = auth.pair(auth.issue_code(), source="tablet")

    auth.revoke(removed.device_id)

    restarted = make_auth(store, now)
    assert restarted.authenticate(removed.token) is None
    assert restarted.authenticate(kept.token) is not None


def test_revoke_all_disconnects_every_paired_device_and_reports_them(tmp_path):
    now = [1_000_000.0]
    store = tmp_path / "paired-devices.json"
    auth = make_auth(store, now)
    devices = [auth.pair(auth.issue_code(), source=name) for name in ("phone", "tablet")]

    assert set(auth.revoke_all()) == {device.device_id for device in devices}

    restarted = make_auth(store, now)
    assert all(restarted.authenticate(device.token) is None for device in devices)


def test_shutdown_clear_keeps_saved_pairings(tmp_path):
    now = [1_000_000.0]
    store = tmp_path / "paired-devices.json"
    auth = make_auth(store, now)
    device = auth.pair(auth.issue_code(), source="phone")

    auth.clear()

    assert make_auth(store, now).authenticate(device.token) is not None


def test_open_home_network_sessions_are_never_saved(tmp_path):
    now = [1_000_000.0]
    store = tmp_path / "paired-devices.json"
    auth = make_auth(store, now)
    session = auth.create_session()

    assert not store.exists() or session.device_id not in store.read_text()
    assert make_auth(store, now).authenticate(session.token) is None


@pytest.mark.parametrize("content", ["not json", "[]", '{"devices": [{"token_sha256": 5}]}'])
def test_damaged_store_starts_empty_instead_of_failing(tmp_path, content):
    store = tmp_path / "paired-devices.json"
    store.write_text(content)
    auth = make_auth(store, [1_000_000.0])

    device = auth.pair(auth.issue_code(), source="phone")

    assert json.loads(store.read_text())["devices"][0]["device_id"] == device.device_id


def test_changes_after_shutdown_clear_cannot_erase_saved_pairings(tmp_path):
    now = [1_000_000.0]
    store = tmp_path / "paired-devices.json"
    auth = make_auth(store, now)
    kept = auth.pair(auth.issue_code(), source="phone")
    leaving = auth.pair(auth.issue_code(), source="tablet")

    auth.clear()
    auth.revoke(leaving.device_id)

    assert make_auth(store, now).authenticate(kept.token) is not None


def test_renewal_is_saved_at_most_daily(tmp_path, monkeypatch):
    now = [1_000_000.0]
    store = tmp_path / "paired-devices.json"
    auth = make_auth(store, now)
    device = auth.pair(auth.issue_code(), source="phone")
    writes = []
    monkeypatch.setattr("ciel.webapp.auth.os.replace",
                        lambda *args, _real=__import__("os").replace: (writes.append(1),
                                                                        _real(*args)))

    for _ in range(50):
        now[0] += 60
        assert auth.authenticate(device.token)
    assert writes == []
    now[0] += DAY
    assert auth.authenticate(device.token)
    assert len(writes) == 1


def test_failed_save_is_not_retried_on_every_request(tmp_path, monkeypatch):
    now = [1_000_000.0]
    store = tmp_path / "paired-devices.json"
    auth = make_auth(store, now)
    device = auth.pair(auth.issue_code(), source="phone")
    attempts = []

    def failing_open(*args, **kwargs):
        attempts.append(1)
        raise OSError("disk full")

    monkeypatch.setattr("ciel.webapp.auth.os.open", failing_open)
    now[0] += 2 * DAY
    for _ in range(100):
        now[0] += 0.35
        assert auth.authenticate(device.token) is not None
    assert len(attempts) == 1


@pytest.mark.parametrize("record", [
    {"token_sha256": "ABC", "device_id": "a" * 32, "csrf": "c", "expires_at": 2e6},
    {"token_sha256": "a" * 64, "device_id": "pc", "csrf": "c", "expires_at": 2e6},
    {"token_sha256": "a" * 64, "device_id": "a" * 32, "csrf": "c" * 129, "expires_at": 2e6},
    {"token_sha256": "a" * 64, "device_id": "a" * 32, "csrf": "c", "expires_at": True},
    {"token_sha256": "a" * 64, "device_id": "a" * 32, "csrf": "c", "expires_at": "2e6"},
])
def test_invalid_saved_records_are_skipped(tmp_path, record):
    store = tmp_path / "paired-devices.json"
    store.write_text(json.dumps({"version": 1, "devices": [record]}))

    auth = make_auth(store, [1_000_000.0])

    assert auth._devices == {}


@pytest.mark.parametrize("expires_at", ["Infinity", "NaN", "1e300"])
def test_saved_expiry_cannot_exceed_the_idle_window(tmp_path, expires_at):
    import hashlib

    now = [1_000_000.0]
    store = tmp_path / "paired-devices.json"
    digest = hashlib.sha256(b"token").hexdigest()
    store.write_text(f'{{"version": 1, "devices": [{{"token_sha256": "{digest}", '
                     f'"device_id": "{"a" * 32}", "csrf": "c", "expires_at": {expires_at}}}]}}')

    auth = make_auth(store, now)
    now[0] += 30 * DAY + 1

    assert auth.authenticate("token") is None


def test_attaching_tightens_existing_store_permissions(tmp_path):
    directory = tmp_path / "web"
    directory.mkdir(mode=0o755)
    directory.chmod(0o755)
    store = directory / "paired-devices.json"
    store.write_text('{"version": 1, "devices": []}')
    store.chmod(0o644)

    make_auth(store, [1_000_000.0])

    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.stat().st_mode) == 0o600


def test_expired_records_are_dropped_from_the_saved_file(tmp_path):
    now = [1_000_000.0]
    store = tmp_path / "paired-devices.json"
    auth = make_auth(store, now)
    old = auth.pair(auth.issue_code(), source="phone")
    now[0] += 31 * DAY
    auth.pair(auth.issue_code(), source="tablet")

    saved = {record["device_id"] for record in json.loads(store.read_text())["devices"]}
    assert old.device_id not in saved and len(saved) == 1


def _handler(path, command="GET", cookie="", body=None, csrf=""):
    raw = json.dumps(body or {}).encode()
    headers = {"Content-Type": "application/json", "Content-Length": str(len(raw))}
    if cookie:
        headers["Cookie"] = "ciel_device=" + cookie
    if csrf:
        headers["X-CSRF-Token"] = csrf
    return SimpleNamespace(
        path=path, command=command, headers=headers, client_address=("192.168.1.20", 1234),
        rfile=io.BytesIO(raw), _json=MagicMock(), response_headers={},
    )


def test_pairing_cookie_lasts_thirty_days_and_session_check_renews_it():
    from ciel.web import DesktopWebUI

    ui = DesktopWebUI("owner", "hey ciel")
    code = ui.channels.auth.issue_code()
    pair = _handler("/api/pair", "POST", body={"code": code})
    assert ui.channels.authorize(pair)
    assert ui.channels.post(pair)
    cookie = pair.response_headers["Set-Cookie"]
    assert "Max-Age=2592000" in cookie and "Secure" in cookie and "HttpOnly" in cookie
    token = cookie.split(";", 1)[0].split("=", 1)[1]

    check = _handler("/api/session", cookie=token)
    assert ui.channels.authorize(check)
    renewed = check.response_headers["Set-Cookie"]
    assert renewed.startswith("ciel_device=" + token + ";")
    assert "Max-Age=2592000" in renewed and "Secure" in renewed


def test_an_open_app_renews_its_cookie_about_daily_while_polling():
    from ciel.web import DesktopWebUI

    now = [1_000_000.0]
    ui = DesktopWebUI("owner", "hey ciel")
    ui.channels.auth.clock = lambda: now[0]
    phone = ui.channels.auth.pair(ui.channels.auth.issue_code(), source="phone")

    first = _handler("/api/state", cookie=phone.token)
    assert ui.channels.authorize(first)
    assert "Max-Age=2592000" in first.response_headers.get("Set-Cookie", "")
    now[0] += 60
    soon = _handler("/api/state", cookie=phone.token)
    assert ui.channels.authorize(soon)
    assert "Set-Cookie" not in soon.response_headers
    now[0] += DAY
    later = _handler("/api/state", cookie=phone.token)
    assert ui.channels.authorize(later)
    assert "Max-Age=2592000" in later.response_headers["Set-Cookie"]


@pytest.mark.parametrize("value", ["true", 1, False])
def test_disconnect_all_needs_an_explicit_true(value):
    from ciel.web import DesktopWebUI

    ui = DesktopWebUI("owner", "hey ciel")
    phone = ui.channels.auth.pair(ui.channels.auth.issue_code(), source="phone")
    local = _handler("/api/revoke", "POST", body={"all": value})
    local.client_address = ("127.0.0.1", 1234)
    assert ui.channels.authorize(local)
    ui.channels.post(local)

    assert ui.channels.auth.authenticate(phone.token) is not None


def test_only_the_pc_can_disconnect_all_phones():
    from ciel.web import DesktopWebUI

    ui = DesktopWebUI("owner", "hey ciel")
    phone = ui.channels.auth.pair(ui.channels.auth.issue_code(), source="phone")

    attempt = _handler("/api/revoke", "POST", cookie=phone.token, body={"all": True},
                       csrf=phone.csrf)
    assert ui.channels.authorize(attempt)
    ui.channels.post(attempt)
    assert attempt._json.call_args.args[0] == 400
    assert ui.channels.auth.authenticate(phone.token) is not None

    local = _handler("/api/revoke", "POST", body={"all": True})
    local.client_address = ("127.0.0.1", 1234)
    assert ui.channels.authorize(local)
    local.headers["X-CSRF-Token"] = local.device.csrf
    ui.channels.post(local)
    assert local._json.call_args.args == (200, {"revoked": True, "devices": 1})
    assert ui.channels.auth.authenticate(phone.token) is None
