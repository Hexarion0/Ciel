"""Private Wi-Fi setup provides a stable CA without modifying system trust."""

import ssl
import stat
import subprocess

import pytest


def test_local_tls_creates_stable_ca_and_certificate_for_pc_address(tmp_path):
    from ciel.webapp.tls import prepare_local_tls

    bundle = prepare_local_tls("192.168.1.100", tmp_path / "tls")
    ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(bundle.certificate, bundle.key)
    check = subprocess.run(
        [
            "openssl",
            "verify",
            "-CAfile",
            str(bundle.ca_certificate),
            "-verify_ip",
            "192.168.1.100",
            str(bundle.certificate),
        ],
        capture_output=True,
    )
    assert check.returncode == 0, check.stderr.decode()
    assert stat.S_IMODE(bundle.key.stat().st_mode) == 0o600
    assert stat.S_IMODE(bundle.key.parent.stat().st_mode) == 0o700
    again = prepare_local_tls("192.168.1.101", tmp_path / "tls")
    assert again.fingerprint == bundle.fingerprint
    with pytest.raises(ValueError):
        prepare_local_tls("8.8.8.8", tmp_path / "public")


def test_android_setup_http_exposes_only_public_certificate_and_instructions(tmp_path):
    import httpx

    from ciel.webapp.network import AndroidSetupServer

    ca = tmp_path / "public.crt"
    ca.write_bytes(b"public certificate fixture")
    server = AndroidSetupServer("127.0.0.1", 0, ca, "https://192.168.1.100:8765", "sha256")
    try:
        with httpx.Client(base_url=server.url, trust_env=False) as client:
            assert "Android" in client.get("/").text
            assert client.get("/ciel-ca.crt").content == b"public certificate fixture"
            for path in ["/api/state", "/api/settings", "/ciel-ca-key.pem", "/../.env"]:
                assert client.get(path).status_code == 404
    finally:
        server.close()


def test_pc_server_adds_separate_trusted_https_listener_without_stopping(tmp_path):
    import httpx

    from ciel.web import DesktopWebUI
    from ciel.webapp.tls import prepare_local_tls

    ui = DesktopWebUI("owner", "hey ciel")
    pc_url = ui.start(lambda text: True, port=0)
    bundle = prepare_local_tls("127.0.0.2", tmp_path / "tls")
    phone_url = ui.enable_lan("127.0.0.2", bundle.certificate, bundle.key)
    try:
        with httpx.Client(trust_env=False) as client:
            assert client.get(pc_url + "/api/state").status_code == 200
        trust = ssl.create_default_context(cafile=str(bundle.ca_certificate))
        with httpx.Client(verify=trust, trust_env=False) as client:
            assert client.get(phone_url + "/").status_code == 200
    finally:
        ui.close()


def test_tls_refresh_keeps_previous_certificate_and_key_valid(tmp_path):
    from ciel.webapp.tls import prepare_local_tls

    first = prepare_local_tls("192.168.1.100", tmp_path / "tls")
    old_certificate, old_key = first.certificate.read_bytes(), first.key.read_bytes()
    prepare_local_tls("192.168.1.101", tmp_path / "tls")
    assert first.certificate.read_bytes() == old_certificate
    assert first.key.read_bytes() == old_key
    ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(first.certificate, first.key)


def test_connection_reports_active_cli_address_instead_of_saved_default(tmp_path):
    from types import SimpleNamespace

    from ciel.config import Settings
    from ciel.webapp.network import LocalConnection

    connection = LocalConnection(
        Settings(_env_file=None),
        SimpleNamespace(phone_url="https://192.168.1.100:8765"),
        tmp_path,
        tmp_path / ".env",
        8765,
    )
    assert connection.snapshot()["host"] == "192.168.1.100"


def test_plain_lan_mode_serves_http_and_omits_certificate_onboarding(tmp_path):
    import httpx

    from ciel.config import Settings
    from ciel.web import DesktopWebUI
    from ciel.webapp.network import LocalConnection

    ui = DesktopWebUI("owner", "hey ciel")
    ui.insecure_lan = True
    url = ui.start(lambda text: True, host="127.0.0.2", port=0)
    try:
        with httpx.Client(trust_env=False) as client:
            assert url.startswith("http://")
            assert client.get(url + "/api/session").status_code == 200
        ui.phone_url = "http://192.168.1.100:8765"
        (tmp_path / "ciel-ca.crt").write_text("unused certificate")
        connection = LocalConnection(Settings(_env_file=None), ui, tmp_path,
                                     tmp_path / ".env", 8765)
        status = connection.snapshot()
        assert status["insecure_lan"] and status["enabled"]
        assert not status["certificate_available"] and not status["fingerprint"]
        connection.start_public_setup()
        assert connection.setup_server is None
    finally:
        ui.close()
