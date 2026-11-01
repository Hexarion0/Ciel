"""Accidental HTTP visits redirect without exposing the HTTPS app or tracebacks."""

import socket
import ssl

import httpx
import pytest


@pytest.mark.parametrize("dynamic", [False, True])
def test_http_on_https_port_redirects_without_processing_private_requests(
    tmp_path, capsys, dynamic,
):
    from ciel.web import DesktopWebUI
    from ciel.webapp.tls import prepare_local_tls

    bundle = prepare_local_tls("127.0.0.2", tmp_path / "tls")
    submitted = []
    ui = DesktopWebUI("owner", "hey ciel")
    if dynamic:
        ui.start(lambda text: submitted.append(text), port=0)
        secure = ui.enable_lan("127.0.0.2", bundle.certificate, bundle.key)
    else:
        secure = ui.start(lambda text: submitted.append(text), host="127.0.0.2", port=0,
                          certificate=str(bundle.certificate), key=str(bundle.key))
    plain = secure.replace("https:", "http:")
    try:
        with httpx.Client(trust_env=False, timeout=5) as client:
            response = client.get(plain + "/api/settings")
            assert response.status_code == 307
            assert response.headers["Location"] == secure + "/"
            assert not response.content
            assert "Set-Cookie" not in response.headers
            assert client.head(plain).status_code == 307
            assert client.post(plain + "/api/message", json={"text": "Hello"}).status_code == 400
            assert client.get(plain, headers={"Host": "evil.example"}).status_code == 403
        trust = ssl.create_default_context(cafile=str(bundle.ca_certificate))
        with httpx.Client(verify=trust, trust_env=False, follow_redirects=True) as client:
            assert client.get(plain).status_code == 200
            assert client.get(secure + "/api/session").status_code == 200
        assert not submitted
    finally:
        ui.close()
    assert "Traceback" not in capsys.readouterr().err


def test_tls_disconnects_are_quiet_and_do_not_block_later_requests(tmp_path, capsys):
    from ciel.web import DesktopWebUI
    from ciel.webapp.tls import prepare_local_tls

    bundle = prepare_local_tls("127.0.0.2", tmp_path / "tls")
    ui = DesktopWebUI("owner", "hey ciel")
    secure = ui.start(lambda text: True, host="127.0.0.2", port=0,
                      certificate=str(bundle.certificate), key=str(bundle.key))
    port = ui._server.server_address[1]
    try:
        # A client rejecting the local CA must not dump a traceback on the host.
        with pytest.raises(httpx.ConnectError):
            with httpx.Client(trust_env=False) as client:
                client.get(secure)
        # An incomplete ClientHello is handled independently of other connections.
        with socket.create_connection(("127.0.0.2", port), timeout=3) as stalled:
            stalled.sendall(b"\x16\x03\x01")
            trust = ssl.create_default_context(cafile=str(bundle.ca_certificate))
            with httpx.Client(verify=trust, trust_env=False) as client:
                assert client.get(secure).status_code == 200
    finally:
        ui.close()
    assert "Traceback" not in capsys.readouterr().err
