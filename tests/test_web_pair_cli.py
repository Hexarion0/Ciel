"""The pairing command talks only to the existing PC-local web server."""

import httpx
import pytest


def test_pair_cli_prints_existing_phone_address_and_new_code(monkeypatch, capsys):
    from ciel.webapp import pair_cli

    calls = []

    def respond(request):
        calls.append((request.method, request.url.path))
        assert request.url.host == "127.0.0.1"
        assert request.url.port == 8765
        if request.url.path == "/api/network":
            return httpx.Response(200, json={
                "enabled": True, "url": "https://192.168.1.100:8765",
                "setup_url": "http://192.168.1.100:8766", "insecure_lan": False,
            })
        return httpx.Response(200, json={"code": "12345678", "expires_in": 300})

    monkeypatch.setattr(pair_cli, "make_client", lambda port: httpx.Client(
        base_url=f"http://127.0.0.1:{port}", transport=httpx.MockTransport(respond),
    ))
    assert pair_cli.main([]) == 0
    output = capsys.readouterr().out
    assert "12345678" in output and "5 minutes" in output
    assert "https://192.168.1.100:8765" in output
    assert "http://192.168.1.100:8766" in output
    assert calls == [("GET", "/api/network"), ("POST", "/api/pair-code")]


def test_pair_cli_prepares_phone_connection_using_single_private_address(monkeypatch, capsys):
    from ciel.webapp import pair_cli

    def respond(request):
        if request.url.path == "/api/network":
            return httpx.Response(200, json={"enabled": False, "addresses": ["192.168.1.100"]})
        if request.url.path == "/api/network/setup":
            assert request.read() == b'{"host":"192.168.1.100"}'
            return httpx.Response(200, json={
                "enabled": True, "url": "https://192.168.1.100:8765", "setup_url": "",
            })
        return httpx.Response(200, json={"code": "12345678"})

    monkeypatch.setattr(pair_cli, "make_client", lambda port: httpx.Client(
        base_url=f"http://127.0.0.1:{port}", transport=httpx.MockTransport(respond),
    ))
    assert pair_cli.main([]) == 0
    assert "12345678" in capsys.readouterr().out


def test_pair_cli_reports_stopped_server_without_traceback(monkeypatch, capsys):
    from ciel.webapp import pair_cli

    def unavailable(request):
        raise httpx.ConnectError("internal fixture", request=request)

    monkeypatch.setattr(pair_cli, "make_client", lambda port: httpx.Client(
        base_url=f"http://127.0.0.1:{port}", transport=httpx.MockTransport(unavailable),
    ))
    assert pair_cli.main(["--port", "8767"]) == 1
    output = capsys.readouterr().err
    assert "ciel web --web-port 8767" in output
    assert "internal fixture" not in output


def test_pair_cli_requires_host_choice_for_multiple_interfaces(monkeypatch, capsys):
    from ciel.webapp import pair_cli

    def respond(request):
        assert request.method == "GET"
        return httpx.Response(200, json={
            "enabled": False, "addresses": ["192.168.1.100", "10.0.0.5"],
        })

    monkeypatch.setattr(pair_cli, "make_client", lambda port: httpx.Client(
        base_url=f"http://127.0.0.1:{port}", transport=httpx.MockTransport(respond),
    ))
    assert pair_cli.main([]) == 1
    assert "--host" in capsys.readouterr().err


@pytest.mark.parametrize("host", ["0.0.0.0", "8.8.8.8", "localhost", "127.0.0.1"])
def test_pair_cli_rejects_non_private_phone_addresses(host):
    from ciel.webapp import pair_cli

    with pytest.raises(SystemExit):
        pair_cli.main(["--host", host])


def test_pair_command_dispatches_without_loading_audio(monkeypatch):
    from ciel import __main__
    from ciel.webapp import pair_cli

    monkeypatch.setattr(pair_cli, "main", lambda arguments: 7 if arguments == ["--help"] else 9)
    assert __main__.main(["pair", "--help"]) == 7
