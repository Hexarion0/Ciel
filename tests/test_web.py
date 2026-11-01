"""Desktop chat/status service uses bounded state and validated local requests."""

import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from ciel.web import DesktopWebUI, WebErrorHandler


@pytest.fixture
def web_server():
    ui = DesktopWebUI("hexarion", "hey ciel")
    submit = MagicMock(return_value=True)
    url = ui.start(submit, port=0)
    with httpx.Client(base_url=url, trust_env=False, timeout=2) as client:
        yield ui, submit, client
    ui.close()


def test_web_chat_progress_keeps_sentences_in_one_reply_and_freezes_interruption():
    ui = DesktopWebUI("owner", "hey ciel")
    ui.user_message("Hello")
    ui.start_sentence()
    ui.progress("Hi there.")
    ui.start_sentence()
    ui.progress("What")
    assert ui.snapshot()["messages"][-1]["text"] == "Hi there. What"
    ui.progress("What's up", interrupted=True)
    ui.progress("What's up with your project?")
    reply = ui.snapshot()["messages"][-1]
    assert reply["text"] == "Hi there. What's up"
    assert reply["interrupted"] and not reply["live"]


def test_empty_unplayed_sentence_does_not_create_a_fake_reply():
    ui = DesktopWebUI("owner", "hey ciel")
    ui.start_sentence()
    ui.finish_reply()
    assert ui.snapshot()["messages"] == []
    ui.text_reply("Voice unavailable, but here is the answer.")
    assert ui.snapshot()["messages"][-1]["text"].startswith("Voice unavailable")


def test_web_state_is_bounded_and_secrets_are_masked():
    ui = DesktopWebUI("owner", "hey ciel")
    for number in range(205):
        ui.user_message(str(number))
    ui.user_message("key nvapi-0123456789secret")
    ui.set_muted(True)
    ui.set_state("processing")
    for number in range(35):
        ui.feedback(str(number))
    snapshot = ui.snapshot()
    assert len(snapshot["messages"]) == 200
    assert len(snapshot["activity"]) == 30
    assert snapshot["muted"] and snapshot["state"] == "processing"
    assert snapshot["messages"][-1]["text"] == "key [REDACTED_SECRET]"
    snapshot["messages"][-1]["text"] = "changed"
    assert ui.snapshot()["messages"][-1]["text"] == "key [REDACTED_SECRET]"


def test_http_serves_page_and_submits_to_the_existing_worker(web_server):
    ui, submit, client = web_server
    response = client.get("/")
    assert response.status_code == 200
    assert "Message Ciel" in response.text
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert client.get("/api/state").json()["owner"] == "hexarion"
    for text in ["hello", "/mute", "/stop", "/exit"]:
        assert client.post("/api/message", json={"text": text}).json() == {"accepted": True}
    assert [call.args[0] for call in submit.call_args_list] == [
        "hello",
        "/mute",
        "/stop",
        "/exit",
    ]
    assert ui.snapshot()["messages"] == []  # Queuing isn't proof of a completed reply.


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "evil.example"},
        {"Origin": "https://evil.example"},
        {"Sec-Fetch-Site": "cross-site"},
    ],
)
def test_http_rejects_requests_from_other_websites(web_server, headers):
    _ui, submit, client = web_server
    assert client.post("/api/message", json={"text": "/exit"}, headers=headers).status_code == 403
    submit.assert_not_called()


@pytest.mark.parametrize("data", [{}, [], {"text": ""}, {"text": 5}, {"text": "x" * 4001}])
def test_http_rejects_invalid_messages(web_server, data):
    _ui, submit, client = web_server
    assert client.post("/api/message", json=data).status_code == 400
    submit.assert_not_called()


def test_http_reports_unavailable_worker_and_does_not_accept_form_posts(web_server):
    _ui, submit, client = web_server
    assert client.post("/api/message", data={"text": "/exit"}).status_code == 415
    submit.return_value = False
    response = client.post("/api/message", json={"text": "hello"})
    assert response.status_code == 503
    assert response.json() == {"accepted": False}


def test_runtime_warnings_are_masked_for_browser():
    ui = DesktopWebUI("owner", "hey ciel")
    handler = WebErrorHandler(ui)
    handler.handle(
        logging.LogRecord(
            "ciel",
            logging.ERROR,
            "file.py",
            1,
            "key=%s",
            ("sk-0123456789secret",),
            None,
        )
    )
    assert ui.snapshot()["activity"][-1]["text"] == "key=[REDACTED_SECRET]"


def test_history_uses_current_conversation():
    ui = DesktopWebUI("owner", "hey ciel")
    ui.load_history(
        [
            SimpleNamespace(role="user", content="hello"),
            SimpleNamespace(role="assistant", content="Hi!"),
        ]
    )
    assert [entry["text"] for entry in ui.snapshot()["messages"]] == ["hello", "Hi!"]


def test_history_preserves_dates_and_omits_internal_playback_notes():
    ui = DesktopWebUI("owner", "hey ciel")
    timestamp = datetime(2026, 10, 8, tzinfo=timezone.utc)
    ui.load_history(
        [
            SimpleNamespace(
                role="assistant",
                content="Hi!\n[Playback was interrupted.]",
                timestamp=timestamp,
            )
        ]
    )
    message = ui.snapshot()["messages"][0]
    assert message["text"] == "Hi!"
    assert message["time"] == timestamp.timestamp()


def test_history_restores_safe_sources_for_assistant_replies():
    ui = DesktopWebUI("owner", "hey ciel")
    ui.load_history(
        [
            SimpleNamespace(
                role="assistant",
                content="Verified answer",
                sources=[
                    {"title": "Public page", "url": "https://example.com/article"},
                    {"title": "Bad", "url": "javascript:alert(1)"},
                ],
            )
        ]
    )
    assert ui.snapshot()["messages"][0]["sources"] == [
        {"title": "Public page", "url": "https://example.com/article"},
    ]


@pytest.mark.parametrize("live", [True, False])
def test_reply_sources_attach_to_live_or_completed_reply_and_snapshot_is_independent(live):
    ui = DesktopWebUI("owner", "hey ciel")
    ui.text_reply("An earlier answer")
    ui.user_message("A current question")
    if live:
        ui.start_sentence()
        ui.progress("An answer")
    else:
        ui.text_reply("An answer")
    sources = [{"title": "<script>literal label</script>", "url": "https://example.com/"}]
    ui.set_reply_sources(sources)
    snapshot = ui.snapshot()
    assert snapshot["messages"][-1]["sources"] == sources
    assert "sources" not in snapshot["messages"][0]
    snapshot["messages"][-1]["sources"][0]["title"] = "Changed snapshot"
    sources[0]["title"] = "Changed input"
    assert ui.snapshot()["messages"][-1]["sources"][0]["title"] == "<script>literal label</script>"


def test_reply_sources_do_not_attach_to_an_older_reply_after_a_new_user_message():
    ui = DesktopWebUI("owner", "hey ciel")
    sources = [{"title": "Reference", "url": "https://example.com/"}]
    ui.set_reply_sources(sources)
    assert ui.snapshot()["messages"] == []
    ui.text_reply("An earlier answer")
    ui.user_message("A newer question")
    ui.set_reply_sources(sources)
    assert all("sources" not in message for message in ui.snapshot()["messages"])


@pytest.mark.parametrize("live", [True, False])
def test_reply_source_identity_rejects_an_older_request_after_a_new_reply(live):
    """An older worker cannot cite the newer user's assistant reply."""
    ui = DesktopWebUI("owner", "hey ciel")
    first_id = ui.user_message("First question")
    assert isinstance(first_id, int)
    ui.text_reply("First answer")
    second_id = ui.user_message("Second question")
    assert second_id > first_id
    if live:
        ui.start_sentence()
        ui.progress("Second answer")
    else:
        ui.text_reply("Second answer")
    old_sources = [{"title": "Old reference", "url": "https://example.com/old"}]
    new_sources = [{"title": "New reference", "url": "https://example.com/new"}]
    ui.set_reply_sources(old_sources, user_message_id=first_id)
    assert "sources" not in ui.snapshot()["messages"][-1]
    ui.set_reply_sources(new_sources, user_message_id=second_id)
    assert ui.snapshot()["messages"][-1]["sources"] == new_sources
    ui.set_reply_sources(old_sources, user_message_id=first_id)
    assert ui.snapshot()["messages"][-1]["sources"] == new_sources


def test_interrupted_reply_retains_sources_attached_for_its_user_message():
    """Citations attached at speech start remain visible after interruption."""
    ui = DesktopWebUI("owner", "hey ciel")
    user_id = ui.user_message("Current question")
    ui.start_sentence()
    ui.progress("First part")
    sources = [{"title": "Reference", "url": "https://example.com/"}]
    ui.set_reply_sources(sources, user_message_id=user_id)
    ui.progress("First part", interrupted=True)
    reply = ui.snapshot()["messages"][-1]
    assert not reply["live"]
    assert reply["interrupted"]
    assert reply["sources"] == sources


def test_http_get_rejects_untrusted_host_and_port_conflicts_are_reported(web_server):
    _ui, _submit, client = web_server
    assert client.get("/api/state", headers={"Host": "evil.example"}).status_code == 403
    second_ui = DesktopWebUI("owner", "hey ciel")
    with pytest.raises(OSError):
        second_ui.start(lambda _text: True, port=client.base_url.port)
    second_ui.close()


@pytest.mark.parametrize(
    "asset,kind",
    [
        ("chat.css", "text/css"),
        ("chat.js", "javascript"),
        ("orb.js", "javascript"),
        ("render.js", "javascript"),
    ],
)
def test_packaged_assets_are_served_without_arbitrary_file_access(web_server, asset, kind):
    _ui, _submit, client = web_server
    response = client.get("/assets/" + asset)
    assert response.status_code == 200
    assert kind in response.headers["Content-Type"]
    assert response.content
    assert client.get("/assets/unknown.js").status_code == 404
    assert client.get("/assets/%2e%2e/web.py").status_code == 404


def test_incremental_state_includes_stream_changes_and_resets_after_switch():
    ui = DesktopWebUI("owner", "hey ciel")
    ui.user_message("Hello")
    initial = ui.snapshot()
    cursor = initial["cursor"]
    assert ui.snapshot(after=cursor)["messages"] == []
    ui.start_sentence()
    ui.progress("Hi")
    update = ui.snapshot(after=cursor)
    assert [m["text"] for m in update["messages"]] == ["Hi"]
    ui.progress("Hi there")
    assert ui.snapshot(after=update["cursor"])["messages"][0]["text"] == "Hi there"
    ui.replace_history([])
    assert ui.snapshot(after=cursor)["reset"] is True


def test_routing_prefixes_keep_browser_request_and_local_pair_code_can_be_renewed(web_server):
    ui, legacy, client = web_server
    received = []
    ui.channels.configure(
        lambda request, audio, current: received.append(request) or True, SimpleNamespace()
    )
    for text in ["/fast hello", "/strong hello", "/deep hello"]:
        response = client.post("/api/message", json={"text": text})
        assert response.status_code == 200
    assert [r.text for r in received] == ["/fast hello", "/strong hello", "/deep hello"]
    legacy.assert_not_called()
    first = client.post("/api/pair-code", json={}).json()["code"]
    second = client.post("/api/pair-code", json={}).json()["code"]
    assert len(first) == 8 and len(second) == 8
    device = ui.channels.auth.pair(second, source="phone")
    assert ui.channels.auth.authenticate(device.token)


def test_lan_mode_keeps_separate_pc_loopback_listener(monkeypatch):
    from ciel import web

    # Two actual loopback addresses exercise dual binding without exposing a test on Wi-Fi.
    class TLS:
        def wrap_socket(self, sock, **kwargs):
            return sock

    monkeypatch.setattr(web, "server_tls", lambda *_args: TLS())
    ui = DesktopWebUI("owner", "hey ciel")
    ui.start(lambda text: True, port=0, host="127.0.0.2", certificate="fixture", key="fixture")
    try:
        port = ui._server.server_address[1]
        with httpx.Client(trust_env=False) as client:
            response = client.get(f"http://127.0.0.1:{port}/api/session")
            assert response.status_code == 200
            assert response.json()["device_id"] == "pc"
    finally:
        ui.close()


def test_configured_https_handshake_and_exact_origin(tmp_path):
    import shutil
    import subprocess

    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("OpenSSL is needed for the local TLS integration check")
    certificate, key = tmp_path / "fixture.pem", tmp_path / "fixture-key.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    ui = DesktopWebUI("owner", "hey ciel")
    url = ui.start(lambda text: True, port=0, certificate=str(certificate), key=str(key))
    try:
        import ssl

        context = ssl.create_default_context(cafile=str(certificate))
        with httpx.Client(base_url=url, verify=context, trust_env=False) as client:
            assert client.get("/api/session").json()["device_id"] == "pc"
            assert client.post("/api/message", json={"text": "Hello"}).status_code == 403
            assert (
                client.post(
                    "/api/message", json={"text": "Hello"}, headers={"Origin": url}
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/api/message",
                    json={"text": "Hello"},
                    headers={"Origin": "https://evil.example"},
                ).status_code
                == 403
            )
    finally:
        ui.close()


def test_authenticated_web_admin_settings_memory_and_confirmed_restart(web_server, tmp_path):
    import threading

    from ciel.config import Settings
    from ciel.webapp.admin import WebAdministration
    from ciel.webapp.settings import WebSettings

    ui, _submit, client = web_server
    settings = Settings(_env_file=None, nim_api_key='hidden-secret')
    preferences = WebSettings(settings, tmp_path / '.env', apply=lambda fields: None)
    restarted = threading.Event()
    forgotten = []
    network = SimpleNamespace(snapshot=lambda: {'host':'192.168.1.100'}, close=lambda: None)
    ui.admin = WebAdministration(ui, preferences, network, restart=restarted.set,
                                 memories=lambda: [{'id':1,'text':'A saved fact'}],
                                 forget=lambda identity: forgotten.append(identity) or True)
    response = client.get('/api/settings')
    assert response.status_code == 200 and 'hidden-secret' not in response.text
    assert client.post('/api/settings', json={'nim_model':'browser-selected'}).status_code == 200
    assert settings.nim_model == 'browser-selected'
    assert client.post('/api/settings', json={'tts_speed':-5}).status_code == 400
    assert client.get('/api/memory').json()['memories'][0]['text'] == 'A saved fact'
    assert client.post('/api/memory/forget', json={'id':1}).status_code == 400
    assert not forgotten
    assert client.post('/api/memory/forget', json={'id':1,'confirmed':True}).json()['forgotten']
    assert forgotten == [1]
    assert client.post('/api/server/restart', json={}).status_code == 400
    assert client.post('/api/server/restart', json={'confirmed':True}).status_code == 200
    assert restarted.wait(1)
