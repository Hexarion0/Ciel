"""Explicit screenshot output uses synthetic PNGs and mocked Telegram HTTP only."""

import asyncio
import json
import struct
import threading
import time
import zlib
from unittest.mock import Mock

import httpx
import pytest

from ciel.chat.state import ChatState
from ciel.chat.telegram import (
    MAX_PHOTO_BYTES,
    ChatReply,
    OutgoingPhoto,
    TelegramClient,
    TelegramError,
    TelegramService,
)

NOW = 1_800_000_000


def png(width=1, height=1):
    """Build a tiny real image without reading a screen or a local file."""
    def chunk(kind, content):
        return (
            struct.pack(">I", len(content)) + kind + content
            + struct.pack(">I", zlib.crc32(kind + content))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + chunk(b"IEND", b"")
    )


def update(update_id=1, text="Send me the screenshot"):
    return {
        "update_id": update_id,
        "message": {
            "from": {"id": 42, "is_bot": False},
            "chat": {"id": 42, "type": "private"},
            "text": text, "date": NOW,
        },
    }


@pytest.fixture
def state(tmp_path):
    selected = ChatState(tmp_path / "chat.db")
    selected.bind(123, 42)
    yield selected
    selected.close()


def test_explicit_photo_reply_remains_text_compatible_without_exposing_bytes():
    image = OutgoingPhoto(data=png(), expires_at=NOW + 900)
    answer = ChatReply("Here is the screen.", photos=(image,))
    assert isinstance(answer, str)
    assert json.loads(json.dumps({"text": answer})) == {"text": "Here is the screen."}
    assert answer.photos == (image,)
    assert "PNG" not in repr(image)
    assert "data=" not in repr(image)
    assert repr(answer) == repr("Here is the screen.")


@pytest.mark.parametrize("data", [
    "https://example.org/screen.png", "/tmp/screen.png", b"not a PNG",
    b"\x89PNG\r\n\x1a\n" + b"x" * MAX_PHOTO_BYTES,
    png(0, 1), png(6000, 6000), png(21, 1),
])
def test_photo_rejects_paths_urls_invalid_images_and_telegram_limit_violations(data):
    with pytest.raises(ValueError):
        OutgoingPhoto(data=data, expires_at=NOW + 900)


@pytest.mark.parametrize("expiry", [float("nan"), float("inf"), "soon"])
def test_photo_requires_finite_expiry(expiry):
    with pytest.raises(ValueError):
        OutgoingPhoto(data=png(), expires_at=expiry)


def test_reply_allows_only_one_typed_photo():
    image = OutgoingPhoto(data=png(), expires_at=NOW + 900)
    with pytest.raises(ValueError):
        ChatReply("too many", photos=(image, image))
    with pytest.raises(ValueError):
        ChatReply("arbitrary URL", photos=("https://example.org/screen.png",))
    with pytest.raises(ValueError):
        OutgoingPhoto(data=png(), expires_at=NOW + 900, caption="😀" * 513)


def test_plain_reply_never_uploads_an_image_even_when_text_mentions_screenshots(state):
    client = Mock()
    service = TelegramService(
        client, state, 123, lambda *_args: "Here is what I see.", clock=lambda: NOW,
    )
    service.process(update())
    client.send.assert_called_once()
    client.send_photo.assert_not_called()


def test_explicit_png_is_uploaded_as_multipart_after_text(state, monkeypatch):
    monkeypatch.setattr("ciel.chat.telegram.time.time", lambda: NOW)
    image = OutgoingPhoto(data=png(), expires_at=NOW + 900, caption="Current screen")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": len(requests)}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = TelegramClient("test:token", client=http)
        service = TelegramService(
            client, state, 123, lambda *_args: ChatReply("Look here.", photos=(image,)),
            clock=lambda: NOW,
        )
        service.process(update())
    assert [request.url.path.rsplit("/", 1)[-1] for request in requests] == [
        "sendMessage", "sendPhoto",
    ]
    assert json.loads(requests[0].content) == {"chat_id": 42, "text": "Look here."}
    upload = requests[1]
    assert upload.headers["content-type"].startswith("multipart/form-data; boundary=")
    assert b'name="photo"; filename="screen.png"' in upload.content
    assert b"Content-Type: image/png" in upload.content
    assert image.data in upload.content
    assert b"Current screen" in upload.content
    assert b'name="chat_id"\r\n\r\n42' in upload.content


@pytest.mark.parametrize("change", ["cancel", "unpair", "replace", "expire", "reset"])
def test_state_changes_after_text_suppress_photo_before_network(state, change):
    stop = threading.Event()
    now = [NOW]
    valid = [True]
    image = OutgoingPhoto(data=png(), expires_at=NOW + 900, valid=lambda: valid[0])
    client = Mock()

    def sent_text(*_args):
        if change == "cancel":
            stop.set()
        elif change == "unpair":
            state.unpair()
        elif change == "replace":
            state.bind(123, 99)
        elif change == "reset":
            valid[0] = False
        else:
            now[0] = NOW + 900

    client.send.side_effect = sent_text
    service = TelegramService(
        client, state, 123, lambda *_args: ChatReply("Screen described.", photos=(image,)),
        stop=stop, clock=lambda: now[0],
    )
    service.process(update())
    client.send.assert_called_once()
    client.send_photo.assert_not_called()


def test_expired_photo_is_never_uploaded(monkeypatch):
    monkeypatch.setattr("ciel.chat.telegram.time.time", lambda: NOW)
    requests = []
    transport = httpx.MockTransport(lambda request: requests.append(request))
    with httpx.Client(transport=transport) as http:
        client = TelegramClient("test:token", client=http)
        assert not client.send_photo(
            42, OutgoingPhoto(data=png(), expires_at=NOW), threading.Event(),
        )
    assert requests == []


@pytest.mark.parametrize("invalid", [lambda: False, Mock(side_effect=RuntimeError("reset"))])
def test_revoked_or_failed_image_lease_never_uploads(monkeypatch, invalid):
    monkeypatch.setattr("ciel.chat.telegram.time.time", lambda: NOW)
    requests = []
    transport = httpx.MockTransport(lambda request: requests.append(request))
    with httpx.Client(transport=transport) as http:
        client = TelegramClient("test:token", client=http)
        assert not client.send_photo(
            42, OutgoingPhoto(data=png(), expires_at=NOW + 900, valid=invalid), threading.Event(),
        )
    assert requests == []


def test_rate_limit_rechecks_expiry_without_retrying_stale_image(monkeypatch):
    now, requests = [NOW], []
    monkeypatch.setattr("ciel.chat.telegram.time.time", lambda: now[0])

    def respond(request):
        requests.append(request)
        now[0] = NOW + 900
        return httpx.Response(429, json={
            "ok": False, "error_code": 429, "parameters": {"retry_after": 1},
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = TelegramClient("test:token", client=http)
        assert not client.send_photo(
            42, OutgoingPhoto(data=png(), expires_at=NOW + 900), threading.Event(),
        )
    assert len(requests) == 1


def test_ambiguous_upload_failure_is_sanitized_and_never_retried(monkeypatch):
    monkeypatch.setattr("ciel.chat.telegram.time.time", lambda: NOW)
    requests = []

    def respond(request):
        requests.append(request)
        raise httpx.ReadTimeout(f"private-token-and-bytes: {request.url}", request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = TelegramClient("test:token", client=http)
        with pytest.raises(TelegramError, match="temporarily unreachable") as error:
            client.send_photo(
                42, OutgoingPhoto(data=png(), expires_at=NOW + 900), threading.Event(),
            )
    assert "private-token-and-bytes" not in str(error.value)
    assert len(requests) == 1


def test_worker_preserves_typed_attachment_and_cancellation_between_text_and_photo(state):
    stop = threading.Event()
    image = OutgoingPhoto(data=png(), expires_at=NOW + 900)
    client = Mock()

    def polls(offset, *, timeout):
        if offset == 0:
            return [update()]
        time.sleep(0.001)
        return []

    client.updates.side_effect = polls
    client.send.side_effect = lambda *_args: stop.set()
    TelegramService(
        client, state, 123, lambda *_args: ChatReply("Ready.", photos=(image,)),
        stop=stop, clock=lambda: NOW,
    ).run()
    client.send.assert_called_once()
    assert isinstance(client.send.call_args.args[1], ChatReply)
    client.send_photo.assert_not_called()


def test_pairing_change_between_text_chunks_stops_later_chunks_and_photo(state, monkeypatch):
    monkeypatch.setattr("ciel.chat.telegram.time.time", lambda: NOW)
    requests = []

    def respond(request):
        requests.append(request)
        state.unpair()
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        service = TelegramService(
            TelegramClient("test:token", client=http), state, 123,
            lambda *_args: ChatReply(
                "x" * 5000, photos=(OutgoingPhoto(data=png(), expires_at=NOW + 900),),
            ), clock=lambda: NOW,
        )
        service.process(update())
    assert len(requests) == 1
    assert requests[0].url.path.endswith("sendMessage")
    receipt = state._db.execute("SELECT status FROM telegram_receipts").fetchone()
    assert receipt == ("cancelled",)


def test_cancel_aborts_in_flight_owned_photo_upload(monkeypatch):
    entered, interrupted = threading.Event(), threading.Event()
    stop = threading.Event()
    results, upload_options = [], []

    class AsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def post(self, _url, **options):
            upload_options.append(options)
            entered.set()
            try:
                await asyncio.sleep(30)
            finally:
                interrupted.set()

        async def aclose(self):
            pass

    monkeypatch.setattr("ciel.chat.telegram.httpx.AsyncClient", AsyncClient)
    client = TelegramClient("test:token")

    def upload():
        results.append(client.send_photo(
            42, OutgoingPhoto(data=png(), expires_at=time.time() + 900), stop,
        ))

    worker = threading.Thread(target=upload, daemon=True)
    worker.start()
    assert entered.wait(1)
    stop.set()
    worker.join(1)
    try:
        assert not worker.is_alive()
        assert interrupted.is_set()
        assert results == [False]
        assert upload_options[0]["files"]["photo"][0] == "screen.png"
        assert "json" not in upload_options[0]
    finally:
        client.close()
