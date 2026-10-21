"""Research progress crosses a bounded queue; Telegram HTTP stays on the polling thread."""

import json
import threading
import time
from queue import Queue
from unittest.mock import MagicMock

import pytest

from ciel.chat import cli
from ciel.chat.session import ChatSession
from ciel.chat.state import ChatState
from ciel.chat.telegram import TelegramError, TelegramService
from ciel.config import Settings
from ciel.internet import InternetResult, InternetSource
from ciel.providers.base import LLMResponse
from tests.test_chat_telegram_interrupts import NOW, update


@pytest.fixture
def state(tmp_path):
    selected = ChatState(tmp_path / "chat.db")
    selected.bind(123, 42)
    yield selected
    selected.close()


def test_progress_is_sent_once_on_polling_thread_before_final_reply(state):
    progress = Queue(maxsize=8)
    progress.put_nowait((threading.Event(), "Stale request progress"))
    started, acknowledged = threading.Event(), threading.Event()
    stop = threading.Event()
    main_thread = threading.get_ident()
    sent = []

    def reply(text, cancel):
        assert threading.get_ident() != main_thread
        progress.put_nowait((cancel, "I'll check the articles."))
        progress.put_nowait((cancel, "Duplicate progress"))
        started.set()
        assert acknowledged.wait(1)
        return "The sourced briefing"

    class Client:
        def updates(self, offset, *, timeout):
            assert threading.get_ident() == main_thread
            if offset == 0:
                return [update(1, "Search Mongolia news")]
            assert started.wait(1)
            time.sleep(0.001)
            return []

        def send(self, owner, text, cancel):
            assert threading.get_ident() == main_thread
            sent.append((owner, text))
            if text == "I'll check the articles.":
                acknowledged.set()
            else:
                stop.set()

    TelegramService(Client(), state, 123, reply, stop=stop, clock=lambda: NOW,
                    progress=progress).run()
    assert sent == [(42, "I'll check the articles."), (42, "The sourced briefing")]
    assert progress.empty()


def test_control_discards_cancelled_progress_and_stale_request_events(state):
    progress = Queue(maxsize=8)
    progress.put_nowait((threading.Event(), "Unrelated old progress"))
    started = threading.Event()
    stop = threading.Event()
    sent = []

    def reply(text, cancel):
        if text == "Search news":
            progress.put_nowait((cancel, "Obsolete progress"))
            started.set()
            assert cancel.wait(1)
            return "Cancelled briefing"
        return "Stopped"

    class Client:
        def updates(self, offset, *, timeout):
            if offset == 0:
                return [update(1, "Search news")]
            if offset == 2:
                assert started.wait(1)
                return [update(2, "stop")]
            time.sleep(0.001)
            return []

        def send(self, owner, text, cancel):
            sent.append(text)
            stop.set()

    TelegramService(Client(), state, 123, reply, stop=stop, clock=lambda: NOW,
                    progress=progress).run()
    assert sent == ["Stopped"]
    assert progress.empty()


def test_unpair_discards_queued_progress_before_any_send(state):
    progress = Queue(maxsize=8)
    started = threading.Event()
    sent = []

    def reply(text, cancel):
        progress.put_nowait((cancel, "Private request progress"))
        started.set()
        assert cancel.wait(1)
        return "Private result"

    class Client:
        def updates(self, offset, *, timeout):
            if offset == 0:
                return [update(1, "Search news")]
            assert started.wait(1)
            state.unpair()
            return []

        def send(self, owner, text, cancel):
            sent.append(text)

    with pytest.raises(TelegramError, match="pairing changed"):
        TelegramService(Client(), state, 123, reply, clock=lambda: NOW, progress=progress).run()
    assert sent == []


def test_chat_session_reports_only_actual_new_research(tmp_path):
    settings = Settings(_env_file=None, memory_db_path=str(tmp_path / "memory.db"),
                        ciel_persona_file="", internet_enabled=True)
    router = MagicMock()
    quote = "The example project supports private text chat."
    router.send.return_value = LLMResponse(json.dumps({"claims": [
        {"text": quote, "quote": quote, "source": 1},
    ]}), "model", "provider")
    session = ChatSession(settings, channel="telegram", session_id="test-owner", router=router)
    session.internet.prepare = MagicMock(return_value=InternetResult(
        attempted=True, query="example project", sources=[
            InternetSource("Example project", "https://example.com/guide", content=quote),
        ],
    ))
    session.progress = MagicMock()
    try:
        assert quote in session.reply("Search the example project")
        session.progress.assert_called_once_with("I'll check that online.")
        session.progress.reset_mock()
        assert quote in session.reply("Tell me what you found")
        session.progress.assert_not_called()
        session.internet.prepare.return_value = InternetResult()
        router.send.return_value = LLMResponse("Hi.", "model", "provider")
        assert session.reply("Hello") == "Hi."
        session.progress.assert_not_called()
    finally:
        session.close()


def test_cli_wires_bounded_progress_without_worker_http_calls(tmp_path, monkeypatch):
    state = ChatState(tmp_path / "chat.db")
    state.bind(123, 42)
    state.close()
    settings = Settings(
        _env_file=None, chat_services="telegram", telegram_bot_token="123:synthetic",
        chat_state_path=str(tmp_path / "chat.db"),
    )
    stop, acknowledged = threading.Event(), threading.Event()
    main_thread = threading.get_ident()
    sent, closed, queues = [], [], []
    original_service = TelegramService

    def service(*args, **kwargs):
        queues.append(kwargs["progress"])
        return original_service(*args, **kwargs)

    class Client:
        def __init__(self, token):
            assert threading.get_ident() == main_thread

        def identity(self):
            return {"id": 123}

        def check_polling(self):
            pass

        def updates(self, offset, *, timeout):
            assert threading.get_ident() == main_thread
            if offset == 0:
                return [update(1, "Search news", sent_at=int(time.time()))]
            time.sleep(0.001)
            return []

        def send(self, owner, text, cancel):
            assert threading.get_ident() == main_thread
            sent.append(text)
            if text == "I'll check the articles.":
                acknowledged.set()
            else:
                stop.set()

        def close(self):
            closed.append("client")

    class Session:
        def __init__(self, settings, **kwargs):
            assert threading.get_ident() != main_thread

        def reply(self, text, cancel_event):
            assert threading.get_ident() != main_thread
            for _ in range(20):
                self.progress("I'll check the articles.")
            assert acknowledged.wait(1)
            return "Briefing"

        def close(self):
            closed.append("session")

    monkeypatch.setattr(cli, "TelegramClient", Client)
    monkeypatch.setattr(cli, "TelegramService", service)
    monkeypatch.setattr("ciel.chat.session.ChatSession", Session)
    assert cli._serve(settings, stop) == 0
    assert queues[0].maxsize == 8
    assert sent == ["I'll check the articles.", "Briefing"]
    assert closed == ["session", "client"]
