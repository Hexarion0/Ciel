"""The Telegram poller receives owner controls while one reply is still running."""

import threading
import time

import pytest

from ciel.chat.state import ChatState
from ciel.chat.telegram import TelegramError, TelegramService

NOW = 1_800_000_000


def update(update_id, text, owner=42, *, sent_at=NOW):
    return {
        "update_id": update_id,
        "message": {
            "from": {"id": owner, "is_bot": False},
            "chat": {"id": owner, "type": "private"},
            "text": text, "date": sent_at,
        },
    }


@pytest.fixture
def state(tmp_path):
    selected = ChatState(tmp_path / "chat.db")
    selected.bind(123, 42)
    yield selected
    selected.close()


@pytest.mark.parametrize("control", ["stop", "wait", "do not search"])
def test_owner_control_cancels_work_and_discards_stale_reply(state, control):
    started, cancelled = threading.Event(), threading.Event()
    stop = threading.Event()
    main_thread = threading.get_ident()
    replies, deliveries, polls = [], [], []

    def reply(text, cancel):
        replies.append(text)
        assert threading.get_ident() != main_thread
        if text == "Search Mongolia news":
            started.set()
            assert cancel.wait(1), "Polling was blocked behind generation"
            cancelled.set()
            return "Stale generated news briefing"
        assert cancelled.is_set()
        assert not cancel.is_set(), "The control inherited the cancelled request's event"
        return "Control accepted"

    class Client:
        def updates(self, offset, *, timeout):
            assert threading.get_ident() == main_thread
            polls.append(timeout)
            if offset == 0:
                return [update(1, "Search Mongolia news")]
            if offset == 2:
                assert started.wait(1)
                return [update(2, control)]
            time.sleep(0.001)
            return []

        def send(self, owner, text, cancel):
            assert threading.get_ident() == main_thread
            deliveries.append((owner, text))
            stop.set()

    service = TelegramService(Client(), state, 123, reply, stop=stop, clock=lambda: NOW)
    service.run()

    assert replies == ["Search Mongolia news", control]
    assert deliveries == [(42, "Control accepted")]
    assert 1 in polls
    assert state.offset(123) == 3


def test_control_drops_older_queued_turns_and_preserves_newer_turns_without_concurrency(state):
    started, cancelled = threading.Event(), threading.Event()
    stop = threading.Event()
    replies, worker_threads, sent = [], [], []
    main_thread = threading.get_ident()
    running = 0

    def reply(text, cancel):
        nonlocal running
        running += 1
        assert running == 1
        worker_threads.append(threading.get_ident())
        replies.append(text)
        try:
            if text == "Search news":
                started.set()
                assert cancel.wait(1)
                cancelled.set()
                return "Discard this stale result"
            assert cancelled.is_set()
            assert not cancel.is_set()
            return text + " reply"
        finally:
            running -= 1

    class Client:
        def updates(self, offset, *, timeout):
            assert threading.get_ident() == main_thread
            if offset == 0:
                return [update(1, "Search news")]
            if offset == 2:
                assert started.wait(1)
                return [update(2, "Older explicit search"), update(2, "Duplicate"),
                        update(3, "stop"), update(4, "New ordinary follow-up")]
            time.sleep(0.001)
            return []

        def send(self, owner, text, cancel):
            assert threading.get_ident() == main_thread
            sent.append(text)
            if text == "New ordinary follow-up reply":
                stop.set()

    TelegramService(Client(), state, 123, reply, stop=stop, clock=lambda: NOW).run()

    assert replies == ["Search news", "stop", "New ordinary follow-up"]
    assert sent == ["stop reply", "New ordinary follow-up reply"]
    assert len(set(worker_threads)) == 1
    assert worker_threads[0] != main_thread
    assert state.offset(123) == 5
    receipts = dict(state._db.execute("SELECT update_id, status FROM telegram_receipts"))
    assert receipts[2] == "cancelled"


def test_ordinary_next_message_never_cancels_the_active_reply(state):
    started, release = threading.Event(), threading.Event()
    stop = threading.Event()
    seen, sent = [], []

    def reply(text, cancel):
        seen.append(text)
        if text == "Search news":
            started.set()
            assert release.wait(1)
            assert not cancel.is_set()
        return text + " reply"

    class Client:
        def updates(self, offset, *, timeout):
            if offset == 0:
                return [update(1, "Search news")]
            if offset == 2:
                assert started.wait(1)
                return [update(2, "What about sports?")]
            release.set()
            time.sleep(0.001)
            return []

        def send(self, owner, text, cancel):
            sent.append(text)
            if text == "What about sports? reply":
                stop.set()

    TelegramService(Client(), state, 123, reply, stop=stop, clock=lambda: NOW).run()
    assert seen == ["Search news", "What about sports?"]
    assert sent == ["Search news reply", "What about sports? reply"]


@pytest.mark.parametrize("invalid", [
    update(2, "stop", owner=999), update(2, "stop", sent_at=NOW - 301),
    update(2, "stop", sent_at=NOW + 31),
])
def test_other_owners_and_stale_controls_cannot_interrupt_work(state, invalid):
    started, release = threading.Event(), threading.Event()
    stop = threading.Event()
    seen, sent = [], []

    def reply(text, cancel):
        seen.append(text)
        started.set()
        assert release.wait(1)
        assert not cancel.is_set()
        return "Completed normally"

    class Client:
        def updates(self, offset, *, timeout):
            if offset == 0:
                return [update(1, "Search news")]
            if offset == 2:
                assert started.wait(1)
                return [invalid]
            release.set()
            time.sleep(0.001)
            return []

        def send(self, owner, text, cancel):
            sent.append((owner, text))
            stop.set()

    TelegramService(Client(), state, 123, reply, stop=stop, clock=lambda: NOW).run()
    assert seen == ["Search news"]
    assert sent == [(42, "Completed normally")]


@pytest.mark.parametrize("next_owner", [None, 99])
def test_pairing_change_cancels_work_and_never_delivers_to_either_owner(state, next_owner):
    started, cancelled = threading.Event(), threading.Event()
    seen, sent = [], []

    def reply(text, cancel):
        seen.append(text)
        started.set()
        assert cancel.wait(1)
        cancelled.set()
        return "Private result belonging to the old owner"

    class Client:
        def updates(self, offset, *, timeout):
            if offset == 0:
                return [update(1, "Search news")]
            assert started.wait(1)
            if next_owner is None:
                state.unpair()
            else:
                state.bind(123, next_owner)
            return [update(2, "stop", owner=next_owner or 42)]

        def send(self, owner, text, cancel):
            sent.append((owner, text))

    with pytest.raises(TelegramError, match="pairing changed"):
        TelegramService(Client(), state, 123, reply, clock=lambda: NOW).run()
    assert cancelled.is_set()
    assert seen == ["Search news"]
    assert sent == []


def test_shutdown_cancels_active_work_and_skips_queued_requests(state):
    started, cancelled = threading.Event(), threading.Event()
    stop = threading.Event()
    seen, sent = [], []

    def reply(text, cancel):
        seen.append(text)
        started.set()
        assert cancel.wait(1)
        cancelled.set()
        return "Do not send after shutdown"

    class Client:
        def updates(self, offset, *, timeout):
            if offset == 0:
                return [update(1, "Search news")]
            if offset == 2:
                assert started.wait(1)
                return [update(2, "Queued memory update")]
            stop.set()
            return []

        def send(self, owner, text, cancel):
            sent.append(text)

    TelegramService(Client(), state, 123, reply, stop=stop, clock=lambda: NOW).run()
    assert cancelled.is_set()
    assert seen == ["Search news"]
    assert sent == []
    receipts = dict(state._db.execute("SELECT update_id, status FROM telegram_receipts"))
    assert receipts == {1: "cancelled", 2: "cancelled"}


def test_shutdown_wait_is_bounded_even_if_a_provider_ignores_cancellation(state, monkeypatch):
    from ciel.chat import telegram

    started, release, finished = (threading.Event() for _ in range(3))
    stop = threading.Event()
    sent = []
    monkeypatch.setattr(telegram, "REPLY_SHUTDOWN_SECONDS", 0.02)

    def reply(text, cancel):
        started.set()
        release.wait(1)
        finished.set()
        return "Late result"

    class Client:
        def updates(self, offset, *, timeout):
            if offset == 0:
                return [update(1, "Search news")]
            assert started.wait(1)
            stop.set()
            return []

        def send(self, owner, text, cancel):
            sent.append(text)

    before = time.monotonic()
    service = TelegramService(Client(), state, 123, reply, stop=stop, clock=lambda: NOW)
    try:
        service.run()
        assert time.monotonic() - before < 0.5
        assert not finished.is_set()
        assert sent == []
        stop.clear()
        with pytest.raises(TelegramError, match="previous chat reply"):
            service.run()
    finally:
        stop.set()
        release.set()
        assert finished.wait(1)


def test_synchronous_process_also_rechecks_owner_before_delivery(state):
    deliveries = []

    def reply(text, cancel):
        state.bind(123, 99)
        return "Old owner's private reply"

    class Client:
        def send(self, owner, text, cancel):
            deliveries.append((owner, text))

    TelegramService(Client(), state, 123, reply, clock=lambda: NOW).process(update(1, "hello"))
    assert deliveries == []


def test_waiting_queue_is_bounded_and_overflow_claim_is_never_retried(state, monkeypatch):
    from ciel.chat import telegram

    monkeypatch.setattr(telegram, "MAX_PENDING_REQUESTS", 2)
    started, release = threading.Event(), threading.Event()
    stop = threading.Event()
    seen, sent = [], []

    def reply(text, cancel):
        seen.append(text)
        if text == "Search news":
            started.set()
            assert release.wait(1)
            assert not cancel.is_set()
        return text + " reply"

    class Client:
        def updates(self, offset, *, timeout):
            if offset == 0:
                return [update(1, "Search news")]
            if offset == 2:
                assert started.wait(1)
                return [update(2, "First queued"), update(3, "Second queued"),
                        update(4, "Overflow"), update(4, "Overflow duplicate")]
            release.set()
            time.sleep(0.001)
            return []

        def send(self, owner, text, cancel):
            sent.append(text)
            if text == "Second queued reply":
                stop.set()

    TelegramService(Client(), state, 123, reply, stop=stop, clock=lambda: NOW).run()
    assert seen == ["Search news", "First queued", "Second queued"]
    assert sum("waiting queue is full" in text for text in sent) == 1
    receipts = dict(state._db.execute("SELECT update_id, status FROM telegram_receipts"))
    assert receipts == {1: "done", 2: "done", 3: "done", 4: "queue_full"}


def test_queued_messages_are_rechecked_for_freshness_before_memory_access(state):
    started, release = threading.Event(), threading.Event()
    stop = threading.Event()
    now = [NOW]
    seen, sent = [], []

    def reply(text, cancel):
        seen.append(text)
        started.set()
        assert release.wait(1)
        return "Long-running result"

    class Client:
        def updates(self, offset, *, timeout):
            if offset == 0:
                return [update(1, "Search news")]
            if offset == 2:
                assert started.wait(1)
                return [update(2, "Queued memory change")]
            now[0] = NOW + 301
            release.set()
            receipt = state._db.execute(
                "SELECT status FROM telegram_receipts WHERE update_id=2",
            ).fetchone()
            if receipt[0] == "ignored":
                stop.set()
            time.sleep(0.001)
            return []

        def send(self, owner, text, cancel):
            sent.append(text)

    TelegramService(Client(), state, 123, reply, stop=stop, clock=lambda: now[0]).run()
    assert seen == ["Search news"]
    assert sent == ["Long-running result"]


def test_reply_worker_exception_is_sanitized_without_replaying_claimed_request(state):
    stop = threading.Event()
    seen, sent = [], []

    def reply(text, cancel):
        seen.append(text)
        raise RuntimeError("private provider credential")

    class Client:
        def updates(self, offset, *, timeout):
            if offset == 0:
                return [update(1, "hello"), update(1, "Duplicate")]
            time.sleep(0.001)
            return []

        def send(self, owner, text, cancel):
            sent.append(text)
            stop.set()

    TelegramService(Client(), state, 123, reply, stop=stop, clock=lambda: NOW).run()
    assert seen == ["hello"]
    assert len(sent) == 1
    assert "partly processed" in sent[0]
    assert "private provider credential" not in sent[0]
