"""Telegram boundary tests use local HTTP mocks and never contact Telegram."""

import json
import logging
import threading
from unittest.mock import Mock

import httpx
import pytest

from ciel.chat.state import ChatAlreadyRunning, ChatState, polling_lock
from ciel.chat.telegram import (
    TelegramClient,
    TelegramError,
    TelegramService,
    message_chunks,
    private_message,
)

NOW = 1_800_000_000
TOKEN = "123456:private-example-token"


def update(update_id=1, owner=42, text="hello", *, now=NOW):
    return {
        "update_id": update_id,
        "message": {
            "from": {"id": owner, "is_bot": False},
            "chat": {"id": owner, "type": "private"},
            "text": text, "date": now,
        },
    }


def test_owner_only_before_conversation_and_duplicate_claim_survives_restart(tmp_path):
    state = ChatState(tmp_path / "chat.db")
    state.bind(123, 42)
    client = Mock()
    reply = Mock(return_value="Ciel reply")
    service = TelegramService(client, state, 123, reply, clock=lambda: NOW)
    service.process(update(1, owner=999, text="remember secret"))
    reply.assert_not_called()
    client.send.assert_not_called()
    service.process(update(2))
    reply.assert_called_once()
    client.send.assert_called_once()
    state.close()
    restored = ChatState(tmp_path / "chat.db")
    service = TelegramService(client, restored, 123, reply, clock=lambda: NOW)
    service.process(update(2))
    assert reply.call_count == 1
    assert restored.offset(123) == 3
    restored.close()


def test_transport_and_real_memory_session_confirm_and_recall_same_updated_fact(tmp_path):
    from ciel.chat.session import ChatSession
    from ciel.config import Settings

    state = ChatState(tmp_path / "chat.db")
    state.bind(123, 42)
    router = Mock()
    settings = Settings(_env_file=None, memory_db_path=str(tmp_path / "facts.db"),
                        ciel_persona_file="", internet_enabled=False)
    session = ChatSession(settings, channel="telegram", session_id="telegram:123:42",
                          router=router)
    client = Mock()
    service = TelegramService(client, state, 123, session.reply, clock=lambda: NOW)
    try:
        for incoming in (
            update(1, text="Remember my favorite game is Standoff 2"),
            update(2, text="Update my favorite game to Portal 2"),
            update(3, text="Yes, update it"),
            update(3, text="Yes, update it"),
            update(4, text="/memory"),
        ):
            service.process(incoming)
        assert client.send.call_count == 4
        assert "Portal 2" in client.send.call_args.args[1]
        assert "Standoff 2" not in client.send.call_args.args[1]
        assert session.store.get_fact("user:favorite_game").metadata["value"] == "Portal 2"
        router.send.assert_not_called()
    finally:
        session.close()
        state.close()


@pytest.mark.parametrize("change", [
    lambda event: event.update(edited_message=event.pop("message")),
    lambda event: event["message"]["chat"].update(type="group"),
    lambda event: event["message"]["chat"].update(id=999),
    lambda event: event["message"]["from"].update(is_bot=True),
    lambda event: event["message"].update(date=NOW - 301),
    lambda event: event["message"].update(date=NOW + 31),
    lambda event: event["message"].update(forward_origin={"type": "user"}),
    lambda event: event["message"].update(text=None),
])
def test_non_owner_sources_and_stale_messages_are_ignored(change):
    event = update()
    change(event)
    assert private_message(event, NOW) is None


def test_bot_binding_cannot_carry_to_different_token_identity(tmp_path):
    state = ChatState(tmp_path / "chat.db")
    state.bind(123, 42)
    reply = Mock()
    service = TelegramService(Mock(), state, 456, reply, clock=lambda: NOW)
    service.process(update())
    reply.assert_not_called()
    with pytest.raises(TelegramError, match="Pair your account"):
        service.run()
    state.close()


def test_claim_precedes_mutation_even_if_reply_crashes(tmp_path):
    state = ChatState(tmp_path / "chat.db")
    state.bind(123, 42)
    client = Mock()
    reply = Mock(side_effect=RuntimeError("private provider data"))
    service = TelegramService(client, state, 123, reply, clock=lambda: NOW)
    service.process(update())
    service.process(update())
    assert reply.call_count == 1
    assert "partly processed" in client.send.call_args.args[1]
    assert "private provider data" not in client.send.call_args.args[1]
    state.close()


def test_reply_delivery_failure_never_reprocesses_memory(tmp_path):
    state = ChatState(tmp_path / "chat.db")
    state.bind(123, 42)
    client = Mock()
    client.send.side_effect = TelegramError("Temporary delivery failure", transient=True)
    reply, report = Mock(return_value="saved"), Mock()
    service = TelegramService(client, state, 123, reply, clock=lambda: NOW, report=report)
    service.process(update())
    service.process(update())
    assert reply.call_count == 1
    assert "will not be run again" in report.call_args.args[0]
    state.close()


def test_chat_commands_reach_verified_owner_session_without_transport_policy_split(tmp_path):
    state = ChatState(tmp_path / "chat.db")
    state.bind(123, 42)
    client, reply = Mock(), Mock(return_value="local chat command reply")
    service = TelegramService(client, state, 123, reply, clock=lambda: NOW)
    commands = ["/help", "/start", "/memory", "/cancel", "/reset", "/fast hello"]
    for index, command in enumerate(commands, start=1):
        service.process(update(index, text=command))
    assert client.send.call_count == len(commands)
    assert [call.args[0] for call in reply.call_args_list] == commands
    state.close()


def test_pairing_ignores_backlog_wrong_codes_groups_and_then_binds(tmp_path):
    state = ChatState(tmp_path / "chat.db")
    client = Mock()
    code = {}
    polls = iter([[update(1, text="old history")], []])

    def poll(offset, timeout=20):
        if not code:
            return next(polls)
        group = update(3, owner=999, text=f"/start {code['value']}")
        group["message"]["chat"]["type"] = "group"
        return [update(2, text="/start wrong"), group,
                update(4, text=f"/start {code['value']}")]

    client.updates.side_effect = poll
    reply = Mock()
    service = TelegramService(client, state, 123, reply, clock=lambda: NOW)
    assert service.pair(lambda value: code.update(value=value)) == 42
    assert len(code["value"]) >= 20
    assert state.owner(123) == 42
    assert state.offset(123) == 5
    reply.assert_not_called()
    client.send.assert_called_once()
    assert client.send.call_args.args[0] == 42
    state.close()


def test_pairing_code_expires_without_binding(tmp_path):
    state = ChatState(tmp_path / "chat.db")
    client = Mock()
    client.updates.return_value = []
    shown = []
    service = TelegramService(client, state, 123, Mock(),
                              clock=lambda: NOW + 301 if shown else NOW)
    assert service.pair(shown.append) is None
    assert state.owner(123) is None
    client.send.assert_not_called()
    state.close()


def test_pairing_saved_even_if_confirmation_delivery_fails(tmp_path):
    state = ChatState(tmp_path / "chat.db")
    client, code = Mock(), {}
    client.updates.side_effect = lambda *_args, **_kwargs: (
        [update(1, text=f"/start {code['value']}")] if code else []
    )
    client.send.side_effect = TelegramError("Delivery failed", transient=True)
    report = Mock()
    service = TelegramService(client, state, 123, Mock(), clock=lambda: NOW, report=report)
    assert service.pair(lambda value: code.update(value=value)) == 42
    assert state.owner(123) == 42
    assert "Pairing saved" in report.call_args.args[0]
    state.close()


def test_poll_backoff_is_bounded_interruptible_and_does_not_run_turns(tmp_path):
    state = ChatState(tmp_path / "chat.db")
    client, stop, reply = Mock(), Mock(), Mock()
    client.updates.side_effect = TelegramError("Temporary failure", transient=True)
    stop.is_set.return_value = False
    stop.wait.return_value = False
    service = TelegramService(client, state, 123, reply, stop=stop, report=Mock())
    with pytest.raises(TelegramError):
        service.poll()
    assert client.updates.call_count == 6
    assert [call.args[0] for call in stop.wait.call_args_list] == [1, 2, 4, 8, 16]
    reply.assert_not_called()
    state.close()


def test_local_singleton_prevents_pair_run_race(tmp_path):
    path = tmp_path / "chat.db"
    with polling_lock(path):
        with pytest.raises(ChatAlreadyRunning):
            with polling_lock(path):
                pytest.fail("second poller was allowed")
    with polling_lock(path):
        pass


def test_unpair_retains_offsets_and_separates_other_bot(tmp_path):
    state = ChatState(tmp_path / "chat.db")
    state.bind(123, 42)
    assert state.claim(123, 20)
    assert state.offset(456) == 0
    state.unpair()
    assert state.owner(123) is None
    state.bind(123, 99)
    assert not state.claim(123, 20)
    assert state.offset(123) == 21
    assert state.bindings() == [(123, 99)]
    state.close()


def test_idle_week_accepts_randomized_lower_ids_without_replaying_receipts(tmp_path):
    state = ChatState(tmp_path / "chat.db")
    state.bind(123, 42)
    state.claim(123, 3, now=NOW)
    state.claim(123, 100_000, now=NOW)
    later = NOW + 8 * 24 * 60 * 60
    client, reply = Mock(), Mock(return_value="fresh reply")
    client.updates.return_value = [update(2, now=later), update(3, now=later)]
    service = TelegramService(client, state, 123, reply, clock=lambda: later)
    for incoming in service.poll():
        service.process(incoming)
    client.updates.assert_called_once_with(0, timeout=20)
    reply.assert_called_once()
    assert state.offset(123) == 4
    assert not state.recover_idle_offset(123, later + 1)
    state.close()


def test_bot_api_uses_plain_text_and_explicit_poll_parameters():
    calls = []

    def handle(request):
        calls.append((request.url.path.rsplit("/", 1)[-1], json.loads(request.content)))
        return httpx.Response(200, json={"ok": True, "result": []})

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = TelegramClient(TOKEN, client=http)
        assert client.updates(10) == []
        client.send(42, "<plain> **text**", threading.Event())
    assert calls[0] == ("getUpdates", {
        "offset": 10, "timeout": 20, "limit": 100, "allowed_updates": ["message"],
    })
    assert calls[1] == ("sendMessage", {"chat_id": 42, "text": "<plain> **text**"})


@pytest.mark.parametrize(("code", "hint", "transient"), [
    (401, "bot token", False), (404, "bot token", False),
    (409, "webhook or another", False), (500, "temporarily unavailable", True),
])
def test_api_errors_are_actionable_and_never_expose_response_secrets(code, hint, transient):
    def handle(request):
        return httpx.Response(code, json={
            "ok": False, "error_code": code, "description": f"private body {TOKEN}",
        })

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = TelegramClient(TOKEN, client=http)
        with pytest.raises(TelegramError, match=hint) as captured:
            client.identity()
    assert captured.value.transient is transient
    assert TOKEN not in str(captured.value)
    assert "private body" not in str(captured.value)


def test_http_exceptions_and_info_debug_logs_cannot_print_bot_token(caplog):
    def handle(request):
        if request.url.path.endswith("getMe"):
            return httpx.Response(200, json={"ok": True, "result": {"id": 123}})
        raise httpx.ReadTimeout(f"request failed {request.url}", request=request)

    caplog.set_level(logging.DEBUG)
    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = TelegramClient(TOKEN, client=http)
        assert client.identity() == {"id": 123}
        with pytest.raises(TelegramError) as captured:
            client.updates(0)
    assert TOKEN not in caplog.text
    assert TOKEN not in str(captured.value)
    assert captured.value.__suppress_context__
    logging.getLogger("httpcore.http11").debug("request target=b'/bot%s/getMe'", TOKEN)
    assert TOKEN not in caplog.text


def test_existing_webhook_is_reported_without_deleting_it():
    calls = []

    def handle(request):
        calls.append(request.url.path.rsplit("/", 1)[-1])
        return httpx.Response(200, json={"ok": True, "result": {"url": "https://example.test"}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = TelegramClient(TOKEN, client=http)
        with pytest.raises(TelegramError, match="webhook configured"):
            client.check_polling()
    assert calls == ["getWebhookInfo"]


def test_unicode_chunking_preserves_all_text_with_legal_lengths():
    text = "hello " + "💙" * 5000 + "end"
    chunks = message_chunks(text)
    assert "".join(chunks) == text
    assert all(len(chunk.encode("utf-16-le")) // 2 <= 4096 for chunk in chunks)


def test_explicit_rate_limit_can_retry_but_ambiguous_network_failure_cannot():
    attempts = []

    def handle(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, json={
                "ok": False, "error_code": 429, "parameters": {"retry_after": 1},
            })
        return httpx.Response(200, json={"ok": True, "result": {}})

    stop = Mock()
    stop.is_set.return_value = False
    stop.wait.return_value = False
    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = TelegramClient(TOKEN, client=http)
        client.send(42, "hello", stop)
    assert len(attempts) == 2
    stop.wait.assert_called_once_with(1)

    def ambiguous(request):
        raise httpx.ReadTimeout("unknown send outcome", request=request)

    with httpx.Client(transport=httpx.MockTransport(ambiguous)) as http:
        client = TelegramClient(TOKEN, client=http)
        with pytest.raises(TelegramError):
            client.send(42, "hello", stop)
    stop.wait.assert_called_once_with(1)


def test_stop_cancels_blocked_owned_http_request_and_closes_it(monkeypatch):
    import asyncio

    from ciel.chat import telegram

    entered, cancelled, closed, stop = (threading.Event() for _ in range(4))
    errors = []

    class BlockingHTTP:
        async def post(self, *_args, **_kwargs):
            entered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def aclose(self):
            closed.set()

    monkeypatch.setattr(telegram.httpx, "AsyncClient", lambda **_kwargs: BlockingHTTP())

    def request():
        client = TelegramClient(TOKEN, stop=stop)
        try:
            client.identity()
        except TelegramError as error:
            errors.append(error)
        finally:
            client.close()

    thread = threading.Thread(target=request, daemon=True)
    thread.start()
    assert entered.wait(1), "HTTP request did not start"
    stop.set()
    thread.join(1)

    assert not thread.is_alive(), "Cancellation waited for the 35-second HTTP timeout"
    assert cancelled.is_set() and closed.is_set()
    assert len(errors) == 1 and TOKEN not in str(errors[0])


def test_owned_client_reuses_one_loop_for_requests_and_cleanup_without_live_network(monkeypatch):
    import asyncio

    from ciel.chat import telegram

    calls, loops = [], []

    class AsyncHTTP:
        async def post(self, url, *, json):
            method = url.rsplit("/", 1)[-1]
            calls.append(method)
            loops.append(asyncio.get_running_loop())
            result = {"getMe": {"id": 123}, "getWebhookInfo": {"url": ""},
                      "getUpdates": [], "sendMessage": {"message_id": 1}}[method]
            return httpx.Response(200, json={"ok": True, "result": result})

        async def aclose(self):
            loops.append(asyncio.get_running_loop())
            calls.append("close")
            assert not {task for task in asyncio.all_tasks()
                        if task is not asyncio.current_task()}

    monkeypatch.setattr(telegram.httpx, "AsyncClient", lambda **_kwargs: AsyncHTTP())
    client = TelegramClient(TOKEN, stop=threading.Event())
    try:
        assert client.identity() == {"id": 123}
        client.check_polling()
        assert client.updates(0) == []
        client.send(42, "hello", threading.Event())
    finally:
        client.close()
    client.close()  # Closing twice cannot reopen or reuse the event loop.
    assert calls == ["getMe", "getWebhookInfo", "getUpdates", "sendMessage", "close"]
    assert len(set(loops)) == 1 and loops[0].is_closed()
