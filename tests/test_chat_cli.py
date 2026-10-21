"""Headless messaging configuration and CLI dispatch regressions."""

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from ciel import __main__
from ciel.chat import autostart, cli
from ciel.config import Settings


@pytest.fixture(autouse=True)
def isolated_user_services(tmp_path, monkeypatch):
    """Never inspect or change a developer's real login service during CLI tests."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


def test_optional_chat_settings_load_comma_selection_and_mask_secret(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "CHAT_SERVICES= Telegram, telegram \nTELEGRAM_BOT_TOKEN=123:private-token\n"
        "CHAT_STATE_PATH=some/chat.db\nCHAT_AUTOSTART=false\n"
    )
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=env)
    assert settings.chat.services == ["telegram"]
    assert settings.chat.state_path == "some/chat.db"
    assert not hasattr(settings.chat, "autostart")
    assert settings.chat.telegram_bot_token.get_secret_value() == "123:private-token"
    assert "private-token" not in repr(settings.chat)
    with patch.dict(os.environ, {}, clear=True):
        assert Settings(_env_file=None).chat.services == []
        with pytest.raises(ValidationError, match="supports telegram"):
            Settings(_env_file=None, chat_services="signal")


def test_chat_dispatch_precedes_voice_initialization(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "main", lambda arguments: called.append(arguments) or 0)
    assert __main__.main(["chat", "pair", "telegram"]) == 0
    assert called == [["pair", "telegram"]]


def test_chat_help_imports_neither_audio_runtime_nor_session():
    code = (
        "import sys; from ciel.__main__ import main; "
        "\ntry: main(['chat','--help'])"
        "\nexcept SystemExit as error: assert error.code == 0"
        "\nassert 'ciel.runtime' not in sys.modules"
        "\nassert 'ciel.chat.session' not in sys.modules"
        "\nassert not any(name.startswith('ciel.audio.') for name in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "_worker" not in completed.stdout
    assert " run " not in completed.stdout
    assert "autostart" not in completed.stdout
    assert "automatically" in completed.stdout
    assert "five minutes" in " ".join(completed.stdout.split())


def test_status_without_token_or_state_does_not_connect(tmp_path, monkeypatch, capsys):
    settings = Settings(_env_file=None, chat_services="", telegram_bot_token=None,
                        chat_state_path=str(tmp_path / "chat.db"))
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "TelegramClient", lambda *_: pytest.fail("network access"))
    assert cli.main(["status"]) == 0
    assert "not paired" in capsys.readouterr().out
    assert not (tmp_path / "chat.db").exists()


def test_run_requires_explicit_channel_selection(tmp_path, monkeypatch, capsys):
    settings = Settings(_env_file=None, chat_services="", telegram_bot_token="123:example",
                        chat_state_path=str(tmp_path / "chat.db"))
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "TelegramClient", lambda *_: pytest.fail("network access"))
    assert cli.main(["_worker"]) == 1
    assert "ciel chat setup" in capsys.readouterr().out


def test_unpair_works_locally_even_after_removing_token(tmp_path, monkeypatch, capsys):
    from ciel.chat.state import ChatState

    path = tmp_path / "chat.db"
    state = ChatState(path)
    state.bind(123, 42)
    state.close()
    settings = Settings(_env_file=None, chat_services="", telegram_bot_token=None,
                        chat_state_path=str(path))
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    assert cli.main(["unpair"]) == 0
    state = ChatState(path)
    assert state.owner(123) is None
    state.close()
    assert "removed" in capsys.readouterr().out


def test_run_constructs_session_only_for_paired_owner_and_closes_it(tmp_path, monkeypatch):
    from ciel.chat.state import ChatState

    path = tmp_path / "chat.db"
    state = ChatState(path)
    state.bind(123, 42)
    state.close()
    settings = Settings(_env_file=None, chat_services="telegram", telegram_bot_token="123:example",
                        chat_state_path=str(path))
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    calls = []
    service_stops = []

    def event(update_id, owner):
        return {"update_id": update_id, "message": {
            "from": {"id": owner, "is_bot": False},
            "chat": {"id": owner, "type": "private"},
            "text": "hello", "date": int(time.time()),
        }}

    class Client:
        def __init__(self, token):
            pass

        def set_stop_event(self, stop):
            service_stops.append(stop)

        def identity(self):
            return {"id": 123}

        def check_polling(self):
            pass

        def updates(self, offset, timeout):
            return [event(1, 999), event(2, 42)]

        def send(self, owner, text, stop):
            calls.append(("send", owner, text))

        def close(self):
            calls.append(("client closed",))

    class Session:
        def __init__(self, actual_settings, **kwargs):
            assert actual_settings is settings
            assert kwargs["channel"] == "telegram"
            assert kwargs["session_id"].startswith("telegram:123:42:")
            calls.append(("session opened",))

        def reply(self, text, cancel_event):
            cancel_event.set()
            service_stops[0].set()
            calls.append(("reply", text))
            return "hello back"

        def close(self):
            calls.append(("session closed",))

    monkeypatch.setattr(cli, "TelegramClient", Client)
    monkeypatch.setitem(sys.modules, "ciel.chat.session", SimpleNamespace(ChatSession=Session))
    assert cli.main(["_worker"]) == 0
    assert calls == [("session opened",), ("reply", "hello"),
                     ("session closed",), ("client closed",)]


@pytest.mark.parametrize("command", [["run"], ["autostart", "disable"], ["autostart", "enable"]])
def test_manual_chat_start_and_autostart_toggle_are_not_public_commands(command):
    with pytest.raises(SystemExit) as error:
        cli.main(command)
    assert error.value.code == 2


def test_pairing_resumes_automatic_worker_after_releasing_pairing_and_polling_locks(
    tmp_path, monkeypatch,
):
    from ciel.chat.state import polling_lock

    settings = Settings(_env_file=None, chat_services="telegram", telegram_bot_token="123:example",
                        chat_state_path=str(tmp_path / "chat.db"))
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    steps = []

    @contextmanager
    def pause():
        steps.append("pause")
        yield autostart.AutostartResult(True, "paused")
        steps.append("pause released")

    def serve(actual, stop, *, pair):
        assert actual is settings and pair
        with polling_lock(settings.chat.state_path):
            steps.append("pair")
        return 0

    def reconcile(configuration, *, project_dir):
        assert configuration == settings.chat
        with polling_lock(configuration.state_path):
            steps.append("resume")
        return autostart.AutostartResult(True, "resumed")

    monkeypatch.setattr(autostart, "pause_for_pairing", pause)
    monkeypatch.setattr(autostart, "reconcile_chat_startup", reconcile)
    monkeypatch.setattr(cli, "_serve", serve)

    assert cli.main(["pair"]) == 0
    assert steps == ["pause", "pair", "pause released", "resume"]


def test_pairing_does_not_succeed_if_automatic_resume_fails(tmp_path, monkeypatch, capsys):
    settings = Settings(_env_file=None, chat_services="telegram", telegram_bot_token="123:example",
                        chat_state_path=str(tmp_path / "chat.db"))
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    @contextmanager
    def pause():
        yield autostart.AutostartResult(True, "paused")

    monkeypatch.setattr(autostart, "pause_for_pairing", pause)
    monkeypatch.setattr(autostart, "reconcile_chat_startup", lambda *a, **kw:
                        autostart.AutostartResult(False, "Automatic resume failed"))
    monkeypatch.setattr(cli, "_serve", lambda *a, **kw: 0)

    assert cli.main(["pair"]) == 1
    assert "Automatic resume failed" in capsys.readouterr().out


def test_unpaired_worker_waits_without_contacting_telegram(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, chat_services="telegram", telegram_bot_token="123:example",
                        chat_state_path=str(tmp_path / "chat.db"))
    stop = MagicMock()
    monkeypatch.setattr(cli, "TelegramClient", lambda *_: pytest.fail("unpaired network access"))

    assert cli._serve(settings, stop) == 0
    stop.wait.assert_called_once_with(5)


def test_worker_retries_transient_identity_or_webhook_check_failures(monkeypatch):
    import threading

    stop = threading.Event()
    waits = []
    monkeypatch.setattr(stop, "wait", lambda delay: waits.append(delay) or False)
    calls = []

    def serve(_settings, actual_stop):
        calls.append(True)
        if len(calls) == 1:
            raise cli.TelegramError("Temporary Telegram connection failure", transient=True)
        actual_stop.set()
        return 0

    monkeypatch.setattr(cli, "_serve", serve)
    assert cli._worker(Settings(_env_file=None), stop) == 0
    assert len(calls) == 2 and waits == [2]


def test_normal_launch_reconciles_chat_before_audio_is_initialized(tmp_path, monkeypatch):
    from ciel import config, platform

    settings = Settings(_env_file=None, chat_services="telegram", telegram_bot_token="123:example",
                        chat_state_path=str(tmp_path / "chat.db"))
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    steps = []

    def automatic_chat(configuration, *, project_dir):
        assert configuration == settings.chat
        steps.append("chat")
        return autostart.AutostartResult(True, "automatic")

    def audio():
        steps.append("audio")
        raise RuntimeError("synthetic unavailable audio")

    monkeypatch.setattr(autostart, "reconcile_chat_startup", automatic_chat)
    monkeypatch.setattr(platform, "get_audio_backend", audio)
    with pytest.raises(RuntimeError, match="synthetic unavailable audio"):
        __main__.main([])
    assert steps == ["chat", "audio"]


def test_pairing_resume_uses_service_selection_saved_during_pairing(tmp_path, monkeypatch):
    initial = Settings(_env_file=None, chat_services="telegram", telegram_bot_token="123:example",
                       chat_state_path=str(tmp_path / "chat.db"))
    updated = Settings(_env_file=None, chat_services="", telegram_bot_token="123:example",
                       chat_state_path=str(tmp_path / "chat.db"))
    monkeypatch.setattr(cli, "get_settings", MagicMock(side_effect=[initial, updated]))

    @contextmanager
    def pause():
        yield autostart.AutostartResult(True, "paused")

    apply = MagicMock(return_value=autostart.AutostartResult(True, "automatic"))
    monkeypatch.setattr(autostart, "pause_for_pairing", pause)
    monkeypatch.setattr(autostart, "reconcile_chat_startup", apply)
    monkeypatch.setattr(cli, "_serve", lambda *a, **kw: 0)

    assert cli.main(["pair"]) == 0
    assert apply.call_args.args[0].services == []


def test_pair_interrupt_cancels_blocked_http_then_restores_worker_after_lock_release(
    tmp_path, monkeypatch,
):
    import asyncio
    import signal
    import threading

    from ciel.chat import telegram
    from ciel.chat.state import polling_lock

    settings = Settings(_env_file=None, chat_services="telegram", telegram_bot_token="123:example",
                        chat_state_path=str(tmp_path / "chat.db"))
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    steps, handlers, loops = [], {}, []
    cancel_inflight = threading.Event()

    class BlockingHTTP:
        async def post(self, url, *, json):
            import httpx

            method = url.rsplit("/", 1)[-1]
            if method == "getMe":
                return httpx.Response(200, json={"ok": True, "result": {"id": 123}})
            if method == "getWebhookInfo":
                return httpx.Response(200, json={"ok": True, "result": {"url": ""}})
            if json["timeout"] == 0:  # Initial backlog drain, before showing the code.
                return httpx.Response(200, json={"ok": True, "result": []})
            loops.append(asyncio.get_running_loop())
            steps.append("http entered")
            handlers[signal.SIGINT](signal.SIGINT, None)
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                steps.append("http cancelled")
                cancel_inflight.set()
                raise

        async def aclose(self):
            assert cancel_inflight.is_set()
            assert not {task for task in asyncio.all_tasks()
                        if task is not asyncio.current_task()}
            steps.append("http closed")

    def register(signum, callback):
        previous = handlers.get(signum)
        handlers[signum] = callback
        return previous

    @contextmanager
    def pause():
        steps.append("service paused")
        try:
            yield autostart.AutostartResult(True, "paused")
        finally:
            steps.append("pairing guard released")

    def resume(_configuration, *, project_dir):
        with polling_lock(settings.chat.state_path):
            assert loops[0].is_closed()
            steps.append("service resumed")
        return autostart.AutostartResult(True, "automatic")

    monkeypatch.setattr(telegram.httpx, "AsyncClient", lambda **_kwargs: BlockingHTTP())
    monkeypatch.setattr(cli.signal, "signal", register)
    monkeypatch.setattr(autostart, "pause_for_pairing", pause)
    monkeypatch.setattr(autostart, "reconcile_chat_startup", resume)

    assert cli.main(["pair"]) == 0
    assert steps == ["service paused", "http entered", "http cancelled", "http closed",
                     "pairing guard released", "service resumed"]
