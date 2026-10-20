"""Headless messaging configuration and CLI dispatch regressions."""

import os
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from ciel import __main__
from ciel.chat import cli
from ciel.config import Settings


def test_optional_chat_settings_load_comma_selection_and_mask_secret(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "CHAT_SERVICES= Telegram, telegram \nTELEGRAM_BOT_TOKEN=123:private-token\n"
        "CHAT_STATE_PATH=some/chat.db\nCHAT_AUTOSTART=true\n"
    )
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=env)
    assert settings.chat.services == ["telegram"]
    assert settings.chat.state_path == "some/chat.db"
    assert settings.chat.autostart
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
    assert "autostart" in completed.stdout
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
    assert cli.main(["run"]) == 1
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

    def event(update_id, owner):
        return {"update_id": update_id, "message": {
            "from": {"id": owner, "is_bot": False},
            "chat": {"id": owner, "type": "private"},
            "text": "hello", "date": int(time.time()),
        }}

    class Client:
        def __init__(self, token):
            pass

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
            calls.append(("reply", text))
            return "hello back"

        def close(self):
            calls.append(("session closed",))

    monkeypatch.setattr(cli, "TelegramClient", Client)
    monkeypatch.setitem(sys.modules, "ciel.chat.session", SimpleNamespace(ChatSession=Session))
    assert cli.main(["run"]) == 0
    assert calls == [("session opened",), ("reply", "hello"),
                     ("session closed",), ("client closed",)]
