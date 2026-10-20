"""Chat setup remains local, private, cancellable, and independent of voice."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from dotenv import dotenv_values

from ciel.chat import autostart, setup
from tests.test_setup_wizard import wizard_factory as wizard_factory


class ChatUI:
    """Collect menu choices and output while isolating terminal interaction."""

    def __init__(self, choices):
        self.choices = iter(choices)
        self.output = []
        self.menus = []

    def note(self, text):
        self.output.append(text)

    banner = step = finish = note

    def summary(self, rows):
        self.output.extend(f"{label}: {value}" for label, value in rows)

    def choose(self, title, choices, default=None):
        self.menus.append((title, choices, default))
        selected = next(self.choices)
        if isinstance(selected, BaseException):
            raise selected
        answer = default if selected is None else selected
        assert answer in dict(choices)
        return answer


def test_chat_service_is_selected_before_its_token_and_summary_has_no_secret(monkeypatch):
    configuration = {"GROQ_API_KEY": "keep-existing-provider"}
    ui = ChatUI(["telegram", "on"])

    def secret_input(prompt):
        assert ui.menus[-1][0] == "Chat service"
        assert "Telegram bot token" in prompt
        return "synthetic-telegram-secret"

    monkeypatch.setattr(setup, "getpass", secret_input)
    setup.configure_chat(ui, configuration)

    assert configuration == {
        "GROQ_API_KEY": "keep-existing-provider", "CHAT_SERVICES": "telegram",
        "TELEGRAM_BOT_TOKEN": "synthetic-telegram-secret", "CHAT_AUTOSTART": "true",
    }
    assert set(dict(ui.menus[0][1])) == {"telegram", "done"}
    review = str(setup.chat_review_rows(configuration))
    assert "token configured" in review
    assert "pairing checked when connecting" in review
    assert "synthetic-" not in review + str(ui.output)


@pytest.mark.parametrize("choice", ["done", "disable"])
def test_keep_or_disable_chat_never_erases_a_saved_token(monkeypatch, choice):
    configuration = {
        "CHAT_SERVICES": "telegram", "TELEGRAM_BOT_TOKEN": "saved-secret",
        "CHAT_AUTOSTART": "true",
    }
    secret_input = MagicMock(side_effect=AssertionError("No token prompt expected"))
    monkeypatch.setattr(setup, "getpass", secret_input)
    ui = ChatUI([choice, None] if choice == "done" else [choice])

    setup.configure_chat(ui, configuration)

    assert configuration["TELEGRAM_BOT_TOKEN"] == "saved-secret"
    assert configuration["CHAT_SERVICES"] == ("telegram" if choice == "done" else "")
    assert configuration["CHAT_AUTOSTART"] == ("true" if choice == "done" else "false")
    secret_input.assert_not_called()


def test_blank_new_token_keeps_chat_disabled_and_does_not_offer_autostart(monkeypatch):
    monkeypatch.setattr(setup, "getpass", lambda _prompt: "")
    configuration = {}
    ui = ChatUI(["telegram"])

    setup.configure_chat(ui, configuration)

    assert configuration == {}
    assert len(ui.menus) == 1


@pytest.mark.parametrize("cancel", ["cancel", KeyboardInterrupt()])
def test_chat_setup_cancellation_preserves_env_and_does_not_change_services(
    tmp_path, monkeypatch, cancel,
):
    monkeypatch.chdir(tmp_path)
    original = "# Original\nGROQ_API_KEY='existing-synthetic-key'\n"
    (tmp_path / ".env").write_text(original)
    ui = ChatUI(["telegram", "on", cancel])
    monkeypatch.setattr(setup, "ConsoleUI", lambda: ui)
    monkeypatch.setattr(setup, "getpass", lambda _prompt: "new-synthetic-token")
    apply = MagicMock()
    monkeypatch.setattr(setup, "apply_chat_autostart", apply)

    assert not setup.run_chat_setup()

    assert (tmp_path / ".env").read_text() == original
    apply.assert_not_called()


def test_chat_setup_saves_before_autostart_and_preserves_other_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = (
        "# Voice stays as configured\nTTS_ENGINE='pocket'\ntelegram_bot_token='saved-token'\n"
    )
    (tmp_path / ".env").write_text(original)
    ui = ChatUI(["telegram", "on", "save"])
    monkeypatch.setattr(setup, "ConsoleUI", lambda: ui)
    monkeypatch.setattr(setup, "getpass", lambda _prompt: "replacement-token")

    def apply(_ui, configuration, *, project_dir):
        saved = dotenv_values(tmp_path / ".env")
        assert saved["CHAT_AUTOSTART"] == configuration["CHAT_AUTOSTART"] == "true"
        assert saved["CHAT_SERVICES"] == "telegram"
        assert saved["telegram_bot_token"] == "replacement-token"
        assert project_dir == tmp_path
        return True

    monkeypatch.setattr(setup, "apply_chat_autostart", apply)

    assert setup.run_chat_setup()

    assert (tmp_path / ".env").read_text().startswith(
        "# Voice stays as configured\nTTS_ENGINE='pocket'\n",
    )
    assert (tmp_path / ".env").stat().st_mode & 0o777 == 0o600
    assert "replacement-token" not in str(ui.output)


def test_main_setup_applies_chat_only_after_final_save(tmp_path, monkeypatch, wizard_factory):
    harness = wizard_factory(
        ["quick", "save", "yes"], chat_choices=["telegram"], autostart_choices=["on"],
        secrets=["", "synthetic-chat-token"],
    )

    def apply(_ui, configuration, *, project_dir):
        assert dotenv_values(tmp_path / ".env")["TELEGRAM_BOT_TOKEN"] == "synthetic-chat-token"
        assert project_dir == tmp_path
        return True

    monkeypatch.setattr("ciel.setup_wizard.apply_chat_autostart", apply)
    from ciel import setup_wizard

    assert setup_wizard.run_setup_wizard()
    assert harness.tts.call_count == 1


@pytest.fixture
def user_service(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(autostart.sys, "platform", "linux")
    monkeypatch.setattr(autostart.shutil, "which", lambda _command: "/usr/bin/systemctl")
    runner = MagicMock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(autostart.subprocess, "run", runner)
    return SimpleNamespace(path=autostart._unit_path(), runner=runner)


def test_login_service_runs_only_chat_in_the_current_environment(tmp_path, user_service):
    result = autostart.set_autostart(True, tmp_path)

    assert result.success
    content = user_service.path.read_text()
    assert '"-m" "ciel" "chat" "run"' in content
    assert f"WorkingDirectory={tmp_path}/\n" in content
    assert "EnvironmentFile" not in content
    assert "TOKEN" not in content and "KEY" not in content
    assert user_service.runner.call_args_list == [
        call(["systemctl", "--user", *arguments], capture_output=True, text=True, timeout=15,
             check=False)
        for arguments in [("show-environment",), ("daemon-reload",),
                          ("enable", autostart.UNIT_NAME)]
    ]
    assert "--now" not in str(user_service.runner.call_args_list)


def test_service_paths_escape_systemd_substitution_and_preserve_venv_symlink(tmp_path):
    python = tmp_path / 'venv $DATA 10% "quoted"' / "python"
    content = autostart.render_unit(tmp_path / "Ciel 10%", python)

    assert 'venv $$DATA 10%% \\"quoted\\"' in content
    assert "WorkingDirectory=" + str(tmp_path / "Ciel 10%%") + "/\n" in content
    with pytest.raises(ValueError):
        autostart.render_unit(Path("/tmp/invalid\nunit"), python)


def test_systemd_unavailable_does_not_claim_enabled_or_create_unit(tmp_path, user_service):
    user_service.runner.return_value = subprocess.CompletedProcess([], 1, "", "unavailable")

    result = autostart.set_autostart(True, tmp_path)

    assert not result.success
    assert "unavailable" in result.message
    assert not user_service.path.exists()


def test_disable_removes_only_managed_chat_service(tmp_path, user_service):
    user_service.path.parent.mkdir(parents=True)
    user_service.path.write_text("# Someone else's unit\n[Service]\n")

    assert not autostart.set_autostart(False, tmp_path).success
    user_service.runner.assert_not_called()
    assert user_service.path.read_text().startswith("# Someone else's")

    user_service.path.write_text(autostart.render_unit(tmp_path))
    assert autostart.set_autostart(False, tmp_path).success
    assert not user_service.path.exists()
    assert user_service.runner.call_args_list[-2].args[0] == [
        "systemctl", "--user", "disable", "--now", autostart.UNIT_NAME,
    ]


def test_status_uses_actual_enablement_without_claiming_an_api_connection(user_service):
    user_service.path.parent.mkdir(parents=True)
    user_service.path.write_text(autostart.render_unit(Path.cwd()))
    user_service.runner.side_effect = [
        subprocess.CompletedProcess([], 0, "", ""),
        subprocess.CompletedProcess([], 0, "enabled\n", ""),
    ]

    assert autostart.autostart_status() == "enabled"


def test_disabling_absent_service_needs_no_systemd_or_home_changes(tmp_path, user_service):
    result = autostart.set_autostart(False, tmp_path)

    assert result.success
    user_service.runner.assert_not_called()
    assert not user_service.path.parent.exists()
