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
    ui = ChatUI(["telegram"])

    def secret_input(prompt):
        assert ui.menus[-1][0] == "Chat service"
        assert "Telegram bot token" in prompt
        return "synthetic-telegram-secret"

    monkeypatch.setattr(setup, "getpass", secret_input)
    setup.configure_chat(ui, configuration)

    assert configuration == {
        "GROQ_API_KEY": "keep-existing-provider", "CHAT_SERVICES": "telegram",
        "TELEGRAM_BOT_TOKEN": "synthetic-telegram-secret",
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
    ui = ChatUI([choice])

    setup.configure_chat(ui, configuration)

    assert configuration["TELEGRAM_BOT_TOKEN"] == "saved-secret"
    assert configuration["CHAT_SERVICES"] == ("telegram" if choice == "done" else "")
    assert configuration["CHAT_AUTOSTART"] == "true"  # Obsolete setting is preserved, ignored.
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
    ui = ChatUI(["telegram", cancel])
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
    ui = ChatUI(["telegram", "save"])
    monkeypatch.setattr(setup, "ConsoleUI", lambda: ui)
    monkeypatch.setattr(setup, "getpass", lambda _prompt: "replacement-token")

    def apply(_ui, configuration, *, project_dir):
        saved = dotenv_values(tmp_path / ".env")
        assert "CHAT_AUTOSTART" not in saved
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
        ["quick", "save", "yes"], chat_choices=["telegram"],
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
    assert '"-m" "ciel" "chat" "_worker"' in content
    assert f"WorkingDirectory={tmp_path}/\n" in content
    assert "EnvironmentFile" not in content
    assert "TOKEN" not in content and "KEY" not in content
    assert user_service.runner.call_args_list == [
        call(["systemctl", "--user", *arguments], capture_output=True, text=True, timeout=15,
             check=False)
        for arguments in [("show-environment",), ("daemon-reload",),
                          ("enable", autostart.UNIT_NAME),
                          ("start", autostart.UNIT_NAME),
                          ("is-active", "--quiet", autostart.UNIT_NAME)]
    ]
    assert "Restart=on-failure" in content and "StartLimitIntervalSec=0" in content


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
        subprocess.CompletedProcess([], 0, "", ""),
    ]

    assert autostart.autostart_status() == "automatic · running"


def test_disabling_absent_service_needs_no_systemd_or_home_changes(tmp_path, user_service):
    result = autostart.set_autostart(False, tmp_path)

    assert result.success
    user_service.runner.assert_not_called()
    assert not user_service.path.parent.exists()


def test_selected_service_ignores_legacy_manual_start_preference(tmp_path, monkeypatch):
    apply = MagicMock(return_value=autostart.AutostartResult(True, "automatic"))
    monkeypatch.setattr(autostart, "set_autostart", apply)
    ui = ChatUI(["done"])
    configuration = {"CHAT_SERVICES": "telegram", "TELEGRAM_BOT_TOKEN": "synthetic-token",
                     "CHAT_AUTOSTART": "false"}
    setup.configure_chat(ui, configuration)

    assert len(ui.menus) == 1
    assert setup.apply_chat_autostart(ui, configuration, project_dir=tmp_path)
    apply.assert_called_once_with(True, tmp_path, restart=True)
    assert "Automatic" in str(setup.chat_review_rows(configuration))
    assert configuration["CHAT_AUTOSTART"] == "false"


def test_repeated_startup_does_not_replace_or_restart_a_running_worker(
    tmp_path, user_service, monkeypatch,
):
    user_service.path.parent.mkdir(parents=True)
    user_service.path.write_text(autostart.render_unit(tmp_path))
    write = MagicMock(side_effect=AssertionError("Unchanged unit must not be replaced"))
    monkeypatch.setattr(autostart, "_write_unit", write)

    assert autostart.set_autostart(True, tmp_path).success

    commands = [item.args[0][2:] for item in user_service.runner.call_args_list]
    assert ["start", autostart.UNIT_NAME] in commands
    assert ["restart", autostart.UNIT_NAME] not in commands
    write.assert_not_called()


def test_failed_worker_start_reports_failure_without_offering_a_manual_start(
    tmp_path, user_service,
):
    def run(arguments, **_kwargs):
        return subprocess.CompletedProcess(arguments, 1 if "start" in arguments else 0, "", "")

    user_service.runner.side_effect = run
    result = autostart.set_autostart(True, tmp_path)

    assert not result.success
    assert "could not start" in result.message
    assert "ciel chat run" not in result.message


def test_unit_replacement_failure_preserves_existing_complete_file(
    tmp_path, user_service, monkeypatch,
):
    user_service.path.parent.mkdir(parents=True)
    original = autostart.render_unit(tmp_path / "old")
    user_service.path.write_text(original)
    monkeypatch.setattr(autostart.os, "replace", MagicMock(side_effect=OSError("interrupted")))

    assert not autostart.set_autostart(True, tmp_path).success
    assert user_service.path.read_text() == original
    assert not list(user_service.path.parent.glob(".ciel-chat-*"))


def test_pairing_guard_stops_worker_and_blocks_reconciliation_until_released(
    tmp_path, user_service,
):
    user_service.path.parent.mkdir(parents=True)
    user_service.path.write_text(autostart.render_unit(tmp_path))
    with autostart.pause_for_pairing() as paused:
        assert paused.success
        assert not autostart.set_autostart(True, tmp_path).success
        assert not any("start" in item.args[0] for item in user_service.runner.call_args_list)
    assert autostart.set_autostart(True, tmp_path).success
    commands = [item.args[0][2:] for item in user_service.runner.call_args_list]
    assert commands.index(["stop", autostart.UNIT_NAME]) < commands.index(
        ["start", autostart.UNIT_NAME],
    )


def test_pairing_guard_releases_lock_when_pairing_body_fails(tmp_path, user_service):
    with pytest.raises(ValueError, match="synthetic pairing error"):
        with autostart.pause_for_pairing() as paused:
            assert paused.success
            raise ValueError("synthetic pairing error")
    assert autostart.set_autostart(True, tmp_path).success


def test_main_setup_reports_incomplete_if_automatic_chat_cannot_start(
    monkeypatch, wizard_factory, capsys,
):
    wizard_factory(["quick", "save", "yes"], secrets=[""])
    monkeypatch.setattr("ciel.setup_wizard.apply_chat_autostart", lambda *a, **kw: False)
    from ciel import setup_wizard

    assert not setup_wizard.run_setup_wizard()
    assert "Automatic chat needs attention" in capsys.readouterr().out


def test_opening_ciel_restarts_chat_after_direct_env_edits_without_exposing_secrets(
    tmp_path, user_service,
):
    env_path = tmp_path / ".env"
    env_path.write_text("TELEGRAM_BOT_TOKEN=123:first-synthetic-secret\n")
    assert autostart.set_autostart(True, tmp_path).success
    first_unit = user_service.path.read_text()
    user_service.runner.reset_mock()
    env_path.write_text("TELEGRAM_BOT_TOKEN=123:replacement-synthetic-secret\n")

    assert autostart.set_autostart(True, tmp_path).success
    second_unit = user_service.path.read_text()
    assert first_unit != second_unit
    assert "first-synthetic-secret" not in first_unit
    assert "replacement-synthetic-secret" not in second_unit
    commands = [item.args[0][2:] for item in user_service.runner.call_args_list]
    assert ["restart", autostart.UNIT_NAME] in commands
    user_service.runner.reset_mock()

    assert autostart.set_autostart(True, tmp_path).success
    assert user_service.path.read_text() == second_unit
    commands = [item.args[0][2:] for item in user_service.runner.call_args_list]
    assert ["restart", autostart.UNIT_NAME] not in commands
