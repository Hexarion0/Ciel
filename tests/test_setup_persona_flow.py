"""Keep AI personality drafts pending until the setup save decision."""

from unittest.mock import MagicMock, patch

import pytest
from dotenv import dotenv_values

from ciel import persona, setup_persona, setup_wizard
from ciel.config import Settings
from ciel.platform.system_info import SystemSnapshot, generate_system_prompt
from tests.test_setup_wizard import wizard_factory as wizard_factory


@pytest.mark.parametrize("disabled", ["", "   ", "${UNSET_CIEL_PROFILE}"])
def test_failed_persona_save_keeps_disabled_profile_disabled(
    tmp_path, monkeypatch, wizard_factory, disabled,
):
    monkeypatch.delenv("UNSET_CIEL_PROFILE", raising=False)
    (tmp_path / ".env").write_text(
        f"NIM_API_KEY=synthetic-key\nCIEL_PERSONA_FILE='{disabled}'\n",
    )
    persona_path = tmp_path / "persona.txt"
    original = "A previously disabled personality.\n"
    persona_path.write_text(original)
    monkeypatch.setattr(
        setup_persona, "run_persona_interview", MagicMock(return_value="A new personality."),
    )
    monkeypatch.setattr(persona, "update_setup_persona_file", MagicMock(return_value=False))
    wizard_factory(["quick", "save", "yes"], persona_choices=["interview"])

    setup_wizard.run_setup_wizard()

    raw = dotenv_values(tmp_path / ".env", interpolate=False)
    assert raw["CIEL_PERSONA_FILE"] == disabled
    assert not Settings(_env_file=tmp_path / ".env").ciel_persona_file.strip()
    assert persona_path.read_text() == original


def test_disabled_persona_is_enabled_only_after_profile_is_saved(
    tmp_path, monkeypatch, wizard_factory,
):
    env_path = tmp_path / ".env"
    env_path.write_text("NIM_API_KEY=synthetic-key\nCIEL_PERSONA_FILE=\n")
    draft = "A reviewed and successfully saved personality."
    monkeypatch.setattr(setup_persona, "run_persona_interview", MagicMock(return_value=draft))
    write_persona = persona.update_setup_persona_file

    def check_before_write(path, content):
        assert dotenv_values(env_path)["CIEL_PERSONA_FILE"] == ""
        return write_persona(path, content)

    monkeypatch.setattr(persona, "update_setup_persona_file", check_before_write)
    wizard_factory(["quick", "save", "yes"], persona_choices=["interview"])

    setup_wizard.run_setup_wizard()

    assert dotenv_values(env_path)["CIEL_PERSONA_FILE"] == "persona.txt"
    assert draft in persona.load_persona_preferences(str(tmp_path / "persona.txt"))


def test_activation_save_failure_keeps_saved_profile_disabled(
    tmp_path, monkeypatch, wizard_factory, capsys,
):
    env_path = tmp_path / ".env"
    env_path.write_text("NIM_API_KEY=synthetic-key\nCIEL_PERSONA_FILE=\n")
    draft = "An accepted personality, stored but not enabled."
    monkeypatch.setattr(setup_persona, "run_persona_interview", MagicMock(return_value=draft))
    save_configuration = setup_wizard._save_configuration
    writes = 0

    def fail_activation(path, configuration):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("Synthetic activation failure")
        save_configuration(path, configuration)

    monkeypatch.setattr(setup_wizard, "_save_configuration", fail_activation)
    wizard_factory(["quick", "save", "yes"], persona_choices=["interview"])

    setup_wizard.run_setup_wizard()

    assert dotenv_values(env_path)["CIEL_PERSONA_FILE"] == ""
    assert draft in persona.load_persona_preferences(str(tmp_path / "persona.txt"))
    assert "could not be enabled" in capsys.readouterr().out


@pytest.mark.parametrize("source", ["file", "environment", "lowercase"])
def test_persona_save_uses_expanded_path_and_preserves_stored_reference(
    tmp_path, monkeypatch, wizard_factory, source,
):
    reference = "${CIEL_PROFILE_LOCATION}"
    field = "CIEL_PERSONA_FILE"
    text = "NIM_API_KEY=synthetic-key\n"
    if source == "file":
        text += "CIEL_PROFILE_LOCATION=custom/persona.txt\n"
    elif source == "lowercase":
        text += "profile_location=custom/persona.txt\n"
        reference = "${profile_location}"
        field = "ciel_persona_file"
    else:
        monkeypatch.setenv("CIEL_PROFILE_LOCATION", "custom/persona.txt")
    text += f"{field}={reference}\n"
    (tmp_path / ".env").write_text(text)
    draft = "A personality saved to the configured location."
    monkeypatch.setattr(setup_persona, "run_persona_interview", MagicMock(return_value=draft))
    wizard_factory(["quick", "save", "yes"], persona_choices=["interview"])

    setup_wizard.run_setup_wizard()

    raw = dotenv_values(tmp_path / ".env", interpolate=False)
    assert raw[field] == reference
    assert draft in persona.load_persona_preferences(str(tmp_path / "custom/persona.txt"))
    assert Settings(_env_file=tmp_path / ".env").ciel_persona_file == "custom/persona.txt"
    assert not (tmp_path / reference).exists()


@pytest.mark.parametrize("reconfigure", [False, True])
def test_accepted_personality_saves_after_review_and_preserves_user_content(
    tmp_path, monkeypatch, wizard_factory, capsys, reconfigure,
):
    persona_path = tmp_path / "persona.txt"
    persona_path.write_text("# My notes\nGive concrete examples.\n", encoding="utf-8")
    assert persona.update_persona_file(str(persona_path), "set", "Use a calm tone.")
    if reconfigure:
        assert persona.update_setup_persona_file(str(persona_path), "Previous setup personality.")
    original = persona_path.read_bytes()
    draft = (
        "Be curious and concise. Ask a useful question when the answer is unclear. "
        "Use dry humor sparingly and explain unfamiliar ideas with concrete examples. "
        "Stay calm when the user is frustrated, acknowledge corrections, and continue the task. "
        "Offer enough context to make a choice without turning ordinary answers into speeches. "
        "Use warm, ordinary language and keep suggestions connected to the current request."
    )
    assert len(draft) > 300
    received = []

    def interview(ui, configuration):
        received.append(configuration.copy())
        assert configuration["NIM_API_KEY"] == "synthetic-new-setup-key"
        return draft

    monkeypatch.setattr(setup_persona, "run_persona_interview", interview)
    harness = wizard_factory(
        ["quick", "save", "yes"], secrets=["synthetic-new-setup-key"],
        persona_choices=["interview"],
    )

    def check_review(title):
        if title == "Ready to save?":
            assert persona_path.read_bytes() == original
            assert not (tmp_path / ".env").exists()

    harness.ui.on_choose = check_review
    setup_wizard.run_setup_wizard()

    saved_persona = persona_path.read_text(encoding="utf-8")
    assert "# My notes\nGive concrete examples." in saved_persona
    assert "Use a calm tone." in saved_persona
    assert draft in saved_persona
    assert "Previous setup personality." not in saved_persona
    assert dotenv_values(tmp_path / ".env")["CIEL_PERSONA_FILE"] == "persona.txt"
    assert len(received) == 1
    assert draft in capsys.readouterr().out
    with patch.dict("os.environ", {}, clear=True):
        settings = Settings(_env_file=tmp_path / ".env")
    prompt = generate_system_prompt(snapshot=SystemSnapshot(), settings=settings)
    assert draft in prompt
    assert "Give concrete examples." in prompt
    assert "Use a calm tone." in prompt


def test_accepted_personality_creates_missing_default_file_only_after_save(
    tmp_path, monkeypatch, wizard_factory,
):
    draft = "Use a friendly, practical tone."
    monkeypatch.setattr(
        setup_persona, "run_persona_interview", MagicMock(return_value=draft),
    )
    harness = wizard_factory(
        ["quick", "save", "yes"], secrets=["synthetic-key"], persona_choices=["interview"],
    )

    def check_review(title):
        if title == "Ready to save?":
            assert not (tmp_path / "persona.txt").exists()

    harness.ui.on_choose = check_review
    setup_wizard.run_setup_wizard()

    assert draft in persona.load_persona_preferences(str(tmp_path / "persona.txt"))
    assert (tmp_path / "persona.txt").stat().st_mode & 0o777 == 0o600


def test_cancel_after_accepting_personality_preserves_both_files(
    tmp_path, monkeypatch, wizard_factory,
):
    env_path = tmp_path / ".env"
    original_env = "# Existing setup\nNIM_API_KEY=synthetic-existing-key\n"
    env_path.write_text(original_env, encoding="utf-8")
    env_path.chmod(0o640)
    persona_path = tmp_path / "persona.txt"
    original_persona = "# Handwritten style\nBe clear.\n"
    persona_path.write_text(original_persona, encoding="utf-8")
    persona_path.chmod(0o640)
    interview = MagicMock(return_value="A reviewed but unsaved personality.")
    monkeypatch.setattr(setup_persona, "run_persona_interview", interview)
    harness = wizard_factory(["quick", "cancel"], persona_choices=["interview"])

    setup_wizard.run_setup_wizard()

    assert env_path.read_text(encoding="utf-8") == original_env
    assert persona_path.read_text(encoding="utf-8") == original_persona
    assert env_path.stat().st_mode & 0o777 == 0o640
    assert persona_path.stat().st_mode & 0o777 == 0o640
    harness.tts.assert_not_called()


def test_environment_save_failure_never_writes_accepted_personality(
    tmp_path, monkeypatch, wizard_factory,
):
    persona_path = tmp_path / "persona.txt"
    persona_path.write_text("Keep this personality.\n", encoding="utf-8")
    interview = MagicMock(return_value="An accepted draft.")
    write_persona = MagicMock()
    monkeypatch.setattr(setup_persona, "run_persona_interview", interview)
    monkeypatch.setattr(persona, "update_setup_persona_file", write_persona)

    def fail_save(*args):
        raise OSError("Synthetic write failure")

    monkeypatch.setattr(setup_wizard, "_save_configuration", fail_save)
    harness = wizard_factory(
        ["quick", "save"], secrets=["synthetic-key"], persona_choices=["interview"],
    )

    with pytest.raises(OSError, match="Synthetic write failure"):
        setup_wizard.run_setup_wizard()

    assert persona_path.read_text(encoding="utf-8") == "Keep this personality.\n"
    assert not (tmp_path / ".env").exists()
    write_persona.assert_not_called()
    harness.tts.assert_not_called()


def test_personality_save_failure_reports_saved_settings_and_preserves_personality(
    tmp_path, monkeypatch, wizard_factory, capsys,
):
    persona_path = tmp_path / "persona.txt"
    persona_path.write_text("Keep this personality.\n", encoding="utf-8")
    monkeypatch.setattr(
        setup_persona, "run_persona_interview", MagicMock(return_value="An accepted draft."),
    )
    monkeypatch.setattr(persona, "update_setup_persona_file", MagicMock(return_value=False))
    wizard_factory(
        ["quick", "save", "yes"], secrets=["synthetic-key"], names=["Example name"],
        persona_choices=["interview"],
    )

    setup_wizard.run_setup_wizard()

    assert dotenv_values(tmp_path / ".env")["CIEL_PREFERRED_NAME"] == "Example name"
    assert persona_path.read_text(encoding="utf-8") == "Keep this personality.\n"
    output = capsys.readouterr().out
    assert "Settings were saved" in output
    assert "custom personality could not be saved" in output
    assert "Custom personality saved." not in output


@pytest.mark.parametrize(
    ("original", "draft"),
    [
        ("Keep this personality.\n", "x" * 4001),
        ("x" * (persona.PERSONA_MAX_BYTES - 10), "A valid new personality."),
    ],
)
def test_oversized_personality_draft_or_combined_file_preserves_original(
    tmp_path, monkeypatch, wizard_factory, capsys, original, draft,
):
    persona_path = tmp_path / "persona.txt"
    persona_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        setup_persona, "run_persona_interview", MagicMock(return_value=draft),
    )
    wizard_factory(
        ["quick", "save", "yes"], secrets=["synthetic-key"], persona_choices=["interview"],
    )

    setup_wizard.run_setup_wizard()

    assert persona_path.read_text(encoding="utf-8") == original
    assert (tmp_path / ".env").exists()
    assert "custom personality could not be saved" in capsys.readouterr().out


@pytest.mark.parametrize("choice", ["keep", "interview"])
def test_keep_or_skipped_interview_preserves_saved_personality(
    tmp_path, monkeypatch, wizard_factory, choice,
):
    persona_path = tmp_path / "persona.txt"
    original = "# Personal notes\nKeep my style.\n"
    persona_path.write_text(original, encoding="utf-8")
    persona_path.chmod(0o640)
    interview = MagicMock(return_value=None)
    monkeypatch.setattr(setup_persona, "run_persona_interview", interview)
    wizard_factory(
        ["quick", "save", "yes"], secrets=["synthetic-key"], persona_choices=[choice],
    )

    setup_wizard.run_setup_wizard()

    assert persona_path.read_text(encoding="utf-8") == original
    assert persona_path.stat().st_mode & 0o777 == 0o640
    assert interview.call_count == (1 if choice == "interview" else 0)


@pytest.mark.parametrize("second_choice", ["interview", "discard"])
def test_returning_to_customize_can_keep_or_discard_the_accepted_draft(
    tmp_path, monkeypatch, wizard_factory, second_choice,
):
    (tmp_path / ".env").write_text("NIM_API_KEY=synthetic-existing-key\n", encoding="utf-8")
    persona_path = tmp_path / "persona.txt"
    original = "Keep my handwritten style.\n"
    persona_path.write_text(original, encoding="utf-8")
    draft = "A reviewed personality that survives an abandoned second interview."
    interview = MagicMock(side_effect=[draft, None])
    monkeypatch.setattr(setup_persona, "run_persona_interview", interview)
    wizard_factory(
        ["quick", "customize", "keep", "keep", "keep", None, None, "save", "yes"],
        secrets=["", "", ""], persona_choices=["interview", second_choice],
    )

    setup_wizard.run_setup_wizard()

    saved_persona = persona_path.read_text(encoding="utf-8")
    if second_choice == "discard":
        assert saved_persona == original
        assert interview.call_count == 1
    else:
        assert original.strip() in saved_persona
        assert draft in saved_persona
        assert interview.call_count == 2
