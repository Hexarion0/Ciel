"""Setup tests isolate terminal input, secrets, devices, and playback."""

import json
import os
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from dotenv import dotenv_values

from ciel import setup_wizard
from ciel.terminal_ui import ConsoleUI


class ScriptedUI:
    """Exercise wizard decisions independently of terminal menu rendering."""

    def __init__(
        self, choices, prompts=(), *, names=(), cities=(), memory_choices=(), persona_choices=(),
        provider_choices=(), more_provider_choices=(),
        chat_choices=(), autostart_choices=(), search_choices=(),
    ):
        self.choices = iter(choices)
        self.prompts = iter(prompts)
        self.names = iter(names)
        self.cities = iter(cities)
        self.memory_choices = iter(memory_choices)
        self.persona_choices = iter(persona_choices)
        self.provider_choices = iter(provider_choices)
        self.more_provider_choices = iter(more_provider_choices)
        self.chat_choices = iter(chat_choices)
        self.autostart_choices = iter(autostart_choices)
        self.search_choices = iter(search_choices)
        self.provider_menus = []
        self.summaries = []
        self.questions = []
        self.memory_defaults = []
        self.on_choose = None

    def banner(self, text):
        print(text)

    step = note = finish = banner

    def summary(self, rows):
        self.summaries.append(rows)
        for label, value in rows:
            print(f"{label}: {value}")

    def choose(self, title, choices, default=None):
        if self.on_choose is not None:
            self.on_choose(title)
        if title == "Remembering things":
            self.memory_defaults.append(default)
            answer = next(self.memory_choices, None)
        elif title == "Personality":
            answer = next(self.persona_choices, None)
        elif title == "API key provider":
            self.provider_menus.append((choices, default))
            answer = next(self.provider_choices, None)
        elif title == "Add another provider?":
            answer = next(self.more_provider_choices, None)
        elif title == "Search provider":
            answer = next(self.search_choices, None)
        elif title in {"Screen connection", "Screen model"}:
            answer = None
        elif title == "Chat service":
            answer = next(self.chat_choices, None)
        elif title == "Chat at login":
            answer = next(self.autostart_choices, None)
        else:
            answer = next(self.choices)
        if isinstance(answer, BaseException):
            raise answer
        selected = default if answer is None else answer
        assert selected in {value for value, _label in choices}, (title, selected)
        return selected

    def prompt(self, label):
        self.questions.append(label)
        if "city" in label.lower():
            answer = next(self.cities, "")
        elif "call you" in label.lower() or "preferred name" in label.lower():
            answer = next(self.names, "")
        else:
            answer = next(self.prompts)
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture
def wizard_factory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def create(
        choices, *, secrets=(), prompts=(), devices=(), names=(), cities=(), memory_choices=(),
        persona_choices=(), provider_choices=(), more_provider_choices=(),
        chat_choices=(), autostart_choices=(), search_choices=(),
    ):
        ui = ScriptedUI(
            choices, prompts, names=names, cities=cities, memory_choices=memory_choices,
            persona_choices=persona_choices, provider_choices=provider_choices,
            more_provider_choices=more_provider_choices,
            chat_choices=chat_choices, autostart_choices=autostart_choices,
            search_choices=search_choices,
        )
        secret_answers = iter(secrets)
        secret_prompts = []

        def secret_input(prompt):
            secret_prompts.append(prompt)
            print(prompt)
            answer = next(secret_answers)
            if isinstance(answer, BaseException):
                raise answer
            return answer

        monkeypatch.setattr(setup_wizard, "ConsoleUI", lambda: ui)
        monkeypatch.setattr(setup_wizard, "getpass", secret_input)
        monkeypatch.setattr("ciel.chat.setup.getpass", secret_input)
        monkeypatch.setattr(setup_wizard, "apply_chat_autostart", MagicMock(return_value=True))
        backend = MagicMock()
        backend.list_devices.return_value = list(devices)
        monkeypatch.setattr(setup_wizard, "get_audio_backend", lambda: backend)
        settings = MagicMock()
        monkeypatch.setattr(setup_wizard, "get_settings", settings)
        tts = MagicMock()
        tts.return_value.speak.return_value = True
        tts.return_value.last_engine = "pocket"
        monkeypatch.setattr(setup_wizard, "TextToSpeech", tts)
        assets = MagicMock(return_value=[])
        monkeypatch.setattr(setup_wizard, "custom_voice_asset_problems", assets)
        return SimpleNamespace(
            ui=ui, tts=tts, settings=settings, assets=assets, backend=backend,
            secret_prompts=secret_prompts,
        )

    return create


@pytest.mark.parametrize("mode", ["quick", "customize"])
@pytest.mark.parametrize("provider, label, key", [
    ("nim", "NVIDIA NIM", "NIM_API_KEY"),
    ("groq", "Groq", "GROQ_API_KEY"),
    ("openrouter", "OpenRouter", "OPENROUTER_API_KEY"),
])
def test_provider_is_chosen_before_its_key_and_personality_interview(
    tmp_path, wizard_factory, monkeypatch, capsys, mode, provider, label, key,
):
    from ciel import setup_persona

    secret = "synthetic-selected-provider-key"
    interview = MagicMock(return_value=None)
    monkeypatch.setattr(setup_persona, "run_persona_interview", interview)
    choices = [mode]
    if mode == "customize":
        choices.extend(["keep", "keep", "keep", None, None])
    harness = wizard_factory(
        [*choices, "save", "yes"], secrets=[secret], provider_choices=[provider],
        persona_choices=["interview"],
    )

    def check_prompt_order(title):
        if title == "API key provider":
            assert not harness.secret_prompts

    harness.ui.on_choose = check_prompt_order
    assert setup_wizard.run_setup_wizard()

    saved = dotenv_values(tmp_path / ".env")
    assert {name: value for name, value in saved.items() if name.endswith("_API_KEY")} == {
        key: secret,
    }
    assert len(harness.ui.provider_menus) == 1
    assert harness.secret_prompts == [f"{label} API key [optional; Enter to skip]: "]
    assert interview.call_args.args[1][key] == secret
    output = capsys.readouterr().out
    assert secret not in output
    assert f"{label} · key configured" in output


@pytest.mark.parametrize("mode", ["quick", "customize"])
@pytest.mark.parametrize("existing", [False, True])
def test_skip_or_keep_provider_does_not_ask_for_secrets(
    tmp_path, wizard_factory, mode, existing,
):
    original = "# Saved connection\ngroq_api_key='synthetic-kept-key'\n"
    if existing:
        (tmp_path / ".env").write_text(original)
    choices = [mode]
    if mode == "customize":
        choices.extend(["keep", "keep", "keep", None, None])
    harness = wizard_factory([*choices, "save", "yes"], provider_choices=["done"])

    assert setup_wizard.run_setup_wizard()

    assert not harness.secret_prompts
    if existing:
        assert (tmp_path / ".env").read_text().startswith(original)
    else:
        assert not any(key.endswith("_API_KEY") for key in dotenv_values(tmp_path / ".env"))
    menu, default = harness.ui.provider_menus[0]
    assert ("done", "Keep saved providers and continue" if existing else "Skip for now") in menu
    assert default == ("done" if existing else "nim")


def test_multiple_selected_providers_preserve_existing_key(tmp_path, wizard_factory, capsys):
    original = "# Saved connection\nnim_api_key='synthetic-original-key'\n"
    (tmp_path / ".env").write_text(original)
    harness = wizard_factory(
        ["quick", "save", "yes"], provider_choices=["groq", "openrouter"],
        more_provider_choices=["yes", "no"],
        secrets=["synthetic-groq-key", "synthetic-openrouter-key"],
    )

    assert setup_wizard.run_setup_wizard()

    saved = dotenv_values(tmp_path / ".env")
    assert (tmp_path / ".env").read_text().startswith(original)
    assert saved["GROQ_API_KEY"] == "synthetic-groq-key"
    assert saved["OPENROUTER_API_KEY"] == "synthetic-openrouter-key"
    assert len(harness.secret_prompts) == 2
    assert len(harness.ui.provider_menus) == 2
    assert "synthetic-" not in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["quick", "customize"])
@pytest.mark.parametrize("provider, label, key", [
    ("brave", "Brave Search", "BRAVE_SEARCH_API_KEY"),
    ("tavily", "Tavily", "TAVILY_API_KEY"),
])
def test_search_provider_is_selected_before_masked_key_without_network_check(
    tmp_path, wizard_factory, capsys, monkeypatch, mode, provider, label, key,
):
    network = MagicMock(side_effect=AssertionError("Setup must not call a search API"))
    monkeypatch.setattr("ciel.search_providers.search_api", network)
    choices = [mode]
    if mode == "customize":
        choices.extend(["keep", "keep", "keep", None, None])
    harness = wizard_factory(
        [*choices, "save", "yes"], provider_choices=["done"],
        search_choices=[provider], secrets=["synthetic-search-key"],
    )
    titles = []

    def on_choose(title):
        titles.append(title)
        if title == "Search provider":
            assert not harness.secret_prompts

    harness.ui.on_choose = on_choose
    assert setup_wizard.run_setup_wizard()

    saved = dotenv_values(tmp_path / ".env")
    assert saved["INTERNET_PROVIDER"] == provider
    assert saved[key] == "synthetic-search-key"
    assert harness.secret_prompts == [f"{label} API key [required]: "]
    output = capsys.readouterr().out
    assert "synthetic-search-key" not in output
    assert f"{label} · key configured; connection untested" in output
    assert titles.index("Search provider") < titles.index("Personality")
    network.assert_not_called()


def test_free_search_needs_no_key_and_preserves_unused_keys(tmp_path, wizard_factory):
    original = (
        "# Preserve unused keys\nINTERNET_PROVIDER=brave\n"
        "brave_search_api_key='synthetic-brave-key'\nTAVILY_API_KEY=synthetic-tavily-key\n"
    )
    (tmp_path / ".env").write_text(original)
    harness = wizard_factory(
        ["quick", "save", "yes"], provider_choices=["done"], search_choices=["ddgs"],
    )

    assert setup_wizard.run_setup_wizard()

    saved = dotenv_values(tmp_path / ".env")
    assert saved["INTERNET_PROVIDER"] == "ddgs"
    assert saved["brave_search_api_key"] == "synthetic-brave-key"
    assert saved["TAVILY_API_KEY"] == "synthetic-tavily-key"
    assert not harness.secret_prompts


def test_search_setup_keeps_existing_selected_key_when_input_is_blank(tmp_path, wizard_factory):
    (tmp_path / ".env").write_text("INTERNET_PROVIDER=tavily\nTAVILY_API_KEY=synthetic-old-key\n")
    harness = wizard_factory(
        ["quick", "save", "yes"], provider_choices=["done"], secrets=[""],
    )
    assert setup_wizard.run_setup_wizard()
    assert harness.secret_prompts == ["Tavily API key [configured; Enter to keep]: "]
    assert dotenv_values(tmp_path / ".env")["TAVILY_API_KEY"] == "synthetic-old-key"


def test_search_setup_requires_key_or_explicit_free_choice(tmp_path, wizard_factory, capsys):
    wizard_factory(
        ["quick", "save", "yes"], provider_choices=["done"],
        search_choices=["brave", "ddgs"], secrets=[""],
    )
    assert setup_wizard.run_setup_wizard()
    assert dotenv_values(tmp_path / ".env")["INTERNET_PROVIDER"] == "ddgs"
    assert "Brave Search needs an API key" in capsys.readouterr().out


def test_cancel_after_search_selection_does_not_save_key(tmp_path, wizard_factory):
    harness = wizard_factory(
        ["quick", "cancel"], provider_choices=["done"],
        search_choices=["tavily"], secrets=["synthetic-unsaved-key"],
    )
    assert not setup_wizard.run_setup_wizard()
    assert harness.secret_prompts
    assert not (tmp_path / ".env").exists()


@pytest.mark.parametrize("answer", ["", "synthetic-replacement-key"])
def test_selected_saved_provider_can_keep_or_replace_key(tmp_path, wizard_factory, answer):
    (tmp_path / ".env").write_text("groq_api_key='synthetic-original-key'\n")
    harness = wizard_factory(
        ["quick", "save", "yes"], provider_choices=["groq"], secrets=[answer],
    )

    assert setup_wizard.run_setup_wizard()

    assert dotenv_values(tmp_path / ".env")["groq_api_key"] == (
        answer or "synthetic-original-key"
    )
    assert harness.secret_prompts == ["Groq API key [configured; Enter to keep]: "]
    assert ("groq", "Groq · key configured") in harness.ui.provider_menus[0][0]


def test_cancel_after_adding_provider_keeps_original_file(tmp_path, wizard_factory):
    original = "# Existing key\nNIM_API_KEY='synthetic-original-key'\n"
    (tmp_path / ".env").write_text(original)
    harness = wizard_factory(
        ["quick", "cancel"], provider_choices=["openrouter"], secrets=["synthetic-new-key"],
    )

    assert not setup_wizard.run_setup_wizard()

    assert len(harness.secret_prompts) == 1
    assert (tmp_path / ".env").read_text() == original


def test_quick_start_reviews_before_saving_default_profile(tmp_path, wizard_factory, capsys):
    harness = wizard_factory([None, None, None], secrets=[""])
    env_path = tmp_path / ".env"

    def check_review(title):
        if title == "Ready to save?":
            assert not env_path.exists()

    harness.ui.on_choose = check_review
    setup_wizard.run_setup_wizard()
    saved = dotenv_values(env_path)
    assert env_path.stat().st_mode & 0o777 == 0o600
    assert saved["TTS_VOICE"] == "ciel" and saved["TTS_ENGINE"] == "pocket"
    assert (saved["STT_MODEL"], saved["STT_DEVICE"], saved["STT_COMPUTE_TYPE"]) == (
        "small.en", "cuda", "int8_float16",
    )
    assert saved["STT_BEAM_SIZE"] == "5" and saved["STT_RETRY_MODEL"] == "medium.en"
    assert saved["AMBIENT_LISTENING"] == "true" and saved["SHOW_AI_TRANSCRIPTS"] == "true"
    assert saved["BARGE_IN_MODE"] == "speech"
    harness.backend.list_devices.assert_not_called()
    harness.settings.cache_clear.assert_called_once()
    harness.tts.assert_called_once_with(
        voice_name="ciel", engine="pocket", enabled=True, output_device=None,
    )
    harness.tts.return_value.close.assert_called_once()
    output = capsys.readouterr().out
    assert "local tools remain available" in output
    assert "No cloud key; local tools available" in output
    assert "Run: ciel" in output and "connection verified" not in output.lower()


@pytest.mark.parametrize("configured", [False, True])
def test_quick_start_keeps_new_and_existing_secrets_hidden(
    tmp_path, wizard_factory, capsys, configured,
):
    secret = "synthetic-private-key-for-setup"
    if configured:
        (tmp_path / ".env").write_text(f'NIM_API_KEY="{secret}"\n')
    harness = wizard_factory(["quick", "save", "yes"], secrets=[] if configured else [secret])
    setup_wizard.run_setup_wizard()
    assert dotenv_values(tmp_path / ".env")["NIM_API_KEY"] == secret
    output = capsys.readouterr().out
    assert secret not in output and "key configured" in output
    assert "connection hasn't been tested yet" in output
    assert len(harness.ui.summaries) == 1


def test_existing_settings_comments_case_and_literals_survive_quick_start(
    tmp_path, wizard_factory, capsys,
):
    from ciel.config import Settings

    original = (
        "# Keep my settings\n"
        "stt_model=base.en\nSTT_DEVICE=cpu\nSTT_COMPUTE_TYPE=int8\nSTT_BEAM_SIZE=1\n"
        "STT_RETRY_MODEL=\nAMBIENT_LISTENING=false\nSHOW_AI_TRANSCRIPTS=false\n"
        "BARGE_IN_MODE=wake\nMEMORY_MAX_SHORT_TERM_TURNS=20\n"
        "TTS_VOICE=en_US-amy-medium\nTTS_ENGINE=piper\n"
        'AUDIO_INPUT_DEVICE=3\nAUDIO_OUTPUT_DEVICE="USB Speaker"\n'
        'CUSTOM_VALUE="literal ${NOT_AN_ENV_VAR}"\n'
    )
    (tmp_path / ".env").write_text(original)
    harness = wizard_factory(["quick", "save", None], secrets=[""])
    harness.tts.return_value.speak.return_value = False
    setup_wizard.run_setup_wizard()
    assert (tmp_path / ".env").read_text().startswith(original)
    output = capsys.readouterr().out
    assert "Audio test failed" in output and "Voice confirmed" not in output
    settings = Settings(_env_file=tmp_path / ".env")
    assert settings.audio.tts_voice == "en_US-amy-medium" and settings.audio.tts_engine == "piper"
    assert (settings.stt_model, settings.stt_device, settings.stt_compute_type) == (
        "base.en", "cpu", "int8",
    )
    assert settings.stt_beam_size == 1 and not settings.stt_retry_model
    assert not settings.ambient_listening and not settings.show_ai_transcripts
    assert settings.barge_in_mode == "wake" and settings.memory.max_short_term_turns == 20
    assert settings.audio.input_device == 3 and settings.audio.output_device == "USB Speaker"


def test_customize_can_reset_devices_and_change_speech_settings(tmp_path, wizard_factory):
    (tmp_path / ".env").write_text(
        "TTS_VOICE=ciel\nTTS_ENGINE=chatterbox_turbo\nAUDIO_INPUT_DEVICE=3\n"
    )
    harness = wizard_factory([
        "customize", "amy", "default", "custom", "wake", "off", "save", "yes",
    ], secrets=["", "", ""], prompts=["USB Speaker"])
    setup_wizard.run_setup_wizard()
    saved = dotenv_values(tmp_path / ".env")
    assert saved["TTS_ENGINE"] == "piper" and saved["TTS_VOICE"] == "en_US-amy-medium"
    assert "AUDIO_INPUT_DEVICE" not in saved and saved["AUDIO_OUTPUT_DEVICE"] == "USB Speaker"
    assert saved["AMBIENT_LISTENING"] == "false" and saved["SHOW_AI_TRANSCRIPTS"] == "false"
    harness.tts.assert_called_once_with(
        voice_name="en_US-amy-medium", engine="piper", enabled=True, output_device="USB Speaker",
    )


@pytest.mark.parametrize(
    ("existing", "choice", "voice", "engine"),
    [
        ("", "keep", "ciel", "pocket"),
        ("TTS_VOICE=en_US-amy-medium\n", "keep", "en_US-amy-medium", "piper"),
        ("TTS_ENGINE=piper\n", "keep", "en_US-amy-medium", "piper"),
        ("TTS_ENGINE=pocket\n", "keep", "ciel", "pocket"),
        ("TTS_VOICE=en_US-custom-medium\nTTS_ENGINE=auto\n", "keep",
         "en_US-custom-medium", "auto"),
        ("TTS_VOICE=en_US-amy-medium\nTTS_ENGINE=piper\n", "turbo", "ciel", "chatterbox_turbo"),
        ("TTS_VOICE=ciel\nTTS_ENGINE=chatterbox_turbo\n", "keep", "ciel", "chatterbox_turbo"),
        ("TTS_VOICE=ciel\n", "keep", "ciel", "chatterbox_turbo"),
        ("", "pocket", "ciel", "pocket"),
        ("TTS_VOICE=ciel\nTTS_ENGINE=pocket\n", "keep", "ciel", "pocket"),
    ],
)
def test_setup_voice_choices(tmp_path, wizard_factory, existing, choice, voice, engine):
    if existing:
        (tmp_path / ".env").write_text(existing)
    harness = wizard_factory([
        "customize", choice, "keep", "keep", None, None, "save", "yes",
    ], secrets=["", "", ""])
    setup_wizard.run_setup_wizard()
    saved = dotenv_values(tmp_path / ".env")
    assert saved["TTS_VOICE"] == voice and saved["TTS_ENGINE"] == engine
    harness.tts.assert_called_once_with(
        voice_name=voice, engine=engine, enabled=True, output_device=None,
    )
    assert harness.assets.call_count == (1 if engine == "chatterbox_turbo" else 0)
    harness.tts.return_value.close.assert_called_once()


def test_review_can_return_to_customize(tmp_path, wizard_factory):
    harness = wizard_factory([
        "quick", "customize", "amy", "keep", "keep", "ambient", "on", "save", "yes",
    ], secrets=["", "", "", ""])
    setup_wizard.run_setup_wizard()
    assert len(harness.ui.summaries) == 2
    assert dotenv_values(tmp_path / ".env")["TTS_ENGINE"] == "piper"


def test_numbered_terminal_menu_completes_customize_flow(tmp_path, wizard_factory, monkeypatch):
    harness = wizard_factory([], secrets=["", "", ""])
    output = StringIO()
    # Customize, personal defaults, skip keys, free search, screen off, skip chat, Amy.
    terminal_input = StringIO("2\n\n\n\n4\n\n\n\n\n3\n1\n1\n1\n1\n1\n1\n")
    monkeypatch.setattr(
        setup_wizard, "ConsoleUI", lambda: ConsoleUI(stream=output, input_stream=terminal_input),
    )
    setup_wizard.run_setup_wizard()
    assert dotenv_values(tmp_path / ".env")["TTS_VOICE"] == "en_US-amy-medium"
    assert "Review your setup" in output.getvalue()
    assert "Voice confirmed" in output.getvalue()
    harness.tts.return_value.close.assert_called_once()


def test_customize_enter_keeps_unchanged_boolean_spellings_and_comments(tmp_path, wizard_factory):
    original = (
        "# Retain exact spelling and inline notes\n"
        "ambient_listening=YES # My listening preference\n"
        "show_ai_transcripts=ON # My captions preference\n"
    )
    (tmp_path / ".env").write_text(original)
    wizard_factory([
        "customize", "keep", "keep", "keep", None, None, "save", "yes",
    ], secrets=["", "", ""])
    setup_wizard.run_setup_wizard()
    assert (tmp_path / ".env").read_text().startswith(original)


@pytest.mark.parametrize("existing", [False, True])
def test_cancel_review_leaves_configuration_untouched(tmp_path, wizard_factory, existing, capsys):
    env_path = tmp_path / ".env"
    original = '# Keep this unchanged\nNIM_API_KEY="synthetic-preserved"\n'
    if existing:
        env_path.write_text(original)
        env_path.chmod(0o640)
    harness = wizard_factory(["quick", "cancel"], secrets=[] if existing else [""])
    setup_wizard.run_setup_wizard()
    if existing:
        assert env_path.read_text() == original
        assert env_path.stat().st_mode & 0o777 == 0o640
    else:
        assert not env_path.exists()
    harness.settings.cache_clear.assert_not_called()
    harness.tts.assert_not_called()
    assert "No changes were saved" in capsys.readouterr().out


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, EOFError])
def test_interrupted_secret_prompt_does_not_create_configuration(
    tmp_path, wizard_factory, interruption, capsys,
):
    harness = wizard_factory(["quick"], secrets=[interruption()])
    setup_wizard.run_setup_wizard()
    assert not (tmp_path / ".env").exists()
    harness.tts.assert_not_called()
    assert "No changes were saved" in capsys.readouterr().out


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, EOFError])
def test_interrupted_customization_preserves_existing_configuration(
    tmp_path, wizard_factory, interruption,
):
    env_path = tmp_path / ".env"
    original = '# original\nNIM_API_KEY="synthetic-original"\n'
    env_path.write_text(original)
    harness = wizard_factory(["customize", interruption()], secrets=["replacement", "", ""])
    setup_wizard.run_setup_wizard()
    assert env_path.read_text() == original
    harness.settings.cache_clear.assert_not_called()
    assert list(tmp_path.glob(".ciel-setup-*")) == []


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, EOFError])
def test_interruption_after_save_keeps_configuration(
    tmp_path, wizard_factory, interruption, capsys,
):
    harness = wizard_factory(["quick", "save", interruption()], secrets=[""])
    setup_wizard.run_setup_wizard()
    assert dotenv_values(tmp_path / ".env")["TTS_ENGINE"] == "pocket"
    harness.tts.return_value.close.assert_called_once()
    assert "Configuration is saved" in capsys.readouterr().out


def test_play_again_retries_then_requires_user_confirmation(wizard_factory, capsys):
    harness = wizard_factory(["quick", "save", "again", "yes"], secrets=[""])
    harness.tts.return_value.speak.side_effect = [False, True]
    setup_wizard.run_setup_wizard()
    assert harness.tts.call_count == 2 and harness.tts.return_value.close.call_count == 2
    output = capsys.readouterr().out
    assert "Audio test failed" in output and "Voice confirmed" in output


def test_output_device_retry_saves_and_uses_the_new_device(tmp_path, wizard_factory):
    device = SimpleNamespace(index=9, name="USB Speaker", max_output_channels=2)
    harness = wizard_factory([
        "quick", "save", "output", "device:9", "yes",
    ], secrets=[""], devices=[device])
    setup_wizard.run_setup_wizard()
    assert dotenv_values(tmp_path / ".env")["AUDIO_OUTPUT_DEVICE"] == "9"
    assert harness.settings.cache_clear.call_count == 2
    assert harness.tts.call_args_list == [
        call(voice_name="ciel", engine="pocket", enabled=True, output_device=None),
        call(voice_name="ciel", engine="pocket", enabled=True, output_device=9),
    ]


def test_retry_can_reset_a_newly_saved_output_device_to_system_default(tmp_path, wizard_factory):
    device = SimpleNamespace(index=9, name="USB Speaker", max_output_channels=2)
    harness = wizard_factory([
        "customize", "amy", "keep", "device:9", None, None, "save",
        "output", "default", "yes",
    ], secrets=["", "", ""], devices=[device])
    setup_wizard.run_setup_wizard()
    assert "AUDIO_OUTPUT_DEVICE" not in dotenv_values(tmp_path / ".env")
    assert harness.settings.cache_clear.call_count == 2
    assert harness.tts.call_args_list == [
        call(voice_name="en_US-amy-medium", engine="piper", enabled=True, output_device=9),
        call(voice_name="en_US-amy-medium", engine="piper", enabled=True, output_device=None),
    ]


def test_playback_exception_keeps_saved_configuration_without_claiming_success(
    tmp_path, wizard_factory, capsys,
):
    harness = wizard_factory(["quick", "save", None], secrets=[""])
    harness.tts.return_value.speak.side_effect = RuntimeError("speaker unavailable")
    setup_wizard.run_setup_wizard()
    assert (tmp_path / ".env").exists()
    harness.tts.return_value.close.assert_called_once()
    output = capsys.readouterr().out
    assert "Audio test failed" in output
    assert "Voice confirmed" not in output and "Audio is not confirmed" in output


def test_missing_custom_assets_save_choice_without_claiming_success(
    tmp_path, wizard_factory, capsys,
):
    (tmp_path / ".env").write_text("TTS_VOICE=ciel\nTTS_ENGINE=chatterbox_turbo\n")
    harness = wizard_factory(["quick", "save", None], secrets=[""])
    harness.assets.return_value = ["reference_audio: /private/reference.wav"]
    setup_wizard.run_setup_wizard()
    assert dotenv_values(tmp_path / ".env")["TTS_ENGINE"] == "chatterbox_turbo"
    harness.tts.assert_not_called()
    output = capsys.readouterr().out
    assert "reference_audio: /private/reference.wav" in output and "choose Amy" in output
    assert "Voice confirmed" not in output and "Audio is not confirmed" in output


@pytest.mark.parametrize("engine", ["", "unsupported-engine"])
def test_invalid_saved_engine_reports_failure_without_overwriting_the_choice(
    tmp_path, wizard_factory, capsys, engine,
):
    (tmp_path / ".env").write_text(f"TTS_VOICE=ciel\nTTS_ENGINE={engine}\n")
    harness = wizard_factory(["quick", "save", None], secrets=[""])
    setup_wizard.run_setup_wizard()
    assert dotenv_values(tmp_path / ".env")["TTS_ENGINE"] == engine
    harness.tts.assert_not_called()
    output = capsys.readouterr().out
    assert "Audio test failed (ValueError)" in output
    assert "Audio is not confirmed" in output


def test_failed_save_leaves_original_configuration_intact(tmp_path, wizard_factory, monkeypatch):
    env_path = tmp_path / ".env"
    original = '# original\nNIM_API_KEY="synthetic-original"\n'
    env_path.write_text(original)
    harness = wizard_factory(["quick", "save"])
    real_set_key = setup_wizard.set_key
    writes = 0

    def fail_after_first_write(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("disk full")
        return real_set_key(*args, **kwargs)

    monkeypatch.setattr(setup_wizard, "set_key", fail_after_first_write)
    with pytest.raises(OSError, match="disk full"):
        setup_wizard.run_setup_wizard()
    assert env_path.read_text() == original
    harness.settings.cache_clear.assert_not_called()
    assert list(tmp_path.glob(".ciel-setup-*")) == []


def test_disabled_voice_output_does_not_play_a_sample(tmp_path, wizard_factory, capsys):
    (tmp_path / ".env").write_text("TTS_ENABLED=false\n")
    harness = wizard_factory(["quick", "save", None], secrets=[""])
    setup_wizard.run_setup_wizard()
    harness.tts.assert_not_called()
    assert dotenv_values(tmp_path / ".env")["TTS_ENABLED"] == "false"
    assert "Voice output is disabled" in capsys.readouterr().out


def test_playback_errors_do_not_echo_a_saved_secret(tmp_path, wizard_factory, capsys):
    secret = "synthetic-never-echo-this"
    (tmp_path / ".env").write_text(f'NIM_API_KEY="{secret}"\n')
    harness = wizard_factory(["quick", "save", None])
    harness.tts.return_value.speak.side_effect = RuntimeError(f"backend failure with {secret}")
    setup_wizard.run_setup_wizard()
    output = capsys.readouterr().out
    assert secret not in output
    assert "RuntimeError" in output


@pytest.mark.parametrize("mode", ["quick", "customize"])
def test_both_setup_modes_ask_personal_questions_and_reload_answers(
    tmp_path, wizard_factory, mode,
):
    from ciel.config import Settings

    choices = (["quick", "save", "yes"] if mode == "quick" else [
        "customize", "keep", "keep", "keep", None, None, "save", "yes",
    ])
    harness = wizard_factory(
        choices, secrets=[""] * 3, names=["  Алекс  "],
        cities=["  Ulaanbaatar, Mongolia  "], memory_choices=["explicit"],
    )
    setup_wizard.run_setup_wizard()

    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=tmp_path / ".env")
    assert settings.ciel_preferred_name == "Алекс"
    assert settings.internet.home_city == "Ulaanbaatar, Mongolia"
    assert settings.memory.learning_mode == "explicit"
    assert len(harness.ui.questions) == 2
    assert harness.ui.memory_defaults == ["confirm"]
    review = dict(harness.ui.summaries[-1])
    assert review["Call you"] == "Алекс"
    assert review["Default city"] == "Ulaanbaatar, Mongolia"
    assert "Memory learning" in review


def test_quick_setup_asks_personal_questions_when_ai_key_already_exists(
    tmp_path, wizard_factory,
):
    (tmp_path / ".env").write_text('NIM_API_KEY="synthetic-existing"\n')
    harness = wizard_factory(
        ["quick", "save", "yes"], names=["Alex"], cities=["Hovd, Mongolia"],
        memory_choices=["automatic"],
    )
    setup_wizard.run_setup_wizard()

    saved = dotenv_values(tmp_path / ".env")
    assert saved["CIEL_PREFERRED_NAME"] == "Alex"
    assert saved["INTERNET_HOME_CITY"] == "Hovd, Mongolia"
    assert saved["MEMORY_LEARNING_MODE"] == "automatic"
    assert len(harness.ui.questions) == 2


@pytest.mark.parametrize("mode", ["quick", "customize"])
def test_personal_questions_enter_preserves_raw_values_comments_and_key_case(
    tmp_path, wizard_factory, mode,
):
    original = (
        "# Preserve my personal settings exactly\n"
        "ciel_preferred_name='  Alex ${UNDEFINED_NAME}  ' # Name note\n"
        "internet_home_city='  Hovd, Mongolia  ' # City note\n"
        "memory_learning_mode=explicit # Learning note\n"
    )
    (tmp_path / ".env").write_text(original)
    choices = (["quick", "save", "yes"] if mode == "quick" else [
        "customize", "keep", "keep", "keep", None, None, "save", "yes",
    ])
    harness = wizard_factory(choices, secrets=[""] * 3)
    setup_wizard.run_setup_wizard()

    assert (tmp_path / ".env").read_text().startswith(original)
    saved = dotenv_values(tmp_path / ".env", interpolate=False)
    assert saved["ciel_preferred_name"] == "  Alex ${UNDEFINED_NAME}  "
    assert saved["internet_home_city"] == "  Hovd, Mongolia  "
    assert saved["memory_learning_mode"] == "explicit"
    assert harness.ui.memory_defaults == ["explicit"]


@pytest.mark.parametrize("existing", [False, True])
def test_personal_questions_allow_no_name_or_default_city(tmp_path, wizard_factory, existing):
    from ciel.config import Settings

    if existing:
        (tmp_path / ".env").write_text(
            "CIEL_PREFERRED_NAME=Old Name\nINTERNET_HOME_CITY=Old City\n",
        )
    answer = "-" if existing else ""
    harness = wizard_factory(
        ["quick", "save", "yes"], secrets=[""], names=[answer], cities=[answer],
    )
    setup_wizard.run_setup_wizard()

    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=tmp_path / ".env")
    assert settings.ciel_preferred_name == ""
    assert settings.internet.home_city == ""
    assert settings.memory.learning_mode == "confirm"
    review = dict(harness.ui.summaries[-1])
    assert review["Call you"] and review["Default city"]
    assert "Old Name" not in review["Call you"]
    assert "Old City" not in review["Default city"]


def test_review_can_revise_personal_answers_before_saving(tmp_path, wizard_factory):
    harness = wizard_factory(
        ["quick", "customize", "keep", "keep", "keep", None, None, "save", "yes"],
        secrets=[""] * 4, names=["First Name", "Revised Name"],
        cities=["First City", "Revised City"], memory_choices=["confirm", "automatic"],
    )
    setup_wizard.run_setup_wizard()

    saved = dotenv_values(tmp_path / ".env")
    assert saved["CIEL_PREFERRED_NAME"] == "Revised Name"
    assert saved["INTERNET_HOME_CITY"] == "Revised City"
    assert saved["MEMORY_LEARNING_MODE"] == "automatic"
    assert len(harness.ui.questions) == 4
    assert dict(harness.ui.summaries[0])["Call you"] == "First Name"
    assert dict(harness.ui.summaries[1])["Call you"] == "Revised Name"


@pytest.mark.parametrize("existing", [False, True])
def test_cancel_discards_personal_answers(tmp_path, wizard_factory, existing):
    env_path = tmp_path / ".env"
    original = "# Personal settings\nCIEL_PREFERRED_NAME=Original\n"
    if existing:
        env_path.write_text(original)
    harness = wizard_factory(
        ["quick", "cancel"], secrets=[""], names=["Changed"],
        cities=["Hovd, Mongolia"], memory_choices=["automatic"],
    )
    setup_wizard.run_setup_wizard()

    if existing:
        assert env_path.read_text() == original
    else:
        assert not env_path.exists()
    harness.settings.cache_clear.assert_not_called()


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, EOFError])
@pytest.mark.parametrize("question", ["name", "city", "memory"])
def test_interrupted_personal_questions_preserve_existing_file(
    tmp_path, wizard_factory, interruption, question,
):
    env_path = tmp_path / ".env"
    original = "# Personal settings\nCIEL_PREFERRED_NAME=Original\n"
    env_path.write_text(original)
    harness = wizard_factory(
        ["quick"], names=[interruption()] if question == "name" else ["Changed"],
        cities=[interruption()] if question == "city" else ["Hovd, Mongolia"],
        memory_choices=[interruption()] if question == "memory" else [],
    )
    setup_wizard.run_setup_wizard()

    assert env_path.read_text() == original
    harness.settings.cache_clear.assert_not_called()
    assert list(tmp_path.glob(".ciel-setup-*")) == []


@pytest.mark.parametrize(
    ("key", "invalid", "valid", "message"),
    [
        ("CIEL_PREFERRED_NAME", "n" * 101, "Alex", "100"),
        ("CIEL_PREFERRED_NAME", "Alex\x1b[0m", "Alex", "control"),
        ("INTERNET_HOME_CITY", "c" * 201, "Hovd, Mongolia", "200"),
        ("INTERNET_HOME_CITY", "Hovd\nMongolia", "Hovd, Mongolia", "control"),
    ],
)
def test_invalid_personal_answers_retry_before_saving(
    tmp_path, wizard_factory, capsys, key, invalid, valid, message,
):
    harness = wizard_factory(
        ["quick", "save", "yes"], secrets=[""],
        names=[invalid, valid] if key == "CIEL_PREFERRED_NAME" else ["Alex"],
        cities=[invalid, valid] if key == "INTERNET_HOME_CITY" else ["Hovd, Mongolia"],
    )
    setup_wizard.run_setup_wizard()

    assert dotenv_values(tmp_path / ".env")[key] == valid
    assert len(harness.ui.questions) == 3
    assert message in capsys.readouterr().out.lower()


def test_invalid_saved_memory_choice_defaults_safely_and_is_repaired_on_save(
    tmp_path, wizard_factory,
):
    (tmp_path / ".env").write_text("MEMORY_LEARNING_MODE=unsupported\n")
    harness = wizard_factory(["quick", "save", "yes"], secrets=[""])
    setup_wizard.run_setup_wizard()

    assert harness.ui.memory_defaults == ["confirm"]
    assert dotenv_values(tmp_path / ".env")["MEMORY_LEARNING_MODE"] == "confirm"


def test_personalization_defaults_only_fill_missing_settings():
    fresh = setup_wizard._initial_configuration({})
    assert fresh["CIEL_PREFERRED_NAME"] == ""
    assert fresh["INTERNET_HOME_CITY"] == ""
    assert fresh["MEMORY_LEARNING_MODE"] == "confirm"
    existing = {
        "CIEL_PREFERRED_NAME": "  Alex  ", "INTERNET_HOME_CITY": "  Hovd  ",
        "MEMORY_LEARNING_MODE": "unsupported",
    }
    configured = setup_wizard._initial_configuration(existing.copy())
    assert all(configured[key] == value for key, value in existing.items())


def test_custom_asset_check_uses_configured_paths(tmp_path):
    config = {
        "TTS_VOICE": "ciel",
        "TTS_VOICE_PROFILES": str(tmp_path / "profiles"),
        "TTS_CHATTERBOX_MODEL": str(tmp_path / "model"),
        "TTS_CHATTERBOX_PYTHON": str(tmp_path / "env/bin/python"),
    }
    problems = setup_wizard.custom_voice_asset_problems(config)
    assert any("Turbo Python environment" in problem for problem in problems)
    assert any("voice profile" in problem for problem in problems)
    assert any("Turbo model file" in problem for problem in problems)
    profile = tmp_path / "profiles/ciel"
    profile.mkdir(parents=True)
    (profile / "voice.json").write_text(json.dumps({
        "engine": "chatterbox_turbo",
        "reference_audio": str(tmp_path / "reference.wav"),
        "reference_transcript": str(tmp_path / "reference.txt"),
    }))
    problems = setup_wizard.custom_voice_asset_problems(config)
    assert any("reference_audio" in problem for problem in problems)
    assert any("reference_transcript" in problem for problem in problems)
    for problem in problems:
        path = tmp_path / problem.split(": ", 1)[1]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test asset")
    assert setup_wizard.custom_voice_asset_problems(config) == []
    (tmp_path / "reference.txt").write_text("")
    assert len(setup_wizard.custom_voice_asset_problems(config)) == 1
    (profile / "voice.json").write_text("invalid json")
    assert "voice profile" in setup_wizard.custom_voice_asset_problems(config)[0]


def test_managed_setup_saves_choices_before_installing_and_playing(
    tmp_path, wizard_factory, monkeypatch,
):
    from ciel import installation

    harness = wizard_factory(["quick", "cpu", "save", "yes"], secrets=[""])

    def prepare(audio, notify):
        saved = dotenv_values(tmp_path / ".env")
        assert saved["STT_DEVICE"] == "cpu"
        assert saved["STT_COMPUTE_TYPE"] == "int8"
        assert saved["TTS_ENGINE"] == "pocket"
        harness.tts.assert_not_called()
        assert audio is harness.settings.return_value.audio

    preparation = MagicMock(side_effect=prepare)
    monkeypatch.setattr(installation, "prepare_components", preparation)
    assert setup_wizard.run_setup_wizard(install_components=True)
    preparation.assert_called_once()
    harness.tts.assert_called_once()


def test_managed_setup_cancellation_installs_nothing(tmp_path, wizard_factory, monkeypatch):
    from ciel import installation

    harness = wizard_factory(["quick", "cpu", "cancel"], secrets=[""])
    preparation = MagicMock()
    monkeypatch.setattr(installation, "prepare_components", preparation)
    assert not setup_wizard.run_setup_wizard(install_components=True)
    assert not (tmp_path / ".env").exists()
    preparation.assert_not_called()
    harness.tts.assert_not_called()


def test_managed_setup_preserves_saved_acceleration(tmp_path, wizard_factory, monkeypatch):
    from ciel import installation

    (tmp_path / ".env").write_text("STT_DEVICE=cuda\nSTT_COMPUTE_TYPE=float16\n")
    wizard_factory(["quick", None, "save", "yes"], secrets=[""])
    monkeypatch.setattr(installation, "prepare_components", MagicMock())
    assert setup_wizard.run_setup_wizard(install_components=True)
    saved = dotenv_values(tmp_path / ".env")
    assert saved["STT_DEVICE"] == "cuda"
    assert saved["STT_COMPUTE_TYPE"] == "float16"


def test_managed_setup_preserves_pending_gpu_when_returning_to_customize(
    tmp_path, wizard_factory, monkeypatch,
):
    from ciel import installation

    wizard_factory([
        "quick", "cuda", "customize", "amy", "keep", "keep", None, None,
        None, "save", "yes",
    ], secrets=["", "", "", ""])
    monkeypatch.setattr(installation, "prepare_components", MagicMock())
    assert setup_wizard.run_setup_wizard(install_components=True)
    saved = dotenv_values(tmp_path / ".env")
    assert saved["STT_DEVICE"] == "cuda"
    assert saved["STT_COMPUTE_TYPE"] == "int8_float16"
    assert saved["TTS_ENGINE"] == "piper"


@pytest.mark.parametrize("interruption", [False, True])
def test_managed_setup_failed_download_keeps_choices_for_retry(
    tmp_path, wizard_factory, monkeypatch, capsys, interruption,
):
    from ciel import installation

    harness = wizard_factory(["quick", "cpu", "save"], secrets=[""])
    error = KeyboardInterrupt() if interruption else RuntimeError("Synthetic download failure")
    monkeypatch.setattr(installation, "prepare_components", MagicMock(side_effect=error))
    assert not setup_wizard.run_setup_wizard(install_components=True)
    assert dotenv_values(tmp_path / ".env")["STT_DEVICE"] == "cpu"
    harness.tts.assert_not_called()
    assert "saved" in capsys.readouterr().out.lower()
