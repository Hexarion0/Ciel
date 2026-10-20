"""Tests for centralized configuration and secret handling."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from ciel.config import (
    DEFAULT_NIM_MODEL,
    AudioConfig,
    ControllerConfig,
    InternetConfig,
    ProviderConfig,
    Settings,
)


def test_default_settings():
    """Verify default settings values when no environment variables are set."""
    settings = Settings(
        _env_file=None,
        ciel_env="development",
        ciel_log_level="INFO",
        wake_word="ciel",
    )
    assert settings.app.env == "development"
    assert settings.app.log_level == "INFO"
    assert settings.audio.wake_word == "ciel"
    assert settings.audio.sample_rate == 16000
    assert settings.providers.nim_api_key is None


@pytest.mark.parametrize("configuration", [Settings, AudioConfig])
def test_fresh_voice_defaults_use_cpu_pocket(configuration):
    with patch.dict(os.environ, {}, clear=True):
        selected = configuration(_env_file=None) if configuration is Settings else configuration()
    audio = selected.audio if configuration is Settings else selected
    assert (audio.tts_engine, audio.tts_voice) == ("pocket", "ciel")
    assert audio.tts_fallback_voice == "en_US-ciel-medium"


@pytest.mark.parametrize("configuration", [Settings, AudioConfig])
def test_fresh_speech_defaults_match_the_desktop_conversation_profile(configuration):
    """Keep direct configuration and environment settings on the same voice profile."""
    with patch.dict(os.environ, {}, clear=True):
        selected = configuration(_env_file=None) if configuration is Settings else configuration()
    audio = selected.audio if configuration is Settings else selected
    assert (audio.stt_model, audio.stt_device, audio.stt_compute_type) == (
        "small.en", "cuda", "int8_float16",
    )
    assert audio.stt_beam_size == 5
    assert audio.stt_retry_model == "medium.en"
    assert audio.stt_retry_min_free_mb == 2048
    assert audio.wake_stt_model == "base.en"
    assert audio.ambient_listening and audio.show_ai_transcripts
    assert not audio.show_transcripts
    assert audio.barge_in_mode == "speech"


def test_explicit_cpu_and_wake_only_configuration_overrides_new_desktop_defaults(tmp_path):
    """Users can retain a lighter recognizer and their chosen listening behavior."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "STT_MODEL=base.en\nSTT_DEVICE=cpu\nSTT_COMPUTE_TYPE=int8\n"
        "STT_BEAM_SIZE=1\nSTT_RETRY_MODEL=\nAMBIENT_LISTENING=false\n"
        "SHOW_AI_TRANSCRIPTS=false\nBARGE_IN_MODE=wake\n"
    )
    with patch.dict(os.environ, {}, clear=True):
        audio = Settings(_env_file=env_file).audio
    assert (audio.stt_model, audio.stt_device, audio.stt_compute_type) == ("base.en", "cpu", "int8")
    assert audio.stt_beam_size == 1
    assert not audio.stt_retry_model
    assert not audio.ambient_listening and not audio.show_ai_transcripts
    assert audio.barge_in_mode == "wake"


@pytest.mark.parametrize("configuration", [Settings, AudioConfig])
@pytest.mark.parametrize(
    ("values", "engine", "voice"),
    [
        ({"tts_engine": "piper"}, "piper", "en_US-amy-medium"),
        ({"tts_engine": "auto"}, "auto", "en_US-amy-medium"),
        ({"tts_engine": "chatterbox_turbo"}, "chatterbox_turbo", "ciel"),
        ({"tts_voice": "en_US-custom-medium"}, "auto", "en_US-custom-medium"),
        ({"tts_engine": "piper", "tts_voice": "custom.onnx"}, "piper", "custom.onnx"),
    ],
)
def test_voice_defaults_preserve_explicit_legacy_choices(configuration, values, engine, voice):
    with patch.dict(os.environ, {}, clear=True):
        selected = (configuration(_env_file=None, **values)
                    if configuration is Settings else configuration(**values))
    audio = selected.audio if configuration is Settings else selected
    assert (audio.tts_engine, audio.tts_voice) == (engine, voice)


def test_legacy_voice_only_environment_keeps_piper(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("TTS_VOICE=en_US-custom-medium\n")
    with patch.dict(os.environ, {}, clear=True):
        audio = Settings(_env_file=env_file).audio
    assert (audio.tts_engine, audio.tts_voice) == ("auto", "en_US-custom-medium")


def test_example_configuration_starts_with_cpu_pocket():
    example = Path(__file__).resolve().parents[1] / ".env.example"
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=example)
    audio = settings.audio
    assert (audio.tts_engine, audio.tts_voice) == ("pocket", "ciel")
    assert audio.tts_pocket_device == "cpu" and audio.tts_pocket_quantization == "none"
    assert (audio.stt_model, audio.stt_device, audio.stt_compute_type) == (
        "small.en", "cuda", "int8_float16",
    )
    assert audio.ambient_listening and audio.show_ai_transcripts
    assert not audio.show_transcripts
    assert audio.barge_in_mode == "speech"
    assert audio.stt_beam_size == 5
    assert audio.stt_retry_model == "medium.en"


def test_fresh_nim_configuration_uses_the_available_model_consistently():
    """Every default route must use the model verified by real inference requests."""
    example = Path(__file__).resolve().parents[1] / ".env.example"
    with patch.dict(os.environ, {}, clear=True):
        fresh = Settings(_env_file=None)
        documented = Settings(_env_file=example)
        controllers = (
            ControllerConfig(), fresh.controller, documented.controller,
        )
        providers = (ProviderConfig(), fresh.providers, documented.providers)
    assert DEFAULT_NIM_MODEL == "nvidia/nemotron-3.5-lightning-30b-a3b"
    for controller in controllers:
        assert controller.model == DEFAULT_NIM_MODEL
        assert controller.timeout_seconds == 8.0
    for provider in providers:
        assert provider.nim_model == DEFAULT_NIM_MODEL
        assert provider.nim_complex_model == DEFAULT_NIM_MODEL
        assert provider.nim_fallback_model == DEFAULT_NIM_MODEL


def test_saved_nim_model_overrides_are_preserved(tmp_path):
    """Changing defaults must preserve intentional account-specific model choices."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "NIM_MODEL=custom-conversation\nNIM_COMPLEX_MODEL=custom-reasoning\n"
        "NIM_FALLBACK_MODEL=custom-backup\nCONTROLLER_MODEL=custom-controller\n",
        encoding="utf-8",
    )
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=env_file)
    assert settings.providers.nim_model == "custom-conversation"
    assert settings.providers.nim_complex_model == "custom-reasoning"
    assert settings.providers.nim_fallback_model == "custom-backup"
    assert settings.controller.model == "custom-controller"


def test_environment_override():
    """Verify settings pick up environment variables."""
    env_vars = {
        "CIEL_ENV": "production",
        "CIEL_LOG_LEVEL": "DEBUG",
        "NIM_API_KEY": "nvapi-test-key-1234567890",
        "WAKE_WORD": "jarvis",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings(_env_file=None)
        assert settings.app.env == "production"
        assert settings.app.log_level == "DEBUG"
        assert settings.app.debug is True
        assert settings.audio.wake_word == "jarvis"
        assert settings.providers.nim_api_key is not None
        # Verify SecretStr masks the secret in str/repr
        assert "nvapi-test-key-1234567890" not in repr(settings.providers.nim_api_key)
        assert settings.providers.nim_api_key.get_secret_value() == "nvapi-test-key-1234567890"


def test_engine_defaults_to_voice_detection():
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=None, tts_voice="en_US-amy-medium")
    assert settings.audio.tts_engine == "auto"


def test_chatterbox_voice_profile_and_fallback_settings_reach_audio_config(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TTS_ENGINE=chatterbox_turbo\nTTS_VOICE=ciel\n"
        "TTS_FALLBACK_VOICE=en_US-amy-medium\nTTS_AUDIO_QUEUE_SIZE=1\n"
    )
    with patch.dict(os.environ, {}, clear=True):
        audio = Settings(_env_file=env_file).audio
    assert audio.tts_engine == "chatterbox_turbo"
    assert audio.tts_voice == "ciel"
    assert audio.tts_fallback_voice == "en_US-amy-medium"
    assert audio.tts_audio_queue_size == 1


def test_tts_playback_latency_is_configurable_and_validated():
    import pytest
    from pydantic import ValidationError

    with patch.dict(os.environ, {}, clear=True):
        assert Settings(_env_file=None).audio.tts_playback_latency == 0.12
        settings = Settings(_env_file=None, tts_playback_latency=0.2)
        assert settings.audio.tts_playback_latency == 0.2
        for latency in (0, -0.1, 1.1, float("nan"), float("inf")):
            with pytest.raises(ValidationError):
                Settings(_env_file=None, tts_playback_latency=latency)


def test_audio_device_indices_and_names_from_env(tmp_path):
    from ciel.config import AudioConfig

    env_path = tmp_path / ".env"
    env_path.write_text('AUDIO_INPUT_DEVICE=3\nAUDIO_OUTPUT_DEVICE="USB Speaker"\n')
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=env_path)
    assert settings.audio_input_device == 3
    assert settings.audio.input_device == 3
    assert settings.audio.output_device == "USB Speaker"
    direct = AudioConfig(input_device=" 4 ", output_device="")
    assert direct.input_device == 4
    assert direct.output_device is None


def test_stt_quality_thresholds_are_exposed_and_validated():
    import pytest
    from pydantic import ValidationError

    settings = Settings(_env_file=None, stt_min_confidence=0.3, stt_retry_confidence=0.6)
    assert settings.audio.stt_min_confidence == 0.3
    assert settings.audio.stt_retry_confidence == 0.6
    with pytest.raises(ValidationError):
        Settings(_env_file=None, stt_min_confidence=1.1)


def test_default_speech_endpoint_allows_quick_conversation_turns():
    from ciel.config import Settings

    audio = Settings(_env_file=None).audio
    assert audio.utterance_silence_seconds == 0.7
    assert audio.utterance_pause_grace_seconds == 0.2
    assert round(audio.utterance_silence_seconds + audio.utterance_pause_grace_seconds, 1) == 0.9


def test_persona_file_can_be_configured_or_disabled():
    with patch.dict(os.environ, {}, clear=True):
        assert Settings(_env_file=None).ciel_persona_file == "persona.txt"
    for path in ("data/custom-persona.txt", ""):
        with patch.dict(os.environ, {"CIEL_PERSONA_FILE": path}, clear=True):
            assert Settings(_env_file=None).ciel_persona_file == path


def test_wake_sensitivity_settings_reach_audio_config():
    with patch.dict(
        os.environ, {"WAKE_MIN_RMS": "0.009", "WAKE_WINDOW_SECONDS": "4"}, clear=True
    ):
        settings = Settings(_env_file=None)
    assert settings.audio.wake_min_rms == 0.009
    assert settings.audio.wake_window_seconds == 4.0


def test_ambient_and_pause_settings_reach_audio_config():
    with patch.dict(
        os.environ,
        {
            "AMBIENT_LISTENING": "true", "AMBIENT_FOLLOWUP_SECONDS": "30",
            "UTTERANCE_PAUSE_GRACE_SECONDS": "1.2",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
    assert settings.audio.ambient_listening
    assert settings.audio.ambient_followup_seconds == 30
    assert settings.audio.utterance_pause_grace_seconds == 1.2


def test_listening_defaults_can_be_selected_in_env(tmp_path):
    env_file = tmp_path / '.env'
    env_file.write_text(
        'AMBIENT_LISTENING=true\nSHOW_TRANSCRIPTS=true\nSHOW_AI_TRANSCRIPTS=true\n'
        'BARGE_IN_MODE=speech\nBARGE_IN_SPEECH_SECONDS=0.32\n'
    )
    with patch.dict(os.environ, {}, clear=True):
        audio = Settings(_env_file=env_file).audio
    assert audio.ambient_listening and audio.show_transcripts
    assert audio.show_ai_transcripts
    assert audio.barge_in_mode == 'speech'
    assert audio.barge_in_speech_seconds == 0.32


def test_invalid_barge_in_mode_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        Settings(_env_file=None, barge_in_mode='anything')


def test_followup_policy_is_configurable_and_validated():
    import pytest
    from pydantic import ValidationError

    with patch.dict(os.environ, {}, clear=True):
        assert Settings(_env_file=None).audio.ambient_followup_policy == 'conversation'
        assert Settings(_env_file=None).audio.ambient_followup_seconds == 300
    with patch.dict(os.environ, {'AMBIENT_FOLLOWUP_POLICY': 'strict'}, clear=True):
        assert Settings(_env_file=None).audio.ambient_followup_policy == 'strict'
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ambient_followup_policy='anything')


def test_internet_defaults_allow_free_automatic_search_without_assuming_a_city():
    """Keep default retrieval bounded and the user's location unset."""
    with patch.dict(os.environ, {}, clear=True):
        internet = Settings(_env_file=None).internet
    assert internet == InternetConfig()
    assert internet.enabled and internet.automatic
    assert internet.home_city == ""
    assert (internet.timeout_seconds, internet.research_timeout_seconds) == (8.0, 20.0)
    assert (internet.max_results, internet.research_max_results) == (5, 8)
    assert (internet.max_pages, internet.research_max_pages) == (2, 4)


def test_internet_environment_settings_reach_structured_config(tmp_path):
    """Load the flat environment controls without losing their values."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "INTERNET_ENABLED=false\nINTERNET_AUTOMATIC=false\n"
        'INTERNET_HOME_CITY="  Paris, France  "\n'
        "INTERNET_TIMEOUT_SECONDS=4.5\nINTERNET_RESEARCH_TIMEOUT_SECONDS=30\n"
        "INTERNET_MAX_RESULTS=3\nINTERNET_RESEARCH_MAX_RESULTS=12\n"
        "INTERNET_MAX_PAGES=0\nINTERNET_RESEARCH_MAX_PAGES=6\n"
    )
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=env_file)
    assert settings.internet_home_city == "Paris, France"
    assert settings.internet == InternetConfig(
        enabled=False,
        automatic=False,
        home_city="Paris, France",
        timeout_seconds=4.5,
        research_timeout_seconds=30,
        max_results=3,
        research_max_results=12,
        max_pages=0,
        research_max_pages=6,
    )


def test_internet_process_environment_overrides_env_file(tmp_path):
    """Allow deployment settings to override the checked-in example values."""
    env_file = tmp_path / ".env"
    env_file.write_text("INTERNET_ENABLED=true\nINTERNET_HOME_CITY=Paris, France\n")
    with patch.dict(
        os.environ,
        {"INTERNET_ENABLED": "false", "INTERNET_HOME_CITY": "Dublin, Ireland"},
        clear=True,
    ):
        internet = Settings(_env_file=env_file).internet
    assert not internet.enabled
    assert internet.home_city == "Dublin, Ireland"


@pytest.mark.parametrize(
    ("field", "invalid_values"),
    [
        ("timeout_seconds", (0, 60.1, float("nan"), float("inf"))),
        ("research_timeout_seconds", (0, 120.1, float("nan"), float("inf"))),
        ("max_results", (0, 11)),
        ("research_max_results", (0, 21)),
        ("max_pages", (-1, 6)),
        ("research_max_pages", (-1, 11)),
        ("home_city", ("x" * 201,)),
    ],
)
def test_internet_limits_reject_invalid_values(field, invalid_values):
    """Reject unbounded or non-finite budgets in both configuration entry points."""
    with patch.dict(os.environ, {}, clear=True):
        for value in invalid_values:
            with pytest.raises(ValidationError):
                Settings(_env_file=None, **{f"internet_{field}": value})
            with pytest.raises(ValidationError):
                InternetConfig(**{field: value})


def test_blank_internet_city_stays_unset():
    """Whitespace does not create a default location."""
    with patch.dict(os.environ, {"INTERNET_HOME_CITY": "  "}, clear=True):
        assert Settings(_env_file=None).internet.home_city == ""
    assert InternetConfig(home_city="  ").home_city == ""


def test_controller_configuration_from_environment():
    with patch.dict(os.environ, {
        "CONTROLLER_ENABLED": "true", "CONTROLLER_MODEL": "custom-light",
        "CONTROLLER_TIMEOUT_SECONDS": "4.0", "CONTROLLER_MAX_TOOL_STEPS": "2",
    }, clear=True):
        controller = Settings(_env_file=None).controller
    assert controller.enabled and controller.model == "custom-light"
    assert controller.timeout_seconds == 4 and controller.max_tool_steps == 2


@pytest.mark.parametrize("options", [
    {"controller_timeout_seconds": 0}, {"controller_max_tool_steps": 6},
])
def test_controller_rejects_unbounded_settings(options):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **options)
