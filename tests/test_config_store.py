"""config.json holds ordinary settings; .env holds only secrets."""

import json
import stat

import pytest
from dotenv import dotenv_values

from ciel.config import Settings
from ciel.config_store import (
    load_configuration,
    save_configuration,
    secret_names,
    split_env_file,
)


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_settings_read_config_json_beside_the_env_file(tmp_path):
    write_json(tmp_path / "config.json", {
        "stt_model": "medium.en", "ambient_listening": False,
        "barge_in_speech_seconds": 0.3, "wake_models": ["hey_ciel.onnx"],
    })
    (tmp_path / ".env").write_text("GROQ_API_KEY=gsk-test\n", encoding="utf-8")

    settings = Settings(_env_file=tmp_path / ".env")

    assert settings.stt_model == "medium.en"
    assert settings.ambient_listening is False
    assert settings.barge_in_speech_seconds == 0.3
    assert settings.wake_models == ["hey_ciel.onnx"]
    assert settings.groq_api_key.get_secret_value() == "gsk-test"


def test_env_file_still_overrides_config_json(tmp_path):
    write_json(tmp_path / "config.json", {"stt_model": "medium.en"})
    (tmp_path / ".env").write_text("STT_MODEL=tiny.en\n", encoding="utf-8")

    assert Settings(_env_file=tmp_path / ".env").stt_model == "tiny.en"


def test_no_env_file_means_no_config_json_either(tmp_path, monkeypatch):
    write_json(tmp_path / "config.json", {"stt_model": "medium.en"})
    monkeypatch.chdir(tmp_path)

    assert Settings(_env_file=None).stt_model == "small.en"


def test_invalid_config_json_names_the_file(tmp_path):
    (tmp_path / "config.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="config.json"):
        Settings(_env_file=tmp_path / ".env")


def test_secret_names_are_the_credential_fields():
    assert secret_names() == {
        "NIM_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "TELEGRAM_BOT_TOKEN",
        "BRAVE_SEARCH_API_KEY", "TAVILY_API_KEY",
    }


def test_save_routes_secrets_to_env_and_the_rest_to_config(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# keys\nSTT_MODEL=tiny.en\nNIM_API_KEY=old\n", encoding="utf-8")

    save_configuration(env, {"STT_MODEL": "medium.en", "NIM_API_KEY": "new",
                             "AMBIENT_LISTENING": "false", "WAKE_THRESHOLD": "0.5"})

    assert dotenv_values(env) == {"NIM_API_KEY": "new"}
    assert "# keys" in env.read_text(encoding="utf-8")
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config == {"stt_model": "medium.en", "ambient_listening": False,
                      "wake_threshold": 0.5}
    for path in (env, tmp_path / "config.json"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_save_none_removes_a_setting_from_both_files(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AUDIO_INPUT_DEVICE=3\n", encoding="utf-8")
    write_json(tmp_path / "config.json", {"audio_input_device": 4, "stt_model": "tiny.en"})

    save_configuration(env, {"AUDIO_INPUT_DEVICE": None})

    assert dotenv_values(env) == {}
    assert json.loads((tmp_path / "config.json").read_text()) == {"stt_model": "tiny.en"}


def test_load_configuration_merges_both_files_as_text(tmp_path):
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=gsk\nSTT_MODEL=tiny.en\n", encoding="utf-8")
    write_json(tmp_path / "config.json", {"stt_model": "medium.en", "tts_enabled": True,
                                          "wake_models": ["a.onnx"]})

    assert load_configuration(env) == {
        "STT_MODEL": "tiny.en", "TTS_ENABLED": "true",
        "WAKE_MODELS": '["a.onnx"]', "GROQ_API_KEY": "gsk",
    }


def test_split_moves_every_ordinary_setting_and_keeps_the_same_settings(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "NIM_API_KEY=nvapi-test\nSTT_MODEL='medium.en'\nAMBIENT_LISTENING=false\n"
        "TTS_ENGINE=piper\nCHAT_SERVICES=telegram\nTELEGRAM_BOT_TOKEN=123:abc\n",
        encoding="utf-8",
    )
    before = Settings(_env_file=env).model_dump()

    split_env_file(env)

    assert set(dotenv_values(env)) == {"NIM_API_KEY", "TELEGRAM_BOT_TOKEN"}
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["stt_model"] == "medium.en"
    assert config["ambient_listening"] is False
    assert "wake_word" in config  # The full file lists defaults too.
    assert not {name.lower() for name in secret_names()} & set(config)
    assert Settings(_env_file=env).model_dump() == before
