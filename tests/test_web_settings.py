"""Browser settings are validated, secret-free, persistent and applied to the host."""

import json

import pytest

from ciel.config import Settings


def test_web_settings_do_not_return_keys_and_save_only_admitted_fields(tmp_path):
    from ciel.webapp.settings import WebSettings

    settings = Settings(_env_file=None, nim_api_key="private-value", nim_model="original-model")
    applied = []
    service = WebSettings(settings, tmp_path / ".env", apply=lambda fields: applied.append(fields))
    snapshot = service.snapshot()
    assert "private-value" not in json.dumps(snapshot)
    assert snapshot["configured"]["nim_api_key"] is True
    saved = service.save({"nim_model": "new-model", "tts_speed": 1.1})
    assert saved["values"]["nim_model"] == "new-model"
    assert saved["needs_restart"] is True
    assert settings.nim_model == "new-model"
    assert "NIM_MODEL=" in (tmp_path / ".env").read_text()
    assert applied == [{"nim_model", "tts_speed"}]
    with pytest.raises(ValueError):
        service.save({"memory_db_path": "/tmp/other-owner.db"})
    with pytest.raises(ValueError):
        service.save({"tts_speed": 0})
    assert settings.audio.tts_speed == 1.1


def test_web_recall_status_reflects_persisted_pause_without_mutating_defaults(tmp_path):
    from ciel.webapp.settings import WebSettings

    settings = Settings(_env_file=None, memory_recall_enabled=True)
    service = WebSettings(
        settings, tmp_path / ".env", apply=lambda fields: None, recall=lambda: False
    )
    assert service.snapshot()["values"]["memory_recall_enabled"] is False
    service.save({"nim_model": "new-model"})
    assert settings.memory_recall_enabled is True
    assert "MEMORY_RECALL_ENABLED" not in (tmp_path / ".env").read_text()
