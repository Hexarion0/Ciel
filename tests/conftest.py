"""Keep ordinary TTS unit tests independent of private live voice configuration."""

import pytest

from ciel.config import Settings


@pytest.fixture(autouse=True)
def isolate_tts_settings(monkeypatch, request):
    """Model tests opt into integration; unit tests start from dependency-light defaults."""
    if request.node.get_closest_marker("integration") is not None:
        return
    monkeypatch.setattr(
        "ciel.audio.tts.get_settings", lambda: Settings(_env_file=None, tts_engine="piper"),
    )
