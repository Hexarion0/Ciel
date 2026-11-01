"""Keep ordinary TTS unit tests independent of private live voice configuration."""

import pytest

from ciel.config import Settings


@pytest.fixture(autouse=True)
def isolate_background_tool_language(monkeypatch, request):
    """Other unit tests exercise tools without asynchronous model-bank requests."""
    if (request.node.get_closest_marker("integration") is not None
            or request.module.__name__.endswith("test_tool_progress")):
        return
    monkeypatch.setattr("ciel.tool_progress.ToolProgress.warm", lambda *args, **kwargs: None)


@pytest.fixture(autouse=True)
def isolate_background_titles(monkeypatch, request):
    """Background conversation titling would add model calls to exact router assertions."""
    if (request.node.get_closest_marker("integration") is not None
            or request.module.__name__.endswith("test_conversation_titles")):
        return
    monkeypatch.setattr("ciel.memory.titles.ConversationTitler.after_reply",
                        lambda *args, **kwargs: None)


@pytest.fixture(autouse=True)
def isolate_screen_runtime(monkeypatch, request, tmp_path_factory):
    """Unit resets/startup must never touch real screenshots or user-service configuration."""
    if request.node.get_closest_marker("integration") is not None:
        return
    runtime = tmp_path_factory.mktemp("screen-runtime")
    runtime.chmod(0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("unit-config")))


@pytest.fixture(autouse=True)
def isolate_tts_settings(monkeypatch, request):
    """Model tests opt into integration; unit tests start from dependency-light defaults."""
    if request.node.get_closest_marker("integration") is not None:
        return
    monkeypatch.setattr(
        "ciel.audio.tts.get_settings", lambda: Settings(_env_file=None, tts_engine="piper"),
    )
