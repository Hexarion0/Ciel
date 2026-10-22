"""Progress reaches the user before tool work and never changes application authority."""

from threading import Event
from unittest.mock import Mock

import pytest

from ciel.actions import ActionRegistry
from ciel.actions.open_app import OpenAppAction
from ciel.config import InternetConfig
from ciel.internet import InternetResult, InternetService
from tests.test_screen_integration import RESULT, install_screen, run_callbacks


@pytest.mark.parametrize("controller", [False, True])
def test_voice_screen_announcement_precedes_work_and_is_not_saved(
    tmp_path, monkeypatch, controller,
):
    events = []
    def answer(_service, request, _messages, **kwargs):
        assert kwargs["progress"]("screen", request.query)
        events.append("screen")
        return RESULT
    install_screen(monkeypatch, on_answer=answer)
    tts = Mock()
    def speak(text, **kwargs):
        events.append(text)
        if kwargs.get("on_start"):
            kwargs["on_start"]()
        return True
    tts.speak.side_effect = speak
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("What anime am I watching?", {"input_source":"keyboard"})],
        controller=controller, tts=tts,
    )
    try:
        assert "anime" in events[0] or "watching" in events[0]
        assert events[1] == "screen"
        assert [t.content for t in turns if t.role == "assistant"] == [RESULT.text]
    finally:
        store.close()


def test_prepare_acknowledges_before_first_search_and_skips_vague_requests():
    service = InternetService(InternetConfig())
    events=[]
    def lookup(*_, **__):
        events.append("lookup")
        return InternetResult(attempted=True)
    service.lookup=Mock(side_effect=lookup)
    def announce(tool, query):
        events.append("ack")
        assert tool == "search" and "Mongolia" in query
        return True
    service.prepare("Search the news in Mongolia", before_lookup=announce)
    assert events == ["ack", "lookup"]
    events.clear()
    service.prepare("Search", before_lookup=announce)
    assert not events


def test_cancel_during_web_announcement_blocks_retrieval():
    service = InternetService(InternetConfig())
    service.lookup=Mock()
    canceled=Event()
    def announce(*_):
        canceled.set()
        return True
    service.prepare("Search Mongolia news", before_lookup=announce, cancel_event=canceled)
    service.lookup.assert_not_called()


def test_action_progress_runs_after_authorization_and_can_cancel_execution(monkeypatch):
    action=OpenAppAction()
    action.execute=Mock(return_value="Launch requested.")
    registry=ActionRegistry([action])
    progress=Mock(return_value=False)
    assert "address" in registry.handle("Open Discord", authorized=False, on_start=progress)
    progress.assert_not_called()
    assert "cancelled" in registry.handle("Open Discord", authorized=True, on_start=progress)
    action.execute.assert_not_called()
