"""Controlled screen turns reuse real recall without changing admission or privacy."""

from unittest.mock import MagicMock

import pytest

from ciel.dialogue import DialogueContext
from ciel.memory import MemoryItem, MemoryType
from ciel.providers.router import ModelRouter
from tests.test_controller_runtime import _seed_legacy_private_summary
from tests.test_screen_integration import (
    TEXT,
    install_screen,
    run_callbacks,
    stub_runtime_internet,
)

RECALLED_STYLE = "The user prefers concise screen explanations."


@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("typed", [False, True])
def test_controlled_visual_callback_recalls_once_and_preserves_context(
    tmp_path, monkeypatch, private, typed,
):
    """Count the actual recall implementation while exercising the real desktop callback."""
    recalled = []
    recall = DialogueContext.recall

    def record_recall(context, query):
        result = recall(context, query)
        recalled.append((query, result))
        return result

    def initialize(store, session_id):
        store.upsert_fact("user:response_style", MemoryItem(
            content=RECALLED_STYLE,
            memory_type=MemoryType.PREFERENCE,
            source="user_explicit",
            metadata={"value": "concise screen explanations"},
        ))
        if private:
            _seed_legacy_private_summary(store, session_id)

    monkeypatch.setattr(DialogueContext, "recall", record_recall)
    created = install_screen(monkeypatch)
    internet = stub_runtime_internet(monkeypatch)
    router = MagicMock()
    info = {"input_source": "keyboard"} if typed else {"wake_verified": True}
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Look at my screen", info)], router=router,
        controller=True, initialize_conversation=initialize,
    )
    try:
        assert len(recalled) == 1
        assert recalled[0][0] == "Look at my screen"
        assert RECALLED_STYLE in "\n".join(recalled[0][1])
        screen = created[0]
        screen.answer.assert_called_once()
        request, messages = screen.answer.call_args.args
        assert request.query == "Look at my screen" and not request.previous
        assert RECALLED_STYLE in messages[0].content
        assert ModelRouter.requires_local(messages) is private
        router.send.assert_not_called()
        router.stream.assert_not_called()
        internet.lookup.assert_not_called()
        assert [turn.content for turn in turns if turn.role == "assistant"] == [TEXT]
        assert turns[-1].sources == []
        if typed:
            tts.begin_stream.assert_not_called()
        tts.speak.assert_called_once()
        assert tts.speak.call_args.args[0] == TEXT
    finally:
        store.close()
