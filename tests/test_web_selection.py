"""Active conversation revisions stay monotonic across A→B→A and process restarts."""

import pytest

from ciel.memory.bridge import start_desktop_conversation
from ciel.memory.models import ConversationTurn
from ciel.memory.store import MemoryStore


def test_selection_rejects_stale_activation_and_keeps_old_turns(tmp_path):
    from ciel.webapp.conversations import ConversationCatalog
    from ciel.webapp.selection import ConversationSelection

    store = MemoryStore(tmp_path / "memory.db")
    a = start_desktop_conversation(store)
    store.save_turn(ConversationTurn("user", "Keep this", session_id=a))
    scope = [a]
    catalog = ConversationCatalog(store, lambda: tuple(scope))
    switches = []

    def switch(identity):
        scope[:] = [identity]
        switches.append(identity)

    selection = ConversationSelection(store, catalog, a, switch)
    first = selection.current()
    second = selection.activate(None, first.revision)
    third = selection.activate(first.handle, second.revision)
    assert first.handle == third.handle
    assert first.revision < second.revision < third.revision
    with pytest.raises(ValueError):
        selection.activate(None, first.revision)
    assert len(switches) == 2
    assert catalog.read(first.handle)["messages"][0]["text"] == "Keep this"
    assert start_desktop_conversation(store) != a
    store.close()


def test_screen_cache_checks_activation_revision_separately_from_reset_generation(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from ciel.config import Settings
    from ciel.screen.service import ScreenResult, ScreenService, screen_scope

    store = MemoryStore(tmp_path / "memory.db")
    identity = start_desktop_conversation(store)
    conversation = SimpleNamespace(store=store, context_session_ids=lambda: (identity,))
    settings = Settings(_env_file=None, memory_db_path=str(tmp_path / "memory.db"))
    service = ScreenService(settings, conversation, backend=Mock(), vision=Mock())
    service._cache = Mock()
    service._cache.contains.return_value = True
    service._epochs = screen_scope(conversation)
    result = ScreenResult("Visible answer", "fixture", "vision", "frame")
    assert service.image_valid(result)
    store.select_desktop_session(identity, store.desktop_selection()[1])
    assert not service.image_valid(result)
    store.close()
