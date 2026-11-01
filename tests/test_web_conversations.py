"""Archive admission and explicit continuation use real temporary SQLite storage."""

import pytest

from ciel.memory.bridge import desktop_conversation_id
from ciel.memory.models import ConversationTurn
from ciel.memory.store import MemoryStore


def save(store, session, role, text):
    return store.save_turn(ConversationTurn(role, text, session_id=session))


def test_catalog_reads_only_owner_archives_without_selecting_them(tmp_path):
    from ciel.webapp.conversations import ConversationCatalog

    store = MemoryStore(tmp_path / "memory.db")
    first = desktop_conversation_id() + ":run:" + "a" * 32
    active = desktop_conversation_id() + ":run:" + "b" * 32
    save(store, first, "user", "A little project")
    save(store, first, "assistant", "Let's work on it.")
    save(store, "telegram:111:222:other", "user", "Someone else's secret")
    catalog = ConversationCatalog(store, lambda: (active,))
    items = catalog.list()["conversations"]
    assert len(items) == 1
    assert items[0]["title"] == "A little project"
    result = catalog.read(items[0]["handle"])
    assert [m["text"] for m in result["messages"]] == ["A little project", "Let's work on it."]
    with pytest.raises(ValueError):
        catalog.read(first)
    store.close()


def test_catalog_revokes_old_pairing_and_paginates(tmp_path):
    from ciel.webapp.conversations import ConversationCatalog

    store = MemoryStore(tmp_path / "memory.db")
    old = "telegram:111:222:owner:run:" + "a" * 32
    scope = [old]
    save(store, old, "user", "Private old pairing")
    catalog = ConversationCatalog(store, lambda: tuple(scope))
    handle = catalog.list()["conversations"][0]["handle"]
    scope[:] = [desktop_conversation_id()]
    with pytest.raises(ValueError):
        catalog.read(handle)
    for n in range(35):
        save(store, desktop_conversation_id() + f":run:{n:032x}", "user", str(n))
    page = catalog.list()
    assert len(page["conversations"]) == 30
    assert len(catalog.list(before=page["next_before"])["conversations"]) == 5
    store.close()
