"""Archived owner recall preserves private routing across channel projections."""

from types import SimpleNamespace

import pytest

from ciel.config import MemoryConfig
from ciel.memory.models import ConversationTurn, MemoryItem, MemoryType
from ciel.memory.store import MemoryStore
from ciel.memory.tools import MemoryTools


@pytest.mark.parametrize("marker_channel", ["desktop", "telegram"])
@pytest.mark.parametrize("transitive", [False, True])
@pytest.mark.parametrize("private_source", ["raw", "summary"])
def test_archived_private_channel_projection_requires_local_recall(
    tmp_path, marker_channel, transitive, private_source,
):
    store = MemoryStore(tmp_path / "synthetic.db")
    desktop = "desktop_session:archive-test"
    telegram = "telegram:12:34:archive-test"
    old_desktop, old_telegram = desktop + ":run:" + "a" * 32, telegram + ":run:" + "a" * 32
    current = (desktop + ":run:" + "b" * 32, telegram + ":run:" + "b" * 32)
    public = desktop + ":run:" + "c" * 32
    marker = old_desktop if marker_channel == "desktop" else old_telegram
    followup = old_telegram if marker_channel == "desktop" else old_desktop
    projection = "bridge-one" if transitive else "bridge-old"
    if transitive:
        store.register_session_projection(projection, [marker, "shared-projection"])
        store.register_session_projection("bridge-two", ["shared-projection", followup])
    else:
        store.register_session_projection(projection, [old_desktop, old_telegram])
    if private_source == "raw":
        store.save_turn(ConversationTurn("user", "/local Keep this discussion local.",
                                         session_id=marker))
    else:
        store.save_memory(MemoryItem(
            "A private archived discussion.", memory_type=MemoryType.CONVERSATION,
            metadata={"session_id": projection, "local_only": True},
        ))
    store.save_turn(ConversationTurn("user", "The violet launch phrase is silver kestrel.",
                                     session_id=followup))
    store.save_turn(ConversationTurn("user", "The public launch label is cobalt.",
                                     session_id=public))
    store.save_turn(ConversationTurn("user", "The foreign launch label is amber.",
                                     session_id="telegram:99:88:archive-test:run:" + "a" * 32))
    tools = MemoryTools(store, MemoryConfig(), SimpleNamespace(context_session_ids=lambda: current))
    try:
        # Raw routing markers and summary metadata both protect the entire projection.
        ordinary = tools.search("violet launch", include_sessions=True)
        assert "silver kestrel" not in " ".join(ordinary["session_excerpts"])
        private = tools.search("violet launch", include_sessions=True, local_only=True)
        assert "silver kestrel" in " ".join(private["session_excerpts"])
        # A private projection cannot hide a separate public launch or grant another owner access.
        public_result = tools.search("public launch", include_sessions=True)
        assert "cobalt" in " ".join(public_result["session_excerpts"])
        foreign_result = tools.search("foreign launch", include_sessions=True, local_only=True)
        assert "amber" not in " ".join(foreign_result["session_excerpts"])
        store.clear_session(followup)
        after_reset = tools.search("violet launch", include_sessions=True, local_only=True)
        assert "silver kestrel" not in " ".join(after_reset["session_excerpts"])
    finally:
        tools.close()
        store.close()
