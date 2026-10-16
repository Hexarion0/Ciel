"""Tests for CIEL SQLite memory architecture and persistence engine."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from ciel.memory import ConversationTurn, MemoryItem, MemoryStore, MemoryType


@pytest.fixture
def temp_store():
    """Create a temporary MemoryStore backed by an isolated SQLite database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name

    store = MemoryStore(db_path=tmp_path)
    yield store
    store.close()
    try:
        Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        pass


def test_memory_crud_operations(temp_store: MemoryStore):
    """Verify inserting, retrieving, updating, and deleting memories."""
    # 1. Insert Fact
    item = MemoryItem(
        content="User's favorite game is Counter-Strike 2",
        memory_type=MemoryType.PREFERENCE,
        source="user_explicit",
        confidence=1.0,
        metadata={"category": "gaming"},
    )
    mem_id = temp_store.save_memory(item)
    assert mem_id is not None
    assert mem_id > 0
    assert temp_store.count_memories() == 1

    # 2. Retrieve
    retrieved = temp_store.get_memory(mem_id)
    assert retrieved is not None
    assert retrieved.content == "User's favorite game is Counter-Strike 2"
    assert retrieved.memory_type == MemoryType.PREFERENCE
    assert retrieved.metadata == {"category": "gaming"}

    # 3. Update
    retrieved.content = "User's favorite game is CS2 and Valorant"
    temp_store.save_memory(retrieved)
    updated = temp_store.get_memory(mem_id)
    assert updated.content == "User's favorite game is CS2 and Valorant"

    # 4. Delete
    deleted = temp_store.delete_memory(mem_id)
    assert deleted is True
    assert temp_store.get_memory(mem_id) is None
    assert temp_store.count_memories() == 0


def test_memory_search_and_filtering(temp_store: MemoryStore):
    """Verify searching memory by keyword and filtering by type."""
    temp_store.save_memory(
        MemoryItem(
            content="User prefers dark mode on all interfaces",
            memory_type=MemoryType.PREFERENCE,
        )
    )
    temp_store.save_memory(
        MemoryItem(
            content="Project CIEL is a desktop voice assistant",
            memory_type=MemoryType.PROJECT,
        )
    )
    temp_store.save_memory(
        MemoryItem(
            content="User is running an NVIDIA GTX 1660 SUPER GPU",
            memory_type=MemoryType.FACT,
        )
    )

    # Search keyword across all types
    gpu_results = temp_store.search_memories("GPU")
    assert len(gpu_results) == 1
    assert "GTX 1660" in gpu_results[0].content

    # Search with type filter
    pref_results = temp_store.search_memories("mode", memory_type=MemoryType.PREFERENCE)
    assert len(pref_results) == 1
    assert "dark mode" in pref_results[0].content

    # Search without results
    empty_results = temp_store.search_memories("nonexistent_keyword_xyz")
    assert len(empty_results) == 0


def test_delete_by_pattern(temp_store: MemoryStore):
    """Verify deleting memories matching a pattern."""
    temp_store.save_memory(MemoryItem(content="Remember to buy milk"))
    temp_store.save_memory(MemoryItem(content="Remember to buy eggs"))
    temp_store.save_memory(MemoryItem(content="User name is Hexarion"))

    deleted_count = temp_store.delete_by_pattern("buy")
    assert deleted_count == 2
    assert temp_store.count_memories() == 1


@pytest.mark.parametrize("query", ["you", ""])
def test_recall_excludes_conversation_summaries_before_limit(temp_store, query):
    """A newer broad-match summary must not displace the user's saved fact."""
    fact = MemoryItem(content="You prefer patient explanations", memory_type=MemoryType.FACT)
    temp_store.save_memory(fact)
    archived = MemoryItem(
        content="You requested a cold commanding tone", memory_type=MemoryType.CONVERSATION,
        source="auto_summarizer", metadata={"session_id": "desktop_session"},
    )
    temp_store.save_memory(archived)

    recalled = temp_store.search_memories(
        query, limit=1, exclude_types=(MemoryType.CONVERSATION,),
    )

    assert [memory.id for memory in recalled] == [fact.id]
    assert temp_store.get_memory(archived.id).content == archived.content


def test_conversation_turns_persistence(temp_store: MemoryStore):
    """Verify storing and retrieving multi-turn conversation history."""
    session_a = "session_123"
    session_b = "session_456"

    # Add turns to session A
    temp_store.save_turn(
        ConversationTurn(session_id=session_a, role="user", content="Hello Ciel")
    )
    temp_store.save_turn(
        ConversationTurn(
            session_id=session_a,
            role="assistant",
            content="Hey there! How can I help?",
            provider="nim",
            model="nemotron",
            latency=0.5,
        )
    )
    temp_store.save_turn(
        ConversationTurn(session_id=session_a, role="user", content="What is my GPU?")
    )

    # Add turn to session B
    temp_store.save_turn(
        ConversationTurn(session_id=session_b, role="user", content="Testing other session")
    )

    # Retrieve session A turns (ordered chronologically)
    turns = temp_store.get_recent_turns(session_id=session_a, limit=10)
    assert len(turns) == 3
    assert turns[0].role == "user"
    assert turns[0].content == "Hello Ciel"
    assert turns[1].role == "assistant"
    assert turns[1].provider == "nim"
    assert turns[2].content == "What is my GPU?"

    # Verify limit parameter
    limited = temp_store.get_recent_turns(session_id=session_a, limit=2)
    assert len(limited) == 2
    assert limited[0].role == "assistant"
    assert limited[1].role == "user"

    # Clear session
    cleared = temp_store.clear_session(session_a)
    assert cleared == 3
    assert len(temp_store.get_recent_turns(session_id=session_a)) == 0
    assert len(temp_store.get_recent_turns(session_id=session_b)) == 1


def test_conversation_sources_survive_restart_without_changing_reply_text(tmp_path):
    """Citation metadata reloads independently from the assistant's spoken content."""
    path = tmp_path / "sources.db"
    sources = [{"title": "Public forecast", "url": "https://example.com/weather"}]
    store = MemoryStore(path)
    turn = ConversationTurn(role="assistant", content="It is sunny.", sources=sources)
    store.save_turn(turn)
    store.close()
    sources[0]["title"] = "Changed outside the turn"

    restarted = MemoryStore(path)
    try:
        restored = restarted.get_recent_turns()[0]
        assert restored.content == "It is sunny."
        assert restored.sources == [{"title": "Public forecast", "url": "https://example.com/weather"}]
        assert restored.to_dict()["sources"] == restored.sources
        assert turn.sources == restored.sources
    finally:
        restarted.close()


def test_existing_conversation_database_is_migrated_with_empty_sources(tmp_path):
    """Upgrading a previous schema preserves old turns and defaults their citations."""
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE conversation_turns ("
            "id INTEGER PRIMARY KEY, session_id TEXT DEFAULT 'default', role TEXT, "
            "content TEXT, timestamp TEXT, provider TEXT DEFAULT '', model TEXT DEFAULT '', "
            "tokens INTEGER DEFAULT 0, latency REAL DEFAULT 0.0);"
        )
        connection.execute(
            "INSERT INTO conversation_turns (id, role, content, timestamp) "
            "VALUES (1, 'assistant', 'Existing reply', '2026-10-01T00:00:00+00:00');"
        )
    store = MemoryStore(path)
    try:
        restored = store.get_recent_turns()[0]
        assert restored.content == "Existing reply"
        assert restored.sources == []
        assert store._get_connection().execute(
            "SELECT sources_json FROM conversation_turns;"
        ).fetchone()[0] == "[]"
    finally:
        store.close()


@pytest.mark.parametrize("stored_sources", ["broken JSON", "null", "{}", "[null]", "\"text\""])
def test_corrupt_conversation_source_json_does_not_break_history(temp_store, stored_sources):
    """Invalid citation data cannot prevent a session from being restored."""
    turn_id = temp_store.save_turn(ConversationTurn(role="assistant", content="Saved reply"))
    with temp_store._get_connection() as connection:
        connection.execute(
            "UPDATE conversation_turns SET sources_json=? WHERE id=?;", (stored_sources, turn_id),
        )
    restored = temp_store.get_recent_turns()[0]
    assert restored.content == "Saved reply"
    assert restored.sources == []


@pytest.mark.parametrize("url", [
    "javascript:alert(1)", "data:text/html,bad", "file:///etc/passwd", "/relative",
    "//example.com/page", "https://user:password@example.com/", "https://@example.com/",
    "https://example.com/\npage", "https://example.com/\x7fpage", "https://example.com/a b",
    "https://example.com\\page", "https://example.com:99999/", "https://example.com:bad/",
])
def test_conversation_sources_reject_unsafe_links(temp_store, url):
    """Only valid HTTP(S) citations with no embedded credentials are persisted."""
    turn = ConversationTurn(
        role="assistant", content="Answer", sources=[{"title": "Bad", "url": url}],
    )
    temp_store.save_turn(turn)
    assert turn.sources == []
    assert temp_store.get_recent_turns()[0].sources == []


def test_conversation_sources_are_bounded_deduplicated_and_copy_metadata(temp_store):
    """Citation lists have finite size and discard provider-specific metadata."""
    sources = [{"title": "", "url": "https://example.com/0", "snippet": "Do not store me"}]
    sources += [{"title": str(index), "url": f"https://example.com/{index}"} for index in range(25)]
    sources += [{"title": "Bad\nlabel", "url": "https://example.com/control"}]
    turn = ConversationTurn(role="assistant", content="Answer", sources=sources)
    temp_store.save_turn(turn)
    restored = temp_store.get_recent_turns()[0]
    assert len(restored.sources) == 20
    assert restored.sources[0] == {"title": "example.com", "url": "https://example.com/0"}
    assert all(set(source) == {"title", "url"} for source in restored.sources)


def test_conversation_default_source_lists_are_independent():
    """The default citation list belongs to one conversation turn."""
    first = ConversationTurn(role="assistant", content="First")
    second = ConversationTurn(role="assistant", content="Second")
    first.sources.append({"title": "Source", "url": "https://example.com/"})
    assert second.sources == []
