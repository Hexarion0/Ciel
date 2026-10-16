"""Regression tests for durable indexed recall and safe shared session state."""

import json
import sqlite3

import pytest

from ciel.memory.models import ConversationTurn, MemoryItem, MemoryType
from ciel.memory.store import MemoryStore


@pytest.mark.parametrize("metadata", ["broken JSON", "null", "[]", '"text"', "123"])
def test_invalid_legacy_metadata_does_not_prevent_restart(tmp_path, metadata):
    """An unreadable metadata field must not disable otherwise intact saved facts."""
    path = tmp_path / "legacy.db"
    store = MemoryStore(path)
    memory_id = store.save_memory(MemoryItem(content="A preserved fact"))
    with store._get_connection() as connection:
        connection.execute("UPDATE memories SET metadata_json=? WHERE id=?", (metadata, memory_id))
    store.close()

    restarted = MemoryStore(path)
    try:
        assert restarted.get_memory(memory_id).metadata == {}
        assert restarted.get_session_summary("default") is None
        assert restarted.clear_session("default") == 0
    finally:
        restarted.close()


def test_recent_turn_query_uses_session_and_id_index():
    """Chronological retrieval should not sort every historical turn in a session."""
    store = MemoryStore(":memory:")
    try:
        plan = store._get_connection().execute(
            "EXPLAIN QUERY PLAN SELECT * FROM conversation_turns "
            "WHERE session_id=? ORDER BY id DESC LIMIT ?", ("default", 10),
        ).fetchall()
        assert not any("TEMP B-TREE" in row["detail"] for row in plan)
    finally:
        store.close()


def summary_item(content, last_turn_id, memory_id=None):
    """Build a summary with its persisted session identity and source cursor."""
    return MemoryItem(
        id=memory_id, content=content, memory_type=MemoryType.CONVERSATION,
        metadata={"session_id": "shared", "last_turn_id": last_turn_id},
    )


def test_session_summary_compare_and_swap_survives_multiple_store_instances(tmp_path):
    """A clear from another connection invalidates an already generated summary."""
    path = tmp_path / "shared.db"
    first, second = MemoryStore(path), MemoryStore(path)
    try:
        generation = first.get_session_generation("shared")
        first.save_turn(ConversationTurn(role="user", content="Private topic", session_id="shared"))
        assert second.clear_session("shared") == 1
        assert not first.save_session_summary(
            summary_item("Stale topic", 1), expected_generation=generation, expected_last_turn_id=0,
        )
        assert first.get_session_summary("shared") is None
        fresh = summary_item("Fresh topic", 2)
        assert first.save_session_summary(
            fresh, expected_generation=first.get_session_generation("shared"),
            expected_last_turn_id=0,
        )
        assert second.get_session_summary("shared").content == "Fresh topic"
    finally:
        first.close()
        second.close()


def test_summary_compare_and_swap_rejects_worker_with_old_cursor():
    """A slower summarizer cannot overwrite the cursor another worker advanced."""
    store = MemoryStore(":memory:")
    try:
        initial = summary_item("Initial", 2)
        assert store.save_session_summary(
            initial, expected_generation=0, expected_last_turn_id=0,
        )
        next_item = summary_item("Newer", 4, initial.id)
        assert store.save_session_summary(
            next_item, expected_generation=0, expected_last_turn_id=2,
        )
        stale = summary_item("Older worker", 3, initial.id)
        assert not store.save_session_summary(
            stale, expected_generation=0, expected_last_turn_id=2,
        )
        assert store.get_session_summary("shared").content == "Newer"
    finally:
        store.close()


def test_forgetting_reads_latest_persisted_fact_instead_of_stale_snapshot():
    """A correction racing with forgetting must not leave its current value exposed."""
    store = MemoryStore(":memory:")
    try:
        store.upsert_fact("user:favorite_food", MemoryItem(
            content="Your favorite food is pasta.", metadata={"value": "pasta"},
        ))
        stale = store.get_fact("user:favorite_food")
        store.upsert_fact("user:favorite_food", MemoryItem(
            content="Your favorite food is sushi.", metadata={"value": "sushi"},
        ))
        assert store.forget_memories([stale]) == 1
        assert store.redact_forgotten("pasta and sushi") == "[forgotten fact] and [forgotten fact]"
    finally:
        store.close()


def test_recall_index_updates_deletes_and_excludes_summaries():
    """All normal memory writes keep indexed candidate selection synchronized."""
    store = MemoryStore(":memory:")
    try:
        item = MemoryItem(content="A mechanical keyboard uses silent switches")
        store.save_memory(item)
        store.save_memory(MemoryItem(
            content="A mechanical keyboard summary", memory_type=MemoryType.CONVERSATION,
        ))
        recalled = store.ranked_recall_candidates({"keyboard"})
        assert [memory.id for _, memory in recalled] == [item.id]
        item.content = "An ergonomic mouse"
        store.save_memory(item)
        assert store.ranked_recall_candidates({"keyboard"}) == []
        assert store.ranked_recall_candidates({"mouse"})[0][1].id == item.id
        store.delete_memory(item.id)
        assert store.ranked_recall_candidates({"mouse"}) == []
    finally:
        store.close()


def test_recall_finds_older_fact_beyond_recent_candidate_cutoff():
    """New unrelated notes cannot push an older exact match out of the candidate set."""
    store = MemoryStore(":memory:")
    try:
        target = MemoryItem(content="The telescope is named Andromeda")
        store.save_memory(target)
        for index in range(2001):
            store.save_memory(MemoryItem(content=f"Unrelated recent note {index}"))
        assert store.ranked_recall_candidates({"telescope"})[0][1].id == target.id
    finally:
        store.close()


def test_recall_backfills_legacy_records_and_ignores_noncanonical_key_duplicates(tmp_path):
    """Migration indexes durable records but recalls only the current keyed fact."""
    path = tmp_path / "legacy.db"
    store = MemoryStore(path)
    old = MemoryItem(content="Your favorite planet is Venus.", metadata={
        "scope": "user", "fact_key": "favorite_planet", "value": "Venus",
    })
    store.save_memory(old)
    store.upsert_fact("user:favorite_planet", MemoryItem(
        content="Your favorite planet is Saturn.", metadata={"value": "Saturn"},
    ))
    # Simulate an older duplicate left by a release without canonical fact keys.
    with store._get_connection() as connection:
        connection.execute(
            "INSERT INTO memories(content,memory_type,source,confidence,created_at,updated_at,"
            "metadata_json) VALUES(?,?,?,?,?,?,?)",
            ("Your favorite planet is Venus.", "fact", "user_explicit", 1,
             "2020-01-01", "2020-01-01",
             json.dumps(old.metadata)),
        )
        connection.execute("DELETE FROM memory_index_state WHERE name='recall_terms'")
    store.close()
    reopened = MemoryStore(path)
    try:
        results = reopened.ranked_recall_candidates({"planet"})
        assert [item.content for _, item in results] == ["Your favorite planet is Saturn."]
        reopened.forget_memories([results[0][1]])
        with reopened._get_connection() as connection:
            connection.execute("DELETE FROM memory_index_state WHERE name='recall_terms'")
    finally:
        reopened.close()
    rebuilt = MemoryStore(path)
    try:
        assert rebuilt.ranked_recall_candidates({"planet"}) == []
        assert rebuilt.get_fact("user:favorite_planet") is None
    finally:
        rebuilt.close()


def test_settings_are_json_persisted_and_default_is_not_saved(tmp_path):
    """Recall preferences survive restart without modifying a missing setting on read."""
    path = tmp_path / "settings.db"
    store = MemoryStore(path)
    assert store.get_setting("recall_enabled", True) is True
    store.set_setting("recall_enabled", False)
    store.close()
    reopened = MemoryStore(path)
    try:
        assert reopened.get_setting("recall_enabled", True) is False
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM memory_settings").fetchone()[0] == 1
    finally:
        reopened.close()


def test_completed_index_migration_is_not_repeated_on_restart(tmp_path, monkeypatch):
    """Normal startup must reuse the persisted word index instead of scanning all facts."""
    path = tmp_path / "indexed.db"
    store = MemoryStore(path)
    store.save_memory(MemoryItem(content="The observatory houses a telescope"))
    store.close()

    def unexpected_reindex(*_args):
        raise AssertionError("Completed recall migration must not run again")

    monkeypatch.setattr(MemoryStore, "_sync_recall_index", unexpected_reindex)
    restarted = MemoryStore(path)
    try:
        assert len(restarted.ranked_recall_candidates({"telescope"})) == 1
    finally:
        restarted.close()


def test_recall_query_is_bounded_and_index_transaction_rolls_back(tmp_path, monkeypatch):
    """Oversized queries stay valid and failed indexing cannot leave a partially saved fact."""
    store = MemoryStore(tmp_path / "atomic.db")
    try:
        assert store.ranked_recall_candidates({f"term{index}" for index in range(2000)}) == []

        def failed_index(*_args):
            raise RuntimeError("Index write failed")

        monkeypatch.setattr(store, "_sync_recall_index", failed_index)
        with pytest.raises(RuntimeError, match="Index write failed"):
            store.save_memory(MemoryItem(content="A partially written note"))
        assert store.count_memories() == 0
    finally:
        store.close()


def test_migration_removes_legacy_mapping_that_resurrected_a_forgotten_duplicate(tmp_path):
    """Older startup recovery must not make a tombstone-covered duplicate authoritative."""
    path = tmp_path / "resurrected.db"
    store = MemoryStore(path)
    store.upsert_fact("user:favorite_planet", MemoryItem(
        content="Your favorite planet is Venus.", metadata={"value": "Venus"},
    ))
    store.upsert_fact("user:favorite_planet", MemoryItem(
        content="Your favorite planet is Saturn.", metadata={"value": "Saturn"},
    ))
    current = store.get_fact("user:favorite_planet")
    connection = store._get_connection()
    with connection:
        duplicate = connection.execute(
            "INSERT INTO memories(content,memory_type,source,confidence,created_at,updated_at,"
            "metadata_json) VALUES(?,?,?,?,?,?,?)",
            ("Your favorite planet is Venus.", "fact", "user_explicit", 1,
             "2020-01-01", "2020-01-01", json.dumps({
                 "scope": "user", "fact_key": "favorite_planet", "value": "Venus",
             })),
        ).lastrowid
    store.forget_memories([current])
    with connection:
        # The older initializer rebuilt keys without checking forgotten values.
        connection.execute(
            "INSERT INTO memory_keys VALUES (?,?)", ("user:favorite_planet", duplicate),
        )
        connection.execute("DELETE FROM memory_index_state")
    store.close()

    migrated = MemoryStore(path)
    try:
        assert migrated.get_fact("user:favorite_planet") is None
        assert migrated.ranked_recall_candidates({"planet"}) == []
        assert migrated.list_memories() == []
    finally:
        migrated.close()


def test_rebuilding_preserves_reauthorized_fact_with_older_forgotten_values(tmp_path):
    """An older tombstone must not hide the current explicitly reauthorized value."""
    path = tmp_path / "reauthorized.db"
    store = MemoryStore(path)
    store.upsert_fact("user:favorite_planet", MemoryItem(
        content="Your favorite planet is Venus.", metadata={"value": "Venus"},
    ))
    store.forget_memories([store.get_fact("user:favorite_planet")])
    store.upsert_fact("user:favorite_planet", MemoryItem(
        content="Your favorite planet is Saturn.", metadata={"value": "Saturn"},
    ))
    with store._get_connection() as connection:
        connection.execute("DELETE FROM memory_index_state")
    store.close()

    migrated = MemoryStore(path)
    try:
        assert migrated.get_fact("user:favorite_planet").metadata["value"] == "Saturn"
        assert [item.metadata["value"] for _, item in
                migrated.ranked_recall_candidates({"planet"})] == ["Saturn"]
    finally:
        migrated.close()


@pytest.mark.parametrize("value", ["LA", "UK", "C", "4"])
def test_forgetting_short_values_redacts_whole_words_and_preserves_substrings(value):
    """Known short names and identifiers stay private without matching inside other words."""
    store = MemoryStore(":memory:")
    try:
        store.upsert_fact("user:short_identifier", MemoryItem(
            content=f"Your identifier is {value}.", metadata={"value": value},
        ))
        store.forget_memories([store.get_fact("user:short_identifier")])

        assert store.redact_forgotten(f"The identifier was {value}.") == (
            "The identifier was [forgotten fact]."
        )
        assert store.redact_forgotten(f"prefix{value}suffix") == f"prefix{value}suffix"
    finally:
        store.close()


def test_forgetting_also_redacts_short_replaced_values():
    """Earlier one-letter corrections must not return through historical conversation."""
    store = MemoryStore(":memory:")
    try:
        for value in ("R", "G"):
            store.upsert_fact("user:favorite_letter", MemoryItem(
                content=f"Your favorite letter is {value}.", metadata={"value": value},
            ))
        store.forget_memories([store.get_fact("user:favorite_letter")])

        assert store.redact_forgotten("R and G are letters in RGB.") == (
            "[forgotten fact] and [forgotten fact] are letters in RGB."
        )
    finally:
        store.close()


def test_hidden_duplicates_do_not_distort_distinctive_term_ranking():
    """Obsolete rows must not inflate corpus size and push rare relevant facts out of recall."""
    store = MemoryStore(":memory:")
    try:
        store.save_memory(MemoryItem(content="Unrelated canonical note", metadata={
            "scope": "user", "fact_key": "unrelated",
        }))
        for index in range(30):
            store.save_memory(MemoryItem(content=f"alpha beta shared note {index}"))
        target = MemoryItem(content="quasar target")
        store.save_memory(target)
        query = {"alpha", "beta", "quasar"}
        baseline = store.ranked_recall_candidates(query, limit=2)
        assert baseline[0][1].id == target.id

        for index in range(100):
            store.save_memory(MemoryItem(content=f"Unrelated hidden duplicate {index}", metadata={
                "scope": "user", "fact_key": "unrelated",
            }))
        ranked = store.ranked_recall_candidates(query, limit=2)

        assert [item.id for _, item in ranked] == [item.id for _, item in baseline]
        assert [score for score, _ in ranked] == pytest.approx([score for score, _ in baseline])
    finally:
        store.close()


def test_guarded_fact_create_rejects_another_stores_competing_create(tmp_path):
    """Approval to create an absent fact cannot replace a concurrently created value."""
    from ciel.memory.store import MemoryConflictError

    path = tmp_path / "guarded-create.db"
    first, second = MemoryStore(path), MemoryStore(path)
    try:
        first.upsert_fact("user:current_project", MemoryItem(
            content="Your current project is Atlas.", metadata={"value": "Atlas"},
        ), expected_content=None)

        with pytest.raises(MemoryConflictError):
            second.upsert_fact("user:current_project", MemoryItem(
                content="Your current project is Borealis.", metadata={"value": "Borealis"},
            ), expected_content=None)

        current = first.get_fact("user:current_project")
        assert current.metadata["value"] == "Atlas"
        assert not current.metadata.get("revisions")
        assert second.ranked_recall_candidates({"borealis"}) == []
    finally:
        first.close()
        second.close()


def test_guarded_fact_update_rejects_another_stores_concurrent_correction(tmp_path):
    """Confirmation of one baseline must not overwrite a later unapproved correction."""
    from ciel.memory.store import MemoryConflictError

    path = tmp_path / "guarded-update.db"
    first, second = MemoryStore(path), MemoryStore(path)
    try:
        first.upsert_fact("user:current_project", MemoryItem(
            content="Your current project is Atlas.", metadata={"value": "Atlas"},
        ))
        baseline = first.get_fact("user:current_project").content
        second.upsert_fact("user:current_project", MemoryItem(
            content="Your current project is Borealis.", metadata={"value": "Borealis"},
        ))

        with pytest.raises(MemoryConflictError):
            first.upsert_fact("user:current_project", MemoryItem(
                content="Your current project is Cygnus.", metadata={"value": "Cygnus"},
            ), expected_content=baseline)

        current = second.get_fact("user:current_project")
        assert current.metadata["value"] == "Borealis"
        assert [revision["value"] for revision in current.metadata["revisions"]] == ["Atlas"]
        assert first.ranked_recall_candidates({"cygnus"}) == []
    finally:
        first.close()
        second.close()


def test_guarded_fact_update_accepts_matching_baseline_and_preserves_revisions():
    """An unchanged approved value remains replaceable through the guarded API."""
    store = MemoryStore(":memory:")
    try:
        original = MemoryItem(
            content="Your current project is Atlas.", metadata={"value": "Atlas"},
        )
        memory_id = store.upsert_fact("user:current_project", original, expected_content=None)
        replacement = MemoryItem(
            content="Your current project is Borealis.", metadata={"value": "Borealis"},
        )

        assert store.upsert_fact(
            "user:current_project", replacement, expected_content=original.content,
        ) == memory_id
        current = store.get_fact("user:current_project")
        assert current.metadata["value"] == "Borealis"
        assert [revision["value"] for revision in current.metadata["revisions"]] == ["Atlas"]
    finally:
        store.close()
