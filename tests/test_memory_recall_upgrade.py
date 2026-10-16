"""Indexed recall, pause controls, duplicate suppression, and referential continuity."""

from datetime import datetime

import pytest

from ciel.memory import MemoryItem, MemoryStore, MemoryType
from ciel.memory.context import contextual_recall_query, recall_context_memories
from ciel.memory.recall import rank_memories, terms
from ciel.providers.base import ChatMessage


@pytest.fixture
def store():
    memory = MemoryStore(":memory:")
    yield memory
    memory.close()


def test_indexed_recall_finds_old_fact_beyond_recent_two_thousand(store):
    old = MemoryItem("My telescope uses an equatorial mount.")
    store.save_memory(old)
    for index in range(2000):
        store.save_memory(MemoryItem(f"Unrelated synthetic note {index}."))

    ranked = rank_memories(store, "telescope mount")

    assert len(ranked) == 1
    assert ranked[0][1].id == old.id


def test_distinctive_query_word_outranks_generic_overlap(store):
    distinctive = MemoryItem("My telescope is an heirloom.")
    store.save_memory(distinctive)
    for index in range(12):
        store.save_memory(MemoryItem(f"Keyboard accessory number {index}."))

    ranked = rank_memories(store, "telescope keyboard", limit=2)

    assert len(ranked) == 2
    assert ranked[0][1].id == distinctive.id


@pytest.mark.parametrize("query", ["café", "cafe\u0301", "ＣＡＦÉ"])
def test_unicode_equivalent_queries_share_the_written_index(store, query):
    store.save_memory(MemoryItem("My café closes on Tuesday."))
    assert rank_memories(store, query)[0][1].content == "My café closes on Tuesday."


def test_explicit_spelling_aliases_and_word_boundaries(store):
    store.save_memory(MemoryItem("Your favorite colour is blue."))
    store.save_memory(MemoryItem("Your profession is an illustrator."))
    store.save_memory(MemoryItem("My scarlet scarf is handmade."))
    assert "colour" in rank_memories(store, "favorite color")[0][1].content
    assert "illustrator" in rank_memories(store, "occupation")[0][1].content
    assert rank_memories(store, "car") == []
    assert terms("_%?") == set()


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_rank_limit_does_not_return_matches(store, limit):
    store.save_memory(MemoryItem("My telescope uses an equatorial mount."))
    assert rank_memories(store, "telescope", limit=limit) == []


def test_conversation_summaries_never_compete_with_durable_facts(store):
    store.save_memory(MemoryItem("The user's telescope is blue.",
                                 memory_type=MemoryType.CONVERSATION))
    store.save_memory(MemoryItem("My telescope is white."))
    assert [item.content for _, item in rank_memories(store, "telescope")] == [
        "My telescope is white.",
    ]


@pytest.mark.parametrize("query", [
    "Continue with that.", "Can you give me more details?", "Could you tell me more?",
    "Explain further.", "Keep going.",
])
def test_common_followups_recall_latest_user_topic_without_assistant_claims(store, query):
    store.save_memory(MemoryItem("My mechanical keyboard uses silent switches."))
    store.save_memory(MemoryItem("My telescope uses an equatorial mount."))
    dialogue = [
        ChatMessage("user", "Tell me about my keyboard"),
        ChatMessage("assistant", "The user wants to know about their telescope."),
        ChatMessage("user", query),
    ]

    context = recall_context_memories(store, query, recent_messages=dialogue)

    assert any("silent switches" in entry for entry in context)
    assert all("telescope" not in entry for entry in context)


@pytest.mark.parametrize("query", [
    "Can you give me more details about my telescope?", "Continue with my telescope.",
    "What about my telescope?",
])
def test_followup_with_explicit_topic_does_not_borrow_an_older_topic(query):
    dialogue = [ChatMessage("user", "Tell me about my keyboard")]
    assert contextual_recall_query(query, dialogue) == query


def test_recall_pause_preserves_data_and_survives_restart(tmp_path):
    path = tmp_path / "memory.db"
    memory = MemoryStore(path)
    memory.upsert_fact("user:current_project", MemoryItem(
        "Your current project is Atlas.", memory_type=MemoryType.PROJECT,
        metadata={"value": "Atlas"},
    ))
    memory.set_setting("recall_enabled", False)
    memory.close()
    reopened = MemoryStore(path)
    try:
        assert recall_context_memories(reopened, "Atlas") == []
        assert reopened.get_fact("user:current_project").metadata["value"] == "Atlas"
        reopened.set_setting("recall_enabled", True)
        assert any("Atlas" in entry for entry in recall_context_memories(reopened, "Atlas"))
    finally:
        reopened.close()


def test_persisted_resume_overrides_default_recall_setting(store):
    store.save_memory(MemoryItem("My telescope uses an equatorial mount."))
    assert recall_context_memories(store, "telescope", enabled=False) == []
    store.set_setting("recall_enabled", True)
    assert recall_context_memories(store, "telescope", enabled=False)


@pytest.mark.parametrize("options", [{"limit": 0}, {"limit": -1}, {"max_chars": 0}])
def test_empty_context_budget_does_not_even_include_profile_anchors(store, options):
    store.upsert_fact("user:preferred_name", MemoryItem("Your preferred name is Alex."))
    assert recall_context_memories(store, "Alex", **options) == []


def test_duplicate_content_is_supplied_once(store):
    store.save_memory(MemoryItem("My telescope is white."))
    store.save_memory(MemoryItem("  My telescope is white!  "))
    context = recall_context_memories(store, "telescope")
    assert len(context) == 1


def test_current_keyed_fact_suppresses_legacy_duplicate_value(store):
    store.save_memory(MemoryItem(
        "Your favorite game is Sable.", memory_type=MemoryType.PREFERENCE,
        metadata={"scope": "user", "fact_key": "favorite_game", "value": "Sable"},
    ))
    store.upsert_fact("user:favorite_game", MemoryItem(
        "Your favorite game is Hades.", memory_type=MemoryType.PREFERENCE,
        metadata={"value": "Hades"},
    ))

    context = "\n".join(recall_context_memories(store, "favorite game"))

    assert context.count("Hades") == 1
    assert "Sable" not in context


def test_context_respects_custom_total_budget_and_bounds_untrusted_labels(store):
    for index in range(8):
        store.save_memory(MemoryItem(
            f"Keyboard project {index}: " + "x" * 10000,
            memory_type=MemoryType.PROJECT,
            source="source" * 1000,
            metadata={"observed_at": "timestamp" * 1000},
        ))
    context = recall_context_memories(store, "keyboard", max_chars=1000)
    assert context
    assert sum(map(len, context)) <= 1000
    assert max(map(len, context)) < 900


def test_timeline_calculation_survives_canonical_anchor_deduplication(store):
    store.upsert_fact("ciel:project_start_date", MemoryItem(
        "CIEL project development began on 2026-09-28.", memory_type=MemoryType.PROJECT,
        metadata={"value": "2026-09-28"},
    ))
    context = recall_context_memories(store, "project", now=datetime(2026, 10, 3))
    assert len(context) == 2
    assert "development day 6" in context[-1]
