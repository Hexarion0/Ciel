"""Regression coverage for dated recall and project timeline anchors."""

from datetime import datetime, timezone

import pytest

from ciel.memory import MemoryItem, MemoryStore, MemoryType
from ciel.memory.context import recall_context_memories


def project_start(value="2026-09-28"):
    """Create a user-confirmed project date without relying on generated summaries."""
    return MemoryItem(
        content=f"CIEL project development began on {value}.",
        memory_type=MemoryType.PROJECT,
        source="user_explicit",
        metadata={"project": "ciel", "fact_key": "project_start_date", "value": value},
    )


def test_project_date_survives_restart_and_informal_query(tmp_path):
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    store.save_memory(project_start())
    store.close()
    restarted = MemoryStore(path)
    try:
        context = recall_context_memories(restarted, "What day are we on?", datetime(2026, 10, 3))
        joined = "\n".join(context)
        assert "2026-09-28" in joined
        assert "5 elapsed calendar days" in joined
        assert "development day 6" in joined
    finally:
        restarted.close()


def test_relative_facts_keep_their_recorded_date_and_exclude_summaries():
    store = MemoryStore(":memory:")
    try:
        store.save_memory(
            MemoryItem(
                content="This is the third day I am developing you.",
                created_at=datetime(2026, 9, 30, 12, tzinfo=timezone.utc),
            )
        )
        store.save_memory(
            MemoryItem(
                content="The third day means today.", memory_type=MemoryType.CONVERSATION,
            )
        )
        context = recall_context_memories(store, "third", datetime(2026, 10, 3))
        assert len(context) == 1
        assert "Recorded at 2026-09-30" in context[0]
        assert "This is the third day I am developing you." in context[0]
        assert "elapsed calendar days" not in context[0]
    finally:
        store.close()


def test_project_anchor_is_not_duplicated_when_query_already_matches():
    store = MemoryStore(":memory:")
    try:
        store.save_memory(project_start())
        context = recall_context_memories(store, "CIEL", datetime(2026, 10, 3))
        assert len(context) == 2  # One saved anchor and one dated calculation.
    finally:
        store.close()


@pytest.mark.parametrize("value", ["not-a-date", "2026-11-01"])
def test_invalid_or_future_project_dates_do_not_invent_elapsed_days(value):
    store = MemoryStore(":memory:")
    try:
        store.save_memory(project_start(value))
        context = recall_context_memories(store, now=datetime(2026, 10, 3))
        assert len(context) == 1
        assert "elapsed calendar days" not in context[0]
    finally:
        store.close()
