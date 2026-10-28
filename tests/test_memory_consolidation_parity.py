"""Deterministic OpenClaw-style promotion gates and explicit guarded owner review."""

import json
import math
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from ciel.config import MemoryConfig
from ciel.memory.bridge import ConversationContext, desktop_conversation_id
from ciel.memory.lifecycle import MAX_CURATED_CHARS, MemoryLifecycle
from ciel.memory.manager import ConversationManager
from ciel.memory.models import ConversationTurn, MemoryItem
from ciel.memory.store import MemoryStore
from ciel.providers.base import LLMResponse

NOTE = "For the Cedar rollout we decided to keep an offline recovery bundle."
OTHER = "My telescope project uses a compact wooden observing platform in the garden."
NOW = datetime(2026, 10, 26, 4, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    memory = MemoryStore(tmp_path / "memory.db", workspace_path=tmp_path / "workspace")
    yield memory
    memory.close()


def capture(lifecycle, text=NOTE, *, session="default", role="user", origin="owner"):
    turn = ConversationTurn(role, text, session_id=session, timestamp=NOW)
    lifecycle.store.save_turn(turn)
    lifecycle.capture_turn(turn, trusted_origin=origin)
    return turn


def recalled(lifecycle, *, queries=3, score=1.0, query_prefix="recovery query"):
    """Supply real-shaped interactive retrieval hits over the live owner daily quote."""
    lifecycle.flush()
    candidates = lifecycle._candidates(lifecycle.snapshot(), limit=None, max_chars=None)
    results = [{"path": item["daily_path"], "start_line": item["start_line"],
                "end_line": item["end_line"], "snippet": item["content"], "score": score}
               for item in candidates]
    for index in range(queries):
        lifecycle.record_recall(results, f"{query_prefix} {index}")
    return results


def provider(kind="add", target="", *, every=False, callback=None):
    result = MagicMock()

    def send(messages, **kwargs):
        payload = json.loads(messages[-1].content)
        if callback:
            callback(payload)
        sources = payload["observations"] if every else payload["observations"][:1]
        actions = [{"action": kind, "text": item["text"], "target": target,
                    "evidence": [{"id": item["id"], "quote": item["text"]}]}
                   for item in sources]
        return LLMResponse(json.dumps({"actions": actions}), "test", "test")

    result.send.side_effect = send
    return result


def curated(store):
    return next((item.text for item in store.workspace.documents() if item.path == "MEMORY.md"), "")


def test_one_off_and_two_queries_never_reach_model_and_can_later_qualify(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic", clock=lambda: NOW)
    capture(lifecycle)
    model = provider()
    assert lifecycle.consolidate(model).status == "ineligible"
    recalled(lifecycle, queries=2)
    assert lifecycle.consolidate(model).status == "ineligible"
    model.send.assert_not_called()
    with store._lock:
        assert store._get_connection().execute(
            "SELECT state FROM memory_observations",
        ).fetchone()[0] == "captured"
    recalled(lifecycle)
    result = lifecycle.consolidate(model)
    assert result.promoted == 1
    model.send.assert_called_once()


def test_repeated_same_query_cannot_manufacture_query_diversity(store):
    lifecycle = MemoryLifecycle(store, clock=lambda: NOW)
    capture(lifecycle)
    results = recalled(lifecycle, queries=1)
    for _ in range(20):
        lifecycle.record_recall(results, "  RECOVERY   QUERY 0 ")
    assert lifecycle.consolidate(provider()).status == "ineligible"
    with store._lock:
        connection = store._get_connection()
        assert connection.execute("SELECT COUNT(*) FROM memory_recall_signals").fetchone()[0] == 1
        assert "recovery" not in connection.execute(
            "SELECT query_hash FROM memory_recall_signals",
        ).fetchone()[0]


def test_low_quality_hits_fail_weighted_gate_even_when_counts_pass(store):
    lifecycle = MemoryLifecycle(store, clock=lambda: NOW)
    capture(lifecycle)
    recalled(lifecycle, queries=10, score=.05)
    model = provider()
    assert lifecycle.consolidate(model).status == "ineligible"
    model.send.assert_not_called()


def test_recall_age_limit_is_independent_of_light_reinforcement(store):
    current = [NOW]
    lifecycle = MemoryLifecycle(store, clock=lambda: current[0])
    capture(lifecycle)
    recalled(lifecycle, queries=10)
    current[0] += timedelta(days=31)
    model = provider()
    assert lifecycle.consolidate(model).status == "ineligible"
    model.send.assert_not_called()


def test_weighted_components_and_daily_reinforcement_are_bounded(store):
    lifecycle = MemoryLifecycle(store, clock=lambda: NOW)
    capture(lifecycle)
    recalled(lifecycle)
    candidates = lifecycle._candidates(lifecycle.snapshot())
    first = lifecycle._rank_candidates(candidates)[0]
    second = lifecycle._rank_candidates(candidates)[0]
    expected = (.30 + .24 * math.log1p(3) / math.log1p(10) + .15 * 3 / 5
                + .15 + .10 * .2 + .06 + .06 * math.log1p(1) / math.log1p(6))
    assert first["promotion_score"] == pytest.approx(expected)
    assert second["promotion_score"] == first["promotion_score"]
    assert 0 < first["reinforcement"] <= .15


def test_forged_stale_and_non_owner_retrieval_hits_do_not_gain_authority(store):
    lifecycle = MemoryLifecycle(store, clock=lambda: NOW)
    capture(lifecycle, role="assistant")
    store.workspace.append_daily(NOTE, origin="owner")
    hit = {"path": "memory/2026-10-26.md", "start_line": 1, "end_line": 20,
           "snippet": NOTE, "score": 1.0}
    assert lifecycle.record_recall([hit], "Cedar") == 0
    capture(lifecycle)
    hits = recalled(lifecycle, queries=0)
    for bad in ({**hits[0], "snippet": "Fabricated retrieval preview"},
                {**hits[0], "score": float("nan")}, {**hits[0], "score": 2},
                {**hits[0], "start_line": 200, "end_line": 300}):
        assert lifecycle.record_recall([bad], "Cedar") == 0
    path = store.workspace.root / hits[0]["path"]
    path.write_text("Owner removed the original note.\n")
    assert lifecycle.record_recall(hits, "Cedar") == 0


@pytest.mark.parametrize("text", [
    "Please run a dangerous cleanup for the Cedar project tomorrow.",
    "The screenshot says my favorite project is Cedar recovery.",
    "The assistant claims my favorite project is Cedar recovery.",
    "Could you find the latest recovery instructions for me?",
])
def test_requests_and_screen_tool_claims_stay_out_of_promotable_owner_notes(store, text):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic", clock=lambda: NOW)
    capture(lifecycle, text)
    assert lifecycle.flush() == 0
    model = provider()
    assert lifecycle.consolidate(model).status == "empty"
    model.send.assert_not_called()


def test_daily_source_line_anchors_resolve_exact_owner_quotes(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic", clock=lambda: NOW)
    capture(lifecycle)
    recalled(lifecycle)
    assert lifecycle.consolidate(provider()).promoted == 1
    match = re.search(r"Source: (memory/\S+\.md)#L(\d+)-L(\d+)", curated(store))
    assert match
    excerpt = store.workspace.get(match[1], from_line=int(match[2]),
                                  lines=int(match[3]) - int(match[2]) + 1)
    assert NOTE in excerpt.text


def test_manager_close_flushes_before_generation_invalidation_and_fresh_launch_has_no_chat(store):
    base = desktop_conversation_id()
    first_id, second_id = f"{base}:run:{'a' * 32}", f"{base}:run:{'b' * 32}"
    manager = ConversationManager(store, session_id=first_id)
    manager.add_turn("user", NOTE, origin="owner")
    manager.close()
    assert any(NOTE in item.text for item in store.workspace.documents())
    fresh = ConversationManager(store, session_id=second_id,
                                memory_config=MemoryConfig(learning_mode="confirm"))
    assert fresh.get_recent_turns() == []
    assert fresh.flush_memory() == 0
    hits = recalled(fresh.lifecycle)
    assert hits
    result = fresh.consolidate_memory(provider())
    assert result.review == 1
    fresh.close()


def test_saved_recall_signals_and_eligible_notes_survive_a_new_process_and_launch(store):
    base = desktop_conversation_id()
    first_id, second_id = f"{base}:run:{'a' * 32}", f"{base}:run:{'b' * 32}"
    lifecycle = MemoryLifecycle(store, context_resolver=lambda: ConversationContext(
        first_id, (first_id,)), learning_mode="automatic", clock=lambda: NOW)
    capture(lifecycle, session=first_id)
    recalled(lifecycle)
    lifecycle.close()
    other = MemoryStore(store.db_path_str, workspace_path=store.workspace.root)
    try:
        restarted = MemoryLifecycle(other, context_resolver=lambda: ConversationContext(
            second_id, (second_id,)), learning_mode="automatic", clock=lambda: NOW)
        assert restarted.consolidate(provider()).promoted == 1
    finally:
        other.close()


def test_paired_owner_identity_survives_launch_rotation_but_never_repairing(store):
    base = desktop_conversation_id()
    current = [ConversationContext("bridge-a", (
        f"{base}:run:{'a' * 32}", f"telegram:12:34:v1:run:{'a' * 32}"))]
    lifecycle = MemoryLifecycle(store, context_resolver=lambda: current[0], clock=lambda: NOW)
    capture(lifecycle, session=current[0].session_ids[0])
    recalled(lifecycle)
    current[0] = ConversationContext("bridge-b", (
        f"{base}:run:{'b' * 32}", f"telegram:12:99:v1:run:{'b' * 32}"))
    assert lifecycle.consolidate(provider()).status == "empty"
    current[0] = ConversationContext("bridge-c", (
        f"{base}:run:{'c' * 32}", f"telegram:12:34:v1:run:{'c' * 32}"))
    assert lifecycle.consolidate(provider()).review == 1


@pytest.mark.parametrize("mode", ["confirm", "explicit"])
def test_review_approval_is_real_offline_idempotent_and_survives_restart(store, mode):
    lifecycle = MemoryLifecycle(store, learning_mode=mode, clock=lambda: NOW)
    capture(lifecycle)
    recalled(lifecycle)
    proposal = lifecycle.consolidate(provider())
    assert proposal.review == 1 and not curated(store)
    lifecycle.close()
    restarted = MemoryLifecycle(store, learning_mode=mode, clock=lambda: NOW)
    reviews = restarted.list_reviews()
    assert reviews[0]["actions"][0]["index"] == 0
    assert reviews[0]["actions"][0]["sources"][0].startswith("memory/")
    result = restarted.apply_review(proposal.run_id)
    assert result.promoted == 1 and result.review == 0
    assert NOTE in curated(store)
    assert restarted.list_reviews() == []
    assert restarted.apply_review(proposal.run_id).status == "duplicate"


@pytest.mark.parametrize("kind", ["merge", "supersede"])
def test_only_explicit_review_can_merge_or_supersede_and_preserves_unrelated_notes(store, kind):
    old = "For the Cedar rollout we decided to use a cloud-only recovery bundle."
    unrelated = "The garden observing platform must fit through the side gate."
    store.workspace.append_curated(old + "\n\n" + unrelated)
    lifecycle = MemoryLifecycle(store, learning_mode="automatic", clock=lambda: NOW)
    capture(lifecycle)
    recalled(lifecycle)
    proposed = lifecycle.consolidate(provider(kind, old))
    assert proposed.review == 1 and proposed.promoted == 0
    assert old in curated(store) and NOTE not in curated(store)
    approved = lifecycle.apply_review(proposed.run_id, [0])
    assert approved.promoted == 1
    assert NOTE in curated(store) and unrelated in curated(store)
    assert (old in curated(store)) == (kind == "merge")
    preimages = (store.workspace.root / ".memory-state/preimages").glob("*.md")
    assert any(old in path.read_text() for path in preimages)


def test_partial_review_and_invalid_selection_leave_remaining_decisions_intact(store):
    lifecycle = MemoryLifecycle(store, clock=lambda: NOW)
    capture(lifecycle)
    capture(lifecycle, OTHER)
    recalled(lifecycle)
    proposed = lifecycle.consolidate(provider(every=True))
    assert proposed.review == 2
    for indexes in ([], [2], [True], [0, 0]):
        assert lifecycle.apply_review(proposed.run_id, indexes).status == "invalid_selection"
        assert curated(store) == ""
    first = lifecycle.apply_review(proposed.run_id, [1])
    assert first.promoted == 1 and first.review == 1
    assert lifecycle.list_reviews()[0]["actions"][0]["index"] == 0
    assert lifecycle.apply_review(proposed.run_id).promoted == 1
    assert NOTE in curated(store) and OTHER in curated(store)


@pytest.mark.parametrize("change", ["source", "reset", "forget", "curated", "scope", "close"])
def test_review_rechecks_source_forgetting_owner_generations_and_curated_preimage(store, change):
    context = [ConversationContext("default", ("default",))]
    lifecycle = MemoryLifecycle(store, context_resolver=lambda: context[0], clock=lambda: NOW)
    capture(lifecycle)
    hits = recalled(lifecycle)
    proposed = lifecycle.consolidate(provider())
    if change == "source":
        (store.workspace.root / hits[0]["path"]).write_text("Owner removed the note.\n")
    elif change == "reset":
        store.clear_session("default")
    elif change == "forget":
        with store._lock:
            connection = store._get_connection()
            with connection:
                connection.execute("INSERT INTO forgotten_memories VALUES (?,?)",
                                   ("test:bundle", json.dumps(["offline recovery bundle"])))
    elif change == "curated":
        store.workspace.append_curated("A manual owner edit.")
    elif change == "scope":
        context[0] = ConversationContext("other", ("other",))
    else:
        lifecycle.close()
    result = lifecycle.apply_review(proposed.run_id)
    assert result.promoted == 0
    assert result.status in {"stale", "conflict", "not_found", "disabled"}
    assert NOTE not in curated(store)


def test_curated_bootstrap_budget_queues_auto_add_and_refuses_oversized_review(store):
    store.workspace.append_curated("x" * (MAX_CURATED_CHARS - 30))
    lifecycle = MemoryLifecycle(store, learning_mode="automatic", clock=lambda: NOW)
    capture(lifecycle)
    recalled(lifecycle)
    proposed = lifecycle.consolidate(provider())
    assert proposed.review == 1 and proposed.promoted == 0
    assert lifecycle.apply_review(proposed.run_id).status == "budget_exceeded"
    assert len(curated(store)) <= MAX_CURATED_CHARS
    assert NOTE not in curated(store)


def test_review_cannot_rewrite_managed_current_facts_even_with_owner_approval(store):
    fact = MemoryItem("For the Cedar rollout we decided to use a cloud-only recovery bundle.",
                      metadata={"key": "project:cedar:recovery"})
    store.save_memory(fact)
    old = fact.content
    lifecycle = MemoryLifecycle(store, learning_mode="automatic", clock=lambda: NOW)
    capture(lifecycle)
    recalled(lifecycle)
    proposed = lifecycle.consolidate(provider("supersede", old))
    assert proposed.review == 1
    before = curated(store)
    assert lifecycle.apply_review(proposed.run_id).status == "conflict"
    assert curated(store) == before
    assert store.get_memory(fact.id).content == old


def test_eligible_old_note_is_not_hidden_by_recent_unqualified_candidates(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic", clock=lambda: NOW)
    capture(lifecycle)
    recalled(lifecycle)
    for index in range(70):
        capture(lifecycle, f"The current observing project has temporary workshop fixture {index}.")
    result = lifecycle.consolidate(provider())
    assert result.promoted == 1
    assert NOTE in curated(store)


def test_unqualified_duplicate_cannot_hide_an_eligible_original_from_the_model(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic", clock=lambda: NOW)
    capture(lifecycle)
    recalled(lifecycle)
    capture(lifecycle)
    result = lifecycle.consolidate(provider())
    assert result.promoted == 1
    assert curated(store).count(NOTE) == 1


def test_promotion_budget_reserves_actual_profile_and_configured_bootstrap_limit(store):
    store.workspace._append("USER.md", "A" * 500, entry_id="profile", origin="owner",
                            session_id="default", local_only=False)
    lifecycle = MemoryLifecycle(store, learning_mode="automatic", clock=lambda: NOW,
                                bootstrap_max_chars=1000, profile_max_chars=500)
    capture(lifecycle)
    recalled(lifecycle)
    proposed = lifecycle.consolidate(provider())
    assert proposed.review == 1 and proposed.promoted == 0
    assert lifecycle.apply_review(proposed.run_id).status == "budget_exceeded"
    assert curated(store) == ""


def test_promoted_curated_entries_keep_bootstrap_trust_provenance(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic", clock=lambda: NOW)
    capture(lifecycle)
    recalled(lifecycle)
    assert lifecycle.consolidate(provider()).promoted == 1
    assert store.workspace.provenance("MEMORY.md")["trusted"]


def test_private_curated_provenance_is_never_sent_to_reflection_provider(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic", clock=lambda: NOW)
    capture(lifecycle)
    recalled(lifecycle)
    store.workspace._append("MEMORY.md", "Private deployment context.", entry_id="private",
                            origin="owner", session_id="default", local_only=True)
    model = provider()
    assert lifecycle.consolidate(model).status == "private"
    model.send.assert_not_called()


def test_daily_edit_extending_an_owner_quote_revokes_the_old_original(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic", clock=lambda: NOW)
    capture(lifecycle)
    hits = recalled(lifecycle)
    path = store.workspace.root / hits[0]["path"]
    path.write_text(path.read_text().replace(NOTE, NOTE + " That decision was later cancelled."))
    model = provider()
    assert lifecycle.consolidate(model).status == "empty"
    model.send.assert_not_called()


@pytest.mark.parametrize("revocation", ["private", "private_uppercase", "reset"])
def test_previous_launch_evidence_rechecks_every_original_paired_channel(store, revocation):
    base = desktop_conversation_id()
    old = ConversationContext("bridge-a", (
        f"{base}:run:{'a' * 32}", f"telegram:12:34:v1:run:{'a' * 32}"))
    current = [old]
    lifecycle = MemoryLifecycle(store, context_resolver=lambda: current[0], clock=lambda: NOW)
    capture(lifecycle, session=old.session_ids[0])
    recalled(lifecycle)
    if revocation.startswith("private"):
        marker = "/LOCAL" if revocation == "private_uppercase" else "/local"
        capture(lifecycle, marker + " The launch phrase is private.", session=old.session_ids[1])
    else:
        store.clear_session(old.session_ids[1])
    current[0] = ConversationContext("bridge-b", (
        f"{base}:run:{'b' * 32}", f"telegram:12:34:v1:run:{'b' * 32}"))
    model = provider()
    assert lifecycle.consolidate(model).status == "empty"
    model.send.assert_not_called()


def test_curated_edit_conflict_can_produce_a_fresh_review_without_relearning_the_quote(store):
    lifecycle = MemoryLifecycle(store, clock=lambda: NOW)
    capture(lifecycle)
    recalled(lifecycle)
    first = lifecycle.consolidate(provider())
    assert first.review == 1
    assert lifecycle.consolidate(provider()).status == "duplicate"
    store.workspace.append_curated("A manual owner edit about the garden.")
    assert lifecycle.apply_review(first.run_id).status == "conflict"
    second = lifecycle.consolidate(provider())
    assert second.review == 1 and second.run_id != first.run_id
    assert [item["run_id"] for item in lifecycle.list_reviews()] == [second.run_id]
    assert lifecycle.apply_review(first.run_id).status == "stale"
    assert lifecycle.apply_review(second.run_id).promoted == 1
    assert "manual owner edit" in curated(store) and NOTE in curated(store)


def test_a_bounded_plan_keeps_unselected_eligible_evidence_for_later_review(store):
    lifecycle = MemoryLifecycle(store, clock=lambda: NOW)
    capture(lifecycle)
    capture(lifecycle, OTHER)
    recalled(lifecycle)
    model = provider()
    first = lifecycle.consolidate(model)
    second = lifecycle.consolidate(model)
    assert first.review == second.review == 1 and first.run_id != second.run_id
    assert lifecycle.consolidate(model).status == "duplicate"
    assert model.send.call_count == 2
    assert len(lifecycle.list_reviews()) == 2
    assert lifecycle.apply_review(first.run_id).promoted == 1
    # The remaining source stays eligible when approving another run changes
    # the curated preimage and it needs a newly reviewed proposal.
    refreshed = lifecycle.consolidate(model)
    assert refreshed.review == 1 and refreshed.run_id not in {first.run_id, second.run_id}
    assert lifecycle.apply_review(refreshed.run_id).promoted == 1
    assert NOTE in curated(store) and OTHER in curated(store)


def test_uppercase_local_marker_is_rechecked_inside_the_writer_transaction(store):
    lifecycle = MemoryLifecycle(store, clock=lambda: NOW)
    capture(lifecycle)
    guard = lifecycle.snapshot()
    capture(lifecycle, "/LOCAL The next deployment discussion stays private.")
    with store._lock:
        assert lifecycle._private_on_connection(store._get_connection(), guard)
    model = provider()
    assert lifecycle.consolidate(model).status == "private"
    model.send.assert_not_called()


@pytest.mark.parametrize("origin", ["system", "untrusted", "tool", "screen"])
def test_source_taint_revokes_promotion_before_the_model_despite_an_unchanged_quote(store, origin):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic", clock=lambda: NOW)
    capture(lifecycle)
    recalled(lifecycle)
    store.workspace.append_daily("An external observation.", now=NOW, origin=origin)
    model = provider()
    assert lifecycle.consolidate(model).status == "empty"
    model.send.assert_not_called()
    assert curated(store) == ""


def test_reset_erases_only_its_observations_signals_and_related_review_plans(store):
    saved = MemoryItem("My favorite color is cobalt.", metadata={"key": "preference:color"})
    store.save_memory(saved)
    lifecycle = MemoryLifecycle(store, clock=lambda: NOW)
    capture(lifecycle)
    capture(lifecycle, "Which recovery choice should we make?")
    recalled(lifecycle)
    first = lifecycle.consolidate(provider())
    other = MemoryLifecycle(
        store, context_resolver=lambda: ConversationContext("other", ("other",)), clock=lambda: NOW,
    )
    capture(other, OTHER, session="other")
    recalled(other)
    second = other.consolidate(provider())
    daily_before = {item.path: item.text for item in store.workspace.documents()
                    if item.path.startswith("memory/")}
    assert store.clear_session("default") == 2
    with store._lock:
        connection = store._get_connection()
        observations = connection.execute("SELECT entry_id,session_id FROM memory_observations")
        remaining = [tuple(row) for row in observations]
        assert len(remaining) == 1 and remaining[0][1] == "other"
        for table in ("memory_recall_signals", "memory_phase_signals"):
            assert {row[0] for row in connection.execute(f"SELECT entry_id FROM {table}")} == {
                remaining[0][0],
            }
        assert [row[0] for row in connection.execute(
            "SELECT run_id FROM memory_reflections",
        )] == [second.run_id]
    assert lifecycle.apply_review(first.run_id).status == "not_found"
    assert other.list_reviews()[0]["run_id"] == second.run_id
    assert store.get_memory(saved.id).content == saved.content
    assert daily_before == {item.path: item.text for item in store.workspace.documents()
                            if item.path.startswith("memory/")}
