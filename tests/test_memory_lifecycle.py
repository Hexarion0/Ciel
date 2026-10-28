"""Durable, provenance-bound daily notes and guarded overnight reflection."""

import json
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from ciel.config import MemoryConfig
from ciel.memory.bridge import ConversationContext
from ciel.memory.lifecycle import ConsolidationResult, MemoryLifecycle
from ciel.memory.manager import ConversationManager
from ciel.memory.models import ConversationTurn, MemoryItem
from ciel.memory.store import MemoryStore
from ciel.providers.base import LLMResponse
from ciel.providers.priority import BackgroundDeferred
from ciel.providers.router import ModelRouter

NOTE = "For the Cedar rollout we decided to keep an offline recovery bundle."


@pytest.fixture
def store(tmp_path):
    memory = MemoryStore(tmp_path / "memory.db", workspace_path=tmp_path / "workspace")
    yield memory
    memory.close()


def stage(lifecycle, text=NOTE, *, role="user", origin="owner", session="default"):
    """Persist an original turn and submit authority from an authenticated input boundary."""
    turn = ConversationTurn(role, text, session_id=session)
    lifecycle.store.save_turn(turn)
    lifecycle.capture_turn(turn, trusted_origin=origin)
    return turn


def qualify(lifecycle):
    """Exercise reflection safety after real-shaped recalls pass the deterministic gate."""
    lifecycle.flush()
    results = [{"path": item["daily_path"], "start_line": item["start_line"],
                "end_line": item["end_line"], "snippet": item["content"], "score": 1.0}
               for item in lifecycle._candidates(lifecycle.snapshot())]
    for query in ("Cedar recovery design", "offline restore choice", "rollout backup decision"):
        lifecycle.record_recall(results, query)


def provider_for(actions=None, callback=None):
    """Build plans using the exact IDs and excerpts sent to a tool-free model."""
    provider = MagicMock()

    def reply(messages, **kwargs):
        payload = json.loads(messages[-1].content)
        if callback:
            callback(payload)
        result = actions(payload) if callable(actions) else actions
        if result is None:
            source = payload["observations"][0]
            result = [{
                "action": "add", "text": source["text"], "target": "",
                "evidence": [{"id": source["id"], "quote": source["text"]}],
            }]
        return LLMResponse(json.dumps({"actions": result}), "test", "test")

    provider.send.side_effect = reply
    return provider


def public_text(store):
    return "\n".join(document.text for document in store.workspace.documents())


def curated(store):
    return next((item.text for item in store.workspace.documents() if item.path == "MEMORY.md"), "")


def test_capture_stages_originals_and_flush_selects_broad_working_notes(store):
    lifecycle = MemoryLifecycle(store)
    stage(lifecycle)
    stage(lifecycle, "Thanks!")
    stage(lifecycle, "What should we do next?")
    stage(lifecycle, "My current browser is fabricated by the assistant.", role="assistant")
    stage(lifecycle, "System job discovered an unconfirmed launch deadline.", origin="cron")
    stage(lifecycle, "RECALLED MEMORY: Owner lives on a fictional planet.")
    assert public_text(store) == ""
    assert lifecycle.flush() == 1
    assert NOTE in public_text(store)
    assert "historical context" in public_text(store)
    assert "Thanks!" not in public_text(store)
    assert "fabricated" not in public_text(store)
    assert "fictional" not in public_text(store)
    assert lifecycle.flush() == 0


def test_roles_and_session_names_do_not_authenticate_owner_evidence(store):
    lifecycle = MemoryLifecycle(store)
    stage(lifecycle, origin="")
    stage(lifecycle, role="tool")
    turn = ConversationTurn("user", NOTE, session_id="desktop_session:plausible")
    store.save_turn(turn)
    assert lifecycle.capture_turn(turn) == 0
    assert lifecycle.flush([turn]) == 0
    assert public_text(store) == ""


def test_only_persisted_exact_owner_original_can_be_staged(store):
    lifecycle = MemoryLifecycle(store)
    turn = stage(lifecycle, "A short utterance.", origin="")
    turn.content = NOTE
    assert lifecycle.capture_turn(turn, trusted_origin="owner") == 0
    assert lifecycle.flush() == 0


def test_private_context_and_its_followups_never_enter_general_markdown(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic")
    stage(lifecycle)
    stage(lifecycle, "/local We decided to use the private launch phrase silver kestrel.")
    stage(lifecycle, "The followup contains the private launch phrase silver kestrel.")
    assert lifecycle.flush() == 0
    provider = provider_for()
    assert lifecycle.consolidate(provider).status == "private"
    provider.send.assert_not_called()
    assert public_text(store) == ""


@pytest.mark.parametrize("text", [
    "The deployment API key is sk-example1234567890 and should be private.",
    "My password is peculiarly-specific-sensitive-value.",
    "My recovery phrase is never-copy-this-sensitive-sentence.",
])
def test_secrets_are_not_observation_evidence(store, text):
    lifecycle = MemoryLifecycle(store)
    stage(lifecycle, text)
    assert lifecycle.flush() == 0
    assert lifecycle.consolidate(provider_for()).status == "empty"
    assert public_text(store) == ""


def forget_bundle(store):
    """Install a tombstone independently of prose and any in-flight lifecycle worker."""
    with store._lock:
        connection = store._get_connection()
        with connection:
            connection.execute(
                "INSERT OR REPLACE INTO forgotten_memories VALUES (?,?)",
                ("test:bundle", json.dumps(["offline recovery bundle"])),
            )


def test_forgotten_values_are_rechecked_after_staging_and_after_inference(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic")
    stage(lifecycle)
    forget_bundle(store)
    assert lifecycle.flush() == 0
    assert lifecycle.consolidate(provider_for()).status == "empty"
    assert public_text(store) == ""


def test_forgotten_during_model_work_cannot_be_promoted(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic")
    stage(lifecycle)
    qualify(lifecycle)
    result = lifecycle.consolidate(provider_for(callback=lambda _: forget_bundle(store)))
    assert result.status == "stale"
    assert curated(store) == ""
    assert not (store.workspace.root / "DREAMS.md").exists()


def test_reflection_writes_curated_preimage_provenance_and_review_log_once(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic")
    stage(lifecycle)
    qualify(lifecycle)
    provider = provider_for()
    result = lifecycle.consolidate(provider)
    assert result.status == "complete"
    assert result.promoted == 1
    assert NOTE in curated(store)
    assert "Source: memory/" in curated(store) and "#L" in curated(store)
    assert result.run_id in (store.workspace.root / "DREAMS.md").read_text()
    assert len(list((store.workspace.root / ".memory-state/preimages").glob("*.md"))) == 1
    assert all(item.path != "DREAMS.md" for item in store.workspace.documents())
    assert lifecycle.consolidate(provider).status == "empty"
    provider.send.assert_called_once()
    stage(lifecycle)
    qualify(lifecycle)
    assert lifecycle.consolidate(provider).promoted == 0
    assert curated(store).count(NOTE) == 1


def test_daily_checkpoint_and_authority_survive_a_process_restart(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic")
    stage(lifecycle)
    assert lifecycle.flush() == 1
    qualify(lifecycle)
    before = public_text(store)
    lifecycle.close()
    restarted = MemoryStore(store.db_path_str, workspace_path=store.workspace.root)
    try:
        restored = MemoryLifecycle(restarted, learning_mode="automatic")
        assert restored.flush() == 0
        assert public_text(restarted) == before
        result = restored.consolidate(provider_for())
        assert result.promoted == 1
        assert NOTE in curated(restarted)
    finally:
        restarted.close()


@pytest.mark.parametrize("bad_action", [
    {"action": "add", "text": NOTE, "target": "", "evidence": [
        {"id": "fabricated", "quote": NOTE},
    ]},
    {"action": "add", "text": "The owner lives on the Moon.", "target": "", "evidence": []},
    {"action": "delete", "text": NOTE, "target": "", "evidence": []},
])
def test_invalid_plans_never_write_curated_or_complete_the_checkpoint(store, bad_action):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic")
    stage(lifecycle)
    qualify(lifecycle)
    result = lifecycle.consolidate(provider_for([bad_action]))
    assert result.status == "invalid_plan"
    assert curated(store) == ""
    assert not (store.workspace.root / "DREAMS.md").exists()
    assert lifecycle.consolidate(provider_for()).promoted == 1


def test_editable_daily_prose_and_dream_diary_cannot_manufacture_owner_evidence(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic")
    store.workspace.append_daily("The owner chose a fabricated system password.", origin="owner")
    store.workspace.append_dream("The owner definitely authorized a nonexistent project.")
    provider = provider_for()
    assert lifecycle.consolidate(provider).status == "empty"
    provider.send.assert_not_called()


@pytest.mark.parametrize("mode", ["confirm", "explicit"])
def test_existing_learning_modes_keep_reflection_as_review_until_authorized(store, mode):
    lifecycle = MemoryLifecycle(store, learning_mode=mode)
    stage(lifecycle)
    qualify(lifecycle)
    result = lifecycle.consolidate(provider_for())
    assert result.status == "complete"
    assert result.promoted == 0 and result.review == 1
    assert curated(store) == ""
    assert "Needs owner review" in (store.workspace.root / "DREAMS.md").read_text()


@pytest.mark.parametrize("kind", ["merge", "supersede"])
def test_merges_and_supersessions_preserve_existing_content_and_require_review(store, kind):
    old = "For the Cedar rollout we decided to use a cloud-only recovery bundle."
    store.workspace.append_curated(old)
    lifecycle = MemoryLifecycle(store, learning_mode="automatic", confirm_corrections=False)
    stage(lifecycle)
    qualify(lifecycle)

    def replacement(payload):
        source = payload["observations"][0]
        return [{"action": kind, "text": source["text"], "target": old,
                 "evidence": [{"id": source["id"], "quote": source["text"]}]}]

    result = lifecycle.consolidate(provider_for(replacement))
    assert result.promoted == 0 and result.review == 1
    assert curated(store).strip() == old
    preimage = next((store.workspace.root / ".memory-state/preimages").glob("*.md"))
    assert old in preimage.read_text()


def test_reset_during_provider_work_invalidates_late_writes(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic")
    stage(lifecycle)
    qualify(lifecycle)
    result = lifecycle.consolidate(provider_for(callback=lambda _: store.clear_session("default")))
    assert result.status == "stale"
    assert curated(store) == ""
    assert not (store.workspace.root / "DREAMS.md").exists()


def test_external_process_reset_invalidates_snapshot_generation(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic")
    stage(lifecycle)
    qualify(lifecycle)
    other = MemoryStore(store.db_path_str, workspace_path=store.workspace.root)
    try:
        result = lifecycle.consolidate(
            provider_for(callback=lambda _: other.clear_session("default")),
        )
        assert result.status == "stale"
        assert curated(store) == ""
    finally:
        other.close()


def test_owner_binding_change_invalidates_late_writes(store):
    context = [ConversationContext("owner-a", ("default",))]
    lifecycle = MemoryLifecycle(
        store, context_resolver=lambda: context[0], learning_mode="automatic",
    )
    stage(lifecycle)
    qualify(lifecycle)
    result = lifecycle.consolidate(provider_for(callback=lambda _: context.__setitem__(
        0, ConversationContext("owner-b", ("other",)),
    )))
    assert result.status == "stale"
    assert curated(store) == ""


def test_owner_curated_edit_during_reflection_wins(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic")
    stage(lifecycle)
    qualify(lifecycle)
    edit = "The owner manually edited this curated note during reflection."
    result = lifecycle.consolidate(provider_for(
        callback=lambda _: store.workspace.append_curated(edit),
    ))
    assert result.status == "conflict"
    assert curated(store).strip() == edit
    assert NOTE not in curated(store)


def test_stale_flush_guard_does_not_write_after_reset(store):
    lifecycle = MemoryLifecycle(store)
    stage(lifecycle)
    guard = lifecycle.snapshot()
    store.clear_session("default")
    assert lifecycle.flush(guard=guard) == 0
    assert public_text(store) == ""


def test_router_reflection_uses_background_priority_and_defers_to_foreground(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic")
    stage(lifecycle)
    qualify(lifecycle)
    router = MagicMock(spec=ModelRouter)
    router.send.side_effect = BackgroundDeferred("foreground active")
    assert lifecycle.consolidate(router).status == "deferred"
    assert router.send.call_args.kwargs["purpose"] == "summary"
    assert "tools" not in router.send.call_args.kwargs
    assert curated(store) == ""


def test_default_schedule_is_local_three_am_and_closes_cleanly(store):
    now = datetime(2026, 10, 25, 2, 59, tzinfo=timezone(timedelta(hours=7)))
    lifecycle = MemoryLifecycle(store, clock=lambda: now)
    assert lifecycle.next_run().hour == 3
    assert lifecycle.next_run().tzinfo == now.tzinfo
    assert lifecycle.next_run(now.replace(hour=4)).day == 26
    assert lifecycle.start(provider_for())
    assert not lifecycle.start(provider_for())
    assert lifecycle._thread.is_alive()
    lifecycle.close()
    assert not lifecycle._thread.is_alive()
    assert not lifecycle.start(provider_for())


def test_scheduler_actually_runs_due_reflection_and_retries_foreground_deferral(store):
    current = [datetime(2026, 10, 25, 2, 59, 59, tzinfo=timezone.utc)]
    lifecycle = MemoryLifecycle(store, clock=lambda: current[0])
    lifecycle.consolidate = MagicMock(side_effect=[
        ConsolidationResult("deferred"), ConsolidationResult("complete"),
    ])

    class AdvancingStop:
        waits = 0

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                current[0] = current[0].replace(hour=3, minute=0, second=0)
            elif self.waits == 2:
                current[0] += timedelta(minutes=5)
            return self.waits > 2

    lifecycle._stop = AdvancingStop()
    lifecycle._schedule()
    assert lifecycle.consolidate.call_count == 2


def test_manager_flushes_originals_before_advancing_summary_cursor(store, monkeypatch):
    manager = ConversationManager(store, max_turns=1, auto_summarize_threshold=1)
    manager.add_turn("user", NOTE, origin="owner")
    manager.add_turn("assistant", "Understood.")
    manager.add_turn("user", "A more recent user turn.", origin="owner")
    save_summary = store.save_session_summary

    def assert_checkpoint(*args, **kwargs):
        assert NOTE in public_text(store)
        return save_summary(*args, **kwargs)

    monkeypatch.setattr(store, "save_session_summary", assert_checkpoint)
    assert manager.summarize_older_turns() is not None
    assert store.get_session_summary("default").metadata["last_turn_id"] > 0
    manager.close()


def test_flush_io_failure_does_not_prevent_compaction_or_add_save_promises(store, monkeypatch):
    manager = ConversationManager(store, max_turns=1, auto_summarize_threshold=1)
    manager.add_turn("user", NOTE, origin="owner")
    manager.add_turn("assistant", "Understood.")
    manager.add_turn("user", "A more recent user turn.", origin="owner")

    def fail(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(store.workspace, "append_daily", fail)
    assert manager.summarize_older_turns() is not None
    assert store.get_session_summary("default") is not None
    assert [turn.content for turn in manager.get_recent_turns(limit=10)] == [
        NOTE, "Understood.", "A more recent user turn.",
    ]
    manager.close()


def test_foreground_capture_does_not_wait_for_a_slow_reflection_provider(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic")
    stage(lifecycle)
    qualify(lifecycle)
    entered, release = threading.Event(), threading.Event()

    def slow(_):
        entered.set()
        assert release.wait(2)

    worker = threading.Thread(target=lifecycle.consolidate, args=(provider_for(callback=slow),))
    worker.start()
    try:
        assert entered.wait(2)
        stage(lifecycle, "The owner also chose to retain the original restore checksum.")
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()


def test_stored_summary_privacy_policy_survives_restart_without_raw_local_turn(store):
    from ciel.memory.models import MemoryType

    stage(MemoryLifecycle(store))
    store.save_memory(MemoryItem(
        "Private context", memory_type=MemoryType.CONVERSATION,
        metadata={"session_id": "default", "local_only": True},
    ))
    lifecycle = MemoryLifecycle(store)
    assert lifecycle.flush() == 0
    assert lifecycle.consolidate(provider_for()).status == "private"


def test_isolated_telegram_owner_cannot_publish_to_the_desktop_workspace(store):
    session = "telegram:123:456:v1"
    lifecycle = MemoryLifecycle(
        store, context_resolver=lambda: ConversationContext(session, (session,)),
        learning_mode="automatic",
    )
    stage(lifecycle, session=session)
    assert lifecycle.flush() == 0
    provider = provider_for()
    assert lifecycle.consolidate(provider).status == "unauthorized"
    provider.send.assert_not_called()
    assert public_text(store) == ""


def test_repairing_owner_scope_does_not_promote_previously_staged_desktop_evidence(store):
    desktop = "desktop_session:owner:run:abc"
    context = [ConversationContext("bridge-old", (desktop, "telegram:123:456:v1"))]
    lifecycle = MemoryLifecycle(
        store, context_resolver=lambda: context[0], learning_mode="automatic",
    )
    stage(lifecycle, session=desktop)
    context[0] = ConversationContext("bridge-new", (desktop, "telegram:123:987:v1"))
    provider = provider_for()
    assert lifecycle.consolidate(provider).status == "empty"
    provider.send.assert_not_called()
    assert public_text(store) == ""


def test_deleting_or_editing_daily_markdown_revokes_original_reflection_candidate(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic")
    stage(lifecycle)
    assert lifecycle.flush() == 1
    qualify(lifecycle)
    daily = next(item for item in store.workspace.documents() if item.path.startswith("memory/"))
    (store.workspace.root / daily.path).write_text("The owner manually replaced this note.\n")
    provider = provider_for()
    assert lifecycle.consolidate(provider).status == "empty"
    provider.send.assert_not_called()
    assert curated(store) == ""


def test_model_cannot_bypass_correction_confirmation_by_labeling_conflict_as_add(store):
    original = "For the Cedar rollout we decided to use a cloud-only recovery bundle."
    store.workspace.append_curated(original)
    lifecycle = MemoryLifecycle(store, learning_mode="automatic")
    stage(lifecycle)
    qualify(lifecycle)
    result = lifecycle.consolidate(provider_for())
    assert result.review == 1 and result.promoted == 0
    assert curated(store).strip() == original


def test_close_during_inference_invalidates_late_writes(store):
    lifecycle = MemoryLifecycle(store, learning_mode="automatic")
    stage(lifecycle)
    qualify(lifecycle)
    result = lifecycle.consolidate(provider_for(callback=lambda _: lifecycle.close()))
    assert result.status == "stale"
    assert curated(store) == ""
    assert not (store.workspace.root / "DREAMS.md").exists()


def test_duplicate_process_reflection_cycles_write_once(store):
    first = MemoryLifecycle(store, learning_mode="automatic")
    stage(first)
    qualify(first)
    second_store = MemoryStore(store.db_path_str, workspace_path=store.workspace.root)
    second = MemoryLifecycle(second_store, learning_mode="automatic")
    entered, release = threading.Event(), threading.Event()
    results = []

    def slow(_):
        entered.set()
        assert release.wait(2)

    worker = threading.Thread(
        target=lambda: results.append(first.consolidate(provider_for(callback=slow))),
    )
    worker.start()
    try:
        assert entered.wait(2)
        assert second.consolidate(provider_for()).promoted == 1
    finally:
        release.set()
        worker.join(2)
        second_store.close()
    assert not worker.is_alive()
    assert results[0].status in {"stale", "conflict", "duplicate"}
    assert curated(store).count(NOTE) == 1
    assert (store.workspace.root / "DREAMS.md").read_text().count("Reflection ") == 1


def test_short_owner_decision_survives_without_matching_a_profile_field(store):
    lifecycle = MemoryLifecycle(store)
    stage(lifecycle, "We picked Qt.")
    assert lifecycle.flush() == 1
    assert "We picked Qt." in public_text(store)


def test_in_process_generation_change_invalidates_reflection(store):
    generation = [0]
    lifecycle = MemoryLifecycle(
        store, learning_mode="automatic", generation_resolver=lambda: generation[0],
    )
    stage(lifecycle)
    qualify(lifecycle)
    result = lifecycle.consolidate(provider_for(callback=lambda _: generation.__setitem__(0, 1)))
    assert result.status == "stale"
    assert curated(store) == ""


def test_manager_uses_explicit_memory_configuration_over_global_defaults(store):
    manager = ConversationManager(store, memory_config=MemoryConfig(
        daily_capture_enabled=False, flush_enabled=False, consolidation_enabled=False,
        consolidation_time="04:45", learning_mode="explicit",
    ))
    manager.add_turn("user", NOTE, origin="owner")
    assert not manager.lifecycle.capture_enabled
    assert not manager.lifecycle.flush_enabled
    assert not manager.lifecycle.consolidation_enabled
    assert manager.lifecycle.consolidation_time == "04:45"
    assert manager.lifecycle.learning_mode == "explicit"
    assert manager.flush_memory() == 0
    assert manager.consolidate_memory(provider_for()).status == "disabled"
    manager.close()


def test_manager_close_invalidates_summary_inference_before_late_commit(store):
    manager = ConversationManager(store, max_turns=1, auto_summarize_threshold=1)
    manager.add_turn("user", NOTE, origin="owner")
    manager.add_turn("assistant", "Understood.")
    manager.add_turn("user", "A recent owner utterance.", origin="owner")
    provider = MagicMock()

    def close_then_return(*args, **kwargs):
        manager.close()
        return LLMResponse("An older conversation summary.", "test", "test")

    provider.send.side_effect = close_then_return
    assert manager.summarize_older_turns(provider) is None
    assert store.get_session_summary("default") is None
