"""Owner CLI reads real notes and backs up every authoritative memory layer."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ciel.config import Settings
from ciel.memory import cli
from ciel.memory.bridge import (
    _ACTIVE_DESKTOP_KEY,
    conversation_context,
    start_desktop_conversation,
)
from ciel.memory.manager import ConversationManager
from ciel.memory.models import MemoryItem
from ciel.memory.store import MemoryStore
from ciel.memory.tools import MemoryTools
from ciel.memory.workspace_search import WorkspaceMemorySearch
from ciel.providers.base import LLMResponse

NOTE = "For Cedar rollout we chose to retain an offline recovery bundle."


@pytest.fixture
def owner(tmp_path, monkeypatch):
    """Isolate the configured owner root and prevent ambient account settings."""
    settings = Settings(
        _env_file=None, memory_db_path=str(tmp_path / "owner.db"),
        memory_workspace_path=str(tmp_path / "owner-notes"), chat_services="",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    store = MemoryStore(settings.memory.db_path, workspace_path=settings.memory.workspace_path)
    yield settings, store
    store.close()


def run_json(capsys, *args):
    """Read one complete structured CLI result."""
    result = cli.main(["--json", *map(str, args)])
    return result, json.loads(capsys.readouterr().out)


def seed_owner(settings, store, text=NOTE, *, eligible=False):
    """Stage evidence under the active persisted desktop owner projection."""
    session = start_desktop_conversation(store)
    manager = ConversationManager(
        store, session_id=session, memory_config=settings.memory,
        context_resolver=lambda: conversation_context(settings, session, store=store),
    )
    manager.add_turn("user", text, origin="owner")
    if eligible:
        manager.flush_memory()
        searcher = WorkspaceMemorySearch(store.workspace)
        try:
            for query in ["Cedar rollout", "offline recovery", "recovery bundle"]:
                hits = searcher.search(query)
                assert manager.lifecycle.record_recall(hits, query) == 1
        finally:
            searcher.close()
    # Leave staged evidence for the CLI checkpoint, as after an interrupted process.
    manager.lifecycle.close()
    return session


def test_configured_workspace_and_db_override_remain_separate(owner, tmp_path, capsys):
    settings, store = owner
    store.workspace.append_daily("Owner notes belong only in the configured workspace.")
    result, status = run_json(capsys, "status")
    assert result == 0
    assert status["workspace"] == str(store.workspace.root)
    alternate = tmp_path / "isolated.db"
    result, status = run_json(capsys, "--db", alternate, "status")
    assert result == 0
    assert status["workspace"] == str(alternate.with_suffix(".workspace"))
    assert not (alternate.with_suffix(".workspace") / "memory").exists()
    assert settings.memory.workspace_path == str(store.workspace.root)
    explicit = tmp_path / "explicit-workspace"
    result, status = run_json(capsys, "--db", alternate, "--workspace", explicit, "status")
    assert result == 0
    assert status["workspace"] == str(explicit)


def test_search_keeps_legacy_fact_shape_and_includes_actual_markdown_source(owner, capsys):
    _, store = owner
    store.upsert_fact("user:current_project", MemoryItem("Cedar"))
    store.workspace.append_daily("Cedar rollout needs an offline recovery bundle.")
    result, matches = run_json(capsys, "search", "Cedar")
    assert result == 0
    assert matches[0]["content"] == "Cedar"
    assert isinstance(matches[0]["id"], int)
    notes = [item for item in matches if item.get("path", "").startswith("memory/")]
    assert notes
    assert "offline recovery bundle" in notes[0]["content"]
    assert notes[0]["start_line"] >= 1
    assert notes[0]["end_line"] >= notes[0]["start_line"]
    assert len(notes[0]["snippet"]) <= 700


def test_index_force_reconciles_manual_edits_and_deletions(owner, capsys):
    _, store = owner
    path = store.workspace.root / "USER.md"
    path.write_text("Owner likes the spring launch window.\n")
    result, report = run_json(capsys, "index", "--force")
    assert result == 0 and report["indexed"] == 1
    result, status = run_json(capsys, "status")
    assert result == 0 and status["index"]["documents"] == 1
    path.write_text("Owner likes the autumn launch window.\n")
    result, matches = run_json(capsys, "search", "autumn")
    assert result == 0 and any(item.get("path") == "USER.md" for item in matches)
    path.unlink()
    result, report = run_json(capsys, "index")
    assert result == 0 and report["removed"] == 1


def test_owner_get_is_line_bounded_and_can_explicitly_read_private_notes(owner, capsys):
    settings, store = owner
    path = store.workspace.append_daily(
        "private first\nprivate second\nprivate third", local_only=True, origin="owner",
    )
    result, excerpt = run_json(capsys, "get", path, "--from", 2, "--lines", 1)
    assert result == 0
    assert excerpt["text"] == "private second"
    assert excerpt["from_line"] == excerpt["to_line"] == 2
    assert excerpt["truncated"]
    # The ordinary tool keeps its default local-only filtering.
    tools = MemoryTools(store, settings.memory)
    assert not tools.get(path).get("text")
    assert tools.search("private")["results"] == []
    result, matches = run_json(capsys, "search", "private second")
    assert result == 0 and any(item.get("local_only") for item in matches)


def test_paused_automatic_recall_does_not_block_explicit_owner_inspection(owner, capsys):
    _, store = owner
    store.workspace.append_curated("Cedar private owner decision context.", local_only=True)
    store.set_setting("recall_enabled", False)
    result, excerpt = run_json(capsys, "get", "memory/private/MEMORY.md")
    assert result == 0 and "Cedar" in excerpt["text"]
    result, matches = run_json(capsys, "search", "Cedar")
    assert result == 0 and matches
    assert store.get_setting("recall_enabled") is False


def test_explicit_cli_search_reinforces_only_real_owner_notes_while_recall_is_paused(
    owner, monkeypatch, capsys,
):
    settings, store = owner
    settings.memory_learning_mode = "automatic"
    seed_owner(settings, store)
    result, report = run_json(capsys, "flush")
    assert result == 0 and report["flushed"] == 1
    store.workspace.append_daily("Cedar offline recovery bundle from an unmanaged note.")
    store.workspace.append_daily("Private Cedar offline recovery bundle.", local_only=True)
    store.set_setting("recall_enabled", False)
    router = MagicMock()

    def plan(messages, **kwargs):
        observations = json.loads(messages[-1].content)["observations"]
        assert len(observations) == 1 and observations[0]["text"] == NOTE
        source = observations[0]
        proposal = {"action": "add", "text": source["text"], "target": "", "evidence": [
            {"id": source["id"], "quote": source["text"]},
        ]}
        return LLMResponse(json.dumps({"actions": [proposal]}), "test", "test")

    router.send.side_effect = plan
    monkeypatch.setattr("ciel.providers.get_model_router", lambda: router)
    result, report = run_json(capsys, "consolidate")
    assert result == 0 and report["status"] == "ineligible"
    router.send.assert_not_called()
    for query in ["Cedar rollout", "retain offline", "chose recovery"]:
        result, matches = run_json(capsys, "search", query)
        assert result == 0 and any(item.get("path", "").startswith("memory/") for item in matches)
    result, report = run_json(capsys, "consolidate")
    assert result == 0 and report["promoted"] == 1
    assert store.get_setting("recall_enabled") is False
    router.send.assert_called_once()


def test_owner_search_remains_available_without_a_valid_active_owner_binding(owner, capsys):
    _, store = owner
    store.workspace.append_daily("Cedar source inspection stays available.")
    store.set_setting(_ACTIVE_DESKTOP_KEY, "an invalid owner binding")
    result, matches = run_json(capsys, "search", "Cedar")
    assert result == 0 and matches


@pytest.mark.parametrize("path", ["../outside.md", "/etc/passwd", ".ciel-memory.json"])
def test_get_rejects_escape_and_provenance_paths(owner, capsys, path):
    assert cli.main(["get", path]) == 1
    assert "failed" in capsys.readouterr().out


@pytest.mark.parametrize("flag,value", [("--from", "0"), ("--lines", "501")])
def test_get_refuses_unbounded_or_invalid_line_requests(owner, flag, value):
    with pytest.raises(SystemExit):
        cli.main(["get", "MEMORY.md", flag, value])


def test_reset_index_preserves_notes_registry_and_private_preimages(owner, capsys):
    _, store = owner
    store.workspace.append_curated("Owner keeps the Cedar recovery bundle.")
    preimage = store.workspace.save_preimage("Prior curated content", entry_id="test")
    run_json(capsys, "index")
    originals = {relative: (store.workspace.root / relative).read_bytes()
                 for relative in ["MEMORY.md", ".ciel-memory.json", preimage]}
    result, status = run_json(capsys, "reset-index")
    assert result == 0 and status["documents"] == 0 and status["chunks"] == 0
    for relative, content in originals.items():
        assert (store.workspace.root / relative).read_bytes() == content


def test_local_maintenance_never_initializes_audio_or_a_completion_provider(
    owner, monkeypatch, capsys,
):
    settings, store = owner
    seed_owner(settings, store)
    store.workspace.append_curated("Cedar owner preferences.")
    settings.memory_semantic_enabled = True
    settings.memory_semantic_model = "unused-local-model"
    router = MagicMock(side_effect=AssertionError("Unexpected completion provider"))
    embeddings = MagicMock(side_effect=AssertionError("Unexpected embedding provider"))
    audio = MagicMock(side_effect=AssertionError("Unexpected audio initialization"))
    monkeypatch.setattr("ciel.providers.get_model_router", router)
    monkeypatch.setattr("ciel.memory.tools._LocalEmbeddings", embeddings)
    monkeypatch.setattr("ciel.audio.TextToSpeech", audio)
    for args in [
        ["status"], ["index", "--force"], ["get", "MEMORY.md"],
        ["search", "Cedar"], ["flush"], ["reset-index"], ["review"],
    ]:
        assert cli.main(args) == 0
        capsys.readouterr()
    router.assert_not_called()
    embeddings.assert_not_called()
    audio.assert_not_called()


def test_manual_flush_uses_current_exact_owner_and_is_idempotent(owner, capsys):
    settings, store = owner
    seed_owner(settings, store)
    result, report = run_json(capsys, "flush")
    assert result == 0 and report["flushed"] == 1
    assert any(NOTE in item.text for item in store.workspace.documents())
    result, report = run_json(capsys, "flush")
    assert result == 0 and report["flushed"] == 0


def test_flush_does_not_sweep_archives_after_active_owner_projection_changes(owner, capsys):
    settings, store = owner
    seed_owner(settings, store)
    start_desktop_conversation(store)
    result, report = run_json(capsys, "flush")
    assert result == 0 and report["flushed"] == 0
    assert not any(NOTE in item.text for item in store.workspace.documents())


def test_explicit_database_maintenance_does_not_read_configured_owner_pairing_state(
    owner, tmp_path, monkeypatch, capsys,
):
    settings, _ = owner
    alternate = tmp_path / "isolated.db"
    with_store = MemoryStore(alternate)
    try:
        seed_owner(settings, with_store)
    finally:
        with_store.close()
    pairing = MagicMock(side_effect=AssertionError("An isolated database has no pairing"))
    monkeypatch.setattr("ciel.memory.bridge._paired_telegram_id", pairing)
    result, report = run_json(capsys, "--db", alternate, "flush")
    assert result == 0 and report["flushed"] == 1
    pairing.assert_not_called()


def test_maintenance_refuses_an_invalid_active_owner_binding(owner, capsys):
    settings, store = owner
    seed_owner(settings, store)
    store.set_setting(_ACTIVE_DESKTOP_KEY, "an invalid owner binding")
    assert cli.main(["flush"]) == 1
    assert "failed" in capsys.readouterr().out
    assert not any(NOTE in item.text for item in store.workspace.documents())


def test_manual_consolidation_alone_uses_the_configured_router(owner, monkeypatch, capsys):
    settings, store = owner
    settings.memory_learning_mode = "automatic"
    seed_owner(settings, store, eligible=True)
    router = MagicMock()

    def plan(messages, **kwargs):
        source = json.loads(messages[-1].content)["observations"][0]
        action = {"action": "add", "text": source["text"], "target": "", "evidence": [
            {"id": source["id"], "quote": source["text"]},
        ]}
        return LLMResponse(json.dumps({"actions": [action]}), "test", "test")

    router.send.side_effect = plan
    get_router = MagicMock(return_value=router)
    monkeypatch.setattr("ciel.providers.get_model_router", get_router)
    result, report = run_json(capsys, "consolidate")
    assert result == 0 and report["status"] == "complete" and report["promoted"] == 1
    get_router.assert_called_once()
    assert NOTE in (store.workspace.root / "MEMORY.md").read_text()


def test_disabled_workspace_and_flush_configuration_are_respected(owner, capsys):
    settings, _ = owner
    settings.memory_workspace_enabled = False
    result, status = run_json(capsys, "status")
    assert result == 0 and status["workspace_enabled"] is False
    assert cli.main(["index"]) == 1
    capsys.readouterr()
    settings.memory_workspace_enabled = True
    settings.memory_flush_enabled = False
    result, report = run_json(capsys, "flush")
    assert result == 0 and report["flushed"] == 0


def test_backup_restores_raw_notes_private_provenance_and_preimages_without_derived_cache(
    owner, tmp_path, capsys,
):
    settings, store = owner
    store.upsert_fact("user:current_project", MemoryItem("Cedar"))
    store.workspace.append_curated("Private owner preference", local_only=True, origin="owner")
    private = store.workspace.append_daily(
        "Private recovery wording", local_only=True, origin="owner",
    )
    raw = store.workspace.root / "memory" / "2026-10-25.md"
    raw.write_text("Unmanaged Cedar rollout observation.\n")
    (store.workspace.root / "USER.md").write_text("Owner profile written directly.\n")
    store.workspace.append_dream("A reviewable reflection with original evidence.")
    preimage = store.workspace.save_preimage("Original curated preimage", entry_id="backup-test")
    searcher = WorkspaceMemorySearch(store.workspace)
    searcher.sync()
    searcher.close()
    destination = tmp_path / "restored.db"
    assert cli.main(["backup", str(destination)]) == 0
    capsys.readouterr()
    backup_root = destination.with_suffix(".workspace")
    assert not (backup_root / ".ciel").exists()
    assert not (backup_root / ".ciel-memory.lock").exists()
    for relative in ["MEMORY.md", "USER.md", "DREAMS.md", ".ciel-memory.json", preimage,
                     private, "memory/private/MEMORY.md", "memory/2026-10-25.md"]:
        original = store.workspace.root / relative
        assert (backup_root / relative).read_bytes() == original.read_bytes()
    assert destination.stat().st_mode & 0o777 == 0o600
    for directory, names, files in os.walk(backup_root):
        assert Path(directory).stat().st_mode & 0o777 == 0o700
        for name in names:
            assert (Path(directory) / name).stat().st_mode & 0o777 == 0o700
        for name in files:
            assert (Path(directory) / name).stat().st_mode & 0o777 == 0o600
    restored = MemoryStore(destination)
    try:
        assert restored.get_fact("user:current_project").content == "Cedar"
        documents = {item.path: item for item in restored.workspace.documents()}
        assert "Unmanaged Cedar" in documents["memory/2026-10-25.md"].text
        assert documents[private].local_only
        assert documents[private].origin == "owner"
        tools = MemoryTools(restored, settings.memory)
        assert tools.search("Private recovery")["results"] == []
    finally:
        restored.close()


@pytest.mark.parametrize("collision", [
    "database", "workspace", "workspace-file", "journal", "wal", "shm",
])
def test_backup_refuses_any_existing_destination_without_removing_it(
    owner, tmp_path, capsys, collision,
):
    _, _store = owner
    destination = tmp_path / "reserved.db"
    sibling = destination.with_suffix(".workspace")
    existing = destination if collision == "database" else (
        Path(str(destination) + f"-{collision}")
        if collision in {"journal", "wal", "shm"} else sibling
    )
    if collision == "workspace":
        existing.mkdir()
        (existing / "sentinel").write_text("Keep this directory.")
    else:
        existing.write_text("Keep this file.")
    assert cli.main(["backup", str(destination)]) == 1
    assert "failed" in capsys.readouterr().out
    assert existing.exists()
    if collision != "database":
        assert not destination.exists()


def test_backup_refuses_destination_inside_source_and_destination_symlink_ancestors(
    owner, tmp_path, capsys,
):
    _, store = owner
    inside = store.workspace.root / "nested.db"
    assert cli.main(["backup", str(inside)]) == 1
    assert not inside.exists()
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    assert cli.main(["backup", str(alias / "escape.db")]) == 1
    assert not (outside / "escape.db").exists()
    assert "failed" in capsys.readouterr().out


def test_backup_refuses_source_symlinks_even_in_excluded_cache_locations(owner, tmp_path, capsys):
    _, store = owner
    outside = tmp_path / "outside.txt"
    outside.write_text("Do not read through this link.")
    (store.workspace.root / "unexpected-cache-link").symlink_to(outside)
    destination = tmp_path / "backup.db"
    assert cli.main(["backup", str(destination)]) == 1
    assert not destination.exists()
    assert not destination.with_suffix(".workspace").exists()
    assert outside.read_text() == "Do not read through this link."
    assert "symlink" in capsys.readouterr().out


def test_backup_cleans_database_and_partial_workspace_after_copy_failure(
    owner, tmp_path, monkeypatch,
):
    _, store = owner
    store.workspace.append_curated("Owner memory that must be copied.")
    original = cli._copy_backup_file
    copies = []

    def fail_after_one(source, destination):
        copies.append(source)
        if len(copies) > 1:
            raise OSError("Synthetic copy failure")
        original(source, destination)

    monkeypatch.setattr(cli, "_copy_backup_file", fail_after_one)
    destination = tmp_path / "failed.db"
    assert cli.main(["backup", str(destination)]) == 1
    assert copies
    assert not destination.exists()
    assert not destination.with_suffix(".workspace").exists()
    assert not list(tmp_path.glob("failed.db-*"))
    assert "Owner memory" in (store.workspace.root / "MEMORY.md").read_text()


def test_backup_refuses_manual_source_edits_after_the_database_snapshot(
    owner, tmp_path, monkeypatch, capsys,
):
    _, store = owner
    store.workspace.append_curated("Owner memory before the backup snapshot.")
    original = cli._copy_backup_file
    edited = False

    def edit_before_copy(source, destination):
        nonlocal edited
        if not edited:
            edited = True
            (store.workspace.root / "MEMORY.md").write_text("Owner edited during backup.\n")
        original(source, destination)

    monkeypatch.setattr(cli, "_copy_backup_file", edit_before_copy)
    destination = tmp_path / "changed.db"
    assert cli.main(["backup", str(destination)]) == 1
    assert "changed" in capsys.readouterr().out
    assert not destination.exists()
    assert not destination.with_suffix(".workspace").exists()


def test_backup_holds_workspace_then_store_lock_through_file_and_database_snapshot(
    owner, tmp_path, monkeypatch,
):
    _, store = owner
    store.workspace.append_curated("Owner lock-order snapshot.")
    original = cli._copy_backup_file
    observed = []

    def inspect_locks(source, destination):
        assert store.workspace._depth > 0
        assert store._lock.locked()
        observed.append(source)
        original(source, destination)

    monkeypatch.setattr(cli, "_copy_backup_file", inspect_locks)
    cli._backup(store, tmp_path / "locked.db")
    assert observed


def test_backup_refuses_a_symlinked_source_database(owner, tmp_path, capsys):
    settings, store = owner
    alias = tmp_path / "source-link.db"
    alias.symlink_to(Path(settings.memory.db_path))
    destination = tmp_path / "backup.db"
    assert cli.main(["--db", str(alias), "backup", str(destination)]) == 1
    assert "symlink" in capsys.readouterr().out
    assert store.get_recent_turns() == []
    assert not destination.exists()
    assert not destination.with_suffix(".workspace").exists()


def test_backup_checks_normalized_source_workspace_boundaries(owner, tmp_path, capsys):
    settings, store = owner
    intermediate = tmp_path / "intermediate"
    intermediate.mkdir()
    settings.memory_workspace_path = str(intermediate / ".." / store.workspace.root.name)
    destination = store.workspace.root / "inside.db"
    assert cli.main(["backup", str(destination)]) == 1
    assert "inside" in capsys.readouterr().out
    assert not destination.exists()
    assert not destination.with_suffix(".workspace").exists()


@pytest.mark.parametrize("action", ["merge", "supersede"])
def test_owner_can_review_and_apply_pending_corrections_without_another_provider_request(
    owner, monkeypatch, capsys, action,
):
    settings, store = owner
    settings.memory_learning_mode = "automatic"
    old = "For Cedar rollout we chose a hosted recovery service."
    store.workspace.append_curated(old)
    seed_owner(settings, store, eligible=True)

    def plan(messages, **kwargs):
        source = json.loads(messages[-1].content)["observations"][0]
        proposal = {"action": action, "text": source["text"], "target": old, "evidence": [
            {"id": source["id"], "quote": source["text"]},
        ]}
        return LLMResponse(json.dumps({"actions": [proposal]}), "test", "test")

    router = MagicMock()
    router.send.side_effect = plan
    factory = MagicMock(return_value=router)
    monkeypatch.setattr("ciel.providers.get_model_router", factory)
    result, report = run_json(capsys, "consolidate")
    assert result == 0 and report["review"] == 1 and report["promoted"] == 0
    factory.assert_called_once()
    factory.side_effect = AssertionError("Review and apply must use no provider")
    result, reviews = run_json(capsys, "review", report["run_id"])
    assert result == 0 and len(reviews) == 1
    proposal = reviews[0]["actions"][0]
    assert proposal["index"] == 0 and proposal["target"] == old
    assert proposal["evidence"][0]["quote"] == NOTE
    assert cli.main(["review", report["run_id"]]) == 0
    readable = capsys.readouterr().out
    assert NOTE in readable and "Source: memory/" in readable and "#L" in readable
    result, applied = run_json(capsys, "apply", report["run_id"], "--action", "0")
    assert result == 0 and applied["promoted"] == 1 and applied["review"] == 0
    curated = (store.workspace.root / "MEMORY.md").read_text()
    assert NOTE in curated
    assert (old in curated) is (action == "merge")
    result, reviews = run_json(capsys, "review")
    assert result == 0 and reviews == []
    factory.assert_called_once()


def test_owner_can_apply_a_subset_then_the_remaining_review_actions(owner, monkeypatch, capsys):
    settings, store = owner
    statements = [NOTE, "My preferred interface uses large text and muted green accents."]
    seed_owner(settings, store, "\n\n".join(statements), eligible=True)

    def plan(messages, **kwargs):
        source = json.loads(messages[-1].content)["observations"][0]
        proposals = [{"action": "add", "text": text, "target": "", "evidence": [
            {"id": source["id"], "quote": source["text"]},
        ]} for text in statements]
        return LLMResponse(json.dumps({"actions": proposals}), "test", "test")

    router = MagicMock()
    router.send.side_effect = plan
    monkeypatch.setattr("ciel.providers.get_model_router", lambda: router)
    result, report = run_json(capsys, "consolidate")
    assert result == 0 and report["review"] == 2
    result, applied = run_json(capsys, "apply", report["run_id"], "--action", "0")
    assert result == 0 and applied["promoted"] == 1 and applied["review"] == 1
    result, reviews = run_json(capsys, "review", report["run_id"])
    assert result == 0 and [item["index"] for item in reviews[0]["actions"]] == [1]
    result, applied = run_json(capsys, "apply", report["run_id"])
    assert result == 0 and applied["promoted"] == 1 and applied["review"] == 0
    curated = (store.workspace.root / "MEMORY.md").read_text()
    assert all(text in curated for text in statements)


@pytest.mark.parametrize("index", ["-1", "16"])
def test_apply_refuses_invalid_action_indexes(owner, index):
    with pytest.raises(SystemExit):
        cli.main(["apply", "example-run", "--action", index])
