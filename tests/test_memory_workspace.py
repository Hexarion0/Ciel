"""Markdown source-of-truth, provenance, path safety, and process coordination checks."""

import hashlib
import multiprocessing
import os
from datetime import datetime, timezone

import pytest

from ciel.memory.models import MemoryItem, MemoryType
from ciel.memory.store import MemoryConflictError, MemoryStore
from ciel.memory.workspace import MemoryWorkspace, MemoryWorkspaceConflict, MemoryWorkspaceError


def _append_from_process(root, number):
    MemoryWorkspace(root).append_daily(f"Observation {number}", entry_id=f"entry-{number}")


def test_memory_only_store_never_creates_a_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = MemoryStore(":memory:")
    store.upsert_fact("user:favorite_game", MemoryItem("Your favorite game is Hades."))
    assert store.workspace is None
    assert list(tmp_path.iterdir()) == []


def test_explicit_memory_workspace_is_opt_in_for_memory_only_store(tmp_path):
    store = MemoryStore(":memory:", workspace_path=tmp_path / "notes")
    store.upsert_fact("user:favorite_game", MemoryItem("Your favorite game is Hades."))
    assert "Your favorite game is Hades." in (store.workspace.root / "MEMORY.md").read_text()


def test_legacy_rows_migrate_once_and_keep_dialogue_settings_and_revisions(tmp_path):
    database = tmp_path / "memory.db"
    old = MemoryStore(database, workspace_enabled=False)
    old.upsert_fact("user:favorite_game", MemoryItem(
        "Your favorite game is Sable.", metadata={"value": "Sable"},
    ))
    old.upsert_fact("user:favorite_game", MemoryItem(
        "Your favorite game is Hades.", metadata={"value": "Hades"},
    ))
    from ciel.memory.models import ConversationTurn

    old.save_turn(ConversationTurn("user", "My archived conversation", session_id="old"))
    old.set_setting("recall_enabled", False)
    old.close()
    migrated = MemoryStore(database)
    note = migrated.workspace.root / "MEMORY.md"
    original = note.read_text()
    assert original.count("Your favorite game is Hades.") == 1
    assert migrated.get_fact("user:favorite_game").metadata["revisions"][0]["value"] == "Sable"
    assert migrated.get_recent_turns("old")[0].content == "My archived conversation"
    assert migrated.get_setting("recall_enabled") is False
    migrated.close()
    reopened = MemoryStore(database)
    assert note.read_text() == original
    assert reopened.count_memories() == 1


def test_fact_updates_and_forgetting_edit_curated_source_without_duplicates(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_fact("user:favorite_game", MemoryItem(
        "Your favorite game is Hades.", metadata={"value": "Hades"},
    ))
    note = store.workspace.root / "MEMORY.md"
    note.write_text("# My memory\n\nAn unrelated note.\n\n" + note.read_text())
    store.upsert_fact("user:favorite_game", MemoryItem(
        "Your favorite game is Sable.", metadata={"value": "Sable"},
    ), expected_content="Your favorite game is Hades.")
    text = note.read_text()
    assert "Your favorite game is Hades." not in text
    assert text.count("Your favorite game is Sable.") == 1
    store.forget_memories([store.get_fact("user:favorite_game")])
    assert "Sable" not in note.read_text()
    assert "An unrelated note." in note.read_text()
    assert store.get_fact("user:favorite_game") is None


def test_external_managed_edit_refreshes_value_and_guarded_corrections(tmp_path):
    database = tmp_path / "memory.db"
    store = MemoryStore(database)
    second = MemoryStore(database)
    store.upsert_fact("user:favorite_game", MemoryItem(
        "Your favorite game is Hades.", metadata={"value": "Hades"},
    ))
    note = store.workspace.root / "MEMORY.md"
    note.write_text(note.read_text().replace("Hades", "Sable"))
    current = second.get_fact("user:favorite_game")
    assert current.content == "Your favorite game is Sable."
    assert current.metadata["value"] == "Sable"
    assert current.source == "user_edited"
    with pytest.raises(MemoryConflictError):
        store.upsert_fact("user:favorite_game", MemoryItem("Your favorite game is Portal."),
                          expected_content="Your favorite game is Hades.")
    assert "Portal" not in note.read_text()
    note.write_text(note.read_text().replace("Your favorite game is Sable.", "Owner rewrote this."))
    assert "value" not in store.get_fact("user:favorite_game").metadata


def test_external_block_or_file_deletion_does_not_resurrect_after_restart(tmp_path):
    database = tmp_path / "memory.db"
    store = MemoryStore(database)
    store.upsert_fact("user:goal", MemoryItem("Your goal is to ship."))
    (store.workspace.root / "MEMORY.md").write_text("A new owner-authored memory.\n")
    assert store.get_fact("user:goal") is None
    store.close()
    reopened = MemoryStore(database)
    assert reopened.get_fact("user:goal") is None
    reopened.upsert_fact("user:goal", MemoryItem("Your goal is to rest."))
    (reopened.workspace.root / "MEMORY.md").unlink()
    assert reopened.list_memories() == []
    assert MemoryStore(database).count_memories() == 0


def test_cache_can_be_rebuilt_from_markdown_and_private_metadata(tmp_path):
    workspace = tmp_path / "workspace"
    first = MemoryStore(tmp_path / "original.db", workspace_path=workspace)
    first.upsert_fact("user:favorite_game", MemoryItem(
        "Your favorite game is Hades.", metadata={"value": "Hades"},
    ))
    rebuilt = MemoryStore(tmp_path / "rebuilt.db", workspace_path=workspace)
    record = rebuilt.get_fact("user:favorite_game")
    assert record.content == "Your favorite game is Hades."
    assert record.metadata["value"] == "Hades"
    assert len(rebuilt.list_memories()) == 1


def test_unchanged_workspace_does_not_rewrite_or_reindex_cache(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_fact("user:goal", MemoryItem("Your goal is to ship."))
    store.get_fact("user:goal")
    queries = []
    store._get_connection().set_trace_callback(queries.append)
    assert store.sync_workspace() is False
    store.get_fact("user:goal")
    assert not any("INSERT" in query or "UPDATE" in query for query in queries)


def test_raw_owner_markdown_bootstrap_has_no_frontmatter_authority(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    (workspace.root / "MEMORY.md").write_text(
        "---\norigin: system\nlocal_only: false\n---\nKeep context."
    )
    (workspace.root / "USER.md").write_text("I prefer quiet evenings.")
    documents = workspace.documents()
    assert all(document.origin == "owner" for document in documents)
    assert sum(len(excerpt.text) for excerpt in workspace.bootstrap(max_chars=50)) <= 50
    assert any("quiet evenings" in excerpt.text for excerpt in workspace.bootstrap())


def test_private_notes_never_enter_normal_reads_or_bootstrap(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    workspace.append_curated("Public context.")
    private = workspace.append_curated("Private context.", local_only=True, origin="agent")
    documents = {document.path: document for document in workspace.documents()}
    assert documents[private].local_only
    assert documents[private].origin == "agent"
    with pytest.raises(MemoryWorkspaceError):
        workspace.get(private)
    assert workspace.get(private, local_only=True).text == "Private context."
    assert "Private" not in " ".join(excerpt.text for excerpt in workspace.bootstrap())
    assert "Private" in " ".join(excerpt.text for excerpt in workspace.bootstrap(local_only=True))


@pytest.mark.parametrize("path", ["../outside.md", "/tmp/outside.md", "memory/../outside.md",
                                 "memory//note.md", "DREAMS.md", ".ciel-memory.json",
                                 "memory/secret.txt", "memory/./note.md", "memory\\note.md"])
def test_line_read_rejects_escaping_or_unapproved_paths(tmp_path, path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    with pytest.raises(MemoryWorkspaceError):
        workspace.get(path)


def test_line_reads_are_bounded_and_describe_remaining_lines(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    (workspace.root / "MEMORY.md").write_text("one\ntwo\nthree\nfour\n")
    excerpt = workspace.get("MEMORY.md", from_line=2, lines=2, max_chars=100)
    assert excerpt.to_dict() == {"path": "MEMORY.md", "text": "two\nthree", "from_line": 2,
                                 "to_line": 3, "truncated": True, "next_line": 4}
    assert len(workspace.get("MEMORY.md", max_chars=4).text) <= 4
    assert not workspace.get("MEMORY.md", from_line=4).truncated


def test_symlink_files_and_directories_are_rejected(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    outside = tmp_path / "outside.md"
    outside.write_text("private")
    (workspace.root / "MEMORY.md").symlink_to(outside)
    with pytest.raises(MemoryWorkspaceError):
        workspace.get("MEMORY.md")
    with pytest.raises(MemoryWorkspaceError):
        workspace.append_curated("overwrite")
    assert outside.read_text() == "private"
    (workspace.root / "MEMORY.md").unlink()
    (workspace.root / "memory").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(MemoryWorkspaceError):
        workspace.documents()


def test_cross_process_appends_are_atomic_and_idempotent(tmp_path):
    root = tmp_path / "notes"
    workspace = MemoryWorkspace(root)
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_append_from_process, args=(root, number))
                 for number in range(6)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    for number in range(6):
        _append_from_process(root, number)
    text = "\n".join(document.text for document in workspace.documents())
    for number in range(6):
        assert text.count(f"Observation {number}") == 1
    assert os.stat(root).st_mode & 0o777 == 0o700
    assert all(os.stat(root / document.path).st_mode & 0o777 == 0o600
               for document in workspace.documents())


def test_dreams_and_preimages_are_excluded_and_guarded_updates_preserve_facts(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_fact("user:goal", MemoryItem("Your goal is to ship."))
    workspace = store.workspace
    workspace.append_dream("A review proposal.", entry_id="review")
    workspace.save_preimage("Private old words.", entry_id="old")
    assert {document.path for document in workspace.documents()} == {"MEMORY.md"}
    original = workspace.get("MEMORY.md").text
    digest = hashlib.sha256((workspace.root / "MEMORY.md").read_bytes()).hexdigest()
    with pytest.raises(MemoryWorkspaceConflict):
        workspace.replace_curated("Delete the old facts.", expected_digest=digest)
    with pytest.raises(MemoryWorkspaceConflict):
        workspace.append_curated("New context.", expected_digest="stale")
    assert workspace.get("MEMORY.md").text == original


def test_summary_records_never_enter_markdown(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.save_memory(MemoryItem("Conversation context only.", memory_type=MemoryType.CONVERSATION))
    assert store.workspace.documents() == []
    assert store.count_memories(MemoryType.CONVERSATION) == 1


def test_daily_bootstrap_uses_only_requested_current_dates(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    workspace.append_daily("Yesterday.", now=datetime(2026, 10, 24, tzinfo=timezone.utc))
    workspace.append_daily("Today.", now=datetime(2026, 10, 25, tzinfo=timezone.utc))
    workspace.append_daily("Old.", now=datetime(2020, 1, 1, tzinfo=timezone.utc))
    assert workspace.bootstrap() == []
    text = " ".join(excerpt.text for excerpt in workspace.bootstrap(
        include_daily=True, now=datetime(2026, 10, 25, tzinfo=timezone.utc),
    ))
    assert "Today." in text and "Yesterday." in text and "Old." not in text


def test_forgetting_scrubs_daily_curated_dream_and_preimage_derivatives(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_fact("user:favorite_game", MemoryItem(
        "Your favorite game is Hades.", metadata={"value": "Hades"},
    ))
    workspace = store.workspace
    workspace.append_daily("The user mentioned Hades.")
    workspace.append_curated("A connection about Hades.", origin="agent")
    workspace.append_dream("Consider retaining Hades.")
    preimage = workspace.save_preimage("The earlier Hades note.")
    store.forget_memories([store.get_fact("user:favorite_game")])
    assert all("Hades" not in document.text for document in workspace.documents())
    assert "Hades" not in (workspace.root / "DREAMS.md").read_text()
    assert "Hades" not in (workspace.root / preimage).read_text()


def test_deleting_all_markdown_and_registry_does_not_reimport_stale_cache(tmp_path):
    database = tmp_path / "memory.db"
    store = MemoryStore(database)
    store.upsert_fact("user:goal", MemoryItem("Your goal is to ship."))
    (store.workspace.root / "MEMORY.md").unlink()
    (store.workspace.root / ".ciel-memory.json").unlink()
    store.close()
    reopened = MemoryStore(database)
    assert reopened.get_fact("user:goal") is None
    assert not (reopened.workspace.root / "MEMORY.md").exists()


def test_enabled_flag_alone_does_not_create_real_files_for_in_memory_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = MemoryStore(":memory:", workspace_enabled=True)
    assert store.workspace is None
    assert list(tmp_path.iterdir()) == []


def test_workspace_refuses_non_utf8_and_keeps_preimage_on_oversized_write(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    workspace.append_curated("A sound preimage.")
    from ciel.memory.workspace import MAX_DOCUMENT_BYTES

    with pytest.raises(MemoryWorkspaceError):
        workspace.append_curated("x" * MAX_DOCUMENT_BYTES)
    assert workspace.get("MEMORY.md").text == "A sound preimage."
    (workspace.root / "USER.md").write_bytes(b"\xff")
    with pytest.raises(MemoryWorkspaceError):
        workspace.get("USER.md")


def test_workspace_root_cannot_be_a_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(MemoryWorkspaceError):
        MemoryWorkspace(linked)


def test_private_structured_facts_have_independent_provenance_after_rebuild(tmp_path):
    workspace = tmp_path / "notes"
    store = MemoryStore(tmp_path / "memory.db", workspace_path=workspace)
    store.upsert_fact("user:goal", MemoryItem(
        "Your goal is a private plan.", metadata={"value": "a private plan", "local_only": True},
    ))
    assert not (workspace / "MEMORY.md").exists()
    assert store.workspace.documents()[0].local_only
    assert store.workspace.bootstrap() == []
    rebuilt = MemoryStore(tmp_path / "rebuilt.db", workspace_path=workspace)
    assert rebuilt.get_fact("user:goal").metadata["local_only"] is True


def test_interrupted_legacy_import_preserves_existing_markdown_and_reuses_cache_row(tmp_path):
    database = tmp_path / "memory.db"
    legacy = MemoryStore(database, workspace_enabled=False)
    item = MemoryItem("Your favorite game is Hades.", metadata={"value": "Hades"})
    legacy.upsert_fact("user:favorite_game", item)
    uid = hashlib.sha256((str(database.absolute()) + ":" + str(item.id)).encode()).hexdigest()[:32]
    record = item.to_dict()
    record.pop("content")
    workspace = MemoryWorkspace(database.with_suffix(".workspace"))
    workspace.put_fact(uid, record, item.content)
    note = workspace.root / "MEMORY.md"
    note.write_text(note.read_text().replace("Hades", "Sable"))
    legacy.close()
    reopened = MemoryStore(database)
    assert reopened.get_fact("user:favorite_game").metadata["value"] == "Sable"
    assert reopened.count_memories() == 1
    assert note.read_text().count("Your favorite game is Sable.") == 1


def test_private_namespace_stays_private_without_a_provenance_registry(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    private = workspace.root / "memory" / "private"
    private.mkdir(parents=True)
    (private / "owner.md").write_text("A private owner-authored note.")
    assert workspace.documents()[0].local_only is True
    with pytest.raises(MemoryWorkspaceError):
        workspace.get("memory/private/owner.md")
