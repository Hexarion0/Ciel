"""Exact-path recall stays independent of unrelated archives without stale results."""

import json
import os
import sqlite3
from unittest.mock import Mock

import pytest

from ciel.memory.workspace import MemoryWorkspace, MemoryWorkspaceError
from ciel.memory.workspace_search import WorkspaceMemorySearch


def write_note(workspace, path, text):
    target = workspace.root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def forbid_archive_scan(monkeypatch, workspace):
    monkeypatch.setattr(workspace, "_document_paths", Mock(
        side_effect=AssertionError("Scoped recall traversed unrelated archives"),
    ))


def indexed_rows(search, table):
    with sqlite3.connect(search.db_path) as connection:
        return connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()


def test_scoped_search_does_not_traverse_archive_and_full_search_still_discovers_it(
    tmp_path, monkeypatch,
):
    workspace = MemoryWorkspace(tmp_path / "notes")
    write_note(workspace, "MEMORY.md", "Cobalt is the preferred color.")
    write_note(workspace, "memory/archive.md", "Violet is the archived color.")
    search = WorkspaceMemorySearch(workspace)
    try:
        with monkeypatch.context() as patch:
            forbid_archive_scan(patch, workspace)
            assert [hit.path for hit in search.search(
                "cobalt", allowed_paths=("MEMORY.md", "USER.md"),
            )] == ["MEMORY.md"]
            assert search.search("violet", allowed_paths=("MEMORY.md",)) == []
            assert search.status()["documents"] == 1
        assert [hit.path for hit in search.search("violet")] == ["memory/archive.md"]
        assert search.status()["documents"] == 2
    finally:
        search.close()


def test_scoped_edits_and_deletes_refresh_without_removing_unselected_index_rows(
    tmp_path, monkeypatch,
):
    workspace = MemoryWorkspace(tmp_path / "notes")
    target = write_note(workspace, "MEMORY.md", "Cobalt chosen.")
    write_note(workspace, "memory/archive.md", "Archived violet remains.")
    search = WorkspaceMemorySearch(workspace)
    try:
        search.sync()
        archive_chunks = [row for row in indexed_rows(search, "workspace_search_chunks")
                          if row[1] == "memory/archive.md"]
        forbid_archive_scan(monkeypatch, workspace)
        status = target.stat()
        target.write_text("Copper chosen.")
        os.utime(target, ns=(status.st_atime_ns, status.st_mtime_ns))
        assert search.search("cobalt", allowed_paths=("MEMORY.md",)) == []
        assert search.search("copper", allowed_paths=("MEMORY.md",))[0].snippet == "Copper chosen."
        target.unlink()
        assert search.search("copper", allowed_paths=("MEMORY.md",)) == []
        assert search.status()["documents"] == 1
        assert indexed_rows(search, "workspace_search_chunks") == archive_chunks
    finally:
        search.close()


def test_scoped_search_rechecks_provenance_and_private_access(tmp_path, monkeypatch):
    workspace = MemoryWorkspace(tmp_path / "notes")
    workspace.append_curated("Cobalt chosen.")
    search = WorkspaceMemorySearch(workspace)
    try:
        forbid_archive_scan(monkeypatch, workspace)
        assert search.search("cobalt", allowed_paths=("MEMORY.md",), allowed_origins=("owner",))
        registry = workspace.root / ".ciel-memory.json"
        state = json.loads(registry.read_text())
        state["files"]["MEMORY.md"].update(origin="imported", local_only=True)
        registry.write_text(json.dumps(state))
        assert search.search("cobalt", allowed_paths=("MEMORY.md",)) == []
        assert search.search("cobalt", local_only=True, allowed_paths=("MEMORY.md",),
                             allowed_origins=("owner",)) == []
        hits = search.search("cobalt", local_only=True, allowed_paths=("MEMORY.md",),
                             allowed_origins=("imported",))
        assert len(hits) == 1 and hits[0].local_only and hits[0].origin == "imported"
    finally:
        search.close()


def test_scoped_rebuild_does_not_certify_unselected_chunk_format(tmp_path, monkeypatch):
    workspace = MemoryWorkspace(tmp_path / "notes")
    write_note(workspace, "MEMORY.md", "Cobalt chosen.")
    write_note(workspace, "memory/archive.md", "Archived violet remains. " * 15)
    first = WorkspaceMemorySearch(workspace, chunk_tokens=8, chunk_overlap=0)
    first.sync()
    old_archive = [row for row in indexed_rows(first, "workspace_search_chunks")
                   if row[1] == "memory/archive.md"]
    first.close()
    second = WorkspaceMemorySearch(workspace, chunk_tokens=16, chunk_overlap=0)
    try:
        with monkeypatch.context() as patch:
            forbid_archive_scan(patch, workspace)
            assert second.search("cobalt", allowed_paths=("MEMORY.md",))
            with sqlite3.connect(second.db_path) as connection:
                assert dict(connection.execute(
                    "SELECT key,value FROM workspace_search_meta",
                ))["chunking"] == "8:0"
            assert [row for row in indexed_rows(second, "workspace_search_chunks")
                    if row[1] == "memory/archive.md"] == old_archive
        second.sync()
        fresh_archive = [row for row in indexed_rows(second, "workspace_search_chunks")
                         if row[1] == "memory/archive.md"]
        assert len(fresh_archive) < len(old_archive)
        assert second.search("violet")[0].path == "memory/archive.md"
    finally:
        second.close()


class LocalEmbeddings:
    provider_id = "synthetic-scope-test"
    local_only = True

    def __init__(self):
        self.calls = []

    def embed(self, texts, *, model):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


def test_scoped_semantic_search_avoids_archive_and_preserves_unselected_vectors(
    tmp_path, monkeypatch,
):
    workspace = MemoryWorkspace(tmp_path / "notes")
    write_note(workspace, "MEMORY.md", "Preferred seaside lodging.")
    archive_text = "Archived violet remains."
    write_note(workspace, "memory/archive.md", archive_text)
    client = LocalEmbeddings()
    search = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="test")
    try:
        search.sync()
        vectors = indexed_rows(search, "workspace_search_embeddings")
        client.calls.clear()
        forbid_archive_scan(monkeypatch, workspace)
        assert [hit.path for hit in search.search(
            "beach hotel", allowed_paths=("MEMORY.md",),
        )] == ["MEMORY.md"]
        assert client.calls == [["beach hotel"]]
        assert indexed_rows(search, "workspace_search_embeddings") == vectors
        write_note(workspace, "MEMORY.md", "Changed seaside lodging.")
        assert search.search("beach hotel", allowed_paths=("MEMORY.md",))
        assert all(archive_text not in texts for texts in client.calls)
        assert all(vector in indexed_rows(search, "workspace_search_embeddings")
                   for vector in vectors)
    finally:
        search.close()


@pytest.mark.parametrize("path", ["../outside.md", "memory/link.md"])
def test_scoped_reader_preserves_containment_and_symlink_checks(tmp_path, path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    outside = tmp_path / "outside.md"
    outside.write_text("Outside secret.")
    (workspace.root / "memory").mkdir()
    (workspace.root / "memory/link.md").symlink_to(outside)
    with pytest.raises(MemoryWorkspaceError):
        workspace.documents(paths=(path,))


def test_empty_scope_does_not_touch_documents_or_remove_index(tmp_path, monkeypatch):
    workspace = MemoryWorkspace(tmp_path / "notes")
    write_note(workspace, "MEMORY.md", "Cobalt chosen.")
    search = WorkspaceMemorySearch(workspace)
    try:
        search.sync()
        before = indexed_rows(search, "workspace_search_chunks")
        monkeypatch.setattr(workspace, "documents", Mock(side_effect=AssertionError("No scope")))
        assert search.search("cobalt", allowed_paths=()) == []
        assert search.search("cobalt", allowed_origins=()) == []
        search.sync(_paths=())
        assert indexed_rows(search, "workspace_search_chunks") == before
    finally:
        search.close()


@pytest.mark.parametrize("marker,value", [("version", "old"), ("chunking", "8:0")])
def test_partial_format_rebuild_retains_semantic_recall_and_full_rebuild_marker(
    tmp_path, monkeypatch, marker, value,
):
    workspace = MemoryWorkspace(tmp_path / "notes")
    write_note(workspace, "MEMORY.md", "Preferred seaside lodging.")
    write_note(workspace, "memory/archive.md", "Archived violet remains.")
    search = WorkspaceMemorySearch(workspace, embedding_client=LocalEmbeddings(),
                                   embedding_model="test")
    try:
        search.sync()
        with sqlite3.connect(search.db_path) as connection:
            connection.execute("UPDATE workspace_search_meta SET value=? WHERE key=?",
                               (value, marker))
        forbid_archive_scan(monkeypatch, workspace)
        assert [hit.path for hit in search.search(
            "beach hotel", allowed_paths=("MEMORY.md",),
        )] == ["MEMORY.md"]
        with sqlite3.connect(search.db_path) as connection:
            assert dict(connection.execute(
                "SELECT key,value FROM workspace_search_meta",
            ))[marker] == value
    finally:
        search.close()


def test_scoped_embedding_identity_change_does_not_reuse_previous_vectors(tmp_path, monkeypatch):
    workspace = MemoryWorkspace(tmp_path / "notes")
    write_note(workspace, "MEMORY.md", "Preferred seaside lodging.")
    write_note(workspace, "memory/archive.md", "Archived violet remains.")
    first = WorkspaceMemorySearch(workspace, embedding_client=LocalEmbeddings(),
                                  embedding_model="old")
    first.sync()
    old_vectors = indexed_rows(first, "workspace_search_embeddings")
    first.close()
    client = LocalEmbeddings()
    second = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="new")
    try:
        forbid_archive_scan(monkeypatch, workspace)
        assert second.search("beach hotel", allowed_paths=("MEMORY.md",))
        assert ["Preferred seaside lodging."] in client.calls
        assert ["Archived violet remains."] not in client.calls
        assert all(vector in indexed_rows(second, "workspace_search_embeddings")
                   for vector in old_vectors)
        assert second.status()["model"] == "new" and second.status()["dimensions"] == 2
    finally:
        second.close()


@pytest.mark.parametrize("path", ["memory/.hidden/note.md", "OTHER.md", "notes/file.md"])
def test_scoped_documents_reject_paths_outside_discoverable_memory_roots(tmp_path, path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    write_note(workspace, path, "Undiscoverable secret.")
    with pytest.raises(MemoryWorkspaceError):
        workspace.documents(paths=(path,))


def test_duplicate_and_missing_selected_paths_do_not_remove_other_notes(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    write_note(workspace, "MEMORY.md", "Cobalt chosen.")
    write_note(workspace, "memory/archive.md", "Archived violet remains.")
    search = WorkspaceMemorySearch(workspace)
    try:
        search.sync()
        assert len(search.search("cobalt", allowed_paths=("MEMORY.md", "MEMORY.md"))) == 1
        assert search.search("violet", allowed_paths=("USER.md",)) == []
        assert search.status()["documents"] == 2
    finally:
        search.close()
