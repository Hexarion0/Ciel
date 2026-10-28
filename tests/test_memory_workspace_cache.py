"""Archive document caching preserves filesystem and provenance freshness."""

import json
import os
from unittest.mock import Mock

import pytest

from ciel.memory.workspace import MemoryWorkspace, MemoryWorkspaceError


def _write(workspace, path, text):
    target = workspace.root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def test_unchanged_thousand_note_archive_does_not_reread_or_rehash(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    for number in range(1000):
        _write(workspace, f"memory/note-{number:04}.md", "Archived cobalt observation.\n" * 20)
    assert len(workspace.documents()) == 1000
    workspace._read = Mock(wraps=workspace._read)
    workspace._digest = Mock(wraps=workspace._digest)

    assert len(workspace.documents()) == 1000
    assert len(workspace.documents()) == 1000

    assert workspace._read.call_count == 0
    assert workspace._digest.call_count == 0


def test_mtime_preserving_in_place_edit_is_seen_by_documents_and_get(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    target = _write(workspace, "MEMORY.md", "Cobalt chosen.\n")
    original = workspace.documents()[0]
    status = target.stat()
    target.write_text("Copper chosen.\n")
    os.utime(target, ns=(status.st_atime_ns, status.st_mtime_ns))

    fresh = workspace.documents()[0]

    assert fresh.modified_ns == original.modified_ns
    assert fresh.digest != original.digest
    assert fresh.text == "Copper chosen.\n"
    assert workspace.get("MEMORY.md").text == "Copper chosen."


def test_provenance_changes_are_independent_of_cached_prose(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    workspace.append_curated("Cobalt chosen.")
    original = workspace.documents()[0]
    registry = workspace.root / ".ciel-memory.json"
    state = json.loads(registry.read_text())
    state["files"]["MEMORY.md"].update(
        origin="imported", local_only=True, session_id="changed-session",
    )
    registry.write_text(json.dumps(state))
    workspace._read = Mock(wraps=workspace._read)

    fresh = workspace.documents()[0]

    assert fresh.text == original.text
    assert fresh.digest == original.digest
    assert fresh.local_only
    assert fresh.origin == "imported"
    assert fresh.session_id == "changed-session"
    assert all(call.args[0].name == ".ciel-memory.json"
               for call in workspace._read.call_args_list)
    with pytest.raises(MemoryWorkspaceError, match="local-only"):
        workspace.get("MEMORY.md")


def test_structured_fact_refresh_does_not_scan_unrelated_archive(tmp_path, monkeypatch):
    workspace = MemoryWorkspace(tmp_path / "notes")
    uid = "a" * 32
    workspace.put_fact(uid, {"metadata": {}}, "Managed cobalt.")
    _write(workspace, "memory/archive.md", "Unrelated archive.")

    def unexpected_archive_scan():
        pytest.fail("Structured fact refresh traversed the archive")

    monkeypatch.setattr(workspace, "_document_paths", unexpected_archive_scan)
    signature = workspace.signature()
    assert workspace.facts()[uid]["content"] == "Managed cobalt."
    _write(workspace, "memory/archive.md", "Updated unrelated archive.")
    assert workspace.signature() == signature
    target = workspace.root / "MEMORY.md"
    target.write_text(target.read_text().replace("cobalt", "copper"))
    assert workspace.signature() != signature
    assert workspace.facts()[uid]["content"] == "Managed copper."


def test_atomic_replacement_with_same_size_and_mtime_is_not_reused(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    target = _write(workspace, "MEMORY.md", "Cobalt chosen.\n")
    original = workspace.documents()[0]
    status = target.stat()
    replacement = workspace.root / "replacement.tmp"
    replacement.write_text("Copper chosen.\n")
    os.utime(replacement, ns=(status.st_atime_ns, status.st_mtime_ns))
    replacement.replace(target)
    workspace._read = Mock(wraps=workspace._read)

    fresh = workspace.documents()[0]

    assert fresh.digest != original.digest
    assert fresh.modified_ns == original.modified_ns
    assert fresh.text == "Copper chosen.\n"
    workspace._read.assert_called_once_with(target)


def test_deletes_new_nested_files_and_cache_eviction_are_fresh(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    deleted = _write(workspace, "memory/old.md", "Old cobalt.")
    workspace.documents()
    deleted.unlink()
    _write(workspace, "memory/new/nested.md", "New copper.")

    assert [document.path for document in workspace.documents()] == ["memory/new/nested.md"]
    assert "memory/old.md" not in workspace._document_cache


@pytest.mark.parametrize("parent_link", [False, True])
def test_cached_documents_cannot_bypass_replaced_symlinks(tmp_path, parent_link):
    workspace = MemoryWorkspace(tmp_path / "notes")
    target = _write(workspace, "memory/note.md", "Owner note.")
    workspace.documents()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "note.md").write_text("Outside secret.")
    if parent_link:
        target.parent.rename(workspace.root / "old-memory")
        (workspace.root / "memory").symlink_to(outside, target_is_directory=True)
    else:
        target.unlink()
        target.symlink_to(outside / "note.md")
    workspace._read = Mock(wraps=workspace._read)

    with pytest.raises(MemoryWorkspaceError, match="symlink"):
        workspace.documents()
    with pytest.raises(MemoryWorkspaceError, match="symlink"):
        workspace.get("memory/note.md")

    workspace._read.assert_not_called()


def test_permission_changes_force_revalidation_and_never_fall_back_on_cached_text(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    target = _write(workspace, "MEMORY.md", "Owner note.")
    workspace.documents()
    target.chmod(0o400)
    original_read = workspace._read
    workspace._read = Mock(wraps=original_read)
    assert workspace.documents()[0].text == "Owner note."
    workspace._read.assert_called_once_with(target)
    target.chmod(0o000)
    workspace._read = Mock(side_effect=PermissionError("No longer readable"))

    with pytest.raises(PermissionError, match="readable"):
        workspace.documents()

    assert "MEMORY.md" not in workspace._document_cache
    target.chmod(0o600)


def test_invalid_utf8_after_warmup_raises_without_cached_fallback(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    target = _write(workspace, "MEMORY.md", "Valid text.")
    workspace.documents()
    status = target.stat()
    target.write_bytes(b"\xff" * status.st_size)
    os.utime(target, ns=(status.st_atime_ns, status.st_mtime_ns))

    for read in (workspace.documents, lambda: workspace.get("MEMORY.md")):
        with pytest.raises(MemoryWorkspaceError, match="UTF-8"):
            read()
    assert "MEMORY.md" not in workspace._document_cache


def test_document_changing_during_read_is_rejected_and_not_cached(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    _write(workspace, "MEMORY.md", "Initial note.")
    original_read = workspace._read

    def change_after_read(path):
        result = original_read(path)
        path.write_text("Changed note.")
        return result

    workspace._read = change_after_read
    with pytest.raises(MemoryWorkspaceError, match="changed"):
        workspace.documents()

    assert "MEMORY.md" not in workspace._document_cache
    workspace._read = original_read
    assert workspace.get("MEMORY.md").text == "Changed note."


def test_registry_cache_is_not_modified_by_mutable_working_copies(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    workspace.append_curated("Owner note.")
    workspace.documents()
    state = workspace._registry()
    state["files"]["MEMORY.md"]["local_only"] = True
    state["files"]["MEMORY.md"]["origin"] = "untrusted"
    workspace._read = Mock(wraps=workspace._read)

    document = workspace.documents()[0]

    assert not document.local_only
    assert document.origin == "owner"
    workspace._read.assert_not_called()


@pytest.mark.parametrize("replacement", [b"\xff", b"{not json}"])
def test_invalid_changed_registry_does_not_use_cached_provenance(tmp_path, replacement):
    workspace = MemoryWorkspace(tmp_path / "notes")
    workspace.append_curated("Owner note.")
    workspace.documents()
    registry = workspace.root / ".ciel-memory.json"
    status = registry.stat()
    registry.write_bytes(replacement)
    os.utime(registry, ns=(status.st_atime_ns, status.st_mtime_ns))

    with pytest.raises(MemoryWorkspaceError):
        workspace.documents()
    with pytest.raises(MemoryWorkspaceError):
        workspace.get("MEMORY.md")
    assert workspace._registry_cache is None


def test_private_namespace_stays_private_after_registry_deletion(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    path = workspace.append_curated("Private note.", local_only=True)
    workspace.documents()
    (workspace.root / ".ciel-memory.json").unlink()

    assert workspace.documents()[0].local_only
    with pytest.raises(MemoryWorkspaceError, match="local-only"):
        workspace.get(path)


def test_document_cache_is_bounded_by_entries_and_text_bytes(tmp_path, monkeypatch):
    import ciel.memory.workspace as workspace_module

    monkeypatch.setattr(workspace_module, "_DOCUMENT_CACHE_ENTRIES", 2)
    monkeypatch.setattr(workspace_module, "_DOCUMENT_CACHE_BYTES", 1200)
    workspace = MemoryWorkspace(tmp_path / "notes")
    for number in range(5):
        _write(workspace, f"memory/note-{number}.md", "Small note.")
    assert len(workspace.documents()) == 5
    assert len(workspace._document_cache) <= 2
    assert workspace._document_cache_bytes <= 1200
    _write(workspace, "memory/large.md", "x" * 2000)
    assert len(workspace.documents()) == 6
    assert "memory/large.md" not in workspace._document_cache
    assert workspace._document_cache_bytes <= 1200


def test_registered_fact_paths_cannot_read_outside_allowed_roots(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    outside = tmp_path / "outside.md"
    outside.write_text("Outside secret.")
    registry = {"version": 1, "files": {}, "entries": {},
                "facts": {"a" * 32: {"path": "../outside.md"}}}
    (workspace.root / ".ciel-memory.json").write_text(json.dumps(registry))
    workspace._read = Mock(wraps=workspace._read)

    with pytest.raises(MemoryWorkspaceError):
        workspace.signature()
    with pytest.raises(MemoryWorkspaceError):
        workspace.facts()
    assert all(call.args[0].is_relative_to(workspace.root)
               for call in workspace._read.call_args_list)


def test_mtime_preserving_registry_edit_refreshes_origin_without_rereading_text(tmp_path):
    workspace = MemoryWorkspace(tmp_path / "notes")
    workspace.append_curated("Owner note.")
    workspace.documents()
    registry = workspace.root / ".ciel-memory.json"
    status = registry.stat()
    original = registry.read_text()
    changed = original.replace('"origin": "owner"', '"origin": "agent"')
    assert len(changed) == len(original)
    registry.write_text(changed)
    os.utime(registry, ns=(status.st_atime_ns, status.st_mtime_ns))
    workspace._read = Mock(wraps=workspace._read)

    assert workspace.documents()[0].origin == "agent"

    workspace._read.assert_called_once_with(registry)


def test_registry_cache_skips_large_provenance_files(tmp_path, monkeypatch):
    import ciel.memory.workspace as workspace_module

    monkeypatch.setattr(workspace_module, "_REGISTRY_CACHE_BYTES", 1)
    workspace = MemoryWorkspace(tmp_path / "notes")
    workspace.append_curated("Owner note.")
    assert workspace.documents()
    assert workspace._registry_cache is None
