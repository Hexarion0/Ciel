"""Behavioral checks for the rebuildable Markdown memory index."""

import hashlib
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from ciel.memory.workspace_search import WorkspaceMemorySearch


@dataclass(frozen=True)
class Document:
    path: str
    text: str
    modified_ns: int
    digest: str
    origin: str = "owner"
    local_only: bool = False
    session_id: str = ""


class FileWorkspace:
    """Small filesystem-backed test double for the separately tested workspace API."""

    def __init__(self, root: Path):
        self.root = root
        self.metadata = {}
        self.get = Mock(return_value={"text": "exact source excerpt"})

    def documents(self, *, paths=None):
        if paths is None:
            selected = [self.root / "MEMORY.md", self.root / "USER.md"]
            selected += sorted((self.root / "memory").rglob("*.md"))
        else:
            selected = [self.root / path for path in paths]
        result = []
        for path in selected:
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                relative = path.relative_to(self.root).as_posix()
                result.append(Document(
                    relative, text, path.stat().st_mtime_ns,
                    hashlib.sha256(text.encode()).hexdigest(),
                    **self.metadata.get(relative, {}),
                ))
        return result

    def write(self, path, text, **metadata):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        self.metadata[path] = metadata
        return target


@pytest.fixture
def workspace(tmp_path):
    return FileWorkspace(tmp_path)


class Embeddings:
    provider_id = "test-embeddings"
    local_only = True

    def __init__(self, vectors=None):
        self.calls = []
        self.vectors = vectors or {}

    def embed(self, texts, *, model):
        self.calls.append((list(texts), model))
        return [self.vectors.get(text, [1.0, 0.0]) for text in texts]


def test_arbitrary_markdown_is_searchable_with_source_lines_and_provenance(workspace):
    workspace.write("memory/experiments/notes.md", "# Plan\n\nUse cobalt for the launch.\n",
                    origin="agent", session_id="voice-123")
    search = WorkspaceMemorySearch(workspace)

    hits = search.search("cobalt")

    assert len(hits) == 1
    assert hits[0].path == "memory/experiments/notes.md"
    assert hits[0].start_line == 1
    assert hits[0].end_line == 3
    assert hits[0].origin == "agent"
    assert hits[0].source == "memory"
    assert "cobalt" in hits[0].to_dict()["snippet"]
    assert search.status()["mode"] == "keyword"


@pytest.mark.parametrize("query", ["café", "CAFE\u0301", "төслийн", "ПРОЕКТ"])
def test_unicode_queries_match_normalized_notes(workspace, query):
    workspace.write("MEMORY.md", unicodedata.normalize("NFD", "Café төслийн проект"))
    assert WorkspaceMemorySearch(workspace).search(query)


@pytest.mark.parametrize("query,text", [
    ("记忆", "这是一段关于记忆管理的笔记。"),
    ("会議", "明日の会議で設計を相談します。"),
    ("기억", "새로운기억저장기능을사용합니다."),
])
def test_unspaced_cjk_notes_support_short_keyword_queries(workspace, query, text):
    workspace.write("memory/languages.md", text)
    assert WorkspaceMemorySearch(workspace).search(query)


def test_exact_filename_lookup_and_keyword_or_fallback(workspace):
    workspace.write("memory/network-router.md", "The gateway address is 192.0.2.7.")
    search = WorkspaceMemorySearch(workspace)
    assert search.search("network-router")[0].path == "memory/network-router.md"
    assert search.search("memory/network-router.md")[0].score == 1
    assert search.search("gateway missingword")[0].path == "memory/network-router.md"
    assert search.search('gateway " OR * : -')[0].path == "memory/network-router.md"


def test_chunking_preserves_lines_for_large_notes(workspace):
    workspace.write("memory/large.md", "".join(
        f"Line {index} observation number {index}.\n" for index in range(100)
    ) + "Final secretword observation.\n")
    search = WorkspaceMemorySearch(workspace, chunk_tokens=40, chunk_overlap=8)
    hit = search.search("secretword")[0]
    assert hit.start_line <= 101 <= hit.end_line
    lines = (workspace.root / hit.path).read_text().splitlines()
    assert "secretword" in "\n".join(lines[hit.start_line - 1:hit.end_line])
    assert len(hit.snippet) <= 700


def test_manual_edit_delete_and_deleted_index_are_reconciled(workspace):
    note = workspace.write("MEMORY.md", "The launch color is cobalt.")
    search = WorkspaceMemorySearch(workspace)
    assert search.search("cobalt")

    note.write_text("The launch color is copper.")
    assert not search.search("cobalt")
    assert search.search("copper")

    search.db_path.unlink()
    assert search.search("copper")
    note.unlink()
    assert not search.search("copper")
    assert search.status()["chunks"] == 0


def test_private_results_are_filtered_before_limit_and_refresh_metadata(workspace):
    workspace.write("memory/aaa-private.md", "Project cobalt launch details. " * 100,
                    local_only=True)
    workspace.write("memory/public.md", "Cobalt schedule is public.")
    search = WorkspaceMemorySearch(workspace, chunk_tokens=20, chunk_overlap=4)

    assert {hit.path for hit in search.search("cobalt", max_results=1)} == {"memory/public.md"}
    assert any(hit.local_only for hit in search.search("cobalt", local_only=True))
    workspace.metadata["memory/public.md"] = {"local_only": True}
    assert not search.search("cobalt")


def test_default_search_never_constructs_or_calls_network_client(workspace, monkeypatch):
    workspace.write("MEMORY.md", "Cobalt is the selected color.")

    def unexpected_network(*args, **kwargs):
        raise AssertionError("Keyword memory must not use the network")

    monkeypatch.setattr("socket.create_connection", unexpected_network)
    search = WorkspaceMemorySearch(workspace)
    assert search.search("cobalt")
    assert search.status()["dimensions"] == 0


def test_reset_preserves_sources_and_unrelated_database_tables(workspace):
    workspace.write("MEMORY.md", "Cobalt is selected.")
    search = WorkspaceMemorySearch(workspace)
    assert search.search("cobalt")
    with sqlite3.connect(search.db_path) as conn:
        conn.execute("CREATE TABLE unrelated (value TEXT)")
        conn.execute("INSERT INTO unrelated VALUES ('retain me')")

    search.reset_index()

    assert (workspace.root / "MEMORY.md").read_text() == "Cobalt is selected."
    assert search.status()["chunks"] == 0
    with sqlite3.connect(search.db_path) as conn:
        assert conn.execute("SELECT value FROM unrelated").fetchone()[0] == "retain me"
    assert search.search("cobalt")


def test_semantic_synonym_recall_and_incremental_embedding_cache(workspace):
    workspace.write("memory/travel.md", "Holiday accommodation is near the water.")
    workspace.write("memory/hardware.md", "Cobalt graphics device.")
    client = Embeddings({"Where is the beach hotel?": [1, 0],
                         "Cobalt graphics device.": [0, 1]})
    search = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="model-a")

    assert search.search("Where is the beach hotel?")[0].path == "memory/travel.md"
    indexed_calls = [texts for texts, _ in client.calls if "Cobalt graphics device." in texts]
    assert len(indexed_calls) == 1
    client.calls.clear()
    search.sync()
    assert client.calls == []
    workspace.write("memory/new.md", "A newly recorded note.")
    search.sync()
    assert client.calls == [(["A newly recorded note."], "model-a")]


def test_hybrid_keeps_exact_keywords_and_diversifies_duplicate_snippets(workspace):
    workspace.write("memory/a.md", "Cobalt router setup uses port nine.")
    workspace.write("memory/b.md", "Cobalt router setup uses port nine.")
    workspace.write("memory/c.md", "Cobalt weather monitoring needs a separate rain gauge.")
    client = Embeddings({
        "Cobalt weather monitoring needs a separate rain gauge.": [0.95, 0.31225],
    })
    search = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="model-a")

    hits = search.search("cobalt", max_results=2)

    assert hits[0].path == "memory/a.md"
    assert hits[1].path == "memory/c.md"
    assert search.search("port nine", max_results=1)[0].path == "memory/a.md"


def test_daily_recency_changes_order_without_aging_out_eligible_hits(workspace):
    workspace.write("memory/2026-08-26.md", "Cobalt deployment details.")
    workspace.write("memory/2026-10-25.md", "Cobalt deployment details.")
    workspace.write("MEMORY.md", "Cobalt deployment details.")
    search = WorkspaceMemorySearch(
        workspace, now=lambda: datetime(2026, 10, 25, tzinfo=timezone.utc),
    )

    hits = search.search("cobalt", min_score=0.9)

    assert [hit.path for hit in hits] == [
        "MEMORY.md", "memory/2026-10-25.md", "memory/2026-08-26.md",
    ]
    assert hits[-1].score == pytest.approx(0.25)


def test_remote_adapter_never_receives_private_notes_or_private_queries(workspace):
    workspace.write("MEMORY.md", "Public cobalt note.")
    workspace.write("memory/private/notes.md", "Private secret cobalt note.", local_only=True)
    client = Embeddings()
    client.local_only = False
    search = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="model-a")

    assert search.search("cobalt")
    assert all("Private" not in text for texts, _ in client.calls for text in texts)
    client.calls.clear()
    assert search.search("Private secret", local_only=True)
    assert client.calls == []


def test_changed_embedding_dimensions_are_reported_without_mixing_vectors(workspace):
    workspace.write("MEMORY.md", "Cobalt chosen.")
    client = Embeddings()
    search = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="model-a")
    search.sync()
    client.vectors["cobalt"] = [1, 0, 0]

    assert search.search("cobalt")  # Keyword access remains usable with an explicit warning.
    assert "dimensions changed" in search.status()["error"]
    assert search.status()["dimensions"] == 2


def test_changed_model_uses_separate_embedding_identity(workspace):
    workspace.write("MEMORY.md", "Cobalt chosen.")
    client = Embeddings()
    first = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="model-a")
    first.sync()
    first.close()
    client.calls.clear()
    second = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="model-b")
    assert second.search("cobalt")
    assert client.calls[0] == (["Cobalt chosen."], "model-b")
    assert second.status()["model"] == "model-b"


def test_failed_transaction_retains_published_index_and_filters_stale_content(workspace):
    workspace.write("MEMORY.md", "Cobalt chosen.")
    search = WorkspaceMemorySearch(workspace)
    search.sync()
    with sqlite3.connect(search.db_path) as conn:
        conn.execute("""
            CREATE TRIGGER reject_new_chunks BEFORE INSERT ON workspace_search_chunks
            BEGIN SELECT RAISE(ABORT, 'simulated publication failure'); END;
        """)
    workspace.write("MEMORY.md", "Copper chosen.")

    assert search.sync(force=True).error
    with sqlite3.connect(search.db_path) as conn:
        assert conn.execute("SELECT text FROM workspace_search_chunks").fetchone()[0] == (
            "Cobalt chosen."
        )
    assert not search.search("cobalt")
    with sqlite3.connect(search.db_path) as conn:
        conn.execute("DROP TRIGGER reject_new_chunks")
    assert search.search("copper")


def test_get_delegates_safe_reader_including_traversal_errors(workspace):
    search = WorkspaceMemorySearch(workspace)
    assert search.get("MEMORY.md", from_line=3, lines=2) == {"text": "exact source excerpt"}
    workspace.get.assert_called_once_with(
        "MEMORY.md", from_line=3, lines=2, max_chars=8000, local_only=False,
    )
    workspace.get.side_effect = ValueError("Path outside memory roots")
    with pytest.raises(ValueError, match="outside"):
        search.get("../../private.txt")


@pytest.mark.parametrize("options", [
    {"query": "x" * 4097}, {"query": "cobalt", "max_results": 0},
    {"query": "cobalt", "max_results": True}, {"query": "cobalt", "min_score": float("nan")},
])
def test_invalid_search_requests_are_bounded(workspace, options):
    with pytest.raises(ValueError):
        WorkspaceMemorySearch(workspace).search(**options)


def test_empty_queries_and_closed_index_do_not_run_inference(workspace):
    client = Embeddings()
    search = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="model-a")
    assert search.search("  ") == []
    assert search.search("***") == []
    assert not client.calls
    search.close()
    with pytest.raises(RuntimeError, match="closed"):
        search.search("cobalt")


def test_real_workspace_sources_privacy_and_safe_excerpt_reader(tmp_path):
    from ciel.memory.workspace import MemoryWorkspace, MemoryWorkspaceError

    workspace = MemoryWorkspace(tmp_path)
    workspace.append_curated("The launch color is cobalt.")
    workspace.append_daily("The private password cue is violet.", local_only=True)
    search = WorkspaceMemorySearch(workspace)

    assert search.search("cobalt")[0].origin == "owner"
    assert not search.search("violet")
    hit = search.search("violet", local_only=True)[0]
    assert hit.local_only
    assert search.get(hit.path, local_only=True).text == "The private password cue is violet."
    with pytest.raises(MemoryWorkspaceError):
        search.get(hit.path)
    with pytest.raises(MemoryWorkspaceError):
        search.get("../outside.md")


def test_repeated_search_keeps_embedding_failure_visible_during_backoff(workspace):
    workspace.write("MEMORY.md", "Cobalt chosen.")
    client = Embeddings()
    search = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="model-a")
    search.sync()
    client.vectors["cobalt"] = [1, 0, 0]
    assert search.search("cobalt")
    assert search.search("cobalt")
    assert "dimensions changed" in search.status()["error"]


def test_source_removed_during_query_inference_cannot_return_stale_hit(workspace):
    target = workspace.write("MEMORY.md", "Cobalt chosen.")
    client = Embeddings()
    search = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="model-a")
    search.sync()
    original_embed = client.embed

    def remove_during_inference(texts, *, model):
        target.unlink()
        return original_embed(texts, model=model)

    client.embed = remove_during_inference
    assert not search.search("cobalt")


def wait_until(predicate):
    """Wait for a real background publication without triggering on-search synchronization."""
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("Background memory publication did not complete")


def test_optional_watcher_updates_edits_deletes_and_stops_on_close(workspace):
    note = workspace.write("MEMORY.md", "Cobalt chosen.")
    search = WorkspaceMemorySearch(workspace)
    assert not search.status()["watching"]
    search.sync()
    assert search.start_watching(0.05)
    watcher = search._watcher
    assert not search.start_watching(0.05)
    assert search._watcher is watcher

    def indexed_text():
        with sqlite3.connect(search.db_path) as conn:
            return [row[0] for row in conn.execute("SELECT text FROM workspace_search_chunks")]

    note.write_text("Copper chosen.")
    wait_until(lambda: indexed_text() == ["Copper chosen."])
    note.unlink()
    wait_until(lambda: indexed_text() == [])
    search.close()
    assert watcher is not None and not watcher.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        search.start_watching(0.05)


def test_watcher_survives_source_errors_and_stops_when_owner_retires(workspace):
    workspace.write("MEMORY.md", "Cobalt chosen.")
    search = WorkspaceMemorySearch(workspace)
    current = threading.Event()
    current.set()
    original_documents = workspace.documents
    calls = []

    def documents():
        calls.append(True)
        if len(calls) == 1:
            raise OSError("Transient read failure")
        return original_documents()

    workspace.documents = documents
    try:
        search.start_watching(0.05, current=current.is_set)
        wait_until(lambda: search.status()["chunks"] == 1)
        assert len(calls) >= 2
        current.clear()
        wait_until(lambda: not search.status()["watching"])
    finally:
        search.close()


@pytest.mark.parametrize("link_directory", [True, False])
def test_index_cache_symlinks_cannot_modify_an_outside_database(workspace, link_directory):
    workspace.write("MEMORY.md", "Cobalt chosen.")
    outside = workspace.root / "outside"
    outside.mkdir()
    outside_db = outside / "search.sqlite3"
    with sqlite3.connect(outside_db) as conn:
        conn.execute("CREATE TABLE unrelated (value TEXT)")
        conn.execute("INSERT INTO unrelated VALUES ('retain me')")
    original = outside_db.read_bytes()
    cache = workspace.root / ".ciel"
    if link_directory:
        cache.symlink_to(outside, target_is_directory=True)
    else:
        cache.mkdir()
        (cache / "search.sqlite3").symlink_to(outside_db)
    search = WorkspaceMemorySearch(workspace)

    with pytest.raises(ValueError, match="symlink"):
        search.sync()
    with pytest.raises(ValueError, match="symlink"):
        search.reset_index()
    assert outside_db.read_bytes() == original


def test_configured_index_path_rejects_symlinked_ancestors(workspace):
    workspace.write("MEMORY.md", "Cobalt chosen.")
    real = workspace.root / "real-cache"
    real.mkdir()
    alias = workspace.root / "alias-cache"
    alias.symlink_to(real, target_is_directory=True)
    search = WorkspaceMemorySearch(workspace, db_path=alias / "nested" / "search.sqlite3")
    with pytest.raises(ValueError, match="symlink"):
        search.status()
    assert list(real.iterdir()) == []


def test_path_and_origin_filters_apply_before_candidates_and_result_limit(workspace):
    for index in range(12):
        workspace.write(f"memory/aaa-{index}.md", "Cobalt launch. " * 20, origin="agent")
    workspace.write("USER.md", "Cobalt is my preferred launch color.", origin="owner")
    workspace.write("MEMORY.md", "Cobalt curated agent claim.", origin="agent")
    search = WorkspaceMemorySearch(workspace, candidate_limit=6)

    hits = search.search("cobalt", max_results=1,
                         allowed_paths=("MEMORY.md", "USER.md"), allowed_origins=("owner",))

    assert [hit.path for hit in hits] == ["USER.md"]
    assert search.search("cobalt", allowed_paths=()) == []
    assert search.search("cobalt", allowed_origins=()) == []
    assert any(hit.origin == "agent" for hit in search.search("cobalt"))


def test_filters_apply_to_filename_and_semantic_candidates(workspace):
    workspace.write("memory/aaa-excluded.md", "Holiday accommodation near the sea.",
                    origin="agent")
    workspace.write("USER.md", "Preferred seaside lodging.", origin="owner")
    client = Embeddings()
    search = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="model-a")

    assert [hit.path for hit in search.search("beach hotel", allowed_paths=("USER.md",),
                                             allowed_origins=("owner",))] == ["USER.md"]
    keyword = WorkspaceMemorySearch(workspace)
    assert keyword.search("aaa-excluded", allowed_paths=("USER.md",)) == []


def test_keyword_refresh_does_not_wait_for_background_embedding(workspace):
    note = workspace.write("MEMORY.md", "Cobalt chosen.")
    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()
    result = []
    client = Embeddings()
    original_embed = client.embed

    def slow_embed(texts, *, model):
        entered.set()
        assert release.wait(3)
        return original_embed(texts, model=model)

    client.embed = slow_embed
    search = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="model-a")
    background = threading.Thread(target=search.sync)
    background.start()
    assert entered.wait(1)
    note.write_text("Copper chosen.")

    def query():
        try:
            result.extend(search.search("copper"))
        finally:
            done.set()

    foreground = threading.Thread(target=query)
    foreground.start()
    try:
        assert done.wait(0.5), "Fresh keyword access waited for background inference"
        assert [hit.path for hit in result] == ["MEMORY.md"]
        assert "Copper" in result[0].snippet
    finally:
        release.set()
        foreground.join(3)
        background.join(3)
        search.close()


@pytest.mark.parametrize("scope", [
    {"allowed_paths": ["MEMORY.md"]}, {"allowed_origins": ("owner",) * 1001},
    {"allowed_paths": (None,)}, {"allowed_origins": ("",)},
])
def test_invalid_search_scopes_are_rejected(workspace, scope):
    with pytest.raises(ValueError, match="scopes"):
        WorkspaceMemorySearch(workspace).search("cobalt", **scope)


def test_live_origin_changes_are_rechecked_after_inference(workspace):
    workspace.write("MEMORY.md", "Cobalt chosen.", origin="owner")
    client = Embeddings()
    original_embed = client.embed

    def downgrade_during_query(texts, *, model):
        if texts == ["cobalt"]:
            workspace.metadata["MEMORY.md"] = {"origin": "agent"}
        return original_embed(texts, model=model)

    client.embed = downgrade_during_query
    search = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="model-a")
    assert search.search("cobalt", allowed_origins=("owner",)) == []
    assert search.search("cobalt")[0].origin == "agent"


def test_first_search_has_bounded_inference_and_only_one_pending_worker(workspace):
    workspace.write("MEMORY.md", "Cobalt chosen.")
    entered = threading.Event()
    release = threading.Event()
    client = Embeddings()
    original_embed = client.embed

    def slow_embed(texts, *, model):
        entered.set()
        assert release.wait(3)
        return original_embed(texts, model=model)

    client.embed = slow_embed
    search = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="model-a")
    try:
        started = time.monotonic()
        assert search.search("cobalt")
        assert time.monotonic() - started < 0.5
        assert entered.is_set()
        worker = search._search_inference
        assert search.search("chosen")
        assert search._search_inference is worker
    finally:
        release.set()
        wait_until(lambda: not search._search_inference.is_alive())
        search.close()


@pytest.mark.parametrize("action", ["close", "reset_index"])
def test_retired_or_reset_index_does_not_publish_inflight_embeddings(workspace, action):
    workspace.write("MEMORY.md", "Cobalt chosen.")
    entered = threading.Event()
    release = threading.Event()
    client = Embeddings()
    original_embed = client.embed

    def slow_embed(texts, *, model):
        entered.set()
        assert release.wait(3)
        return original_embed(texts, model=model)

    client.embed = slow_embed
    search = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="model-a")
    background = threading.Thread(target=search.sync)
    background.start()
    try:
        assert entered.wait(1)
        getattr(search, action)()
    finally:
        release.set()
        background.join(3)
    with sqlite3.connect(search.db_path) as conn:
        assert conn.execute("SELECT count(*) FROM workspace_search_embeddings").fetchone()[0] == 0
    search.close()


def test_watcher_retirement_discards_inflight_embeddings(workspace):
    workspace.write("MEMORY.md", "Cobalt chosen.")
    entered = threading.Event()
    release = threading.Event()
    current = threading.Event()
    current.set()
    client = Embeddings()
    original_embed = client.embed

    def slow_embed(texts, *, model):
        entered.set()
        assert release.wait(3)
        return original_embed(texts, model=model)

    client.embed = slow_embed
    search = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="model-a")
    try:
        search.start_watching(0.05, current=current.is_set)
        assert entered.wait(1)
        current.clear()
        release.set()
        wait_until(lambda: not search._watcher.is_alive())
        with sqlite3.connect(search.db_path) as conn:
            count = conn.execute("SELECT count(*) FROM workspace_search_embeddings").fetchone()[0]
            assert count == 0
    finally:
        release.set()
        search.close()


def test_local_vectors_are_not_reused_by_a_remote_adapter_with_the_same_identity(workspace):
    workspace.write("memory/private/note.md", "Identical cobalt text.", local_only=True)
    local_client = Embeddings()
    local = WorkspaceMemorySearch(workspace, embedding_client=local_client,
                                  embedding_model="model-a")
    local.sync()
    local.close()
    workspace.write("MEMORY.md", "Identical cobalt text.")
    remote_client = Embeddings()
    remote_client.local_only = False
    remote = WorkspaceMemorySearch(workspace, embedding_client=remote_client,
                                   embedding_model="model-a")

    remote.sync()

    assert remote_client.calls == [(["Identical cobalt text."], "model-a")]
    remote.close()


@pytest.mark.parametrize("suffix", ["-journal", "-wal", "-shm"])
def test_sqlite_sidecar_symlinks_are_rejected_before_opening_database(workspace, suffix):
    workspace.write("MEMORY.md", "Cobalt chosen.")
    cache = workspace.root / ".ciel"
    cache.mkdir()
    outside = workspace.root / "outside-data"
    outside.write_bytes(b"retain me")
    (cache / ("search.sqlite3" + suffix)).symlink_to(outside)
    search = WorkspaceMemorySearch(workspace)

    with pytest.raises(ValueError, match="symlink"):
        search.sync()

    assert outside.read_bytes() == b"retain me"


def test_late_semantic_scores_are_cached_for_subsequent_query(workspace):
    workspace.write("MEMORY.md", "Holiday accommodation near the sea.")
    release = threading.Event()
    client = Embeddings()
    original_embed = client.embed

    def slow_embed(texts, *, model):
        assert release.wait(3)
        return original_embed(texts, model=model)

    client.embed = slow_embed
    search = WorkspaceMemorySearch(workspace, embedding_client=client, embedding_model="model-a")
    try:
        assert search.search("beach hotel") == []
        release.set()
        wait_until(lambda: not search._search_inference.is_alive())
        client.calls.clear()
        assert search.search("beach hotel")[0].path == "MEMORY.md"
        assert client.calls == []
    finally:
        release.set()
        search.close()


def test_failed_metadata_publication_cannot_crowd_out_allowed_live_origins(workspace):
    for index in range(12):
        workspace.write(f"memory/aaa-{index}.md", "Cobalt.", origin="owner")
    workspace.write("USER.md", "Cobalt is my preferred color for this project.", origin="owner")
    search = WorkspaceMemorySearch(workspace, candidate_limit=6)
    search.sync()
    for index in range(12):
        workspace.metadata[f"memory/aaa-{index}.md"] = {"origin": "agent"}
    with sqlite3.connect(search.db_path) as conn:
        conn.execute("""
            CREATE TRIGGER reject_metadata BEFORE INSERT ON workspace_search_documents
            BEGIN SELECT RAISE(ABORT, 'simulated metadata publication failure'); END;
        """)

    hits = search.search("cobalt", max_results=1, allowed_origins=("owner",))

    assert [hit.path for hit in hits] == ["USER.md"]
