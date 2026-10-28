"""Setting replacement compares JSON snapshots atomically across independent writers."""

import multiprocessing
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from ciel.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    memory = MemoryStore(tmp_path / "settings.db")
    yield memory
    memory.close()


def test_missing_setting_can_be_replaced_without_touching_other_settings(store):
    store.set_setting("other", {"keep": True})
    assert store.compare_setting("task", None, {"revision": "first"})
    assert store.get_setting("task") == {"revision": "first"}
    assert store.get_setting("other") == {"keep": True}


def test_stale_or_wrong_expected_setting_never_overwrites_newer_value(store):
    store.set_setting("task", {"revision": "current"})
    assert not store.compare_setting("task", {"revision": "old"}, {"revision": "stale"})
    assert not store.compare_setting("task", None, {"revision": "resurrected"})
    assert not store.compare_setting("missing", {}, {"revision": "unexpected"})
    assert store.get_setting("task") == {"revision": "current"}
    assert store.get_setting("missing") is None


def test_persisted_null_matches_none_like_a_missing_setting(store):
    store.set_setting("task", None)
    assert store.compare_setting("task", None, {"revision": "first"})


@pytest.mark.parametrize("persisted,expected", [
    ({"flag": True}, {"flag": 1}),
    ({"revision": 1}, {"revision": 1.0}),
    ({"items": [False]}, {"items": [0]}),
    ({"items": [1, 2]}, {"items": [2, 1]}),
])
def test_comparison_preserves_json_types_and_array_order(store, persisted, expected):
    store.set_setting("task", persisted)
    assert not store.compare_setting("task", expected, {"revision": "wrong"})
    assert store.get_setting("task") == persisted


def test_comparison_ignores_object_key_order_and_json_whitespace(store):
    with store._get_connection() as connection:
        connection.execute(
            "INSERT INTO memory_settings(name,value_json) VALUES (?,?)",
            ("task", '{ "b": [2, 3], "a": { "first": "one", "second": "two" } }'),
        )
    expected = {"a": {"second": "two", "first": "one"}, "b": [2, 3]}
    assert store.compare_setting("task", expected, {"revision": "next"})
    assert store.get_setting("task") == {"revision": "next"}


@pytest.mark.parametrize("raw", ["{broken JSON", "NaN", '{"revision": NaN}'])
def test_corrupt_persisted_json_does_not_match_missing_or_get_overwritten(store, raw):
    with store._get_connection() as connection:
        connection.execute("INSERT INTO memory_settings VALUES (?,?)", ("task", raw))
    assert not store.compare_setting("task", None, {"revision": "wrong"})
    row = store._get_connection().execute(
        "SELECT value_json FROM memory_settings WHERE name=?", ("task",),
    ).fetchone()
    assert row["value_json"] == raw
    assert not store._get_connection().in_transaction
    assert store.compare_setting("other", None, {"revision": "valid"})


def test_invalid_new_json_is_rejected_before_any_setting_is_changed(store):
    store.set_setting("task", {"revision": "current"})
    with pytest.raises(ValueError):
        store.compare_setting("task", {"revision": "current"}, {"invalid": float("nan")})
    assert store.get_setting("task") == {"revision": "current"}
    assert not store._get_connection().in_transaction


def test_failed_setting_write_rolls_back_and_releases_transaction(store):
    store.set_setting("task", {"revision": "current"})
    connection = store._get_connection()
    connection.execute(
        "CREATE TRIGGER reject_task BEFORE UPDATE ON memory_settings "
        "BEGIN SELECT RAISE(ABORT, 'synthetic write failure'); END",
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.compare_setting("task", {"revision": "current"}, {"revision": "next"})
    assert not connection.in_transaction
    assert store.get_setting("task") == {"revision": "current"}
    connection.execute("DROP TRIGGER reject_task")
    assert store.compare_setting("task", {"revision": "current"}, {"revision": "next"})


def test_competing_store_instances_only_replace_the_same_old_snapshot_once(tmp_path):
    path = tmp_path / "shared.db"
    first, second = MemoryStore(path), MemoryStore(path)
    first.set_setting("task", {"revision": "old"})
    ready = Barrier(2)

    def replace(memory, revision):
        try:
            ready.wait(timeout=5)
            return revision, memory.compare_setting("task", {"revision": "old"},
                                                     {"revision": revision})
        finally:
            memory.close()

    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            pending = [workers.submit(replace, memory, revision)
                       for memory, revision in [(first, "one"), (second, "two")]]
            results = [future.result(timeout=5) for future in pending]
        assert sum(changed for _revision, changed in results) == 1
        winner = next(revision for revision, changed in results if changed)
        assert first.get_setting("task") == {"revision": winner}
    finally:
        first.close()
        second.close()


def _replace_from_process(path, revision, start, ready, results):
    memory = MemoryStore(path)
    try:
        ready.put(True)
        if not start.wait(timeout=5):
            raise TimeoutError("Competing setting writers were not started")
        changed = memory.compare_setting("task", {"revision": "old"}, {"revision": revision})
        results.put((revision, changed))
    finally:
        memory.close()


def test_competing_processes_only_replace_the_same_old_snapshot_once(tmp_path):
    path = tmp_path / "shared-process.db"
    memory = MemoryStore(path)
    memory.set_setting("task", {"revision": "old"})
    context = multiprocessing.get_context("spawn")
    start, ready, results = context.Event(), context.Queue(), context.Queue()
    processes = [context.Process(target=_replace_from_process,
                                 args=(path, revision, start, ready, results))
                 for revision in ["one", "two"]]
    try:
        for process in processes:
            process.start()
        for _process in processes:
            assert ready.get(timeout=5)
        start.set()
        replacements = [results.get(timeout=5) for _process in processes]
        for process in processes:
            process.join(timeout=5)
            assert process.exitcode == 0
        assert sum(changed for _revision, changed in replacements) == 1
        winner = next(revision for revision, changed in replacements if changed)
        assert memory.get_setting("task") == {"revision": winner}
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        ready.close()
        results.close()
        memory.close()
