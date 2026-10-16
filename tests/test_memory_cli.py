"""Memory owner tools run locally, preserve facts on pause, and make consistent backups."""

import json
from unittest.mock import MagicMock

import pytest

from ciel import __main__
from ciel.memory import ConversationTurn, MemoryItem, MemoryStore
from ciel.memory.cli import main
from ciel.memory.service import MemoryService


@pytest.fixture
def database(tmp_path):
    """Create a synthetic corrected fact and one conversation turn."""
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    store.upsert_fact("user:current_project", MemoryItem(content="Atlas"))
    store.upsert_fact("user:current_project", MemoryItem(content="Borealis"))
    store.save_turn(ConversationTurn(role="user", content="A synthetic question"))
    store.close()
    return path


def test_stats_and_search_are_machine_readable(database, capsys):
    assert main(["--db", str(database), "--json", "stats"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["durable_memories"] == 1
    assert stats["conversation_turns"] == 1
    assert main(["--db", str(database), "--json", "search", "Borealis"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["content"] == "Borealis"


def test_history_exposes_current_value_and_revisions(database, capsys):
    assert main(["--db", str(database), "history", "user:current_project"]) == 0
    history = json.loads(capsys.readouterr().out)
    assert history["current"]["content"] == "Borealis"
    assert history["revisions"][0]["content"] == "Atlas"


def test_project_inspection_filters_exact_project_identity(database, capsys):
    store = MemoryStore(database)
    service = MemoryService(store, learning_mode="automatic")
    service.handle("Project Atlas decision: stay offline")
    service.handle("Project Atlas next step: test restart")
    service.handle("Project Atlas Two next step: write proposal")
    store.close()
    assert main(["--db", str(database), "--json", "project", "Atlas"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2


def test_pause_resume_and_explicit_inspection_keep_data(database, capsys):
    assert main(["--db", str(database), "pause"]) == 0
    assert main(["--db", str(database), "show", "user:current_project"]) == 0
    assert "Borealis" in capsys.readouterr().out
    store = MemoryStore(database)
    assert store.get_setting("recall_enabled") is False
    assert store.count_memories() == 1
    store.close()
    assert main(["--db", str(database), "resume"]) == 0


def test_backup_is_complete_private_and_does_not_overwrite(database, tmp_path, capsys):
    backup = tmp_path / "backup.db"
    assert main(["--db", str(database), "backup", str(backup)]) == 0
    assert backup.stat().st_mode & 0o777 == 0o600
    restored = MemoryStore(backup)
    assert restored.get_fact("user:current_project").content == "Borealis"
    assert len(restored.get_recent_turns()) == 1
    restored.close()
    assert main(["--db", str(database), "backup", str(backup)]) == 1
    assert "failed" in capsys.readouterr().out


def test_clear_session_preserves_facts_and_forget_is_targeted(database):
    assert main(["--db", str(database), "clear-session", "default"]) == 0
    store = MemoryStore(database)
    assert store.get_fact("user:current_project") is not None
    assert not store.get_recent_turns()
    store.close()
    assert main(["--db", str(database), "forget", "nonexistent"]) == 1
    assert main(["--db", str(database), "forget", "user:current_project"]) == 0
    store = MemoryStore(database)
    assert store.count_memories() == 0
    assert "Borealis" not in store.redact_forgotten("Atlas and Borealis")
    store.close()


def test_memory_command_does_not_initialize_audio(monkeypatch, capsys):
    audio = MagicMock(side_effect=AssertionError("Memory inspection must not start audio"))
    monkeypatch.setattr("ciel.audio.TextToSpeech", audio)
    monkeypatch.setattr("sys.argv", ["ciel", "memory", "--help"])
    with pytest.raises(SystemExit) as error:
        __main__.main()
    assert error.value.code == 0
    assert "pause" in capsys.readouterr().out
    audio.assert_not_called()


@pytest.mark.parametrize("limit", ["0", "-1", "501"])
def test_list_rejects_unbounded_limits(database, limit):
    with pytest.raises(SystemExit):
        main(["--db", str(database), "list", "--limit", limit])
