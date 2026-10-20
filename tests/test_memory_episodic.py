"""Older user dialogue is searchable context, scoped to the owner's active channels."""

from datetime import datetime, timezone

from ciel.config import Settings
from ciel.dialogue import DialogueContext
from ciel.memory import ConversationManager, ConversationTurn, MemoryItem, MemoryStore
from ciel.memory.episodic import recall_episodes
from ciel.memory.service import MemoryService


def test_old_topic_is_recalled_beyond_short_term_window_without_promoting_it_to_fact(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    settings = Settings(_env_file=None, memory_db_path=str(tmp_path / "memory.db"))
    conversation = ConversationManager(store=store, session_id="desktop", max_turns=4)
    memory = MemoryService(store)
    context = DialogueContext(settings, store, conversation, memory)
    try:
        conversation.add_turn("user", "I planned to test the nebula emulator with a tiny ROM.")
        for index in range(40):
            conversation.add_turn("user", f"Unrelated conversation number {index}.")
        recalled = context.recall("What did I say about the nebula emulator?")
        assert any("tiny ROM" in entry for entry in recalled)
        assert any("historical" in entry.lower() and "not confirmed" in entry.lower()
                   for entry in recalled)
        assert store.count_memories() == 0
    finally:
        store.close()


def test_archive_scope_excludes_other_owner_and_assistant_claims_and_keeps_provenance(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        store.save_turn(ConversationTurn(
            "user", "I chose the silver falcon keyboard.", session_id="telegram:7:42:v1",
            timestamp=datetime(2025, 3, 1, tzinfo=timezone.utc),
        ))
        store.save_turn(ConversationTurn(
            "user", "Private falcon secret from someone else.", session_id="telegram:7:99:v1",
        ))
        store.save_turn(ConversationTurn(
            "assistant", "You bought a golden falcon keyboard.", session_id="telegram:7:42:v1",
        ))
        recalled = recall_episodes(store, "falcon keyboard", session_ids=["telegram:7:42:v1"])
        assert len(recalled) == 1
        assert "silver falcon" in recalled[0]
        assert "2025-03-01" in recalled[0] and '"channel": "telegram"' in recalled[0]
        assert "secret" not in str(recalled) and "golden" not in str(recalled)
    finally:
        store.close()


def test_archive_suppresses_replaced_and_forgotten_values(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    memory = MemoryService(store)
    try:
        memory.handle("Remember my favorite game is Stand of Two")
        store.save_turn(ConversationTurn("user", "My favorite game is Stand of Two.", "voice"))
        memory.handle("Update my favorite game to Standoff 2")
        memory.handle("Yes, update it")
        assert recall_episodes(store, "favorite game", session_ids=["voice"]) == []
        store.save_turn(ConversationTurn("user", "I play Standoff 2 after lunch.", "voice"))
        memory.handle("Forget my favorite game")
        assert recall_episodes(store, "Standoff 2 favorite game", session_ids=["voice"]) == []
    finally:
        store.close()


def test_index_finds_old_records_and_refreshes_after_edit_and_delete(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        old_id = store.save_turn(ConversationTurn("user", "My nebula engine uses Alpha.", "voice"))
        for index in range(400):
            store.save_turn(ConversationTurn("user", f"Ordinary filler {index}.", "voice"))
        assert "Alpha" in str(recall_episodes(store, "nebula engine", session_ids=["voice"]))
        with store._lock, store._get_connection() as connection:
            connection.execute("UPDATE conversation_turns SET content=? WHERE id=?",
                               ("My orbital engine uses Beta.", old_id))
        assert recall_episodes(store, "nebula", session_ids=["voice"]) == []
        assert "Beta" in str(recall_episodes(store, "orbital", session_ids=["voice"]))
        store.clear_session("voice")
        assert recall_episodes(store, "orbital", session_ids=["voice"]) == []
    finally:
        store.close()


def test_archive_obeys_pause_and_budget_and_ignores_known_secret_tokens(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    settings = Settings(_env_file=None, memory_recall_enabled=False)
    conversation = ConversationManager(store=store, session_id="voice")
    try:
        store.save_turn(ConversationTurn(
            "user", "Nebula notes with nvapi-PRIVATE1234567890", "voice",
        ))
        store.save_turn(ConversationTurn(
            "user", "Nebula notes with 123456:PRIVATE_TELEGRAM_TOKEN_EXAMPLE", "voice",
        ))
        store.save_turn(ConversationTurn("user", "Nebula notes with a clean example.", "voice"))
        assert "PRIVATE" not in str(recall_episodes(store, "Nebula", session_ids=["voice"]))
        assert recall_episodes(store, "Nebula", session_ids=["voice"], max_chars=10) == []
        context = DialogueContext(settings, store, conversation, MemoryService(store))
        assert context.recall("Nebula") == []
        store.set_setting("recall_enabled", False)
        assert recall_episodes(store, "Nebula", session_ids=["voice"]) == []
    finally:
        store.close()


def test_current_fact_retains_priority_over_historical_dialogue(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    settings = Settings(_env_file=None)
    conversation = ConversationManager(store=store, session_id="voice", max_turns=2)
    try:
        store.save_memory(MemoryItem(
            "Your current project is Ciel.", metadata={
                "scope": "user", "fact_key": "current_project", "value": "Ciel",
            },
        ))
        conversation.add_turn("user", "I was considering a nebula engine project.")
        for index in range(10):
            conversation.add_turn("user", f"Unrelated message {index}.")
        context = DialogueContext(settings, store, conversation, MemoryService(store))
        recalled = context.recall("Tell me about the nebula engine project")
        assert "current project is Ciel" in recalled[0]
        assert any("considering a nebula" in entry for entry in recalled[1:])
        assert sum(map(len, recalled)) <= settings.memory.recall_max_chars
    finally:
        store.close()


def test_filtered_newer_matches_do_not_hide_an_older_valid_episode(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    memory = MemoryService(store)
    try:
        memory.handle("Remember my favorite game is Obsolete Game")
        store.save_turn(ConversationTurn("user", "Nebula project uses a tiny ROM.", "voice"))
        for _index in range(20):
            store.save_turn(ConversationTurn(
                "user", "Nebula project involves Obsolete Game.", "voice",
            ))
        memory.handle("Update my favorite game to Portal 2")
        memory.handle("Yes, update it")
        recalled = recall_episodes(store, "nebula project", session_ids=["voice"])
        assert len(recalled) == 1 and "tiny ROM" in recalled[0]
    finally:
        store.close()


def test_long_episode_snippet_contains_matching_passage_and_stays_bounded(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        store.save_turn(ConversationTurn(
            "user", "Unrelated introduction. " * 40 + "Nebula engine uses a tiny ROM.", "voice",
        ))
        recalled = recall_episodes(store, "nebula engine", session_ids=["voice"], max_chars=900)
        assert len(recalled) == 1 and "Nebula engine uses a tiny ROM" in recalled[0]
        assert sum(map(len, recalled)) <= 900
    finally:
        store.close()


def test_old_topic_recall_keeps_the_users_adjacent_decision(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        store.save_turn(ConversationTurn(
            "user", "I want to build a tiny Nebula emulator.", "voice",
        ))
        store.save_turn(ConversationTurn("assistant", "Parser or execution first?", "voice"))
        store.save_turn(ConversationTurn(
            "user", "I'll start with the instruction parser.", "voice",
        ))
        for index in range(40):
            store.save_turn(ConversationTurn("user", f"Unrelated topic {index}.", "voice"))
        recalled = recall_episodes(store, "Nebula emulator decision", session_ids=["voice"])
        assert "instruction parser" in str(recalled)
        assert "Parser or execution first" not in str(recalled)
    finally:
        store.close()


def test_browsing_preference_bridges_processes_and_clears_only_its_origin(tmp_path):
    from ciel.memory.bridge import ConversationContext

    path = tmp_path / "memory.db"
    settings = Settings(_env_file=None, memory_db_path=str(path))
    voice_store, chat_store = MemoryStore(path), MemoryStore(path)
    def shared():
        return ConversationContext("shared-summary", ("voice", "telegram:7:42:v1"))

    voice = ConversationManager(store=voice_store, session_id="voice", context_resolver=shared)
    chat = ConversationManager(store=chat_store, session_id="telegram:7:42:v1",
                               context_resolver=shared)
    voice_context = DialogueContext(settings, voice_store, voice, MemoryService(voice_store))
    chat_context = DialogueContext(settings, chat_store, chat, MemoryService(chat_store))
    try:
        voice_context.pause_web(True)
        assert chat_context.web_paused()
        chat_context.pause_web(False)
        assert not voice_context.web_paused()
        chat_context.clear_web_preference()
        assert chat_context.web_paused()
        isolated = ConversationManager(store=chat_store, session_id="telegram:7:99:v1")
        other = DialogueContext(settings, chat_store, isolated, MemoryService(chat_store))
        assert not other.web_paused()
    finally:
        voice_store.close()
        chat_store.close()


def test_binding_change_during_recall_discards_old_context(tmp_path, monkeypatch):
    from ciel.memory.bridge import ConversationContext

    store = MemoryStore(tmp_path / "memory.db")
    selected = ["telegram:7:42:v1"]

    def context_selection():
        return ConversationContext("shared-summary", tuple(selected))

    conversation = ConversationManager(
        store=store, session_id="voice", context_resolver=context_selection,
    )
    try:
        store.save_turn(ConversationTurn("user", "My nebula project used a tiny ROM.", selected[0]))
        original = conversation.get_recent_turns

        def change_binding(*args, **kwargs):
            recent = original(*args, **kwargs)
            selected[:] = ["telegram:7:99:v1"]
            return recent

        monkeypatch.setattr(conversation, "get_recent_turns", change_binding)
        context = DialogueContext(
            Settings(_env_file=None), store, conversation, MemoryService(store),
        )
        assert context.recall("What about that nebula project?") == []
    finally:
        store.close()
