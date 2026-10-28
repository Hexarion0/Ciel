"""Owner-scoped Markdown recall reaches the same user-facing voice/chat context."""

import json
import threading
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ciel.chat.session import ChatSession
from ciel.config import Settings
from ciel.controller import AIController, ControllerDecision
from ciel.dialogue import DialogueContext
from ciel.internet import InternetResult
from ciel.memory import ConversationManager, ConversationTurn, MemoryItem, MemoryStore, MemoryType
from ciel.memory.bridge import (
    ConversationContext,
    desktop_conversation_id,
    start_desktop_conversation,
)
from ciel.memory.service import MemoryService
from ciel.memory.tools import MemoryTools, memory_lookup_request
from ciel.providers.base import ChatMessage, LLMResponse, ToolCall


class RecordingRouter:
    """Configured model double; no audio, credentials, embedding model, or network."""

    def __init__(self):
        self.calls = []
        self.decisions = []
        self.native_calls = []
        self.answer = "The saved launch code is cobalt."

    def send(self, messages, **kwargs):
        self.calls.append((list(messages), dict(kwargs)))
        if kwargs.get("purpose") == "tool_progress":
            return LLMResponse("{}", "fixture", "fixture")
        if kwargs.get("purpose") == "controller":
            payload = self.decisions.pop(0) if self.decisions else {
                "action": "answer", "confidence": 1,
            }
            return LLMResponse(json.dumps(payload), "fixture", "fixture")
        if kwargs.get("tools") and self.native_calls:
            return LLMResponse("", "fixture", "fixture",
                               tool_calls=(self.native_calls.pop(0),))
        return LLMResponse(self.answer, "fixture", "fixture")


@pytest.fixture
def configured(tmp_path):
    settings = Settings(
        _env_file=None, memory_db_path=str(tmp_path / "facts.sqlite3"),
        memory_workspace_path=str(tmp_path / "workspace"), ciel_persona_file="",
        internet_enabled=False, memory_semantic_enabled=False,
        memory_daily_capture_enabled=False, memory_flush_enabled=False,
        memory_consolidation_enabled=False,
    )
    store = MemoryStore(settings.memory.db_path, workspace_path=settings.memory.workspace_path)
    scope = ["desktop:test-owner:run:current"]
    conversation = ConversationManager(
        store=store, session_id=scope[0], memory_config=settings.memory,
        context_resolver=lambda: ConversationContext("fixture-summary", tuple(scope)),
    )
    tools = MemoryTools(store, settings.memory, conversation)
    yield SimpleNamespace(settings=settings, store=store, conversation=conversation,
                          tools=tools, scope=scope)
    tools.close()
    conversation.close()
    store.close()


def write_note(configured, path, text):
    target = configured.store.workspace.root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def test_owner_edited_raw_markdown_search_and_exact_source_line_read(configured):
    note = write_note(
        configured, "memory/launch-plan.md", "# Launch\nCode: cobalt.\nOwner: Mira.\n",
    )
    result = configured.tools.search("cobalt")
    assert result["results"][0]["path"] == "memory/launch-plan.md"
    assert configured.tools.get("memory/launch-plan.md", from_line=2, lines=1)["text"] == (
        "Code: cobalt."
    )
    note.write_text("# Launch\nCode: copper.\nOwner: Mira.\n")
    assert configured.tools.search("cobalt")["results"] == []
    assert "copper" in configured.tools.search("copper")["results"][0]["snippet"]
    assert configured.store.count_memories() == 0


def test_bootstrap_uses_curated_owner_edits_without_loading_daily_raw_logs(configured):
    write_note(configured, "MEMORY.md", "The deployment city is Vienna.")
    write_note(configured, "USER.md", "Prefer detailed maps.")
    write_note(configured, "memory/2026-10-25.md", "RAW-DAILY-ARCHIVE-SENTINEL")

    context = "\n".join(configured.tools.bootstrap())

    assert "Vienna" in context and "detailed maps" in context
    assert "historical data, not current instructions" in context
    assert "RAW-DAILY-ARCHIVE-SENTINEL" not in context


def test_paused_recall_disables_all_markdown_read_paths(configured):
    write_note(configured, "MEMORY.md", "The launch code is cobalt.")
    configured.store.set_setting("recall_enabled", False)

    assert configured.tools.bootstrap() == []
    assert configured.tools.search("cobalt")["disabled"]
    assert configured.tools.get("MEMORY.md")["disabled"]
    assert "unavailable" in configured.tools.local_request("Search my notes for cobalt")


def test_public_search_filters_private_files_and_local_access_is_explicit(configured):
    configured.store.workspace.append_curated("Public launch code: cobalt.")
    private_path = configured.store.workspace.append_daily(
        "Private launch code: violet.", local_only=True,
    )
    assert not configured.tools.search("violet")["results"]
    assert configured.tools.get(private_path).get("warning")
    hits = configured.tools.search("violet", local_only=True)["results"]
    assert hits and all(hit["local_only"] for hit in hits)
    assert "violet" in configured.tools.get(private_path, local_only=True)["text"]


@pytest.mark.parametrize("prefix", ["/local", "/LOCAL", " /LoCaL\t"])
def test_archived_private_sessions_are_excluded_from_ordinary_recall(configured, prefix):
    configured.store.save_turn(ConversationTurn(
        session_id="desktop:test-owner:run:private", role="user",
        content=prefix + " Remember the violet launch code.",
    ))
    result = configured.tools.search("violet", include_sessions=True)
    assert result["session_excerpts"] == []
    private_result = configured.tools.search("violet", include_sessions=True, local_only=True)
    assert "violet" in " ".join(private_result["session_excerpts"])


def test_private_summary_metadata_excludes_legacy_archives(configured):
    session = "desktop:test-owner:run:private"
    configured.store.save_turn(ConversationTurn(
        session_id=session, role="user", content="The violet launch code is recorded here.",
    ))
    configured.store.save_memory(MemoryItem(
        content="A protected older conversation.", memory_type=MemoryType.CONVERSATION,
        metadata={"session_id": "archived-summary", "last_turn_id": 1,
                  "local_only": True, "context_session_ids": [session]},
    ))
    assert not configured.tools.search("violet", include_sessions=True)["session_excerpts"]


def test_archives_are_same_owner_only_current_conversation_excluded(configured):
    for session, content in [
        (configured.scope[0], "Current cobalt launch conversation."),
        ("desktop:test-owner:run:previous", "Earlier cobalt launch decision."),
        ("desktop:other-owner:run:previous", "Foreign cobalt launch decision."),
        ("telegram:77:88", "Telegram cobalt launch decision."),
    ]:
        configured.store.save_turn(ConversationTurn(
            session_id=session, role="user", content=content,
        ))

    recalled = " ".join(configured.tools.search(
        "cobalt", include_sessions=True,
    )["session_excerpts"])

    assert "Earlier cobalt" in recalled
    assert "Current cobalt" not in recalled
    assert "Foreign cobalt" not in recalled
    assert "Telegram cobalt" not in recalled


def test_all_telegram_scope_cannot_read_desktop_workspace_or_archives(configured):
    configured.scope[:] = ["telegram:77:88:run:current"]
    write_note(configured, "MEMORY.md", "Desktop cobalt launch decision.")
    configured.store.save_turn(ConversationTurn(
        session_id="telegram:77:88:run:previous", role="user", content="Old cobalt code.",
    ))
    assert configured.tools.bootstrap() == []
    assert configured.tools.search("cobalt", include_sessions=True)["disabled"]
    assert configured.tools.get("MEMORY.md")["disabled"]


def test_scope_change_during_search_discards_every_excerpt(configured, monkeypatch):
    write_note(configured, "MEMORY.md", "The code is cobalt.")
    index = configured.tools._searcher()
    original_search = index.search

    def change_owner(*args, **kwargs):
        hits = original_search(*args, **kwargs)
        configured.scope[:] = ["desktop:other-owner:run:current"]
        return hits

    monkeypatch.setattr(index, "search", change_owner)
    result = configured.tools.search("cobalt", include_sessions=True)
    assert not result["results"]
    assert "owner" in result["warning"]
    assert "cobalt" not in json.dumps(result)


def test_cancel_during_search_discards_evidence_before_controller(configured, monkeypatch):
    write_note(configured, "MEMORY.md", "The code is cobalt.")
    canceled = threading.Event()
    original_search = configured.tools._searcher().search

    def cancel_search(*args, **kwargs):
        hits = original_search(*args, **kwargs)
        canceled.set()
        return hits

    monkeypatch.setattr(configured.tools._searcher(), "search", cancel_search)
    router = RecordingRouter()
    result = configured.tools.run(
        ControllerDecision(action="memory_search", confidence=1, query="cobalt"),
        "What was the launch code?", [], AIController(router, configured.settings.controller),
        cancel_event=canceled,
    )
    assert result == []
    assert not router.calls


def test_retired_current_callback_never_sends_results_to_controller(configured, monkeypatch):
    write_note(configured, "MEMORY.md", "The code is cobalt.")
    current = [True]
    original_search = configured.tools._searcher().search

    def retire_search(*args, **kwargs):
        hits = original_search(*args, **kwargs)
        current[0] = False
        return hits

    monkeypatch.setattr(configured.tools._searcher(), "search", retire_search)
    router = RecordingRouter()
    result = configured.tools.run(
        ControllerDecision(action="memory_search", confidence=1, query="cobalt"),
        "What was the launch code?", [], AIController(router, configured.settings.controller),
        current=lambda: current[0],
    )
    assert result == []
    assert not router.calls


def test_default_tool_loop_detects_persisted_reset_before_another_model_call(
    configured, monkeypatch,
):
    write_note(configured, "MEMORY.md", "The code is cobalt.")
    router = RecordingRouter()
    original = configured.tools.get

    def reset_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        configured.conversation.clear_session()
        return result

    monkeypatch.setattr(configured.tools, "get", reset_after_read)
    evidence = configured.tools.run(
        ControllerDecision(action="memory_get", confidence=1, memory_path="MEMORY.md"),
        "Read the launch code", [], AIController(router, configured.settings.controller),
    )
    assert evidence == []
    assert not router.calls


@pytest.mark.parametrize("action", ["search", "read_page"])
def test_controller_cannot_send_a_personal_archive_question_to_public_tools(configured, action):
    router = RecordingRouter()
    router.decisions = [{"action": action, "confidence": 1,
                         "query": "https://example.org/private-rollout"}]
    question = "What did we decide for the Cedar rollout?"
    decision = AIController(router, configured.settings.controller).decide(
        question, [], addressed=True, active=True, internet_enabled=True,
    )
    assert decision.action == "memory_search"
    assert decision.query == question
    assert decision.memory_corpus == "all"


def test_controller_can_follow_search_with_exact_read(configured):
    write_note(configured, "memory/launch.md", "# Launch\nCode: cobalt.\nOwner: Mira.")
    router = RecordingRouter()
    router.decisions = [
        {"action": "memory_get", "confidence": 1, "memory_path": "memory/launch.md",
         "memory_from": 2, "memory_lines": 1},
        {"action": "answer", "confidence": 1},
    ]
    evidence = configured.tools.run(
        ControllerDecision(action="memory_search", confidence=1, query="cobalt"),
        "What was the launch code?", [ChatMessage("user", "What was the launch code?")],
        AIController(router, configured.settings.controller),
    )
    assert [item["tool"] for item in evidence] == ["memory_search", "memory_get"]
    assert evidence[1]["result"]["text"] == "Code: cobalt."
    assert all(kwargs["purpose"] == "controller" for _, kwargs in router.calls)


@pytest.mark.parametrize("fields", [
    {"action": "memory_search", "query": ""},
    {"action": "memory_get", "memory_path": ""},
    {"action": "memory_get", "memory_path": "MEMORY.md", "memory_from": 0},
    {"action": "memory_get", "memory_path": "MEMORY.md", "memory_from": True},
    {"action": "memory_get", "memory_path": "MEMORY.md", "memory_lines": 201},
    {"action": "memory_get", "memory_path": "MEMORY.md", "memory_lines": "10"},
    {"action": "memory_search", "query": "cobalt", "memory_corpus": "all-disk-files"},
    {"action": "memory_save", "query": "Invented new memory"},
])
def test_memory_decision_fields_are_strict_and_bounded(fields):
    with pytest.raises(ValidationError):
        ControllerDecision(confidence=1, **fields)


@pytest.mark.parametrize("path", ["../outside.md", "/etc/passwd", "sessions/old.jsonl"])
def test_model_selected_get_cannot_escape_memory_roots(configured, path):
    result = configured.tools.execute(ControllerDecision(
        action="memory_get", confidence=1, memory_path=path,
    ))
    assert not result["text"] and result.get("warning")


def test_dialogue_recall_respects_prompt_budget_after_json_expansion(configured):
    configured.settings.memory_recall_max_chars = 600
    write_note(configured, "MEMORY.md", 'Cobalt "quoted"\n' * 600)
    memory = MemoryService(configured.store)
    context = DialogueContext(
        configured.settings, configured.store, configured.conversation, memory,
    )
    try:
        recalled = context.recall("cobalt")
        assert sum(map(len, recalled)) <= 600
    finally:
        context.tools.close()


def test_bootstrap_serialized_output_stays_inside_its_requested_budget(configured):
    write_note(configured, "MEMORY.md", 'Cobalt "quoted"\n' * 600)
    write_note(configured, "USER.md", 'Always "prefer" detail.\n' * 600)
    result = configured.tools.bootstrap(max_chars=4000)
    assert sum(map(len, result)) <= 4000


def test_controller_memory_evidence_fits_configured_prompt_budget(configured):
    write_note(configured, "memory/launch.md", "Cobalt launch observations. " * 1000)
    router = RecordingRouter()
    evidence = configured.tools.run(
        ControllerDecision(action="memory_get", confidence=1, memory_path="memory/launch.md"),
        "What did we record for launch?", [],
        AIController(router, configured.settings.controller),
    )
    assert evidence
    assert len(json.dumps(evidence, ensure_ascii=False)) <= (
        configured.settings.memory.recall_max_chars
    )
    assert evidence[0]["result"].get("truncated")


def test_chat_explicit_search_reads_arbitrary_markdown_without_structured_fact(configured):
    write_note(configured, "memory/owner-plan.md", "The launch code is cobalt.")
    router = RecordingRouter()
    chat = ChatSession(configured.settings, channel="text", session_id="owner-chat", router=router)
    try:
        assert "cobalt" in chat.reply("Search my notes for the launch code")
        assert not [kwargs for _, kwargs in router.calls
                    if kwargs.get("purpose") != "tool_progress"]
        assert configured.store.count_memories() == 0
    finally:
        chat.close()


def test_chat_recall_reaches_fake_configured_router_with_raw_markdown_evidence(configured):
    write_note(configured, "memory/owner-plan.md", "The launch code is cobalt.")
    router = RecordingRouter()
    chat = ChatSession(configured.settings, channel="text", session_id="owner-chat", router=router)
    try:
        reply = chat.reply("What did we decide for the launch code?")
        final_calls = [messages for messages, kwargs in router.calls
                       if kwargs.get("purpose") == "conversation"]
        assert reply == router.answer
        assert final_calls
        assert "cobalt" in final_calls[-1][0].content
        assert "memory/owner-plan.md" in final_calls[-1][0].content
    finally:
        chat.close()


def test_fresh_launch_never_replays_previous_raw_conversation_automatically(configured):
    legacy = desktop_conversation_id()
    configured.store.save_turn(ConversationTurn(
        session_id=legacy, role="user", content="RAW-OLD-CONVERSATION-SENTINEL cobalt.",
    ))
    current = start_desktop_conversation(configured.store)
    router = RecordingRouter()
    chat = ChatSession(configured.settings, channel="text", session_id=current, router=router)
    try:
        assert chat.reply("Hello, how are you today?") == router.answer
        assert "RAW-OLD-CONVERSATION-SENTINEL" not in repr(router.calls)
        assert "RAW-OLD-CONVERSATION-SENTINEL" in " ".join(
            chat.context.tools.search("cobalt", include_sessions=True)["session_excerpts"],
        )
        recalled = "\n".join(chat.context.recall("What did we decide for the cobalt launch?"))
        assert "RAW-OLD-CONVERSATION-SENTINEL" in recalled
    finally:
        chat.close()


def test_unpaired_telegram_chat_never_receives_owner_profile_or_workspace(configured):
    service = MemoryService(configured.store)
    assert service.handle("Remember my name is PrivateOwnerName")
    write_note(configured, "MEMORY.md", "OWNER-WORKSPACE-SENTINEL cobalt.")
    router = RecordingRouter()
    chat = ChatSession(configured.settings, channel="telegram", session_id="telegram:77:88",
                       router=router)
    try:
        chat.reply("Hello, how are you today?")
        assert "OWNER-WORKSPACE-SENTINEL" not in repr(router.calls)
        assert "PrivateOwnerName" not in repr(router.calls)
        direct = chat.reply("Search my notes for cobalt")
        assert "OWNER-WORKSPACE-SENTINEL" not in direct
        assert "cobalt" not in direct.casefold()
    finally:
        chat.close()


def test_chat_pause_blocks_direct_owner_markdown_lookup(configured):
    write_note(configured, "MEMORY.md", "OWNER-WORKSPACE-SENTINEL cobalt.")
    configured.store.set_setting("recall_enabled", False)
    chat = ChatSession(configured.settings, channel="text", session_id="owner-chat",
                       router=RecordingRouter())
    try:
        answer = chat.reply("Search my notes for cobalt")
        assert "OWNER-WORKSPACE-SENTINEL" not in answer
        assert "unavailable" in answer
    finally:
        chat.close()


def test_chat_explicit_search_honors_cancellation_during_source_lookup(configured, monkeypatch):
    write_note(configured, "memory/owner-plan.md", "The launch code is cobalt.")
    chat = ChatSession(configured.settings, channel="text", session_id="owner-chat",
                       router=RecordingRouter())
    canceled = threading.Event()
    original_search = chat.context.tools._searcher().search

    def cancel_search(*args, **kwargs):
        hits = original_search(*args, **kwargs)
        canceled.set()
        return hits

    monkeypatch.setattr(chat.context.tools._searcher(), "search", cancel_search)
    try:
        assert chat.reply("Search my notes for the launch code", cancel_event=canceled) == ""
        assert not [turn for turn in chat.conversation.get_recent_turns()
                    if turn.role == "assistant"]
    finally:
        chat.close()


@pytest.mark.parametrize("paused", [False, True])
def test_personal_history_question_never_falls_through_to_public_search(
    configured, monkeypatch, paused,
):
    configured.settings.internet_enabled = True
    configured.store.set_setting("recall_enabled", not paused)
    write_note(configured, "memory/owner-plan.md", "The launch code is cobalt.")
    router = RecordingRouter()
    chat = ChatSession(configured.settings, channel="text", session_id="owner-chat", router=router)

    def unexpected_public_search(*args, **kwargs):
        pytest.fail("A private archive question was sent to public search")

    monkeypatch.setattr(chat.internet, "lookup", unexpected_public_search)
    try:
        reply = chat.reply("Can you search my memories for our private launch code?")
        assert reply == router.answer
        final_calls = [messages for messages, kwargs in router.calls
                       if kwargs.get("purpose") == "conversation"]
        assert final_calls
        assert all(not {"web_search", "read_page"} & {
            tool["function"]["name"] for tool in kwargs.get("tools", [])
        } for _, kwargs in router.calls)
        if paused:
            assert "cobalt" not in final_calls[-1][0].content
        else:
            assert "cobalt" in final_calls[-1][0].content
    finally:
        chat.close()


@pytest.mark.parametrize("query", [
    "Remember to search the web for the latest Python release",
    "How does computer memory work?",
    "What is the latest Total Recall remake news?",
    "Search online for good ways to organize my notes.",
    "Could you search the web for research on how AI models use previous conversations?",
])
def test_public_questions_are_not_mistaken_for_personal_archive_lookups(query):
    assert not memory_lookup_request(query)


def test_explicit_public_search_about_notes_still_reaches_the_web_service(configured, monkeypatch):
    configured.settings.internet_enabled = True
    chat = ChatSession(configured.settings, channel="text", session_id="owner-chat",
                       router=RecordingRouter())
    queries = []

    def public_search(query, **kwargs):
        queries.append(query)
        return InternetResult(attempted=True, query=query,
                              reply="Here is public organizing advice.")

    monkeypatch.setattr(chat.internet, "lookup", public_search)
    try:
        query = "Search online for good ways to organize my notes."
        chat.router.native_calls.append(ToolCall(
            "organizing-search", "web_search", json.dumps({"query": query}),
        ))
        assert chat.reply(query) == "Here is public organizing advice."
        assert queries == [query]
    finally:
        chat.close()


def test_reset_during_local_note_read_discards_result_without_assistant_commit(
    configured, monkeypatch,
):
    write_note(configured, "memory/owner-plan.md", "The launch code is cobalt.")
    chat = ChatSession(configured.settings, channel="text", session_id="owner-chat",
                       router=RecordingRouter())
    original = chat.context.memory_reply

    def reset_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        chat.conversation.clear_session()
        return result

    monkeypatch.setattr(chat.context, "memory_reply", reset_after_read)
    try:
        assert chat.reply("Search my notes for the launch code") == ""
        assert chat.conversation.get_recent_turns() == []
    finally:
        chat.close()


@pytest.mark.parametrize(("origins", "trusted"), [
    (("owner",), True), (("agent",), True), (("owner", "agent"), True),
    (("tool",), False), (("owner", "tool"), False),
])
def test_ordinary_context_uses_curated_trusted_sources_only(configured, origins, trusted):
    for index, origin in enumerate(origins):
        configured.store.workspace.append_curated(
            f"CURATED-SENTINEL cobalt launch code {index}.", origin=origin,
        )
    write_note(configured, "memory/2026-10-25.md", "DAILY-SENTINEL cobalt launch code.")
    configured.store.workspace.append_dream("DREAM-SENTINEL cobalt launch code.")
    context = DialogueContext(configured.settings, configured.store, configured.conversation,
                              MemoryService(configured.store))
    try:
        ordinary = "\n".join(context.recall("cobalt launch code"))
        assert ("CURATED-SENTINEL" in ordinary) is trusted
        assert "DAILY-SENTINEL" not in ordinary
        assert "DREAM-SENTINEL" not in ordinary
        explicit = context.tools.search("cobalt launch code")
        assert any("DAILY-SENTINEL" in hit["snippet"] for hit in explicit["results"])
        assert "DREAM-SENTINEL" not in json.dumps(explicit)
    finally:
        context.tools.close()
