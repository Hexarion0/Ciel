"""Research continues across channels from observations, without a global time limit."""

import json
import threading
from unittest.mock import Mock

from ciel.chat.session import ChatSession
from ciel.config import ControllerConfig, InternetConfig, Settings
from ciel.controller import AIController, ControllerDecision
from ciel.internet import InternetResult, InternetSource
from ciel.memory import ConversationManager, MemoryStore
from ciel.providers.base import ChatMessage, LLMResponse, ToolCall
from ciel.research import ResearchCancellation, evidence_payload, previous_observations
from tests.test_chat_session import RecordingRouter
from tests.test_controller import _web_model_responses


def test_chat_reuses_actual_website_observations_after_restart_without_fetching(tmp_path):
    settings = Settings(_env_file=None, memory_db_path=str(tmp_path / "facts.db"),
                        ciel_persona_file="", internet_enabled=True)
    quote = "RETRO.MN runs public Counter-Strike 2 community game servers."
    lookup = InternetResult(attempted=True, query="retro.mn", sources=[
        InternetSource("RETRO.MN — CS2 Network", "https://retro.mn/", content=quote),
    ])
    router = RecordingRouter()
    router.answer = json.dumps({"claims": [
        {"text": quote, "quote": quote, "source": 1},
    ]})
    router.tool_calls.append(ToolCall("retro-page", "read_page", '{"url": "https://retro.mn/"}'))
    chat = ChatSession(settings, channel="telegram", session_id="owner", router=router)
    chat.internet.lookup = Mock(return_value=lookup)
    assert "Counter-Strike 2" in chat.reply("whats retro.mn")
    chat.internet.lookup.assert_called_once()
    chat.close()
    reopened = ChatSession(settings, channel="telegram", session_id="owner", router=router)
    reopened.internet.prepare = Mock(side_effect=AssertionError("No fresh search requested"))
    reopened.internet.lookup = Mock(side_effect=AssertionError("No fresh fetch requested"))
    try:
        assert "Counter-Strike 2" in reopened.reply("tell me what u seen")
        report_call = next(messages for messages in reversed(router.messages)
                           if not messages[0].content.startswith("WEB GOAL CHECK"))
        assert quote in "\n".join(m.content for m in report_call)
        assert "previously retrieved observations" in str(report_call)
    finally:
        reopened.close()


def test_failed_lookup_does_not_replay_an_older_successful_topic(tmp_path):
    store = MemoryStore(tmp_path / "facts.db")
    conversation = ConversationManager(store)
    conversation.add_turn("assistant", "Old topic", web_evidence=evidence_payload(
        InternetResult(attempted=True, query="old topic", sources=[
            InternetSource("Old", "https://example.com/old", content="Old evidence here."),
        ]),
    ))
    conversation.add_turn("assistant", "Nothing found", web_evidence=evidence_payload(
        InternetResult(attempted=True, query="new failed topic"),
    ))
    try:
        result = previous_observations("tell me what u seen", conversation)
        assert result.query == "new failed topic"
        assert "didn't retrieve usable evidence" in result.reply
        assert not result.sources
    finally:
        store.close()


def test_unrelated_topic_breaks_implicit_research_continuation(tmp_path):
    store = MemoryStore(tmp_path / "facts.db")
    conversation = ConversationManager(store)
    conversation.add_turn("assistant", "A search result", web_evidence=evidence_payload(
        InternetResult(attempted=True, query="retro.mn", sources=[
            InternetSource("Retro", "https://retro.mn", content="Counter-Strike network."),
        ]),
    ))
    conversation.add_turn("user", "Let's talk about my cat instead.")
    conversation.add_turn("assistant", "Tell me about your cat.")
    try:
        assert previous_observations("Tell me more", conversation) is None
    finally:
        store.close()


def test_research_can_progress_beyond_the_old_three_step_limit():
    router = Mock()
    internet = Mock()
    internet.config = InternetConfig()
    controller = AIController(router, ControllerConfig(max_tool_steps=3))
    quote = "The example project supports private text chat."
    internet.lookup.side_effect = [InternetResult(attempted=True, sources=[
        InternetSource(f"Guide {i}", f"https://example.org/{i}", content=(
            quote if i == 5 else f"The example project has reference page {i}."
        )),
    ]) for i in range(1, 6)]
    replies = []
    for i in range(1, 5):
        replies.extend([
            LLMResponse('{"claims": []}', "model", "provider"),
            LLMResponse(json.dumps({"action": "search", "confidence": 1,
                                    "query": f"example project reference {i + 1}"}),
                        "model", "provider"),
        ])
    replies.append(LLMResponse(json.dumps({"claims": [
        {"text": quote, "quote": quote, "source": 1},
    ]}), "model", "provider"))
    router.send.side_effect = _web_model_responses(replies)
    result = controller.run_web(
        ControllerDecision(action="search", confidence=1, query="example project reference 1"),
        "Search the example project private chat support",
        [ChatMessage("system", "Helpful"), ChatMessage("user", "Search the example project")],
        internet, lambda _: None, lambda: True,
    )
    assert result.text == quote
    assert internet.lookup.call_count == 5


def test_other_channel_pause_cancels_an_active_research_event():
    parent, allowed = threading.Event(), [True]
    cancel = ResearchCancellation(parent, lambda: allowed[0])
    assert not cancel.is_set()
    allowed[0] = False
    cancel._next_check = 0
    assert cancel.is_set()
    assert not parent.is_set()


def test_failed_translation_check_keeps_independently_supported_english_findings():
    english = "The example project supports private text chat."
    mongolian = "Улаанбаатар хотод шинэ номын сан нээгдлээ."
    lookup = InternetResult(attempted=True, query="example project", sources=[
        InternetSource("Example project", "https://example.org/project", content=english),
        InternetSource("Library", "https://example.mn/library", content=mongolian),
    ])
    router, internet = Mock(), Mock()
    internet.config = InternetConfig()
    router.send.side_effect = _web_model_responses([LLMResponse(json.dumps({"claims": [
        {"text": english, "quote": english, "source": 1},
        {"text": "A new library opened in Ulaanbaatar.", "quote": mongolian, "source": 2},
    ]}), "model", "provider"), TimeoutError("Translation service stalled")], relevant_claims=(1, 2))
    report = AIController(router, ControllerConfig()).run_web(
        ControllerDecision(action="search", query="example project", confidence=1),
        "Tell me what you found", [ChatMessage("system", "Be helpful."),
                                  ChatMessage("user", "Tell me what you found")],
        internet, lambda _: None, lambda: True, initial_lookup=lookup, evidence_only=True,
        channel="text",
    )
    assert english in report.text
    assert "new library opened" not in report.text
    assert "remaining details" in report.text
    internet.lookup.assert_not_called()


def test_reused_observations_do_not_receive_a_new_retrieval_time(tmp_path):
    store = MemoryStore(tmp_path / "facts.db")
    conversation = ConversationManager(store)
    original = InternetResult(attempted=True, query="retro.mn",
                              retrieved_at="2026-01-01T00:00:00+00:00", sources=[
        InternetSource("Retro", "https://retro.mn", content="Counter-Strike community network."),
    ])
    conversation.add_turn("assistant", "A website", web_evidence=evidence_payload(original))
    try:
        reused = previous_observations("Tell me what you saw", conversation)
        assert evidence_payload(reused)["retrieved_at"] == original.retrieved_at
    finally:
        store.close()
