"""Text responses can develop an answer while voice keeps its spoken delivery rules."""

import json
from threading import Event
from unittest.mock import Mock, patch

import pytest

from ciel.chat.session import ChatSession
from ciel.config import Settings
from ciel.persona import build_advanced_persona
from ciel.providers.base import LLMResponse, ToolCall


class DepthRouter:
    """Record main responses and expose local reasoning preferences without network access."""

    def __init__(self) -> None:
        self.calls = []
        self.reasoning_mode = "auto"
        self.answer = "Hello."
        self.tool_call = None

    def set_reasoning_mode(self, mode: str) -> None:
        self.reasoning_mode = mode

    def send(self, messages, **kwargs):
        self.calls.append((list(messages), kwargs))
        if self.tool_call is not None:
            call, self.tool_call = self.tool_call, None
            return LLMResponse("", "synthetic-model", "synthetic", tool_calls=(call,))
        return LLMResponse(self.answer, "synthetic-model", "synthetic")


@pytest.fixture
def chat(tmp_path, monkeypatch):
    monkeypatch.setattr("ciel.chat.session.ToolProgress.warm", lambda *_args, **_kwargs: None)
    with patch.dict("os.environ", {}, clear=True):
        settings = Settings(
            _env_file=None, ciel_persona_file="", internet_enabled=False,
            memory_db_path=str(tmp_path / "chat.db"),
            memory_workspace_path=str(tmp_path / "workspace"),
            memory_daily_capture_enabled=False, memory_flush_enabled=False,
            memory_consolidation_enabled=False,
        )
    session = ChatSession(settings, channel="telegram", session_id="synthetic-owner",
                          router=DepthRouter())
    yield session
    session.close()


def persona(**kwargs) -> str:
    return build_advanced_persona(
        "test-login", "12:00", "Linux", "test", "unknown", 0, 0, 0, 4, 1.0, 8.0,
        **kwargs,
    )


def test_voice_keeps_default_spoken_length_and_formatting():
    prompt = persona()
    assert prompt == persona(channel="voice")
    assert "Usually use one to three short spoken sentences;\nexplain more when needed." in prompt
    assert "No thinking tags, internal analysis, markdown decoration, bullet symbols" in prompt
    assert "Keep commands or code brief when explicitly requested." in prompt
    assert "fenced code blocks" not in prompt


def test_text_prompt_adapts_depth_without_a_sentence_limit():
    prompt = persona(channel="text")
    assert "one to three" not in prompt
    assert "spoken sentences" not in prompt
    assert "Match the depth to the user's request" in prompt
    assert "Greetings and simple questions can be brief" in prompt
    assert "explain the reasoning, tradeoffs, examples" in prompt
    assert "Do not impose a sentence limit" in prompt
    assert "fenced code blocks" in prompt
    assert "Provide complete examples or code when requested" in prompt
    assert "with no thinking tags, internal analysis" in prompt
    assert "Never invent a tool result" in prompt


def test_text_preferences_and_capability_authority_still_apply():
    prompt = persona(channel="text", persona_preferences="Keep replies brief unless I ask.")
    assert "Keep replies brief unless I ask." in prompt
    assert "following the user's latest style and length preferences" in prompt
    assert "Style preferences change delivery only" in prompt
    assert "Generated replies cannot save, update, or delete facts" in prompt


def test_chat_passes_larger_budget_and_preserves_complete_explanation(chat):
    explanation = ("A detailed explanation with a concrete example. " * 180).strip()
    chat.router.answer = explanation + "\n\n```python\ndef example():\n    return 42\n```"
    result = chat.reply("Explain this design in detail, with a complete Python example.")
    assert result == chat.router.answer
    assert chat.conversation.get_recent_turns()[-1].content == result
    messages, options = chat.router.calls[-1]
    assert options["max_tokens"] == 4096
    assert options["purpose"] == "conversation"
    assert "one to three" not in messages[0].content
    assert "Provide complete examples or code when requested" in messages[0].content


def test_chat_keeps_response_budget_across_native_tool_continuation(chat, monkeypatch):
    chat.router.tool_call = ToolCall(
        "read-note", "memory_search", json.dumps({"query": "synthetic design note"}),
    )
    monkeypatch.setattr(chat.context.tools, "search", lambda *_args, **_kwargs: {"results": []})
    chat.router.answer = "Here is the explanation of the design."
    assert chat.reply("Explain the design we discussed in detail.") == chat.router.answer
    assert len(chat.router.calls) == 2
    assert all(options["max_tokens"] == 4096 for _, options in chat.router.calls)
    assert any(message.role == "tool" for message in chat.router.calls[-1][0])


@pytest.mark.parametrize("command,mode", [
    ("/reasoning", "auto"), ("/reasoning on", "on"),
    ("/reasoning off", "off"), (" /REASONING\tAUTO ", "auto"),
])
def test_reasoning_commands_are_local_and_do_not_touch_conversation(
    chat, monkeypatch, command, mode,
):
    chat.reply("Remember my favorite game is Portal")
    chat.reply("Update my favorite game to Portal 2")
    assert chat.memory.has_pending_confirmation
    turns = chat.conversation.get_recent_turns()
    facts = chat.store.list_memories()
    blocked = Mock(side_effect=AssertionError("Reasoning commands must stay local"))
    for target, name in (
        (chat, "_refresh_conversation"), (chat.context, "memory_reply"),
        (chat.internet, "lookup"), (chat.screen, "clear"), (chat.router, "send"),
        (chat.memory, "cancel_proposal"),
    ):
        monkeypatch.setattr(target, name, blocked)
    reply = chat.reply(command)
    assert f"Reasoning preference: {mode}" in reply
    assert "supported models" in reply
    assert chat.router.reasoning_mode == mode
    assert chat.conversation.get_recent_turns() == turns
    assert chat.store.list_memories() == facts
    assert chat.memory.has_pending_confirmation
    blocked.assert_not_called()
    # Session cleanup still owns the pending proposal.
    monkeypatch.undo()


@pytest.mark.parametrize("command", ["/reasoning maybe", "/reasoning on now"])
def test_invalid_reasoning_mode_does_not_generate_or_change_preference(chat, command):
    assert "Use /reasoning" in chat.reply(command)
    assert chat.router.reasoning_mode == "auto"
    assert not chat.router.calls
    assert chat.conversation.get_total_turn_count() == 0


def test_reasoning_status_does_not_claim_support_for_unknown_router(chat):
    del chat.router.reasoning_mode
    assert "aren't available" in chat.reply("/reasoning")
    assert "aren't available" in chat.reply("/reasoning on")
    assert not chat.router.calls


def test_canceled_reasoning_control_cannot_change_mode(chat):
    canceled = Event()
    canceled.set()
    assert chat.reply("/reasoning on", cancel_event=canceled) == ""
    assert chat.router.reasoning_mode == "auto"
    assert not chat.router.calls


def test_reasoning_control_is_listed_in_chat_help(chat):
    assert "/reasoning [auto|on|off]" in chat.reply("/help")
