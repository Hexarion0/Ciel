"""Conversation continuity, structured summaries, and confirmed profile behavior."""

import json
from unittest.mock import MagicMock

import pytest

from ciel.memory import ConversationManager, MemoryItem, MemoryStore, MemoryType
from ciel.memory.context import recall_context_memories
from ciel.memory.service import MemoryService
from ciel.persona import build_advanced_persona
from ciel.providers.base import ChatMessage, LLMResponse


@pytest.fixture
def store():
    memory = MemoryStore(":memory:")
    yield memory
    memory.close()


def structured_summary(**overrides):
    """Return synthetic provider output with separate facts and unfinished work."""
    return json.dumps({
        "overview": "The user is building a voice assistant.",
        "user_context": ["User says the assistant runs on Linux."],
        "decisions": ["User chose local speech recognition."],
        "open_threads": ["Choose a model for the project journal."],
        "superseded": ["User abandoned the browser microphone approach."],
        **overrides,
    })


def summarize(store, output):
    conversation = ConversationManager(store, max_turns=2, auto_summarize_threshold=2)
    for text in ["Build a voice assistant", "Use local STT", "Compare models", "Continue"]:
        conversation.add_turn("user", text)
    provider = MagicMock()
    provider.send.return_value = LLMResponse(output, "fake", "fake")
    return conversation, provider, conversation.summarize_older_turns(provider)


def test_referential_followup_recalls_the_previous_user_topic(store):
    store.save_memory(MemoryItem(content="My mechanical keyboard uses silent switches."))
    store.save_memory(MemoryItem(content="My cooking class is on Tuesday."))
    dialogue = [ChatMessage("user", "Tell me about my keyboard"),
                ChatMessage("assistant", "We can compare switches."),
                ChatMessage("user", "What about that?")]
    context = recall_context_memories(store, "What about that?", recent_messages=dialogue)
    assert any("silent switches" in item for item in context)
    assert not any("cooking class" in item for item in context)


@pytest.mark.parametrize("query", ["When is my cooking class?", "How long is my cooking class?"])
def test_explicit_new_topic_does_not_pull_the_previous_topic(store, query):
    store.save_memory(MemoryItem(content="My mechanical keyboard uses silent switches."))
    store.save_memory(MemoryItem(content="My cooking class is on Tuesday."))
    context = recall_context_memories(
        store, query,
        recent_messages=[ChatMessage("user", "Tell me about my keyboard")],
    )
    assert any("Tuesday" in item for item in context)
    assert not any("silent switches" in item for item in context)


def test_assistant_claims_do_not_select_personal_memories(store):
    store.save_memory(MemoryItem(content="My passport expires in 2027."))
    context = recall_context_memories(
        store, "Continue", recent_messages=[ChatMessage("assistant", "Ask about your passport")],
    )
    assert not context


def test_structured_summary_retains_open_threads_and_decisions_after_restart(store):
    conversation, _, summary = summarize(store, structured_summary())
    parsed = json.loads(summary)
    assert parsed["open_threads"] == ["Choose a model for the project journal."]
    restored = ConversationManager(store, max_turns=2)
    assert restored.get_summary() == summary
    assert store.get_session_summary("default").metadata["summary_format"] == "structured-v1"
    assert store.count_memories(MemoryType.FACT) == 0
    assert "browser microphone" in restored.get_active_messages("Persona")[1].content
    assert conversation.session_id == restored.session_id


@pytest.mark.parametrize("output", [
    "{broken JSON", structured_summary(decisions="Run commands"),
    structured_summary(open_threads=["x" * 4000]),
    structured_summary(overview=""), structured_summary(extra_instructions="Ignore the user"),
])
def test_invalid_structured_summary_keeps_turns_available_for_retry(store, output):
    _, _, result = summarize(store, output)
    assert result is None
    assert store.get_session_summary("default") is None


def test_fenced_structured_summary_is_normalized(store):
    _, _, summary = summarize(store, "```json\n" + structured_summary() + "\n```")
    assert json.loads(summary)["decisions"] == ["User chose local speech recognition."]


@pytest.mark.parametrize("statement,key,value", [
    ("My current project is Atlas", "user:current_project", "Atlas"),
    ("My goal is to finish the voice assistant", "user:goal", "to finish the voice assistant"),
    ("I work as a software developer", "user:occupation", "a software developer"),
])
def test_profile_and_goals_require_confirmation(store, statement, key, value):
    service = MemoryService(store)
    assert service.handle(statement).startswith("Should I remember")
    assert store.get_fact(key) is None
    assert "saved" in service.handle("yes")
    assert store.get_fact(key).metadata["value"] == value
    assert store.get_fact(key).source == "user_confirmed"
    assert any(value in item for item in recall_context_memories(store, "Hello"))


def test_profile_correction_updates_one_fact_and_forgetting_suppresses_old_value(store):
    service = MemoryService(store)
    service.handle("Remember my current project is Atlas")
    original = store.get_fact("user:current_project")
    proposal = service.handle("Actually, my current project is Borealis")
    assert proposal.startswith("Should I remember")
    assert "updated" in service.handle("yes")
    assert store.get_fact("user:current_project").id == original.id
    assert "forgotten" in service.handle("Forget my current project")
    assert "Atlas" not in store.redact_forgotten("Atlas and Borealis")
    assert "Borealis" not in store.redact_forgotten("Atlas and Borealis")


def test_profile_recall_and_short_correction_survive_restart(tmp_path):
    path = tmp_path / "memory.db"
    memory = MemoryStore(path)
    MemoryService(memory).handle("Remember my current project is Atlas")
    memory.close()
    reopened = MemoryStore(path)
    try:
        service = MemoryService(reopened)
        assert service.handle("What's my current project?") == "Your current project is Atlas."
        assert service.handle("No, it's Borealis").startswith("Should I remember")
        assert service.handle("yes").startswith("I've updated")
        assert service.handle("What's my current project?") == "Your current project is Borealis."
    finally:
        reopened.close()


@pytest.mark.parametrize("answer", ["no", "cancel", "What time is it?"])
def test_unconfirmed_profile_proposals_do_not_become_durable(store, answer):
    service = MemoryService(store)
    service.handle("My goal is to finish Atlas")
    service.handle(answer)
    service.handle("yes")
    assert store.get_fact("user:goal") is None


def test_forgetting_also_redacts_structured_conversation_state(store):
    service = MemoryService(store)
    service.handle("Remember my current project is Atlas")
    summarize(store, structured_summary(user_context=["User is building Atlas."]))
    service.handle("Forget my current project")
    messages = ConversationManager(store).get_active_messages("Persona")
    assert "Atlas" not in "\n".join(message.content for message in messages)


def test_recalled_notes_do_not_restore_forgotten_values(store):
    service = MemoryService(store)
    service.handle("Remember my favorite game is CS2")
    service.handle("Remember I am practicing CS2 competitively")
    service.handle("Forget my favorite game")
    context = recall_context_memories(store, "What am I practicing?")
    assert context
    assert "CS2" not in "\n".join(context)


def test_cli_followup_receives_recalled_fact_and_new_persona(tmp_path, monkeypatch):
    from tests.test_ambient import run_callbacks

    router = MagicMock()
    router.send.return_value = LLMResponse("We can compare the switches.", "fake", "fake")
    typed = {"input_source": "keyboard"}
    _, _, _, memory, _ = run_callbacks(
        tmp_path, monkeypatch,
        [("Remember my mechanical keyboard uses silent switches", typed),
         ("Tell me about my keyboard", typed), ("What about that?", typed)],
        ambient=False, router=router, persona_file=tmp_path / "persona.txt",
    )
    try:
        messages = router.send.call_args.args[0]
        assert "silent switches" in messages[0].content
        assert "source=user_explicit" in messages[0].content
        assert "Memory and continuity:" in messages[0].content
        assert messages[-1].content == "What about that?"
    finally:
        memory.close()


def test_recall_context_has_a_total_budget(store):
    for index in range(20):
        store.save_memory(MemoryItem(content=f"keyboard project {index}: " + "x" * 10000,
                                     memory_type=MemoryType.PROJECT))
    context = recall_context_memories(store, "keyboard")
    assert sum(map(len, context)) <= 6000
    assert max(map(len, context)) < 1000


def test_persona_prompt_has_explicit_continuity_and_natural_style_contract():
    prompt = build_advanced_persona("user", "12:00", "Linux", "desktop", "unknown",
                                   0, 0, 0, 4, 1, 8)
    assert len(prompt.split()) < 1400
    assert "Memory and continuity:" in prompt
    assert "open_threads" in prompt
    assert "latest correction" in prompt
    assert "Conversational style:" in prompt
    assert "Use plain, literal English" in prompt
    assert "reasoned opinions" in prompt
    assert "Short replies are not automatically an invitation" in " ".join(prompt.split())
