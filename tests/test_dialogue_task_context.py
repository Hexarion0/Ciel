"""Current-task projections use only the authorized, unchanged conversation scope."""

from unittest.mock import Mock

import pytest

from ciel.config import Settings
from ciel.dialogue import DialogueContext
from ciel.memory import ConversationManager, MemoryStore
from ciel.memory.service import MemoryService


@pytest.fixture
def context(tmp_path):
    settings = Settings(_env_file=None, memory_db_path=str(tmp_path / "memory.db"),
                        memory_workspace_enabled=False, memory_semantic_enabled=False)
    store = MemoryStore(settings.memory.db_path, workspace_enabled=False)
    conversation = ConversationManager(store, session_id="synthetic-owner")
    result = DialogueContext(settings, store, conversation, MemoryService(store))
    yield result
    result.tools.close()
    conversation.close()
    store.close()


def seed_search(context):
    context.conversation.add_turn("user", "Search Amber Harbor episode 12 light novel continuation")
    context.conversation.add_turn("assistant", "The search provider was unavailable.")


def test_task_projection_retains_subject_and_outcome_without_rewriting_history(context):
    seed_search(context)
    task = context.task_context("Again.")
    assert task.kind == "retry" and task.task_type == "web"
    assert "Amber Harbor" in task.request and "12" in task.request
    assert "light novel continuation" in task.request
    assert len(context.conversation.get_recent_turns()) == 2


def test_current_turn_can_already_be_saved_before_projection(context):
    seed_search(context)
    context.conversation.add_turn("user", "Again.")
    assert context.task_context("Again.").request.count("Amber Harbor") == 1


def test_owner_scope_denial_does_not_project_an_earlier_task(context):
    seed_search(context)
    context.tools._owner_allowed = Mock(return_value=False)
    task = context.task_context("Again.")
    assert task.kind == "new" and task.request == "Again."
    assert task.task_type == "conversation"


def test_reset_during_task_read_drops_the_old_projection(context, monkeypatch):
    seed_search(context)
    recent = context.conversation.get_recent_turns

    def changed(*args, **kwargs):
        turns = recent(*args, **kwargs)
        context.conversation.clear_session()
        return turns

    monkeypatch.setattr(context.conversation, "get_recent_turns", changed)
    task = context.task_context("Again.")
    assert task.kind == "new" and task.request == "Again."


def test_task_prompt_is_interpretation_data_without_action_authority(context):
    seed_search(context)
    prompt = context.system_prompt([], generate=lambda **_kwargs: "Persona", query="Again.")
    assert "Application current task" in prompt
    assert "Amber Harbor" in prompt
    assert "not tool permissions or factual evidence" in prompt
