"""Conversation context shared by desktop speech and headless text sessions."""

import re
import time
from collections.abc import Callable
from typing import Any

from ciel.config import Settings
from ciel.memory import context as memory_context
from ciel.memory.episodic import recall_episodes
from ciel.memory.manager import ConversationManager
from ciel.memory.read import answer_memory_read
from ciel.memory.service import MemoryService
from ciel.memory.store import MemoryStore
from ciel.providers.base import ChatMessage


class DialogueContext:
    """Build current-fact context without coupling conversation to a channel's I/O."""

    def __init__(
        self, settings: Settings, store: MemoryStore, conversation: ConversationManager,
        memory: MemoryService, *, semantic_recall: Any = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.conversation = conversation
        self.memory = memory
        self.semantic_recall = semantic_recall

    def pause_web(self, paused: bool) -> None:
        """Record an explicit browsing control for the owner's cross-channel conversation."""
        self.store.set_setting(f"web_pause:{self.conversation.session_id}", {
            "paused": bool(paused), "updated_at": time.time_ns(),
        })

    def web_paused(self) -> bool:
        """Read the latest explicit browsing control from the currently authorized channels."""
        choices = []
        for session in self.conversation.context_session_ids():
            value = self.store.get_setting(f"web_pause:{session}")
            if (isinstance(value, dict) and type(value.get("paused")) is bool
                    and type(value.get("updated_at")) is int):
                choices.append(value)
        return max(choices, key=lambda item: item["updated_at"])["paused"] if choices else False

    def clear_web_preference(self) -> None:
        """Clear only this channel's browsing control when its conversation is reset."""
        self.store.set_setting(f"web_pause:{self.conversation.session_id}", None)

    def recall(self, query: str) -> list[str]:
        """Recall dated, current facts with the existing follow-up and city context."""
        if self.store.get_setting("recall_enabled", self.settings.memory.recall_enabled) is False:
            return []
        sessions = self.conversation.context_session_ids()
        recent_turns = self.conversation.get_recent_turns(limit=8)
        recent = [
            ChatMessage(turn.role, turn.content)
            for turn in recent_turns
        ]
        location_hint = query
        if re.match(
            r"^(?:what about|how about|and\b|also\b|tomorrow\b|next week\b|"
            r"try again|retry|check again|search again|double[- ]check)\b", query, re.I,
        ):
            location_hint += " " + next((
                message.content for message in reversed(recent)
                if message.role == "user" and message.content.strip() != query.strip()
            ), "")
        if re.search(
            r"\b(?:weather|forecast|near me|nearby|around here|local restaurants?)\b",
            location_hint, re.I,
        ):
            query += "\nhome city live based location"
        facts = memory_context.recall_context_memories(
            self.store, query, recent_messages=recent,
            enabled=self.settings.memory.recall_enabled,
            limit=self.settings.memory.recall_limit,
            max_chars=self.settings.memory.recall_max_chars,
            semantic_recall=self.semantic_recall,
        )
        episodes = recall_episodes(
            self.store, memory_context.contextual_recall_query(query, recent),
            session_ids=sessions, exclude_ids=(turn.id for turn in recent_turns),
            max_chars=min(2000, self.settings.memory.recall_max_chars
                          - sum(len(entry) for entry in facts)),
        )
        if sessions != self.conversation.context_session_ids():
            return []
        if self.store.get_setting("recall_enabled", self.settings.memory.recall_enabled) is False:
            return []
        return [*facts, *episodes]

    def system_prompt(
        self, recalled: list[str], *, generate: Callable[..., str],
    ) -> str:
        """Supply persona, current preferred name, and truthful memory-write status."""
        profile_name = (
            self.store.get_fact("user:preferred_name") if self.memory.recall_enabled else None
        )
        settings = self.settings
        if profile_name is not None:
            settings = settings.model_copy(update={
                "ciel_preferred_name": self.store.redact_forgotten(
                    str(profile_name.metadata.get("value", "")),
                ),
            })
        prompt = generate(settings=settings, memories=recalled)
        prompt += (
            "\nApplication memory status: saved-fact recall is "
            + ("enabled" if self.memory.recall_enabled else "paused")
            + f"; learning mode={settings.memory.learning_mode}; "
            + ("conflicting replacements require confirmation."
               if settings.memory.confirm_corrections else
               "clear supported corrections follow the configured learning mode.")
        )
        return prompt + (
            "\nNo durable memory write was performed for this turn. This conversational "
            "response cannot save, replace, or forget a fact. Do not claim a new memory "
            "update succeeded or promise that these words will save it. If a requested "
            "memory change was not handled, ask for the specific fact and its new value. "
            "Earlier application acknowledgements describe earlier saves only."
        )

    def memory_reply(
        self, text: str, *, recent_messages: list[ChatMessage] | None = None,
        authorized: bool = True,
    ) -> str | None:
        """Resolve local reads and guarded writes before any provider-generated reply."""
        reply = answer_memory_read(
            self.store, text, recent_messages=recent_messages,
            recall_enabled=self.settings.memory.recall_enabled,
        )
        if reply is not None:
            self.memory.cancel_proposal()
            return reply
        return self.memory.handle(text, authorized=authorized)
