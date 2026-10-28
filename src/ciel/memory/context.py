"""Build dated, bounded memory context without replaying archived conversation summaries."""

import re
import unicodedata
from datetime import date, datetime

from ciel.memory.models import MemoryItem, MemoryType
from ciel.memory.recall import SemanticRecall, rank_memories, terms
from ciel.memory.store import MemoryStore
from ciel.providers.base import ChatMessage

PROFILE_KEYS = (
    "user:preferred_name", "user:preferred_language", "user:response_style",
    "user:current_project", "user:goal", "user:occupation",
)
FOLLOWUP = re.compile(
    r"^(?:continue|go on|tell me more|more details|what about (?:that|it|this)|"
    r"how (?:about|long)|(?:give me )?more (?:details|information)|"
    r"explain (?:more|further)|elaborate|keep going|and\b|also\b|but\b|so\b)|"
    r"\b(?:that|this|it|those|them)\b", re.I,
)
REFERENCE_TERMS = set(
    "continue go more detail details long take takes last same still again it this that those them "
    "one ones thing things else next also but so explain work works mean means much cost give "
    "another with further information elaborate keep going".split()
)


def contextual_recall_query(query: str, recent_messages: list[ChatMessage]) -> str:
    """Resolve short references from user dialogue, without borrowing assistant claims."""
    reference = re.sub(r"^(?:can|could|would)\s+you\s+", "", query.strip(), flags=re.I)
    if (
        len(query.split()) > 12 or not FOLLOWUP.search(reference)
        or terms(query) - REFERENCE_TERMS
    ):
        return query
    for message in reversed(recent_messages[-8:]):
        if message.role != "user" or message.content.strip() == query.strip():
            continue
        previous = message.content[:500].strip()
        if previous and terms(previous) - REFERENCE_TERMS:
            return f"{query}\n{previous}"
    return query


def recall_context_memories(
    store: MemoryStore, query: str = "", now: datetime | None = None,
    *, recent_messages: list[ChatMessage] | None = None,
    enabled: bool = True, limit: int = 4, max_chars: int = 6000,
    semantic_recall: SemanticRecall | None = None, local_only: bool = False,
) -> list[str]:
    """Recall ranked facts plus bounded profile anchors, unless recall has been paused.

    ``limit`` bounds query matches; confirmed profile and project anchors are added separately.
    Pausing recall preserves stored facts and conversation history.
    """
    if limit <= 0 or max_chars <= 0 or store.get_setting("recall_enabled", enabled) is False:
        return []
    current_date = (now or datetime.now()).date()
    recalled: list[MemoryItem] = []
    if query:
        contextual_query = contextual_recall_query(query, recent_messages or [])
        recalled.extend(item for _, item in rank_memories(
            store, contextual_query, limit=limit, semantic_recall=semantic_recall,
        ))
    # A small confirmed profile supplies continuity even for informal conversation.
    for key in PROFILE_KEYS:
        item = store.get_fact(key)
        if item is not None:
            recalled.append(item)
    # Project anchors must survive informal questions that don't match their exact wording.
    recalled.extend(store.list_memories(memory_type=MemoryType.PROJECT, limit=3))
    start = store.get_fact("ciel:project_start_date")
    if start is not None:
        recalled.append(start)
    recalled.extend(store.list_memories(memory_type=MemoryType.PREFERENCE, limit=3))

    context: list[str] = []
    seen: set[int | str] = set()
    seen_keys: set[str] = set()
    seen_content: set[str] = set()
    for memory in recalled:
        fact_key = memory.metadata.get("fact_key")
        identity_key = None
        if isinstance(fact_key, str) and fact_key:
            scope = memory.metadata.get("project") or memory.metadata.get("scope", "user")
            identity_key = f"{scope}:{fact_key}"
            # Legacy duplicate rows can coexist with the authoritative keyed record.
            # Always use its current value rather than resurfacing an obsolete version.
            current = store.get_fact(identity_key)
            if current is not None:
                memory = current
            if identity_key in seen_keys:
                continue
        if memory.metadata.get("local_only") and not local_only:
            continue
        identity = memory.id if memory.id is not None else memory.content
        if identity in seen:
            continue
        seen.add(identity)
        if identity_key:
            seen_keys.add(identity_key)
        recorded_at = memory.metadata.get("observed_at") or (
            memory.created_at.astimezone().isoformat(timespec="seconds")
        )
        content = store.redact_forgotten(memory.content)[:600]
        normalized_content = " ".join(unicodedata.normalize("NFKC", content).casefold().split())
        normalized_content = normalized_content.rstrip(".!?")
        if normalized_content and normalized_content not in seen_content:
            seen_content.add(normalized_content)
            context.append(
                f"Recorded at {str(recorded_at)[:80]}: {content} "
                f"[type={memory.memory_type.value}; source={str(memory.source)[:80]}]"
            )
        if (
            memory.memory_type == MemoryType.PROJECT
            and memory.metadata.get("project") == "ciel"
            and memory.metadata.get("fact_key") == "project_start_date"
        ):
            try:
                start_date = date.fromisoformat(str(memory.metadata["value"]))
            except (KeyError, ValueError):
                continue
            if store.redact_forgotten(start_date.isoformat()) != start_date.isoformat():
                continue
            elapsed = (current_date - start_date).days
            if elapsed >= 0:
                context.append(
                    f"CIEL timeline as of {current_date.isoformat()}: confirmed project "
                    f"start date {start_date.isoformat()}; {elapsed} elapsed calendar days; "
                    f"development day {elapsed + 1} counting the start date as day one. "
                    "Use this dated calculation instead of old relative day-count statements."
                )
    bounded: list[str] = []
    size = 0
    for entry in context:
        if size + len(entry) > max_chars:
            continue
        bounded.append(entry)
        size += len(entry)
    if store.get_setting("recall_enabled", enabled) is False:
        return []
    return bounded
