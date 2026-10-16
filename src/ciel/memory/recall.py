"""Rank indexed durable memories without excluding older facts from retrieval."""

from typing import Protocol

from ciel.memory.models import MemoryItem
from ciel.memory.recall_terms import ALIASES, GROUPS, STOP_WORDS, terms
from ciel.memory.store import MemoryStore

__all__ = ["ALIASES", "GROUPS", "STOP_WORDS", "rank_memories", "terms"]


class SemanticRecall(Protocol):
    """Optional local supplementary ranking; lexical recall works independently."""

    def hybrid_rank(
        self, query: str, lexical: list[tuple[float, MemoryItem]], *, limit: int = 4,
    ) -> list[tuple[float, MemoryItem]]:
        """Fuse semantic hits with the current durable lexical matches."""
        ...


def rank_memories(
    store: MemoryStore,
    query: str,
    limit: int = 4,
    *, semantic_recall: SemanticRecall | None = None,
) -> list[tuple[float, MemoryItem]]:
    """Rank all durable memories by distinctive shared terms using the persisted term index."""
    if limit <= 0:
        return []
    query_terms = terms(query)
    lexical = store.ranked_recall_candidates(query_terms, limit=limit) if query_terms else []
    if semantic_recall is not None:
        return semantic_recall.hybrid_rank(query, lexical, limit=limit)
    return lexical
