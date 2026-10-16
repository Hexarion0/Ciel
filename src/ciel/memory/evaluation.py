"""Evaluate optional local semantic recall on synthetic memories before enabling it."""

import statistics
import time
from datetime import datetime, timezone
from typing import Any

from ciel.memory.models import MemoryItem
from ciel.memory.recall import rank_memories
from ciel.memory.semantic import LocalSemanticRecall
from ciel.memory.store import MemoryStore

DATASET_VERSION = 1
_MEMORIES = (
    "I keep my telescope on the balcony.",
    "I avoid dairy because I am lactose intolerant.",
    "Project Atlas must run without an internet connection.",
    "My laptop has sixteen gigabytes of RAM.",
    "The next step for Project Atlas is testing restart recovery.",
    "We abandoned the hosted database plan for Project Atlas.",
    "I prefer brief answers with minimal detail.",
    "I take the train to work.",
    "My bicycle is stored in the garage.",
    "I am practicing the piano for a recital.",
    "My cooking class is on Tuesday.",
    "My preferred language is Mongolian.",
)
_CASES: tuple[tuple[str, int | None], ...] = (
    ("Where do I store my stargazing equipment?", 0),
    ("Which food ingredient upsets my digestion?", 1),
    ("Does Atlas need to function offline?", 2),
    ("How much system memory does my notebook have?", 3),
    ("What remains unfinished for Atlas?", 4),
    ("Which cloud storage approach did we reject?", 5),
    ("How verbose should your replies be?", 6),
    ("How do I commute to the office?", 7),
    ("Where is my bike parked?", 8),
    ("Which musical instrument am I rehearsing?", 9),
    ("Where is my telescope?", 0),
    ("What day is my cooking class?", 10),
    ("What is my preferred language?", 11),
    ("What is Atlas restart recovery next step?", 4),
    ("Explain quantum vacuum fluctuations", None),
    ("Translate the word volcano into Italian", None),
    ("Describe the history of Byzantine mosaics", None),
)


def quality_gate_passed(report: Any, model: str) -> bool:
    """Require real completed evaluation, strict improvement, and no lexical regression."""
    if not isinstance(report, dict) or not model:
        return False
    identity = model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"
    required_counts = (
        "lexical_hits", "hybrid_hits", "lexical_regressions",
        "lexical_false_positives", "hybrid_false_positives",
    )
    if any(type(report.get(key)) is not int for key in required_counts):
        return False
    return bool(
        report.get("model") == identity
        and report.get("dataset_version") == DATASET_VERSION
        and report.get("query_count") == len(_CASES)
        and report.get("index_complete") is True
        and 0 <= report["lexical_hits"] < report["hybrid_hits"] <= len(_CASES)
        and report["lexical_regressions"] == 0
        and 0 <= report["hybrid_false_positives"] <= report["lexical_false_positives"]
    )


def evaluate_semantic_recall(
    *, model: str, base_url: str = "http://127.0.0.1:11434", timeout: float = 1.5,
) -> dict[str, Any]:
    """Compare top-one accuracy and latency using synthetic data and the actual local model.

    Real personal data never enters the benchmark. Failed indexing does not yield approval.
    This small acceptance set measures this feature, not general model quality.
    """
    store = MemoryStore(":memory:")
    try:
        ids = [store.save_memory(MemoryItem(content=content)) for content in _MEMORIES]
        backend = LocalSemanticRecall(store, model=model, base_url=base_url, timeout=timeout)
        indexed = backend.reindex()
        result: dict[str, Any] = {
            "model": backend.model, "dataset_version": DATASET_VERSION,
            "tested_at": datetime.now(timezone.utc).isoformat(),
            "query_count": len(_CASES), "index_complete": False,
            "lexical_hits": 0, "hybrid_hits": 0, "lexical_regressions": 0,
            "lexical_false_positives": 0, "hybrid_false_positives": 0,
            "quality_improved": False,
        }
        if indexed.error or indexed.failed or indexed.indexed + indexed.unchanged != len(ids):
            result["error"] = indexed.error or "The synthetic index was incomplete"
            return result
        result["index_complete"] = True
        lexical_ms: list[float] = []
        hybrid_ms: list[float] = []
        for query, expected_index in _CASES:
            expected_id = ids[expected_index] if expected_index is not None else None
            started = time.perf_counter()
            lexical = rank_memories(store, query, limit=4)
            lexical_ms.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            hybrid = backend.hybrid_rank(query, lexical, limit=4)
            hybrid_ms.append((time.perf_counter() - started) * 1000)
            if backend._unavailable_until > time.monotonic():
                result["index_complete"] = False
                result["error"] = "Local embeddings became unavailable during evaluation"
                return result
            lexical_id = lexical[0][1].id if lexical else None
            hybrid_id = hybrid[0][1].id if hybrid else None
            lexical_hit, hybrid_hit = lexical_id == expected_id, hybrid_id == expected_id
            result["lexical_hits"] += int(lexical_hit)
            result["hybrid_hits"] += int(hybrid_hit)
            result["lexical_regressions"] += int(lexical_hit and not hybrid_hit)
            if expected_id is None:
                result["lexical_false_positives"] += int(lexical_id is not None)
                result["hybrid_false_positives"] += int(hybrid_id is not None)
        result["lexical_accuracy"] = result["lexical_hits"] / len(_CASES)
        result["hybrid_accuracy"] = result["hybrid_hits"] / len(_CASES)
        result["lexical_median_ms"] = statistics.median(lexical_ms)
        result["hybrid_median_ms"] = statistics.median(hybrid_ms)
        result["quality_improved"] = quality_gate_passed(result, backend.model)
        return result
    finally:
        store.close()
