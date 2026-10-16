"""Semantic recall requires completed local evaluation, measured gains, and no regressions."""

from dataclasses import replace

import pytest

from ciel.memory import MemoryStore
from ciel.memory.cli import main
from ciel.memory.evaluation import DATASET_VERSION, evaluate_semantic_recall, quality_gate_passed
from ciel.memory.semantic import SemanticIndexReport


def passing_report():
    """Represent real evaluator output without pretending mock vectors prove model quality."""
    return {
        "model": "test-embed:latest", "dataset_version": DATASET_VERSION,
        "query_count": 17, "index_complete": True,
        "lexical_hits": 7, "hybrid_hits": 10, "lexical_regressions": 0,
        "lexical_false_positives": 0, "hybrid_false_positives": 0,
    }


@pytest.mark.parametrize("change", [
    {"index_complete": False}, {"hybrid_hits": 7}, {"hybrid_hits": 6},
    {"lexical_regressions": 1}, {"hybrid_false_positives": 1},
    {"dataset_version": 0}, {"query_count": 1}, {"model": "different-model:latest"},
    {"lexical_hits": True}, {"hybrid_hits": 999},
])
def test_quality_gate_rejects_failed_equal_regressed_stale_or_malformed_reports(change):
    assert not quality_gate_passed({**passing_report(), **change}, "test-embed")


def test_quality_gate_accepts_strict_improvement_for_the_tested_model_only():
    assert quality_gate_passed(passing_report(), "test-embed")
    assert not quality_gate_passed(passing_report(), "another")
    assert not quality_gate_passed(None, "test-embed")


def test_failed_actual_index_does_not_produce_a_passing_evaluation(monkeypatch):
    monkeypatch.setattr(
        "ciel.memory.semantic.LocalSemanticRecall.reindex",
        lambda self: SemanticIndexReport(failed=12, error="Model is unavailable"),
    )
    report = evaluate_semantic_recall(model="test-embed")
    assert not report["index_complete"]
    assert not report["quality_improved"]
    assert not quality_gate_passed(report, "test-embed")


def test_cli_records_only_passing_quality_and_revokes_a_later_failure(
    tmp_path, monkeypatch, capsys,
):
    database = tmp_path / "memory.db"
    report = passing_report()
    monkeypatch.setattr("ciel.memory.evaluation.evaluate_semantic_recall", lambda **kwargs: report)
    command = ["--db", str(database), "semantic-evaluate", "--model", "test-embed"]
    assert main(command) == 0
    store = MemoryStore(database)
    assert store.get_setting("semantic_quality_report") == report
    store.close()
    report = {**report, "hybrid_hits": 7}
    assert main(command) == 1
    store = MemoryStore(database)
    assert store.get_setting("semantic_quality_report") is None
    store.close()
    capsys.readouterr()


def test_semantic_index_failure_is_explicit_and_never_enables_recall(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "ciel.memory.semantic.LocalSemanticRecall.reindex",
        lambda self: replace(SemanticIndexReport(), failed=1, error="Model is unavailable"),
    )
    database = tmp_path / "memory.db"
    assert main(["--db", str(database), "semantic-index", "--model", "test-embed"]) == 1
    assert "Model is unavailable" in capsys.readouterr().out
    store = MemoryStore(database)
    assert store.get_setting("semantic_quality_report") is None
    store.close()
