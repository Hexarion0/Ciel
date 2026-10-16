"""Compare lexical and real local CPU semantic recall using synthetic labeled memories.

Run after separately installing an embedding model into local Ollama:
    python scripts/benchmark_memory_recall.py --model your-installed-small-model

The benchmark never reads personal memories, downloads models, or enables the feature.
Its JSON report includes measured top-one accuracy, latency, and quality-gate outcome.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from ciel.memory.evaluation import evaluate_semantic_recall, quality_gate_passed


def main(argv: list[str] | None = None) -> int:
    """Print a reviewable report; exit nonzero when measured quality has not improved."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="An already installed local embedding model")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--output", type=Path, help="Also save the JSON report to this path")
    args = parser.parse_args(argv)
    try:
        report: dict[str, Any] = evaluate_semantic_recall(
            model=args.model, base_url=args.base_url, timeout=args.timeout,
        )
    except ValueError as exc:
        parser.error(str(exc))
    report["quality_gate_passed"] = quality_gate_passed(report, args.model)
    report["automatically_enabled"] = False
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["quality_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
