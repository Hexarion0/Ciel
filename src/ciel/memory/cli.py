"""Local memory inspection and maintenance without starting audio or contacting a provider."""

import argparse
import json
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path

from ciel.config import get_settings
from ciel.memory.models import MemoryItem, MemoryType
from ciel.memory.projects import list_project_memories
from ciel.memory.recall import rank_memories
from ciel.memory.store import MemoryStore


def _positive_limit(value: str) -> int:
    """Keep terminal listings bounded even when a flag is supplied by a script."""
    number = int(value)
    if not 1 <= number <= 500:
        raise argparse.ArgumentTypeError("limit must be between 1 and 500")
    return number


def _lookup(store: MemoryStore, reference: str) -> MemoryItem | None:
    """Accept an owner-visible record ID or a stable namespaced fact key."""
    return store.get_memory(int(reference)) if reference.isdecimal() else store.get_fact(reference)


def _print_items(items: list[MemoryItem], as_json: bool) -> None:
    """Print structured exports or readable, dated records."""
    if as_json:
        print(json.dumps([item.to_dict() for item in items], ensure_ascii=False, indent=2))
    elif not items:
        print("No matching saved memories.")
    else:
        for item in items:
            print(f"#{item.id} [{item.memory_type.value}] {item.content}")
            print(f"  Updated {item.updated_at.isoformat()} | source: {item.source}")


def _backup(store: MemoryStore, destination: Path) -> None:
    """Create a consistent owner-readable snapshot, refusing to overwrite any file."""
    destination = destination.expanduser()
    descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        with sqlite3.connect(destination) as target:
            with store._lock:
                store._get_connection().backup(target)
        print(f"Memory backup saved to {destination.resolve()}")
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    """Run explicit owner controls against the configured database."""
    parser = argparse.ArgumentParser(prog="ciel memory", description=__doc__)
    parser.add_argument("--db", help="Database path (defaults to MEMORY_DB_PATH)")
    parser.add_argument("--json", action="store_true", help="Print inspection results as JSON")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("stats", help="Show counts and saved-memory recall status")
    listing = commands.add_parser("list", help="List the most recent durable memories")
    listing.add_argument("--limit", type=_positive_limit, default=50)
    listing.add_argument("--type", choices=[kind.value for kind in MemoryType
                                          if kind != MemoryType.CONVERSATION])
    search = commands.add_parser("search", help="Search durable memories with local ranked recall")
    search.add_argument("query", nargs="+")
    search.add_argument("--limit", type=_positive_limit, default=20)
    project = commands.add_parser("project", help="Inspect the journal for one project")
    project.add_argument("name", nargs="+")
    project.add_argument("--limit", type=_positive_limit, default=50)
    for command in ("show", "history", "forget"):
        subparser = commands.add_parser(command, help=f"{command.capitalize()} a fact by ID or key")
        subparser.add_argument("reference")
    commands.add_parser("pause", help="Pause saved-fact recall; keep data and conversation")
    commands.add_parser("resume", help="Resume saved-fact recall")
    backup = commands.add_parser("backup", help="Create a consistent SQLite backup in a new file")
    backup.add_argument("destination", type=Path)
    clear = commands.add_parser("clear-session", help="Erase one session's dialogue and summary")
    clear.add_argument("session_id")
    for command in ("semantic-index", "semantic-evaluate"):
        semantic = commands.add_parser(
            command, help="Index facts or evaluate a selected local CPU embedding model",
        )
        semantic.add_argument("--model", help="Installed Ollama model (or MEMORY_SEMANTIC_MODEL)")
        semantic.add_argument("--host", help="HTTP loopback Ollama endpoint")
        semantic.add_argument("--timeout", type=float, help="Per-request deadline in seconds")
    args = parser.parse_args(argv)
    settings = get_settings().memory
    store: MemoryStore | None = None
    try:
        store = MemoryStore(args.db or settings.db_path)
        if args.command == "stats":
            counts = {kind.value: store.count_memories(kind) for kind in MemoryType}
            with store._lock:
                row = store._get_connection().execute(
                    "SELECT COUNT(*) AS turns, COUNT(DISTINCT session_id) AS sessions "
                    "FROM conversation_turns;"
                ).fetchone()
            result = {
                "durable_memories": sum(value for kind, value in counts.items()
                                        if kind != MemoryType.CONVERSATION.value),
                "by_type": counts,
                "conversation_turns": row["turns"], "sessions": row["sessions"],
                "recall_enabled": store.get_setting("recall_enabled", settings.recall_enabled),
                "learning_mode": settings.learning_mode,
                "confirm_corrections": settings.confirm_corrections,
            }
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"Saved facts: {result['durable_memories']}")
                print(f"Conversation turns: {row['turns']} in {row['sessions']} sessions")
                print("Saved-memory recall: " + ("enabled" if result["recall_enabled"]
                                                 else "paused"))
                print(f"Learning mode: {settings.learning_mode}")
        elif args.command == "list":
            kind = MemoryType(args.type) if args.type else None
            items = store.search_memories(
                "", memory_type=kind, limit=args.limit, exclude_types=(MemoryType.CONVERSATION,),
            )
            _print_items(items, args.json)
        elif args.command == "search":
            _print_items([item for _, item in rank_memories(
                store, " ".join(args.query), limit=args.limit,
            )], args.json)
        elif args.command == "project":
            _print_items(list_project_memories(
                store, " ".join(args.name), limit=args.limit,
            ), args.json)
        elif args.command in {"show", "history", "forget"}:
            item = _lookup(store, args.reference)
            if item is None or item.memory_type == MemoryType.CONVERSATION:
                print("No saved fact matches that ID or key.")
                return 1
            if args.command == "show":
                _print_items([item], args.json)
            elif args.command == "history":
                result = {
                    "current": item.to_dict(), "revisions": item.metadata.get("revisions", []),
                }
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"Forgot {store.forget_memories([item])} saved fact.")
        elif args.command in {"pause", "resume"}:
            store.set_setting("recall_enabled", args.command == "resume")
            print("Saved-memory recall " + ("paused. Facts are kept." if args.command == "pause"
                                             else "resumed."))
        elif args.command == "backup":
            _backup(store, args.destination)
        elif args.command == "clear-session":
            count = store.clear_session(args.session_id)
            print(f"Erased {count} conversation turns and session summaries. Saved facts are kept.")
        elif args.command in {"semantic-index", "semantic-evaluate"}:
            from ciel.memory.evaluation import evaluate_semantic_recall, quality_gate_passed
            from ciel.memory.semantic import LocalSemanticRecall

            model = args.model if args.model is not None else settings.semantic_model
            host = args.host if args.host is not None else settings.semantic_host
            timeout = (
                args.timeout if args.timeout is not None else settings.semantic_timeout_seconds
            )
            if not model:
                print("Select an installed local embedding model with --model.")
                return 1
            if args.command == "semantic-index":
                report = LocalSemanticRecall(
                    store, model=model, base_url=host, timeout=timeout,
                ).reindex()
                print(json.dumps(asdict(report), indent=2))
                return 1 if report.error or report.failed else 0
            evaluated = evaluate_semantic_recall(model=model, base_url=host, timeout=timeout)
            print(json.dumps(evaluated, indent=2))
            if quality_gate_passed(evaluated, model):
                store.set_setting("semantic_quality_report", evaluated)
                return 0
            # A new failed evaluation invalidates a previous acceptance for this model.
            previous = store.get_setting("semantic_quality_report")
            if isinstance(previous, dict) and previous.get("model") == evaluated.get("model"):
                store.set_setting("semantic_quality_report", None)
            return 1
        return 0
    except (OSError, sqlite3.Error, ValueError) as error:
        print(f"Memory operation failed: {error}")
        return 1
    finally:
        if store is not None:
            store.close()
