"""Local owner memory controls; only explicit consolidation may contact a provider."""

import argparse
import json
import os
import shlex
import shutil
import sqlite3
import stat
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from ciel.config import get_settings
from ciel.memory.models import MemoryItem, MemoryType
from ciel.memory.projects import list_project_memories
from ciel.memory.recall import rank_memories
from ciel.memory.store import MemoryStore
from ciel.memory.tools import MemoryTools
from ciel.memory.workspace_search import WorkspaceMemorySearch

if TYPE_CHECKING:
    from ciel.config import Settings
    from ciel.memory.manager import ConversationManager


def _positive_limit(value: str) -> int:
    """Keep terminal listings bounded even when a flag is supplied by a script."""
    number = int(value)
    if not 1 <= number <= 500:
        raise argparse.ArgumentTypeError("limit must be between 1 and 500")
    return number


def _positive_line(value: str) -> int:
    """Accept a bounded source-line address rather than an unbounded read."""
    number = int(value)
    if not 1 <= number <= 1_000_000:
        raise argparse.ArgumentTypeError("line must be between 1 and 1000000")
    return number


def _action_index(value: str) -> int:
    """Use the same bounded, zero-based indexes shown in an owner review."""
    number = int(value)
    if not 0 <= number < 16:
        raise argparse.ArgumentTypeError("action index must be between 0 and 15")
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


def _no_symlink_components(path: Path) -> None:
    """Reject existing links before selecting a source or destination namespace."""
    for component in (*reversed(path.parents), path):
        if component.is_symlink():
            raise ValueError("Memory backup paths cannot contain symlinks")


def _backup_sources(root: Path) -> list[Path]:
    """Inventory authoritative memory only, rejecting links and special files."""
    files = []
    for directory, names, filenames in os.walk(root, followlinks=False):
        for name in [*names, *filenames]:
            source = Path(directory) / name
            mode = source.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError("Memory workspace backups cannot contain symlinks")
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError("Memory workspace backups require regular files")
        for name in filenames:
            relative = (Path(directory) / name).relative_to(root)
            parts = relative.parts
            if (
                relative.as_posix() in {"MEMORY.md", "USER.md", "DREAMS.md", ".ciel-memory.json"}
                or parts[0] == "memory" and relative.suffix == ".md"
                and not any(part.startswith(".") for part in parts)
                or parts[:2] == (".memory-state", "preimages") and relative.suffix == ".md"
            ):
                files.append(relative)
    return sorted(files)


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    """Detect replacement or edits even when a file retains its size and modification time."""
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _copy_backup_file(source: Path, destination: Path) -> None:
    """Copy one regular source privately, detecting a concurrent manual edit."""
    missing = []
    parent = destination.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    reader = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(reader)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Memory backup source must be a regular file")
        writer = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(writer, "wb") as target, os.fdopen(reader, "rb", closefd=False) as origin:
            shutil.copyfileobj(origin, target)
            target.flush()
            os.fsync(target.fileno())
        after = os.fstat(reader)

        if (_file_identity(before) != _file_identity(after)
                or _file_identity(after) != _file_identity(source.lstat())):
            raise ValueError("A memory source changed during backup; retry the backup")
    finally:
        os.close(reader)


def _backup(store: MemoryStore, destination: Path) -> None:
    """Snapshot SQLite and sibling Markdown together, with no overwrites or derived cache."""
    _no_symlink_components(destination.expanduser().absolute())
    destination = Path(os.path.abspath(destination.expanduser()))
    workspace_destination = destination.with_suffix(".workspace")
    if destination == workspace_destination:
        raise ValueError("Backup database and Markdown directory need distinct paths")
    _no_symlink_components(destination)
    _no_symlink_components(workspace_destination)
    workspace = store.workspace
    if not store._is_in_memory:
        _no_symlink_components(Path(store.db_path_str).expanduser().absolute())
    source_root = Path(os.path.abspath(workspace.root)) if workspace is not None else None
    if workspace is not None and (
        destination.is_relative_to(source_root)
        or workspace_destination.is_relative_to(source_root)
    ):
        raise ValueError("A backup destination cannot be inside the source memory workspace")
    reserved = [destination, workspace_destination, *[
        Path(str(destination) + suffix) for suffix in ("-journal", "-wal", "-shm")
    ]]
    if any(path.exists() or path.is_symlink() for path in reserved):
        raise FileExistsError("A backup destination or its sibling workspace already exists")
    created_database = False
    created_workspace = False
    try:
        with workspace.locked() if workspace is not None else nullcontext():
            store.sync_workspace()
            sources = _backup_sources(workspace.root) if workspace is not None else []
            identities = {
                relative: _file_identity((workspace.root / relative).lstat())
                for relative in sources
            }
            descriptor = os.open(
                destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600,
            )
            os.close(descriptor)
            created_database = True
            if workspace is not None:
                workspace_destination.mkdir(mode=0o700)
                created_workspace = True
            with store._lock, sqlite3.connect(destination) as target:
                store._get_connection().backup(target)
                if workspace is not None:
                    for relative in sources:
                        _copy_backup_file(
                            workspace.root / relative, workspace_destination / relative,
                        )
                    if _backup_sources(workspace.root) != sources or any(
                        _file_identity((workspace.root / relative).lstat()) != identity
                        for relative, identity in identities.items()
                    ):
                        raise ValueError("A memory source changed during backup; retry the backup")
        print(f"Memory backup saved to {destination}")
        if created_workspace:
            print(f"Markdown memory saved to {workspace_destination}")
    except BaseException:
        try:
            if created_workspace:
                shutil.rmtree(workspace_destination)
        finally:
            if created_database:
                for path in [destination, *[Path(str(destination) + suffix)
                                           for suffix in ("-journal", "-wal", "-shm")]]:
                    path.unlink(missing_ok=True)
        raise


def _search_sources(
    store: MemoryStore, tools: MemoryTools, searcher: WorkspaceMemorySearch | None,
    query: str, limit: int, as_json: bool,
    conversation: "ConversationManager | None" = None,
) -> None:
    """Keep legacy fact records first while returning real Markdown source excerpts too."""
    items = [item for _, item in rank_memories(store, query, limit=limit)]
    results = [item.to_dict() for item in items]
    if searcher is not None:
        hits = searcher.search(query, max_results=min(limit, 100), local_only=True)
        if conversation is not None:
            try:
                conversation.lifecycle.record_recall(
                    [hit for hit in hits if not hit.local_only], query,
                )
            except (OSError, sqlite3.Error, ValueError):
                # A missing/reset owner binding cannot take explicit source inspection away.
                pass
        for hit in hits:
            result = hit.to_dict()
            result["content"] = tools._clean(result["snippet"])
            result["snippet"] = result["content"]
            results.append(result)
    if as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        if items:
            _print_items(items, False)
        sources = [item for item in results if "path" in item]
        if not items and not sources:
            print("No matching saved memories.")
        for item in sources:
            print(f"{item['path']}:{item['start_line']}-{item['end_line']}")
            print("  " + item["snippet"].replace("\n", "\n  "))


def _owner_conversation(
    store: MemoryStore, settings: "Settings", *, bridge_owner: bool = True,
) -> "ConversationManager":
    """Bind maintenance to the current owner projection, never every archived session."""
    from ciel.memory.bridge import (
        ConversationContext,
        _active_desktop_id,
        conversation_context,
    )
    from ciel.memory.manager import ConversationManager

    def owner_context() -> ConversationContext:
        current = _active_desktop_id(settings, store)
        if current is None:
            raise ValueError("The active owner conversation binding is unavailable")
        if not bridge_owner:
            return ConversationContext(current, (current,))
        return conversation_context(settings, current, store=store)

    session = owner_context().session_ids[0]
    return ConversationManager(
        store, session_id=session, memory_config=settings.memory,
        context_resolver=owner_context,
    )


def _print_reviews(
    reviews: list[dict], as_json: bool, command_prefix: str = "ciel memory",
) -> None:
    """Expose proposed edits and their original owner quotes before confirmation."""
    if as_json:
        print(json.dumps(reviews, ensure_ascii=False, indent=2))
    elif not reviews:
        print("No memory changes await owner review.")
    else:
        for review in reviews:
            print(f"Review {review['run_id']} ({review['created_at']})")
            for action in review["actions"]:
                print(f"  [{action['index']}] {action['action']}: {action['text']}")
                if action["target"]:
                    print(f"    Existing content: {action['target']}")
                for source in action["evidence"]:
                    print(f"    Owner quote ({source['id']}): {source['quote']}")
                for source in action["sources"]:
                    print(f"    Source: {source}")
            print(f"  Apply all: {command_prefix} apply {review['run_id']}")
            print("  Apply selected: add --action INDEX for each approved change.")


def main(argv: list[str] | None = None) -> int:
    """Run explicit owner controls against the configured database."""
    parser = argparse.ArgumentParser(prog="ciel memory", description=__doc__)
    parser.add_argument("--db", help="Database path (defaults to MEMORY_DB_PATH)")
    parser.add_argument(
        "--workspace", help="Markdown workspace path (isolated beside --db by default)",
    )
    parser.add_argument("--json", action="store_true", help="Print inspection results as JSON")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("stats", help="Show counts and saved-memory recall status")
    commands.add_parser("status", help="Show Markdown memory, index, and lifecycle status")
    index = commands.add_parser("index", help="Reconcile the local Markdown keyword index")
    index.add_argument("--force", action="store_true", help="Rebuild all derived search entries")
    commands.add_parser("reset-index", help="Clear derived search data while preserving all notes")
    get = commands.add_parser("get", help="Read a bounded range from a real memory file")
    get.add_argument("path")
    get.add_argument("--from", dest="from_line", type=_positive_line, default=1)
    get.add_argument("--lines", type=_positive_limit, default=100)
    commands.add_parser("flush", help="Preserve useful staged owner context in dated notes")
    commands.add_parser("consolidate", help="Reflect on owner notes using the configured AI route")
    review = commands.add_parser(
        "review", help="Inspect pending proposals and original owner quotes",
    )
    review.add_argument("run_id", nargs="?", help="Limit inspection to one reflection run")
    apply = commands.add_parser(
        "apply", help="Confirm pending memory changes without an AI request",
    )
    apply.add_argument("run_id", help="Reflection run shown by review")
    apply.add_argument(
        "--action", dest="action_indexes", type=_action_index, action="append", metavar="INDEX",
        help="Approve one zero-based action index; repeat for several (default: all pending)",
    )
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
    backup = commands.add_parser(
        "backup", help="Back up SQLite and sibling Markdown memory together",
    )
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
    owner_command = ["ciel", "memory"]
    if args.db:
        owner_command.extend(["--db", args.db])
    if args.workspace:
        owner_command.extend(["--workspace", args.workspace])
    command_prefix = shlex.join(owner_command)
    owner_settings = get_settings()
    settings = owner_settings.memory
    store: MemoryStore | None = None
    searcher: WorkspaceMemorySearch | None = None
    conversation = None
    try:
        database_path = args.db or settings.db_path
        if args.command == "backup" and database_path != ":memory:":
            # Reject an alias before opening it can migrate or modify the source database.
            _no_symlink_components(Path(database_path).expanduser().absolute())
        workspace_path = args.workspace or (None if args.db else settings.workspace_path or None)
        store = MemoryStore(
            database_path, workspace_path=workspace_path,
            workspace_enabled=settings.workspace_enabled,
        )
        tools = MemoryTools(store, settings)
        if args.command in {"status", "index", "reset-index", "get", "search"}:
            # Owner inspection stays available while automated recall is paused.
            # Keyword indexing never creates an embedding client or a provider request.
            searcher = (
                WorkspaceMemorySearch(store.workspace) if store.workspace is not None else None
            )
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
        elif args.command == "status":
            result = {
                "workspace_enabled": store.workspace is not None,
                "workspace": str(store.workspace.root) if store.workspace is not None else None,
                "index": searcher.status() if searcher is not None else None,
                "daily_capture_enabled": settings.daily_capture_enabled,
                "flush_enabled": settings.flush_enabled,
                "consolidation_enabled": settings.consolidation_enabled,
                "consolidation_time": settings.consolidation_time,
                "learning_mode": settings.learning_mode,
                "recall_enabled": store.get_setting("recall_enabled", settings.recall_enabled),
            }
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            elif store.workspace is None:
                print("Markdown memory is disabled.")
            else:
                print(f"Markdown memory: {store.workspace.root}")
                print(f"Search: {result['index']['documents']} documents, "
                      f"{result['index']['chunks']} chunks ({result['index']['mode']})")
                print(f"Consolidation: {settings.consolidation_time} local time "
                      + ("(enabled)" if settings.consolidation_enabled else "(disabled)"))
        elif args.command in {"index", "reset-index", "get"}:
            if searcher is None:
                print("Markdown memory is disabled.")
                return 1
            if args.command == "get":
                result = searcher.get(
                    args.path, from_line=args.from_line, lines=args.lines,
                    max_chars=8000, local_only=True,
                ).to_dict()
                result["text"] = tools._clean(result["text"])
                if args.json:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    print(f"{result['path']}:{result['from_line']}-{result['to_line']}")
                    print(result["text"])
                    if result["truncated"]:
                        print("[Excerpt truncated; request a later line range to continue.]")
            elif args.command == "index":
                report = searcher.sync(force=args.force)
                result = report.to_dict()
                print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else
                      f"Indexed {report.indexed} changed documents; {report.unchanged} unchanged, "
                      f"{report.removed} removed, {report.chunks} chunks.")
                return 1 if report.error else 0
            else:
                searcher.reset_index()
                result = searcher.status()
                print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else
                      "Derived memory index reset. Markdown notes are kept.")
        elif args.command in {"flush", "consolidate", "review", "apply"}:
            conversation = _owner_conversation(store, owner_settings, bridge_owner=args.db is None)
            if args.command == "flush":
                count = conversation.flush_memory()
                result = {"flushed": count}
                print(json.dumps(result, indent=2) if args.json else
                      f"Preserved {count} new daily working notes.")
            elif args.command == "consolidate":
                from ciel.providers import get_model_router

                result = asdict(conversation.consolidate_memory(get_model_router()))
                print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else
                      f"Memory reflection: {result['status']}; {result['promoted']} promoted, "
                      f"{result['review']} awaiting owner review.")
                if result["review"] and not args.json:
                    print(f"Inspect proposals: {command_prefix} review {result['run_id']}")
                return 0 if result["status"] in {
                    "complete", "empty", "duplicate", "ineligible",
                } else 1
            elif args.command == "review":
                reviews = conversation.list_memory_reviews()
                if args.run_id:
                    reviews = [item for item in reviews if item["run_id"] == args.run_id]
                _print_reviews(reviews, args.json, command_prefix)
                return 0 if reviews or not args.run_id else 1
            else:
                result = asdict(conversation.apply_memory_review(args.run_id, args.action_indexes))
                print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else
                      f"Owner review: {result['status']}; {result['promoted']} applied, "
                      f"{result['review']} awaiting owner review.")
                return 0 if result["status"] in {"complete", "empty", "duplicate"} else 1
        elif args.command == "list":
            kind = MemoryType(args.type) if args.type else None
            items = store.search_memories(
                "", memory_type=kind, limit=args.limit, exclude_types=(MemoryType.CONVERSATION,),
            )
            _print_items(items, args.json)
        elif args.command == "search":
            try:
                conversation = _owner_conversation(
                    store, owner_settings, bridge_owner=args.db is None,
                )
            except (OSError, sqlite3.Error, ValueError):
                # Notes remain readable even when a current owner cannot authorize reinforcement.
                pass
            _search_sources(
                store, tools, searcher, " ".join(args.query), args.limit, args.json, conversation,
            )
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
        if conversation is not None:
            if args.command == "search":
                conversation.lifecycle.close()
            else:
                conversation.close()
        if searcher is not None:
            searcher.close()
        if store is not None:
            store.close()
