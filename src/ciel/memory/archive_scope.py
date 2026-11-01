"""Shared owner archive admission and transitive protected-session exclusion."""

from ciel.memory.store import MemoryStore
from ciel.providers.base import ChatMessage
from ciel.providers.router import ModelRouter


def admitted_archive_sessions(
    store: MemoryStore,
    active_scope: tuple[str, ...],
    *,
    include_private: bool,
) -> tuple[str, ...]:
    """Select older archives using the current exact channel prefixes."""
    if not active_scope:
        return ()
    prefixes = tuple(item.partition(":run:")[0] for item in active_scope)
    with store._lock:
        connection = store._get_connection()
        rows = connection.execute(
            "SELECT DISTINCT session_id FROM conversation_turns ORDER BY session_id;",
        ).fetchall()
        private_sessions = set()
        if not include_private:
            private_sessions.update(
                row["session_id"]
                for row in connection.execute(
                    "SELECT DISTINCT session_id, content FROM conversation_turns "
                    "WHERE role='user' AND instr(lower(content), '/local') > 0;",
                )
                if ModelRouter.requires_local([ChatMessage("user", row["content"])])
            )
            summaries = connection.execute(
                "SELECT metadata_json FROM memories WHERE memory_type='conversation';",
            ).fetchall()
            for row in summaries:
                metadata = store._decode_mapping(row["metadata_json"])
                if metadata.get("local_only"):
                    if isinstance(metadata.get("session_id"), str):
                        private_sessions.add(metadata["session_id"])
                    members = metadata.get("context_session_ids", [])
                    if isinstance(members, list):
                        private_sessions.update(item for item in members if isinstance(item, str))
            # Active routing treats every channel in a projection as private after
            # /local. Preserve that boundary in archives even before a summary exists.
            peers: dict[str, set[str]] = {}
            for row in connection.execute(
                "SELECT summary_session_id,member_session_id FROM memory_session_members",
            ):
                summary, member = row["summary_session_id"], row["member_session_id"]
                peers.setdefault(summary, set()).add(member)
                peers.setdefault(member, set()).add(summary)
            pending = list(private_sessions)
            while pending:
                for peer in peers.get(pending.pop(), ()):
                    if peer not in private_sessions:
                        private_sessions.add(peer)
                        pending.append(peer)
    permitted = [
        row["session_id"]
        for row in rows
        if row["session_id"] not in active_scope
        and any(
            row["session_id"] == prefix or row["session_id"].startswith(prefix + ":run:")
            for prefix in prefixes
        )
        and row["session_id"] not in private_sessions
    ]
    return tuple(permitted)
