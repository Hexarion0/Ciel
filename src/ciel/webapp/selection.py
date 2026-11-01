"""Explicit active selection, separate from read-only browser archive views."""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from ciel.memory.bridge import desktop_conversation_id
from ciel.memory.store import MemoryStore
from ciel.webapp.conversations import ConversationCatalog


@dataclass(frozen=True)
class ActiveSelection:
    """A conversation handle and its monotonically increasing activation revision."""

    handle: str
    revision: int


class ConversationSelection:
    """Serialize continuation with the runtime's cancellation and rebind callback."""

    def __init__(
        self,
        store: MemoryStore,
        catalog: ConversationCatalog,
        session_id: str,
        switch: Callable[[str], None],
    ) -> None:
        self.store, self.catalog, self.session_id, self.switch = store, catalog, session_id, switch
        self._lock = threading.RLock()

    def current(self) -> ActiveSelection:
        """Read both persistent selection fields together."""
        identity, revision = self.store.desktop_selection()
        return ActiveSelection(self.catalog.handle(identity or self.session_id), revision)

    def activate(self, handle: str | None, expected_revision: int) -> ActiveSelection:
        """Activate only an admitted desktop projection; never delete stored history."""
        with self._lock:
            if handle is None:
                identity = f"{desktop_conversation_id()}:run:{uuid4().hex}"
            else:
                members = self.catalog.resolve(handle)
                identities = [s for s in members if s.startswith("desktop_session:")]
                if len(identities) != 1 or ":run:" not in identities[0]:
                    raise ValueError("This archived channel cannot be continued on desktop.")
                identity = identities[0]
            revision = self.store.select_desktop_session(identity, expected_revision)
            self.switch(identity)
            self.session_id = identity
            return ActiveSelection(self.catalog.handle(identity), revision)
