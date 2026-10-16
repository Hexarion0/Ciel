"""Contracts for trusted, locally installed assistant actions."""

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class ActionError(Exception):
    """An expected action failure safe to describe to the user."""


@dataclass
class ActionContext:
    """Cooperative deadline; handlers must bound blocking I/O by remaining time."""

    cancel_event: threading.Event | None = None
    deadline: float = field(default_factory=lambda: time.monotonic() + 3.0)

    def remaining(self) -> float:
        """Reject stale work before I/O or a side effect and return its time budget."""
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ActionError("That action was canceled.")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ActionError("That action timed out.")
        return remaining


class Action(ABC):
    """One trusted action; conversational text is never executable code."""

    name: str
    description: str
    mutating: bool = False
    requires_confirmation: bool = False

    @abstractmethod
    def match(self, text: str) -> dict[str, str] | None:
        """Recognize one complete command, returning only its arguments."""

    def match_with_context(
        self, text: str, recent_requests: list[str],
    ) -> dict[str, str] | None:
        """Optionally resolve a reference from recent accepted user turns, never permission."""
        return self.match(text)

    @abstractmethod
    def validate(self, arguments: dict[str, str]) -> None:
        """Raise ActionError for missing, extra, or invalid arguments."""

    @abstractmethod
    def execute(self, arguments: dict[str, str], context: ActionContext) -> str:
        """Perform the action within the cooperative deadline."""
