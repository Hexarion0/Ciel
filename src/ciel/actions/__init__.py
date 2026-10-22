"""Discover trusted actions shipped inside the CIEL package."""

import importlib
import pkgutil
import re
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable

from ciel.actions.base import Action, ActionContext, ActionError
from ciel.logging import get_logger

logger = get_logger("actions")


class ActionRegistry:
    """Match, validate, authorize, and execute a bounded local action request."""

    def __init__(self, actions: list[Action]) -> None:
        self.actions = {action.name: action for action in actions}
        if len(self.actions) != len(actions):
            raise ValueError("Duplicate action names")
        self._pending: tuple[str, dict[str, str], float] | None = None
        self._recent_requests: deque[tuple[str, float]] = deque(maxlen=3)

    @classmethod
    def discover(cls) -> "ActionRegistry":
        """Load ACTION exports from this installed package, never user speech or paths."""
        actions = []
        for module in pkgutil.iter_modules(__path__):
            if module.name.startswith("_") or module.name == "base":
                continue
            try:
                candidate = getattr(importlib.import_module(f"{__name__}.{module.name}"),
                                    "ACTION", None)
                if isinstance(candidate, Action):
                    actions.append(candidate)
            except Exception:
                logger.warning("Could not load action module %s", module.name)
        return cls(actions)

    def cancel_pending(self) -> None:
        """Clear confirmation when dialogue ends or another task takes over."""
        self._pending = None
        self._recent_requests.clear()

    @property
    def has_pending_confirmation(self) -> bool:
        """Expose pending confirmations without sharing their execution arguments."""
        return self._pending is not None and time.monotonic() <= self._pending[2]

    def observe_request(self, text: str) -> None:
        """Keep bounded RAM-only reference context; this never authorizes execution."""
        self._recent_requests.append((text[:500], time.monotonic()))

    def handle(
        self, text: str, *, authorized: bool, cancel_event: threading.Event | None = None,
        on_start: Callable[[str], bool] | None = None,
    ) -> str | None:
        """Execute an exact local intent; inferred follow-ups cannot mutate the desktop."""
        clean = text.strip().rstrip(".!?")
        clean = re.sub(r"^(?:please\s+)", "", clean, flags=re.I)
        recent = [
            request for request, started in self._recent_requests
            if 0 <= time.monotonic() - started <= 60.0
        ]
        self.observe_request(clean)
        pending, self._pending = self._pending, None
        if pending is not None and authorized and time.monotonic() <= pending[2]:
            if clean.casefold() in {"yes", "yes please", "confirm"}:
                return self.execute(pending[0], pending[1], authorized=True,
                                    confirmed=True, cancel_event=cancel_event, on_start=on_start)
            if clean.casefold() in {"no", "cancel", "no thanks", "never mind"}:
                return "Okay, I canceled that action."
        matches = []
        for action in self.actions.values():
            try:
                arguments = action.match_with_context(clean, recent)
                if arguments is not None:
                    matches.append((action, arguments))
            except Exception:
                logger.warning("Action matcher %s failed", action.name)
        if len(matches) > 1:
            return "That could mean several actions. Please ask for one specific action."
        if not matches:
            return None
        action, arguments = matches[0]
        return self.execute(action.name, arguments, authorized=authorized,
                            cancel_event=cancel_event, on_start=on_start)

    def execute(
        self, name: str, arguments: dict[str, str], *, authorized: bool,
        confirmed: bool = False, cancel_event: threading.Event | None = None,
        on_start: Callable[[str], bool] | None = None,
    ) -> str:
        """Validate every entry path, with explicit authorization for side effects."""
        action = self.actions.get(name)
        if action is None:
            return "That action is not available."
        context = ActionContext(cancel_event=cancel_event)
        try:
            context.remaining()
            if not isinstance(arguments, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in arguments.items()
            ):
                raise ActionError("Action arguments must be named text values.")
            action.validate(arguments)
            if (action.mutating or action.requires_confirmation) and not authorized:
                raise ActionError("Please address Ciel directly to request that action.")
            if action.requires_confirmation and not confirmed:
                self._pending = (name, arguments.copy(), time.monotonic() + 60.0)
                details = ", ".join(f"{key}={value}" for key, value in arguments.items())
                return f"Confirm {name} ({details})? Say yes to proceed or no to cancel."
            context.remaining()
            if on_start is not None and not on_start(name):
                raise ActionError("That action was cancelled.")
            context.remaining()
            return action.execute(arguments, context)
        except ActionError as error:
            return str(error)
        except PermissionError:
            return "I don't have permission to perform that action."
        except (TimeoutError, subprocess.TimeoutExpired):
            return "That action timed out."
        except Exception:
            logger.warning("Action %s failed", name)
            return "I couldn't complete that action."
