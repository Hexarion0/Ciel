"""Short, varied intent messages before application-authorized tool work."""

from __future__ import annotations

import json
import random
import re
import threading
import time
from collections.abc import Callable

from ciel.internet import _bounded
from ciel.providers.base import ChatMessage

_FALLBACKS = {
    "screen_anime": ("Let me look at your screen to see what you're watching.",
                     "Lemme take a look at the anime on your screen."),
    "screen_error": ("Let me take a closer look at that error.",
                     "Lemme see what's happening on your screen."),
    "screen": ("Lemme take a look.", "Let me check your screen.", "Lemme see."),
    "screenshot": ("Let me get that screenshot for you.", "I'll get that screen capture."),
    "search_news": ("Let me check the latest reporting.", "I'll look through the news."),
    "search": ("Lemme look that up.", "Let me check a few sources.", "I'll check online."),
    "read_page": ("Let me read that page.", "I'll take a look through that page."),
    "open_app": ("I'll open that for you.", "Let me bring that up."),
    "local": ("Let me check that.", "Lemme see."),
}
_PURPOSES = {
    "screen_anime": "see which anime you are watching on your screen",
    "screen_error": "look at the error on your screen",
    "screen": "take a look at your screen to help with your question",
    "screenshot": "get the screenshot you asked for",
    "search_news": "check the latest news stories for you",
    "search": "look online to help answer your question",
    "read_page": "read the web page you asked about",
    "open_app": "open the app you asked for",
    "local": "check the system information you asked for",
}


def purpose_for(tool: str, query: str) -> str:
    """Derive a nonpersonal intent category; original wording never enters the model call."""
    if tool == "screen":
        if re.search(r"\b(?:anime|watching|episode|show|movie)\b", query, re.I):
            return "screen_anime"
        if re.search(r"\b(?:error|crash|broken|warning)\b", query, re.I):
            return "screen_error"
    if tool == "search" and re.search(r"\b(?:news|headlines|reporting)\b", query, re.I):
        return "search_news"
    return tool if tool in _PURPOSES else "local"


def valid_line(value: object) -> str:
    """Progress states intended work; it cannot claim results or introduce another tool."""
    if not isinstance(value, str):
        return ""
    line = " ".join(value.strip().split())
    if (not 3 <= len(line) <= 150 or len(line.split()) > 26
            or any(char in line for char in '{}[]<>')
            or re.search(r"\b(?:requested|API|retrieval|tool)\b|desktop capture", line, re.I)
            or re.search(r"https?://|\b(?:already|done|finished|found|opened|captured|"
                         r"saved|updated|deleted|scheduled|verified|successfully|delete|execute|"
                         r"download|upload|schedule|pay|purchase)\b", line, re.I)
            or not re.match(r"(?:let me|lemme|i(?:'ll| will| am going to|'m going to))\b",
                            line, re.I)):
        return ""
    return line


class ToolProgress:
    """Reuse AI-written variants by intent, with a bounded cold-start fallback.

    Generation receives a synthetic intent category, not screen pixels, personal
    dialogue, retrieved pages, or tool output. A line never authorizes an action.
    """

    def __init__(self, router, *, budget: float = 12.0) -> None:
        self.router = router
        self.budget = budget
        self._variants: dict[tuple[str, bool], tuple[str, ...]] = {}
        self._last: dict[str, str] = {}
        self._lock = threading.Lock()
        self._jobs: dict[bool, tuple[threading.Event, threading.Event]] = {}
        self._retry_after: dict[bool, float] = {}
        self._closed = False

    def warm(self, *, local_only: bool = False) -> None:
        """Generate a generic bank in the background, never on a tool's critical path."""
        with self._lock:
            previous = self._jobs.get(local_only)
            if (self._closed or (previous and not previous[1].is_set())
                    or time.monotonic() < self._retry_after.get(local_only, 0)
                    or all((purpose, local_only) in self._variants for purpose in _PURPOSES)):
                return
            canceled, finished = threading.Event(), threading.Event()
            self._jobs[local_only] = (canceled, finished)

        def generate() -> None:
            try:
                response = _bounded(lambda: self.router.send([
                    ChatMessage("system", "Write two different short casual English lines Ciel "
                                "can say BEFORE each supplied tool purpose. Use natural companion "
                                "phrasing and contractions. Start with Let me, Lemme, or I'll. "
                                "Speak directly to 'you'. Avoid technical terms such as requested "
                                "desktop capture, tool, API, or retrieval. "
                                "Describe only the intended work, no results, extra actions, "
                                "questions, or names. Return only a JSON object mapping EVERY "
                                "supplied key to an array of two strings. No markdown."),
                    ChatMessage("user", json.dumps(_PURPOSES), local_only=local_only),
                ], purpose="tool_progress", temperature=0.8, max_tokens=700,
                    cancel_event=canceled), time.monotonic() + self.budget, canceled)
                clean = response.content.strip()
                fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", clean, re.I | re.S)
                parsed = json.loads(fenced.group(1) if fenced else clean)
                if not isinstance(parsed, dict):
                    return
                for purpose in _PURPOSES:
                    values = parsed.get(purpose, [])
                    lines = tuple(dict.fromkeys(valid_line(v) for v in values if valid_line(v))) \
                        if isinstance(values, list) else ()
                    if len(lines) >= 2 and not canceled.is_set():
                        with self._lock:
                            self._variants[(purpose, local_only)] = lines
            except Exception:
                pass  # Offline/slow AI never blocks a tool or makes progress disappear.
            finally:
                canceled.set()
                with self._lock:
                    self._retry_after[local_only] = time.monotonic() + 60
                finished.set()

        threading.Thread(target=generate, name="ciel-tool-lines", daemon=True).start()

    def line(self, tool: str, query: str, *, local_only: bool = False,
             cancel_event: threading.Event | None = None,
             current: Callable[[], bool] = lambda: True, proposed: str = "") -> str:
        """Return an intent-specific sentence without adding an unbounded model wait."""
        if (cancel_event is not None and cancel_event.is_set()) or not current():
            return ""
        purpose = purpose_for(tool, query)
        ready = valid_line(proposed)
        if ready and ready != self._last.get(purpose):
            self._last[purpose] = ready
            return ready
        self.warm(local_only=local_only)
        if (cancel_event is not None and cancel_event.is_set()) or not current():
            return ""
        with self._lock:
            options = self._variants.get((purpose, local_only), _FALLBACKS[purpose])
        available = [line for line in options if line != self._last.get(purpose)] or list(options)
        chosen = random.choice(available)
        self._last[purpose] = chosen
        return chosen

    def announce(self, tool: str, query: str, emit: Callable[[str], object], **kwargs) -> bool:
        """Emit immediately before actual work; canceled announcements never start tools."""
        line = self.line(tool, query, **kwargs)
        if not line:
            return False
        emit(line)
        canceled = kwargs.get("cancel_event")
        return not (canceled is not None and canceled.is_set()) and kwargs.get(
            "current", lambda: True,
        )()

    def close(self) -> None:
        """Cancel bank generation when its owning conversation service stops."""
        with self._lock:
            self._closed = True
            jobs = list(self._jobs.values())
        for canceled, _ in jobs:
            canceled.set()
        for _, finished in jobs:
            finished.wait(0.2)
