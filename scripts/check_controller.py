"""Check hosted NIM controller choices with synthetic inputs; never load conversation data.

Run from the repository root: .venv/bin/python scripts/check_controller.py
This makes short API requests using the configured NIM key and controller model.
It does not use the microphone, execute local actions, or search the public web.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time

from ciel.config import InternetConfig, get_settings
from ciel.controller import AIController, ControllerDecision
from ciel.internet import InternetResult, InternetSource
from ciel.providers.base import ChatMessage
from ciel.providers.router import ModelRouter


class _DecisionProbe:
    """Record model health before application safety rules can replace a decision."""

    def __init__(self, router: ModelRouter, timeout: float) -> None:
        self.router = router
        self.deadline = time.monotonic() + timeout
        self._checks: list[tuple[bool, str | None]] = []
        self._lock = threading.Lock()

    def send(self, *args, **kwargs):
        """Preserve routing while validating each actual controller model response."""
        with self._lock:
            index = len(self._checks)
            self._checks.append((False, "Model decision did not complete"))
        try:
            response = self.router.send(*args, **kwargs)
            content = response.content.strip()
            fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.I | re.S)
            decision = ControllerDecision.model_validate_json(
                fenced.group(1) if fenced else content,
            )
            error = None if decision.confidence >= 0.5 else "Model confidence is below 0.5"
            if time.monotonic() >= self.deadline:
                error = "Model decision exceeded the controller deadline"
        except Exception as failure:
            with self._lock:
                self._checks[index] = (False, type(failure).__name__)
            raise
        with self._lock:
            self._checks[index] = (error is None, error)
        return response

    def result(self) -> tuple[bool, str | None]:
        """Fail closed for unfinished, invalid, low-confidence, or failed model calls."""
        with self._lock:
            checks = self._checks[:]
        if not checks:
            return False, "No model decision was requested"
        errors = [error for valid, error in checks if not valid]
        return not errors, "; ".join(error for error in errors if error) or None


def main() -> int:
    """Print measured choices and exit nonzero if a routing or evidence check fails."""
    settings = get_settings()
    if not settings.providers.nim_api_key:
        print("NIM_API_KEY is required for this live check.")
        return 1
    router = ModelRouter()
    anime = "Where does Amber Works season 4 part 2 continue in the light novel?"
    # Expected action and query terms test behavior, not exact model wording.
    cases = [
        ("greeting after pending request", "Good evening, Ciel!",
         "Find the opening date for the fictional Marbled Lantern Festival.",
         "answer", [], {"previous_reply": None, "expected_intent": "greeting"}),
        ("missing subject", "Please look up the festival date. I forgot which festival it was.",
         "", "clarify", [], {}),
        ("new task after pending request", "Tell me a short joke.",
         "Find the opening date for the fictional Marbled Lantern Festival.",
         "answer", [], {"previous_reply": None, "expected_intent": "request"}),
        ("ambiguous acceptance", "Yes, go ahead.", "Find the Marbled Lantern Festival date.",
         "clarify", [], {"previous_reply": "I can look again or explain festivals in general."}),
        ("reading point", anime, "", "search", ["amber", "4", "2"], {}),
        ("reading follow-up", "So what volume should I read?", anime,
         "search", ["amber", "4", "2"], {}),
        ("retry", "Okay, try again", anime, "search", ["amber", "4", "2"], {}),
        ("explanation", "Why can't you search it on the internet?", anime, "answer", [], {}),
        ("background", "Mom, what's for dinner?", "", "silent", [],
         {"addressed": False, "active": False}),
        ("other listener", "Dad, can you turn on the TV?", anime, "silent", [],
         {"addressed": False}),
        ("quiet", "Don't answer this, stay silent.", "", "silent", [], {}),
        ("clock", "What time is it?", "", "local", [], {}),
        ("correction", "Research again, finished season 4 part 2 btw", anime,
         "search", ["amber", "4", "2"], {}),
        ("missing city", "What's the weather tomorrow?", "", "clarify", [], {}),
        ("configured city", "Weather tomorrow?", "", "search", ["paris"],
         {"home_city": "Paris, France"}),
        ("explicit city", "Weather in Rome tomorrow?", "", "search", ["rome"],
         {"home_city": "Paris, France"}),
        ("remembered city", "Weather tomorrow?", "", "search", ["helsinki"],
         {"memories": ["Your home city is Helsinki, Finland."]}),
        ("different retry topic", "Okay, try again", "What are the latest stable Rust releases?",
         "search", ["rust"], {}),
        ("automatic off", "What is the latest Python release?", "", "answer", [],
         {"automatic_search": False}),
        ("memory", "Remember my favorite game is CS2.", "", "local", [], {}),
    ]
    selected = set(sys.argv[1:])
    if selected:
        cases = [case for case in cases if case[0] in selected]
        if not cases:
            print("No matching synthetic cases.")
            return 1
    failures = 0
    for name, text, previous, action, query_terms, overrides in cases:
        options = dict(overrides)
        previous_reply = options.pop("previous_reply", "I couldn't verify that online.")
        expected_intent = options.pop("expected_intent", None)
        messages = [ChatMessage("user", previous)] if previous else []
        if previous and previous_reply:
            messages.append(ChatMessage("assistant", previous_reply))
        started = time.monotonic()
        probe = _DecisionProbe(router, settings.controller.timeout_seconds)
        controller = AIController(probe, settings.controller)
        choice = controller.decide(text, messages, **{
            "addressed": True, "active": True, "internet_enabled": True, **options,
        })
        model_decision_valid, model_error = probe.result()
        passed = (
            model_decision_valid and choice.confidence >= 0.5 and choice.action == action
            and all(term in choice.query.casefold() for term in query_terms)
        )
        passed = passed and (expected_intent is None or choice.intent == expected_intent)
        failures += not passed
        print(json.dumps({"case": name, "passed": passed,
                          "model_decision_valid": model_decision_valid,
                          "model_error": model_error,
                          "seconds": round(time.monotonic() - started, 2),
                          **choice.model_dump()}), flush=True)

    if selected:
        print(f"{len(cases) - failures}/{len(cases)} checks passed.")
        return 1 if failures else 0

    quote = "Amber Works season 4 part 2 ends at light novel volume 11."

    class SyntheticWeb:
        config = InternetConfig()

        def lookup(self, query: str, **_kwargs) -> InternetResult:
            return InternetResult(attempted=True, query=query, sources=[InternetSource(
                "Synthetic test page", "https://example.com/amber", content=quote,
            )])

    messages = [ChatMessage("system", "Be concise. This is a fictional test story."),
                ChatMessage("user", "Where does Amber Works season 4 part 2 end in the novel?")]
    controller = AIController(router, settings.controller)
    reply = controller.run_web(
        ControllerDecision(action="search", confidence=1,
                           query="Amber Works season 4 part 2 light novel continuation"),
        messages[-1].content, messages, SyntheticWeb(), lambda _text: None, lambda: True,
    )
    passed = controller.last_lookup.get("outcome") == "supported" and "11" in reply.text
    failures += not passed
    print(json.dumps({"case": "validated report", "passed": passed, "text": reply.text}),
          flush=True)
    print(f"{len(cases) + 1 - failures}/{len(cases) + 1} checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
