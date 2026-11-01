#!/usr/bin/env python3
"""Evaluate synthetic task/search conversations without reading personal history.

Default operation is offline. ``--live`` explicitly opts into the configured response
provider; web reads remain fixtures, memory uses a temporary database, and screen/audio
operations are never invoked. Limits count logical router calls (a provider's own fallback
attempts can add HTTP requests), with cooperative cancellation and a wall-clock deadline.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ciel.chat.session import ChatSession
from ciel.config import Settings
from ciel.internet import InternetResult, InternetSource, _bounded
from ciel.providers.base import LLMResponse

_EVALUATION_WORKERS = threading.BoundedSemaphore(1)
_MODEL_WORKERS = threading.BoundedSemaphore(2)


@dataclass(frozen=True)
class EvaluationCase:
    """Synthetic owner dialogue and observable requirements, independent of model output."""

    name: str
    history: tuple[tuple[str, str], ...]
    current: str
    required_goal: tuple[str, ...]
    forbidden_goal: tuple[str, ...]
    fact: str
    required_answer: tuple[str, ...]
    forbidden_answer: tuple[str, ...] = ()


CASES = (
    EvaluationCase(
        "accepted_corrected_lookup_offer",
        (("user", "Hey Ciel, do you know about Tinsura anime mod in Minecraft?"),
         ("assistant", "I'm not familiar with that mod."),
         ("user", "I meant Tensura."),
         ("assistant", "I haven't heard of a Tensura mod for Minecraft either. "
          "Want me to look it up?")),
        "Yes, sure.", ("Tensura", "mod", "Minecraft"), ("Tinsura", "Yes, sure"),
        "Tensura is a Minecraft mod inspired by the anime series.", ("Tensura", "Minecraft"),
    ),
    EvaluationCase(
        "spoken_name_typed_correction",
        (("user", "Hey Ciel, can you search TensorFlow for me?"),
         ("assistant", "What do you want to know about TensorFlow?")),
        "Tensura*", ("Tensura",), ("TensorFlow", "season 3"),
        "Tensura is the short name for That Time I Got Reincarnated as a Slime.",
        ("Slime",),
    ),
    EvaluationCase(
        "title_correction_retains_episode_and_medium",
        (("user", "Search where Amber Harbour episode 12 continues in the light novel."),
         ("assistant", "Could you spell the title?")),
        "Amber Harbor*", ("Amber Harbor", "12", "light novel"), ("Amber Harbour",),
        "Amber Harbor anime episode 12 ends at light novel volume 3. "
        "Continue reading from light novel volume 4.", ("volume 4",),
    ),
    EvaluationCase(
        "light_node_medium_repair",
        (("user", "Search Amber Harbor episode 12 continuation on the light node."),
         ("assistant", "Did you mean the light novel or the manga?")),
        "light novel continuation", ("Amber Harbor", "12", "light novel"), (),
        "Amber Harbor anime episode 12 ends at light novel volume 3. "
        "Continue reading from light novel volume 4.", ("volume 4",),
    ),
    EvaluationCase(
        "latest_season_replaces_old_scope",
        (("user", "Search Amber Harbor season 4 anime to light novel continuation."),
         ("assistant", "Which season did you finish?")),
        "Actually season 5, the light novel.", ("Amber Harbor", "season 5", "light novel"),
        ("season 4",), "Amber Harbor anime season 5 ends at light novel volume 11. "
        "Continue reading from light novel volume 12.", ("volume 12",),
    ),
    EvaluationCase(
        "retry_keeps_explicit_corrected_name",
        (("user", "Search Amber Harbour episode 12 light novel continuation."),
         ("assistant", "Could you spell the title?"), ("user", "Amber Harbor*"),
         ("assistant", "The lookup timed out. Please try again.")),
        "Try again.", ("Amber Harbor", "12", "light novel"), ("Amber Harbour",),
        "Amber Harbor anime episode 12 ends at light novel volume 3. "
        "Continue reading from light novel volume 4.", ("volume 4",),
    ),
    EvaluationCase(
        "continue_after_interruption_keeps_medium",
        (("user", "Search Amber Harbor episode 12 continuation on the light node."),
         ("assistant", "Did you mean the novel or the manga?"), ("user", "The light novel."),
         ("assistant", "I was checking the guide. [Playback was interrupted.]")),
        "Continue.", ("Amber Harbor", "12", "light novel"), (),
        "Amber Harbor anime episode 12 ends at light novel volume 3. "
        "Continue reading from light novel volume 4.", ("volume 4",),
    ),
    EvaluationCase(
        "two_part_question_is_not_a_title_only_lookup",
        (), "Search whether CedarDB supports Linux and whether it can run without internet.",
        ("CedarDB", "Linux", "internet"), (),
        "CedarDB supports Linux desktop systems. CedarDB can run without internet access.",
        ("Linux", "without internet"),
    ),
    EvaluationCase(
        "partial_evidence_keeps_supported_finding",
        (), "Search whether CedarDB supports Linux and whether video calling is included.",
        ("CedarDB", "Linux", "video"), (),
        "CedarDB supports Linux desktop systems. "
        "The reference guide does not document video calling support.",
        ("Linux", "video"),
    ),
    EvaluationCase(
        "bare_spoken_name_repair_preserves_mod_context",
        (("user", "Hey Ciel, do you know about like Tenzura mod in Minecraft?"),
         ("assistant", "I'm not familiar with that mod. Can you tell me more?")),
        "Tensura.", ("Tensura", "Minecraft", "mod"), ("Tenzura",),
        "Tensura is a Minecraft mod inspired by That Time I Got Reincarnated as a Slime.",
        ("Tensura",),
    ),
    EvaluationCase(
        "gameplay_explanation_advances_beyond_definition",
        (("user", "Do you know the Ambercraft Minecraft mod?"),
         ("assistant", "Ambercraft is a Minecraft mod with races and skills.")),
        "I'm currently playing the mod. Research it so I can understand how to play.",
        ("Ambercraft", "Minecraft", "play"), (),
        "Ambercraft is a Minecraft mod with races and skills. "
        "Open the quest journal to see the first progression tasks. "
        "Complete training tasks to earn points for unlocking skills.",
        ("quest journal", "training", "skills"),
        ("Ambercraft is a Minecraft mod with races and skills",),
    ),
    EvaluationCase(
        "phonetic_title_retry_uses_synopsis",
        (("user", "What is Osinoko anime about?"),
         ("assistant", "I couldn't find usable evidence for that title.")),
        "Can you try again?", ("Osinoko", "anime"), (),
        "Oshi no Ko follows a doctor reborn as the child of an idol. "
        "The story explores the entertainment industry.",
        ("Oshi no Ko", "idol"),
    ),
)


class DeadlineCancellation(threading.Event):
    """Keep the caller's cancellation and the evaluation deadline authoritative."""

    def __init__(self, parent: threading.Event | None, deadline: float) -> None:
        super().__init__()
        self.parent, self.deadline = parent, deadline

    def is_set(self) -> bool:
        return (super().is_set() or time.monotonic() >= self.deadline
                or (self.parent is not None and self.parent.is_set()))

    def wait(self, timeout: float | None = None) -> bool:
        deadline = min(self.deadline, time.monotonic() + timeout) if timeout is not None \
            else self.deadline
        while not self.is_set() and time.monotonic() < deadline:
            super().wait(min(0.05, max(0, deadline - time.monotonic())))
        return self.is_set()


class BudgetRouter:
    """Count foreground model calls and avoid unrelated progress/summary generation."""

    def __init__(self, router: Any, *, deadline: float, maximum: int) -> None:
        self.router, self.deadline, self.maximum = router, deadline, maximum
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def send(self, messages, **kwargs) -> LLMResponse:
        purpose = kwargs.get("purpose", "conversation")
        if purpose not in {"conversation", "verification"}:
            return LLMResponse("{}", "fixture-progress", "local")
        with self._lock:
            if time.monotonic() >= self.deadline or len(self.calls) >= self.maximum:
                raise TimeoutError("Synthetic evaluation request limit reached")
            record = {"purpose": purpose, "tools_enabled": bool(kwargs.get("tools"))}
            self.calls.append(record)
        canceled = DeadlineCancellation(kwargs.get("cancel_event"), self.deadline)
        kwargs.update(cancel_event=canceled, temperature=0)
        started = time.monotonic()
        try:
            response = _bounded(lambda: self.router.send(messages, **kwargs),
                                self.deadline, canceled, workers=_MODEL_WORKERS)
            record.update(provider=response.provider, model=response.model,
                          tool_names=[call.name for call in response.tool_calls],
                          response=response.content[:12000])
            queries = []
            for call in response.tool_calls:
                if call.name == "web_search":
                    try:
                        arguments = json.loads(call.arguments)
                        queries.extend([arguments.get("query", ""),
                                        *arguments.get("search_queries", [])])
                    except (ValueError, TypeError, AttributeError):
                        pass
            try:
                queries.extend(json.loads(response.content).get("search_queries", []))
            except (ValueError, TypeError, AttributeError):
                pass
            record["search_queries"] = [q for q in queries if isinstance(q, str) and q.strip()]
            return response
        finally:
            record["seconds"] = round(time.monotonic() - started, 3)

    def emitted_queries(self) -> set[str]:
        """Snapshot model proposals before the corresponding external lookup runs."""
        with self._lock:
            return {" ".join(query.split()) for record in self.calls
                    for query in record.get("search_queries", [])}


class FixtureInternet:
    """Serve only synthetic reference bytes and record real lookup objectives."""

    def __init__(self, case: EvaluationCase, config: Any, budget: BudgetRouter) -> None:
        self.case, self.config = case, config
        self.budget = budget
        self.calls: list[dict[str, Any]] = []

    def lookup(self, query: str, **kwargs) -> InternetResult:
        proposals = self.budget.emitted_queries()
        outgoing = [query, *kwargs.get("search_queries", [])]
        self.calls.append({"query": query, "objective": kwargs.get("objective", ""),
                           "search_queries": kwargs.get("search_queries", []),
                           "read_page": bool(kwargs.get("read_page")),
                           "queries_written_by_model": all(
                               isinstance(probe, str) and " ".join(probe.split()) in proposals
                               for probe in outgoing
                           )})
        return InternetResult(attempted=True, query=query, sources=[InternetSource(
            "Synthetic evaluation reference", "https://example.org/synthetic/reference",
            content=self.case.fact,
        )])

    def reset_session(self) -> None:
        """The fixture has no network session or persistent search state."""


def evaluate(case: EvaluationCase, router: Any, *, deadline: float, maximum: int) -> dict:
    """Run a real chat request with only synthetic history and reference sources."""
    started = time.monotonic()
    budget = BudgetRouter(router, deadline=deadline, maximum=maximum)
    result: dict[str, Any] = {"case": case.name, "status": "failed", "checks": {}}
    with TemporaryDirectory(prefix="ciel-search-evaluation-") as directory:
        settings = Settings(
            _env_file=None, memory_db_path=str(Path(directory) / "evaluation.db"),
            memory_workspace_path=str(Path(directory) / "notes"),
            memory_workspace_enabled=False, memory_semantic_enabled=False,
            memory_recall_enabled=False, memory_consolidation_enabled=False,
            ciel_persona_file="", internet_enabled=True, chat_services="",
        )
        session = ChatSession(settings, channel="evaluation", session_id="synthetic-owner",
                              router=budget)
        internet = FixtureInternet(case, settings.internet, budget)
        session.internet = internet
        try:
            for role, text in case.history:
                session.conversation.add_turn(role, text)
            canceled = DeadlineCancellation(None, deadline)
            answer = _bounded(lambda: session.reply(case.current, canceled), deadline, canceled,
                              workers=_EVALUATION_WORKERS)
            turns = session.conversation.get_recent_turns(limit=30)
            goal = " ".join(call["objective"] for call in internet.calls).casefold()
            searches = [call for call in internet.calls if not call["read_page"]]
            sources = turns[-1].sources if turns and turns[-1].role == "assistant" else []
            result["checks"] = {
                "searched": bool(searches),
                "queries_written_by_model": bool(searches) and all(
                    call["queries_written_by_model"] for call in searches
                ),
                "task_scope_preserved": all(term.casefold() in goal
                                            for term in case.required_goal),
                "stale_scope_absent": not any(term.casefold() in goal
                                              for term in case.forbidden_goal),
                "useful_supported_answer": all(term.casefold() in answer.casefold()
                                               for term in case.required_answer),
                "no_repeated_background": not any(term.casefold() in answer.casefold()
                                                  for term in case.forbidden_answer),
                "cited_fixture": bool(sources) and all(
                    source["url"] == "https://example.org/synthetic/reference"
                    for source in sources
                ),
                "no_tool_limit_reply": "tool limit" not in answer.casefold(),
                "raw_owner_turns_preserved": [turn.content for turn in turns
                                               if turn.role == "user"] == [
                    *(text for role, text in case.history if role == "user"), case.current,
                ],
                "within_deadline": time.monotonic() < deadline,
            }
            result.update(answer=answer, lookups=internet.calls, sources=sources)
            result["status"] = "passed" if all(result["checks"].values()) else "failed"
        except Exception as error:
            result["error"] = type(error).__name__
        finally:
            session.close()
    result.update(model_calls=budget.calls, seconds=round(time.monotonic() - started, 3))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Allow configured model calls")
    parser.add_argument("--cases", nargs="+", choices=[case.name for case in CASES])
    parser.add_argument("--case-timeout", type=float, default=30)
    parser.add_argument("--total-timeout", type=float, default=240)
    parser.add_argument("--max-requests", type=int, default=6,
                        help="Maximum logical foreground router calls per case")
    parser.add_argument("--output", type=Path,
                        default=Path("data/diagnostics/conversation-search-evaluation.json"))
    args = parser.parse_args(argv)
    if (not 1 <= args.case_timeout <= 60 or not 1 <= args.total_timeout <= 600
            or not 1 <= args.max_requests <= 12):
        parser.error("Timeouts must be 1–60/1–600 seconds; request limit must be 1–12.")
    cases = [case for case in CASES if not args.cases or case.name in args.cases]
    report: dict[str, Any] = {
        "live": args.live, "synthetic_history_only": True, "real_web": False,
        "limits": {"case_seconds": args.case_timeout, "total_seconds": args.total_timeout,
                   "logical_model_requests_per_case": args.max_requests},
        "cases": [],
    }
    if args.live:
        from ciel.providers.router import ModelRouter

        router = ModelRouter()
        deadline = time.monotonic() + args.total_timeout
        for case in cases:
            if time.monotonic() >= deadline:
                report["cases"].append({"case": case.name, "status": "not_run_deadline"})
                continue
            result = evaluate(case, router, deadline=min(
                deadline, time.monotonic() + args.case_timeout,
            ), maximum=args.max_requests)
            report["cases"].append(result)
            print(json.dumps({"case": case.name, "status": result["status"],
                              "seconds": result["seconds"]}), flush=True)
    else:
        report["cases"] = [{"case": case.name, "status": "not_run", "current": case.current,
                            "required_goal": case.required_goal,
                            "required_answer": case.required_answer} for case in cases]
        print("Offline manifest only; add --live to evaluate the configured model.")
    report["passed"] = sum(case["status"] == "passed" for case in report["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Report: {args.output}")
    return int(args.live and report["passed"] != len(cases))


if __name__ == "__main__":
    raise SystemExit(main())
