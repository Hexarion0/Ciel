import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal

from ciel.config import get_settings
from ciel.latency import mark
from ciel.logging import get_logger
from ciel.providers.base import (
    ChatMessage,
    LLMResponse,
    LLMStreamChunk,
    ToolCallAccumulator,
    require_text_messages,
)
from ciel.providers.intents import answer_clock_query
from ciel.providers.manager import ProviderManager
from ciel.providers.priority import BackgroundDeferred, RequestPriority

logger = get_logger("providers.router")


class ComplexityLevel(str, Enum):
    """Query complexity categories."""

    SIMPLE = "simple"  # Greetings, time, short factual queries, basic arithmetic
    MEDIUM = "medium"  # Explanations, summaries, multi-step questions
    COMPLEX = "complex"  # Implementation, architecture, explicit deep work


@dataclass
class RoutingDecision:
    """Outcome of query analysis."""

    complexity: ComplexityLevel
    provider_name: str
    model_name: str | None
    reason: str
    override_applied: str | None = None
    reasoning_enabled: bool = False


class ModelRouter:
    """Analyzes prompt complexity and routes queries to optimal AI providers/models."""

    # Heuristic patterns for complexity classification
    CODE_KEYWORDS = re.compile(
        r"\b(code|function|class|algorithm|python|javascript|rust|bug|error|refactor|sql|debug|regex|api)\b",
        re.IGNORECASE,
    )

    def __init__(self, manager: ProviderManager | None = None) -> None:
        self.manager = manager or ProviderManager()
        self.priority = RequestPriority()
        self._reasoning_lock = threading.Lock()
        self._reasoning_mode: Literal["auto", "on", "off"] = get_settings().providers.nim_reasoning

    @property
    def reasoning_mode(self) -> Literal["auto", "on", "off"]:
        """Return this router's session-scoped reasoning preference."""
        with self._reasoning_lock:
            return self._reasoning_mode

    def set_reasoning_mode(self, mode: Literal["auto", "on", "off"]) -> None:
        """Change future requests without changing an in-flight request or saved settings."""
        if mode not in {"auto", "on", "off"}:
            raise ValueError("Reasoning mode must be auto, on, or off")
        with self._reasoning_lock:
            self._reasoning_mode = mode

    def classify_complexity(self, prompt: str) -> tuple[ComplexityLevel, str, str | None]:
        """Classify prompt into SIMPLE, MEDIUM, or COMPLEX with explanation and override info."""
        clean = prompt.strip()

        # 1. Manual Overrides
        if re.match(r"^/fast(?:\s|$)", clean):
            return ComplexityLevel.SIMPLE, "Manual /fast override requested", "fast"
        if re.match(r"^/(?:strong|deep)(?:\s|$)", clean):
            return ComplexityLevel.COMPLEX, "Manual /strong override requested", "strong"

        # 2. Simple Heuristics (short greetings, time, simple math)
        conversational = re.sub(
            r"^(?:hi|hello|hey)(?:\s+ciel)?[,! ]+\s*",
            "",
            clean,
            flags=re.I,
        )
        greeting = re.fullmatch(
            r"(?:hi|hello|hey|thanks|thank you|good (?:morning|afternoon|evening))"
            r"(?:\s+ciel)?[.!?]*",
            clean,
            re.I,
        )
        arithmetic = re.fullmatch(
            r"(?:what is\s+)?-?\d+(?:\.\d+)?\s*[+\-*/]\s*-?\d+(?:\.\d+)?[.?]?",
            clean,
            re.I,
        )
        if answer_clock_query(conversational) is not None or greeting or arithmetic:
            return ComplexityLevel.SIMPLE, "Short conversational query or basic math", None

        # 3. Complex Heuristics (code requests, architecture, deep analysis)
        technical = bool(self.CODE_KEYWORDS.search(clean))
        action = bool(
            re.search(
                r"\b(?:implement|write|refactor|debug|fix|optimize|redesign|review)\b",
                clean,
                re.I,
            )
        )
        architecture = bool(
            re.search(
                r"\b(?:architecture|architect|sharding|concurrency|distributed|microservices)\b",
                clean,
                re.I,
            )
        )
        embedded_code = chr(96) * 3 in clean or bool(re.search(r"\b(?:def|class)\s+\w+[:(]", clean))
        if (technical and action) or architecture or embedded_code:
            return (
                ComplexityLevel.COMPLEX,
                "Detected code implementation, debugging, or architecture work",
                None,
            )

        # 4. Default: Medium complexity
        return ComplexityLevel.MEDIUM, "Standard explanatory or multi-turn query", None

    def route(
        self, prompt: str, *, purpose: str = "conversation", local_only: bool = False,
    ) -> RoutingDecision:
        """Determine the optimal provider and model for a given prompt."""
        mark(f"{purpose}.routing_started")
        if purpose in {"controller", "tool_progress"}:
            complexity, reason, override = (
                ComplexityLevel.SIMPLE, "Compact AI controller decision", None,
            )
        elif purpose == "verification":
            complexity, reason, override = (
                ComplexityLevel.MEDIUM, "Bounded evidence check on the normal responder", None,
            )
        elif purpose in {"summary", "speech_gate"}:
            complexity, reason, override = (
                ComplexityLevel.SIMPLE,
                f"Background {purpose} uses the economical route",
                None,
            )
        elif purpose == "conversation":
            complexity, reason, override = self.classify_complexity(prompt)
        else:
            raise ValueError(f"Unknown routing purpose: {purpose}")

        # Choose an available provider first; its model names are provider-specific.
        # Private context never enters the configured cloud fallback chain.
        if local_only:
            provider_name = "ollama"
            reason += " → Local context requires Ollama"
        else:
            if purpose in {"controller", "tool_progress"}:
                order = ("nim", "groq", "openrouter", "ollama")
            elif purpose != "conversation" and complexity == ComplexityLevel.SIMPLE:
                order = ("groq", "nim", "openrouter", "ollama")
            else:
                order = ("nim", "openrouter", "groq", "ollama")
            selected = get_settings().providers.primary_provider
            if selected != 'auto':
                order = (selected, *(name for name in order if name != selected))
            provider_name = next(
                (name for name in order if getattr(self.manager, name).is_configured()),
                order[0],
            )
            reason += f" → Configured {provider_name} route"

        if provider_name == "nim":
            settings = get_settings()
            if purpose in {"controller", "tool_progress"}:
                model_name = settings.controller.model
            elif complexity == ComplexityLevel.COMPLEX:
                model_name = settings.providers.nim_complex_model
            else:
                model_name = settings.providers.nim_model
        else:
            model_name = getattr(self.manager, provider_name).default_model
        reasoning_mode = self.reasoning_mode
        reasoning_enabled = purpose == "conversation" and (
            reasoning_mode == "on"
            or (reasoning_mode == "auto" and complexity == ComplexityLevel.COMPLEX)
        )
        decision = RoutingDecision(
            complexity=complexity,
            provider_name=provider_name,
            model_name=model_name,
            reason=reason,
            override_applied=override,
            reasoning_enabled=reasoning_enabled,
        )
        logger.info(
            "🔀 Routed %s [%s] to provider '%s' (Model: %s) — %s",
            purpose,
            decision.complexity.value.upper(),
            decision.provider_name,
            decision.model_name or "default",
            decision.reason,
        )
        mark(f"{purpose}.routing_finished", provider=decision.provider_name,
             model=decision.model_name)
        return decision

    def send(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = 512,
        *,
        purpose: str = "conversation",
        cancel_event: threading.Event | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Route and execute a chat completion with automatic fallback."""
        require_text_messages(messages)
        mark(f"{purpose}.route_call_started")
        latest_user_text = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        routing_text = (
            self._routing_text(messages) if purpose == "conversation" else latest_user_text
        )
        local_only = self.requires_local(messages)
        decision = self.route(routing_text, purpose=purpose, local_only=local_only)

        # Strip override command prefixes from messages if present
        clean_messages = self._clean_override_prefixes(messages)

        options: dict[str, Any] = {
            "preferred_provider": decision.provider_name, "model": decision.model_name,
            "temperature": temperature, "max_tokens": max_tokens or 512,
            "nim_reasoning": decision.reasoning_enabled,
        }
        if tools is not None:
            options["tools"] = tools
        if tool_choice is not None:
            options["tool_choice"] = tool_choice
        if local_only:
            options["allowed_providers"] = ("ollama",)
        elif purpose in {"controller", "tool_progress"}:
            # Keep NIM's dedicated decision model lightweight, while still allowing
            # another configured provider to recover within the caller's deadline.
            options["allow_model_fallback"] = False

        def collect(canceled: threading.Event | None) -> LLMResponse:
            started = time.monotonic()
            response = LLMResponse("", "", "")
            tool_calls = ToolCallAccumulator()
            for chunk in self.manager.stream_with_fallback(
                clean_messages, cancel_event=canceled, **options,
            ):
                response.content += chunk.delta
                tool_calls.add(chunk.tool_calls)
                response.model, response.provider = chunk.model, chunk.provider
            if canceled is not None and canceled.is_set():
                raise BackgroundDeferred("Provider work was canceled")
            response.latency = time.monotonic() - started
            response.tool_calls = tool_calls.finish()
            return response

        if purpose in {"summary", "tool_progress"}:
            with self.priority.background() as canceled:
                if cancel_event is not None:
                    parent = canceled

                    class CombinedCancellation(threading.Event):
                        def is_set(self) -> bool:
                            return parent.is_set() or cancel_event.is_set()

                    return collect(CombinedCancellation())
                return collect(canceled)
        with self.priority.foreground():
            # Gates still return complete JSON, but their blocked reads can now
            # be canceled immediately when a user resumes speaking.
            if purpose == "speech_gate" or cancel_event is not None:
                return collect(cancel_event)
            return self.manager.send_with_fallback(clean_messages, **options)

    def stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = 512,
        *,
        cancel_event: threading.Event | None = None,
        purpose: str = "conversation",
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Route and stream chat completion tokens with automatic fallback."""
        require_text_messages(messages)
        mark(f"{purpose}.route_call_started")
        local_only = self.requires_local(messages)
        routing_text = self._routing_text(messages) if purpose == "conversation" else next(
            (message.content for message in reversed(messages) if message.role == "user"), "",
        )
        decision = self.route(routing_text, purpose=purpose, local_only=local_only)
        clean_messages = self._clean_override_prefixes(messages)

        with self.priority.foreground():
            yield from self.manager.stream_with_fallback(
                clean_messages,
                preferred_provider=decision.provider_name,
                model=decision.model_name,
                temperature=temperature,
                max_tokens=max_tokens or 512,
                cancel_event=cancel_event,
                nim_reasoning=decision.reasoning_enabled,
                **({"tools": tools} if tools is not None else {}),
                **({"tool_choice": tool_choice} if tool_choice is not None else {}),
                **({"allowed_providers": ("ollama",)} if local_only else {}),
            )

    @staticmethod
    def requires_local(messages: list[ChatMessage]) -> bool:
        """Keep legacy private context on Ollama, including continuations and summaries."""
        return any(
            message.local_only or (
                message.role == "user"
                and re.search(r"(?im)^\s*/local(?:\s|$)", message.content) is not None
            )
            for message in messages
        )

    @staticmethod
    def _routing_text(messages: list[ChatMessage]) -> str:
        """Carry recent task context only across explicit or referring continuations."""
        users = [message.content for message in messages if message.role == "user"]
        followups: list[str] = []
        for text in reversed(users[-8:]):
            if re.fullmatch(
                r"(?:continue|go on|keep going|explain more)[.!?]*",
                text.strip(),
                re.I,
            ):
                continue
            if re.fullmatch(
                r"(?:(?:(?:can|could|would) you|please)\s+)?"
                r"(?:fix|debug|refactor|optimize|review|rewrite|improve|simplify|test|implement)"
                r"\s+(?:it|that|this)(?:\s+(?:please|too|again))?[.!?]*",
                text.strip(),
                re.I,
            ):
                followups.append(text[:4096])
                continue
            # The current standalone request (and any current /fast or /strong)
            # wins immediately; old topics never contribute to its complexity.
            if not followups:
                return text
            return "\n".join([text[-4096:], *reversed(followups)])
        return users[-1] if users else ""

    @staticmethod
    def _clean_override_prefixes(messages: list[ChatMessage]) -> list[ChatMessage]:
        """Strip active overrides and legacy /local prefixes during historical replay."""
        cleaned: list[ChatMessage] = []
        for m in messages:
            if m.role == "user":
                content = re.sub(r"^/(fast|strong|deep|local)(?:\s+|$)", "", m.content)
                cleaned.append(replace(m, content=content))
            else:
                cleaned.append(m)
        return cleaned
