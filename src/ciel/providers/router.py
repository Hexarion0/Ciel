import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

from ciel.config import get_settings
from ciel.latency import mark
from ciel.logging import get_logger
from ciel.providers.base import ChatMessage, LLMResponse, LLMStreamChunk
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

    def classify_complexity(self, prompt: str) -> tuple[ComplexityLevel, str, str | None]:
        """Classify prompt into SIMPLE, MEDIUM, or COMPLEX with explanation and override info."""
        clean = prompt.strip()

        # 1. Manual Overrides
        if re.match(r"^/fast(?:\s|$)", clean):
            return ComplexityLevel.SIMPLE, "Manual /fast override requested", "fast"
        if re.match(r"^/(?:strong|deep)(?:\s|$)", clean):
            return ComplexityLevel.COMPLEX, "Manual /strong override requested", "strong"
        if re.match(r"^/local(?:\s|$)", clean):
            return ComplexityLevel.SIMPLE, "Manual /local override requested", "local"

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
        if purpose == "controller":
            complexity, reason, override = (
                ComplexityLevel.SIMPLE, "Compact AI controller decision", None,
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

        # Route by complexity and provider availability
        if override == "local" or local_only:
            decision = RoutingDecision(
                complexity=complexity,
                provider_name="ollama",
                model_name="llama3.2",
                reason=f"{reason} → Local context requires Ollama" if local_only else reason,
                override_applied=override,
            )
        elif purpose == "controller":
            decision = RoutingDecision(
                complexity=complexity, provider_name="nim",
                model_name=get_settings().controller.model, reason=reason,
            )
        elif complexity == ComplexityLevel.SIMPLE:
            # Prefer ultra-fast inference (Groq) if available, otherwise NIM default
            if self.manager.groq.is_configured():
                decision = RoutingDecision(
                    complexity=complexity,
                    provider_name="groq",
                    model_name="llama-3.1-8b-instant",
                    reason=f"{reason} → Ultra-low latency via Groq",
                    override_applied=override,
                )
            else:
                nim_model = get_settings().providers.nim_model
                decision = RoutingDecision(
                    complexity=complexity,
                    provider_name="nim",
                    model_name=nim_model,
                    reason=f"{reason} → Default NIM provider ({nim_model})",
                    override_applied=override,
                )
        elif complexity == ComplexityLevel.COMPLEX:
            # Complex reasoning, coding, architecture -> Ultra model
            if self.manager.nim.is_configured():
                nim_complex_model = get_settings().providers.nim_complex_model
                decision = RoutingDecision(
                    complexity=complexity,
                    provider_name="nim",
                    model_name=nim_complex_model,
                    reason=f"{reason} → Deep reasoning/coding via NIM ({nim_complex_model})",
                    override_applied=override,
                )
            elif self.manager.openrouter.is_configured():
                decision = RoutingDecision(
                    complexity=complexity,
                    provider_name="openrouter",
                    model_name="meta-llama/llama-3.1-70b-instruct",
                    reason=f"{reason} → OpenRouter gateway",
                    override_applied=override,
                )
            else:
                decision = RoutingDecision(
                    complexity=complexity,
                    provider_name="groq",
                    model_name="llama-3.1-8b-instant",
                    reason=f"{reason} → Groq fallback",
                    override_applied=override,
                )
        else:
            # Medium: Standard explanatory / conversation -> Default super model
            if self.manager.nim.is_configured():
                nim_model = get_settings().providers.nim_model
                decision = RoutingDecision(
                    complexity=complexity,
                    provider_name="nim",
                    model_name=nim_model,
                    reason=f"{reason} → High-capacity conversational model via NIM ({nim_model})",
                    override_applied=override,
                )
            elif self.manager.openrouter.is_configured():
                decision = RoutingDecision(
                    complexity=complexity,
                    provider_name="openrouter",
                    model_name="meta-llama/llama-3.1-70b-instruct",
                    reason=f"{reason} → OpenRouter gateway",
                    override_applied=override,
                )
            else:
                decision = RoutingDecision(
                    complexity=complexity,
                    provider_name="groq",
                    model_name="llama-3.1-8b-instant",
                    reason=f"{reason} → Groq fallback",
                    override_applied=override,
                )

        if (
            purpose != "controller" and not local_only and override != "local"
            and not getattr(self.manager, decision.provider_name).is_configured()
        ):
            for candidate in ("nim", "groq", "openrouter", "ollama"):
                if getattr(self.manager, candidate).is_configured():
                    decision.provider_name = candidate
                    decision.model_name = None
                    decision.reason += f" → Configured fallback: {candidate}"
                    break
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
    ) -> LLMResponse:
        """Route and execute a chat completion with automatic fallback."""
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

        options = {
            "preferred_provider": decision.provider_name, "model": decision.model_name,
            "temperature": temperature, "max_tokens": max_tokens or 512,
        }
        if local_only:
            options["allowed_providers"] = ("ollama",)
        elif purpose == "controller":
            options["allowed_providers"] = ("nim",)
            options["allow_model_fallback"] = False

        def collect(canceled: threading.Event | None) -> LLMResponse:
            started = time.monotonic()
            response = LLMResponse("", "", "")
            for chunk in self.manager.stream_with_fallback(
                clean_messages, cancel_event=canceled, **options,
            ):
                response.content += chunk.delta
                response.model, response.provider = chunk.model, chunk.provider
            if canceled is not None and canceled.is_set():
                raise BackgroundDeferred("Provider work was canceled")
            response.latency = time.monotonic() - started
            return response

        if purpose == "summary":
            with self.priority.background() as canceled:
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
    ) -> Iterator[LLMStreamChunk]:
        """Route and stream chat completion tokens with automatic fallback."""
        mark("conversation.route_call_started")
        local_only = self.requires_local(messages)
        decision = self.route(self._routing_text(messages), local_only=local_only)
        clean_messages = self._clean_override_prefixes(messages)

        with self.priority.foreground():
            yield from self.manager.stream_with_fallback(
                clean_messages,
                preferred_provider=decision.provider_name,
                model=decision.model_name,
                temperature=temperature,
                max_tokens=max_tokens or 512,
                cancel_event=cancel_event,
                **({"allowed_providers": ("ollama",)} if local_only else {}),
            )

    @staticmethod
    def requires_local(messages: list[ChatMessage]) -> bool:
        """Keep private context on Ollama, including continuations and summaries."""
        return any(
            message.local_only or (
                message.role == "user"
                and re.search(r"(?m)^\s*/local(?:\s|$)", message.content) is not None
            )
            for message in messages
        )

    @staticmethod
    def _routing_text(messages: list[ChatMessage]) -> str:
        """Continue the last substantive user request through repeated follow-ups."""
        users = [message.content for message in messages if message.role == "user"]
        for text in reversed(users):
            if not re.fullmatch(
                r"(?:continue|go on|keep going|explain more)[.!?]*",
                text.strip(),
                re.I,
            ):
                return text
        return users[-1] if users else ""

    @staticmethod
    def _clean_override_prefixes(messages: list[ChatMessage]) -> list[ChatMessage]:
        """Strip /fast, /strong, /local command prefixes from message text."""
        cleaned: list[ChatMessage] = []
        for m in messages:
            if m.role == "user":
                content = re.sub(r"^/(fast|strong|deep|local)(?:\s+|$)", "", m.content)
                cleaned.append(ChatMessage(role=m.role, content=content, local_only=m.local_only))
            else:
                cleaned.append(m)
        return cleaned
