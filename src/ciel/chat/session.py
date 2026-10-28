"""Headless text conversation over the same persona, memory, and provider components."""

from __future__ import annotations

import getpass
import platform
import re
import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ciel.agent import MainAgent, ToolResult
from ciel.agent_tools import MainTools
from ciel.config import Settings
from ciel.controller import (
    conversation_control,
    conversation_only_request,
    is_explicit_web_request,
)
from ciel.conversation import contains_retired_local_command, strip_internal_reply_notes
from ciel.dialogue import DialogueContext
from ciel.internet import InternetService, _bounded, public_urls
from ciel.logging import get_logger
from ciel.memory import ConversationManager, MemoryStore
from ciel.memory.bridge import conversation_context, current_conversation_id
from ciel.memory.models import normalize_sources
from ciel.memory.service import MemoryService
from ciel.persona import (
    build_advanced_persona,
    load_persona_preferences,
)
from ciel.providers import get_model_router
from ciel.providers.intents import answer_clock_query
from ciel.providers.router import ModelRouter
from ciel.research import (
    evidence_followup,
    evidence_payload,
    previous_observations,
)
from ciel.screen.service import ScreenRequest, ScreenResult, ScreenService, screen_scope
from ciel.tool_progress import ToolProgress
from ciel.web_answers import web_answer_messages

logger = get_logger("chat.session")
REQUEST_TIMEOUT_SECONDS = 60.0
TEXT_RESPONSE_MAX_TOKENS = 4096
_OVERRIDE = re.compile(r"^/(?:fast|strong|deep)\s+", re.I)
_REASONING = re.compile(r"^/reasoning(?:\s|$)", re.I)
_REMINDER = re.compile(
    r"^(?:(?:please|can you|could you|would you)\s+)*(?:remind me\b|"
    r"(?:set|create|schedule|add)\s+(?:(?:a|an|the)\s+)?reminder\b)", re.I,
)
_DESKTOP_ACTION = re.compile(
    r"^(?:(?:please|can you|could you|would you)\s+)*"
    r"(?:open|launch|close|execute|run|shut down|shutdown|reboot)\s+", re.I,
)
_HELP = (
    "You can continue your voice conversations here and use the same saved facts. "
    "Memory changes follow your configured learning and confirmation choices.\n\n"
    "/memory — show saved personal details\n"
    "/cancel — discard a pending memory change\n"
    "/reset — clear this chat's history, keeping voice history and saved facts\n"
    "/fast, /strong, /deep — use your existing AI routing choices\n"
    "/reasoning [auto|on|off] — show or change the reasoning preference for supported models\n\n"
    "Ask me to look at your screen, or send a screenshot of your PC.\n"
    "Reminders, voice messages, and desktop actions are not available in chat yet."
)


class _RequestCancellation(threading.Event):
    """Combine a per-request deadline with the service's shutdown event."""

    def __init__(self, parent: threading.Event | None) -> None:
        super().__init__()
        self.parent = parent

    def is_set(self) -> bool:
        return super().is_set() or (self.parent is not None and self.parent.is_set())

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                break
            super().wait(0.05 if remaining is None else min(0.05, remaining))
        return self.is_set()


def _text_prompt(*, settings: Settings, memories: list[str]) -> str:
    """Use the common persona without probing audio devices or loading speech models."""
    return build_advanced_persona(
        user_name=getpass.getuser(),
        time_str=datetime.now().strftime("%A, %B %d, %Y, %I:%M %p"),
        os_distro=platform.system(), desktop_env="not queried for text chat",
        gpu_name="not queried", gpu_temp_c=0, gpu_free_mb=0, gpu_total_mb=0,
        cpu_cores=0, ram_used_gb=0, ram_total_gb=0,
        memory_db=settings.memory.db_path, recalled_memories=memories,
        preferred_name=settings.ciel_preferred_name,
        persona_preferences=load_persona_preferences(settings.ciel_persona_file),
        internet_enabled=settings.internet.enabled, channel="text",
    )


class ChatSession:
    """Serialize one owner's typed turns with a shared view of explicitly paired history.

    Transport adapters must verify owner identity before constructing this object.
    Application-owned memory handlers establish writes; generated text cannot do so.
    """

    def __init__(
        self, settings: Settings, *, channel: str, session_id: str,
        router: Any = None, store: MemoryStore | None = None,
    ) -> None:
        self.settings = settings
        self.channel = channel
        self.store = store or MemoryStore(
            db_path=settings.memory.db_path,
            workspace_path=settings.memory.workspace_path or None,
            workspace_enabled=settings.memory.workspace_enabled,
        )
        self.router = router if router is not None else get_model_router()
        self._base_session_id = session_id
        session_id = current_conversation_id(settings, session_id, store=self.store)
        self.conversation = ConversationManager(
            store=self.store, session_id=session_id,
            max_turns=settings.memory.max_short_term_turns,
            max_context_chars=settings.memory.max_context_chars,
            memory_config=settings.memory,
            context_resolver=lambda: conversation_context(settings, session_id, store=self.store),
        )
        self.memory = MemoryService(
            self.store, learning_mode=settings.memory.learning_mode,
            recall_enabled=settings.memory.recall_enabled,
            confirm_corrections=settings.memory.confirm_corrections,
        )
        self.context = DialogueContext(settings, self.store, self.conversation, self.memory)
        self.conversation.start_memory_lifecycle(self.router)
        self.internet = InternetService(settings.internet)
        self.progress: Callable[[str], None] = lambda _: None
        self._search_paused = False
        self._lock = threading.RLock()
        self.screen = ScreenService(settings, self.conversation)
        self.tool_progress = ToolProgress(self.router)
        self.tool_progress.warm(local_only=self.conversation.local_only_context())

    def _refresh_conversation(self) -> None:
        """Move a persistent paired chat to the current launch without deleting archives."""
        session_id = current_conversation_id(self.settings, self._base_session_id, store=self.store)
        if session_id == self.conversation.session_id:
            return
        self.memory.cancel_proposal()
        self.screen.close()
        self.context.tools.close()
        self.conversation.close()
        self.conversation = ConversationManager(
            store=self.store, session_id=session_id,
            max_turns=self.settings.memory.max_short_term_turns,
            max_context_chars=self.settings.memory.max_context_chars,
            memory_config=self.settings.memory,
            context_resolver=lambda: conversation_context(self.settings, session_id,
                                                          store=self.store),
        )
        self.context = DialogueContext(self.settings, self.store, self.conversation, self.memory)
        self.conversation.start_memory_lifecycle(self.router)
        self.screen = ScreenService(self.settings, self.conversation)
        self.internet.reset_session()
        self._search_paused = False

    def _finish(
        self, text: str, *, provider: str = "local", model: str = "chat",
        sources: list[dict[str, str]] | None = None,
        web_evidence: dict | None = None,
    ) -> str:
        self.conversation.add_turn("assistant", text, provider=provider, model=model,
                                   sources=sources, web_evidence=web_evidence)
        return text

    def reply(self, text: str, cancel_event: threading.Event | None = None) -> str:
        """Answer a verified owner's text, without ambient gating or audio initialization."""
        with self._lock:
            if cancel_event is not None and cancel_event.is_set():
                return ""
            if _REASONING.match(text.strip()):
                return self._reasoning_command(text.strip())
            self._refresh_conversation()
            text = text.strip()
            if not text:
                return "Please send a text message."
            if len(text) > 4096:
                return "Please split that into messages of at most 4,096 characters."
            if contains_retired_local_command(text):
                return "/local is no longer available. Send your message normally."
            if text == "/reset":
                self.memory.cancel_proposal()
                self._search_paused = False
                self.context.clear_web_preference()
                self.conversation.clear_session()
                self.internet.reset_session()
                return (
                    "This chat's history was cleared. Voice history and saved facts are kept. "
                    "I can still remember what we discussed in voice."
                )
            if text in {"/start", "/help"}:
                self.memory.cancel_proposal()
                return _HELP
            if text == "/cancel":
                self.memory.cancel_proposal()
                self.screen.clear()
                return "The pending memory change was cancelled."
            if text.startswith("/") and text != "/memory" and not _OVERRIDE.match(text):
                self.memory.cancel_proposal()
                return "I don't recognize that chat command. Use /help to see the available ones."
            query = "What do you remember about me?" if text == "/memory" else text
            local_text = _OVERRIDE.sub("", query)
            dialogue = self.conversation.get_gate_messages(limit=8)
            self.conversation.add_turn("user", query, origin="owner")
            control = conversation_control(local_text)
            if control is not None:
                if control == "stop":
                    self.screen.clear()
                self._search_paused = True
                self.context.pause_web(True)
                self.memory.cancel_proposal()
                if not conversation_only_request(local_text):
                    return self._finish({
                        "wait": "Take your time.", "stop": "Got it, quiet now.",
                        "no_search": "Okay, I won't search.",
                    }[control])
            elif is_explicit_web_request(local_text):
                self._search_paused = False
                self.context.pause_web(False)
                self.screen.clear()
            memory_scope = screen_scope(self.conversation)
            local = self.context.memory_reply(local_text, recent_messages=dialogue)
            if local is not None:
                if ((cancel_event is not None and cancel_event.is_set())
                        or memory_scope != screen_scope(self.conversation)):
                    return ""
                self.screen.clear()
                return self._finish(local, model="memory")
            clock = answer_clock_query(local_text)
            if clock is not None:
                self.screen.clear()
                return self._finish(clock, model="clock")
            if _REMINDER.match(local_text):
                self.screen.clear()
                return self._finish(
                    "Reminders aren't available in chat yet. I haven't scheduled one.",
                )
            visual = self.screen.resolve(local_text)
            if visual is not None:
                canceled = _RequestCancellation(cancel_event)
                scope = screen_scope(self.conversation)
                messages = self.conversation.get_active_messages(system_prompt=
                    self.context.system_prompt([], generate=_text_prompt))
                try:
                    result = self.screen.answer(
                        visual, messages, cancel_event=canceled,
                        current=lambda: scope == screen_scope(self.conversation),
                        progress=lambda tool, query: self.tool_progress.announce(
                            tool, query, self.progress, cancel_event=canceled,
                            current=lambda: scope == screen_scope(self.conversation),
                            local_only=ModelRouter.requires_local(messages),
                        ),
                    )
                    if (result is None or canceled.is_set()
                            or scope != screen_scope(self.conversation)
                            or not self.screen.still_valid(result)):
                        return ""
                    return self._finish_screen(result, visual)
                except Exception as error:
                    if canceled.is_set() or scope != screen_scope(self.conversation):
                        return ""
                    logger.warning("Screen request failed (%s).", type(error).__name__)
                    return self._finish("I couldn't access the screen just now. Please try again.",
                                        model="screen")
            if (_DESKTOP_ACTION.match(local_text) and not public_urls(local_text)
                    and not evidence_followup(local_text)):
                return self._finish("Desktop actions aren't available through chat yet.")

            canceled = _RequestCancellation(cancel_event)
            try:
                return self._generate(query, canceled)
            except Exception as error:
                if cancel_event is not None and cancel_event.is_set():
                    return ""
                logger.warning("Chat reply failed (%s).", type(error).__name__)
                return self._finish(
                    "That request took too long. Please try again."
                    if canceled.is_set() or isinstance(error, TimeoutError) else
                    "I couldn't finish that request. Please try again."
                )

    def _reasoning_command(self, text: str) -> str:
        """Inspect or change the router preference without generating or storing a turn."""
        parts = text.lower().split()
        if len(parts) > 2 or (len(parts) == 2 and parts[1] not in {"auto", "on", "off"}):
            return "Use /reasoning to check, or /reasoning auto, /reasoning on, /reasoning off."
        mode = getattr(self.router, "reasoning_mode", None)
        setter = getattr(self.router, "set_reasoning_mode", None)
        if mode not in {"auto", "on", "off"} or (len(parts) == 2 and not callable(setter)):
            return "Reasoning controls aren't available for this chat's model router."
        if len(parts) == 2:
            setter(parts[1])
            mode = self.router.reasoning_mode
        return (
            f"Reasoning preference: {mode}. This applies to supported models; "
            "it does not change the selected model."
        )

    def _finish_screen(self, result: ScreenResult, request: ScreenRequest) -> str:
        """Persist an admitted visual observation and retain its temporary attachment."""
        turn_id = self.conversation.add_turn(
            "assistant", result.text, provider=result.provider, model=result.model,
        )
        if not self.screen.mark(turn_id, request=request, **(
            {"expected_revision": result.task_revision} if result.task_revision else {}
        )):
            return ""
        if result.image is not None:
            from ciel.chat.telegram import ChatReply, OutgoingPhoto

            return ChatReply(result.text, photos=(OutgoingPhoto(
                data=result.image, expires_at=result.expires_at,
                valid=lambda: self.screen.image_valid(result),
            ),))
        return result.text

    def _native_screen_request(self, query: str) -> ScreenRequest | None:
        """Keep a native visual call within the owner's actual visual task."""
        return self.screen.resolve_native(query)

    def _generate(self, query: str, canceled: threading.Event) -> str:
        from ciel.memory.tools import memory_lookup_request

        self.context.tools.start()
        scope = self.conversation.context_session_ids()
        version = screen_scope(self.conversation)
        task = self.context.task_context(query)
        explanation = self.context.failure_explanation(task)
        if explanation is not None:
            if (canceled.is_set() or scope != self.conversation.context_session_ids()
                    or version != screen_scope(self.conversation)):
                return ""
            return self._finish(explanation)
        contextual_web = task.task_type == "web" and task.kind in {
            "retry", "continue", "correction", "detail", "followup",
        }
        if (contextual_web and not canceled.is_set()
                and scope == self.conversation.context_session_ids()
                and version == screen_scope(self.conversation)):
            self.context.pause_web(False)
        self._search_paused = self.context.web_paused()
        recalled = self.context.recall(query)
        prompt = self.context.system_prompt(recalled, generate=_text_prompt, task=task)
        prompt += (
            f"\nCurrent channel: {self.channel} text chat. This channel supports conversation, "
            "clock questions, saved memory, public web retrieval when enabled, and admitted "
            "screen requests. No desktop action, reminder, scheduled delivery, voice playback, "
            "or arbitrary file access was performed. Those features are unavailable in this "
            "channel. Do not claim success or promise future work. The latest user message "
            "was typed intentionally. Earlier screen discussions are historical text; "
            "current visual access requires a successful screen result for this turn."
        )
        messages = self.conversation.get_active_messages(system_prompt=prompt)
        visual_revision = [self.screen.task_revision() if self.screen.active() else None]

        def current() -> bool:
            return (not canceled.is_set() and scope == self.conversation.context_session_ids()
                    and version == screen_scope(self.conversation)
                    and (visual_revision[0] is None or
                         visual_revision[0] == self.screen.task_revision()))

        if not current():
            return ""
        private = ModelRouter.requires_local(messages)
        previous = None if memory_lookup_request(query) else previous_observations(
            query, self.conversation,
        )
        messages[0].content += (
            "\nApplication internet status for this turn: "
            + ("paused at the user's request. Do not search unless explicitly asked again."
               if self._search_paused else
               "blocked by the local-only conversation policy." if private else
               "disabled in settings." if not self.settings.internet.enabled else
               "prior retrieved evidence is supplied below." if previous is not None else
               "no lookup has run. Use an available native tool when evidence is needed.")
        )
        if previous is not None and previous.attempted:
            evidence = web_answer_messages(messages, previous, channel="text")
            messages[0].content = evidence[0].content
            messages[-1:-1] = evidence[1:-1]
        model_cancel = _RequestCancellation(canceled)

        def announce(tool: str, topic: str) -> bool:
            return self.tool_progress.announce(
                tool, topic, self.progress, cancel_event=model_cancel,
                current=current, local_only=private,
            )

        def check_screen() -> ToolResult:
            visual = self._native_screen_request(_OVERRIDE.sub("", query))
            if visual is None:
                return ToolResult(
                    {"error": "No current visual request was admitted."},
                    terminal_text="Did you want me to look at your screen?", is_error=True,
                )
            def screen_current() -> bool:
                return (not canceled.is_set()
                        and scope == self.conversation.context_session_ids()
                        and version == screen_scope(self.conversation))

            result = self.screen.answer(
                visual, messages, cancel_event=model_cancel, current=screen_current,
                progress=lambda tool, topic: self.tool_progress.announce(
                    tool, topic, self.progress, cancel_event=model_cancel,
                    current=screen_current, local_only=private,
                ),
            )
            if result is not None:
                visual_revision[0] = result.task_revision or self.screen.task_revision()
            if (result is None or model_cancel.is_set() or not current()
                    or not self.screen.still_valid(result)):
                return ToolResult({"error": "The visual request is no longer current."},
                                  is_error=True)
            return ToolResult(
                result.text, terminal_text=result.text, source="screen", value=result,
                metadata={"screen_request": visual},
            )

        tools = MainTools(
            self.settings, self.router, messages, query, context=self.context,
            internet=self.internet, current=current, cancel_event=model_cancel,
            web_paused=self.context.web_paused, screen=check_screen, progress=announce,
            channel="text", previous=previous,
            objective=task.request, contextual_web=contextual_web,
            explanation_only=task.kind == "explanation",
        )
        agent = MainAgent(
            self.router, tools.schemas, tools.execute, is_current=tools.is_current,
            buffer_answer=tools.buffer_answer, answer_filter=tools.answer_filter,
            answer_instructions=tools.answer_instructions,
            response_deadline=tools.response_deadline,
            max_steps=max(3, min(5, self.settings.controller.max_tool_steps)),
        )
        try:
            response = _bounded(
                lambda: agent.send(messages, temperature=0.7, max_tokens=TEXT_RESPONSE_MAX_TOKENS,
                                   cancel_event=model_cancel),
                time.monotonic() + REQUEST_TIMEOUT_SECONDS, canceled,
            )
        finally:
            model_cancel.set()
        if canceled.is_set():
            raise TimeoutError("Chat request cancelled")
        if not current() or agent.last_result.canceled:
            return ""
        visual_result = tools.screen_result
        if visual_result is not None:
            if (tools.screen_request is None or not self.screen.still_valid(visual_result)):
                return ""
            return self._finish_screen(visual_result, tools.screen_request)
        reply = strip_internal_reply_notes(response.content).strip()
        if not reply:
            raise ValueError("Provider returned no chat text")
        lookup = tools.lookup
        sources = normalize_sources([source.to_dict() for source in agent.last_result.sources])
        if not reply.rstrip().endswith("?"):
            self.screen.clear()
        self._finish(reply, provider=response.provider, model=response.model, sources=sources,
                     web_evidence=evidence_payload(lookup))
        self.conversation.schedule_summary(router_or_provider=self.router)
        if sources:
            reply += "\n\nSources:\n" + "\n".join(
                f"{index}. {source['title']}: {source['url']}"
                for index, source in enumerate(sources, start=1)
            )
        return reply

    def close(self) -> None:
        """Discard unconfirmed changes and close this thread's database connection."""
        self.memory.cancel_proposal()
        self.screen.close()
        self.tool_progress.close()
        self.context.tools.close()
        self.conversation.close()
        self.store.close()
