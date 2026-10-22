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

from ciel.config import Settings
from ciel.controller import (
    AIController,
    ControllerDecision,
    conversation_control,
    conversation_only_request,
    is_explicit_web_request,
)
from ciel.conversation import contains_retired_local_command, strip_internal_reply_notes
from ciel.dialogue import DialogueContext
from ciel.internet import InternetResult, InternetService, _bounded, public_urls
from ciel.logging import get_logger
from ciel.memory import ConversationManager, MemoryStore
from ciel.memory.bridge import conversation_context
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
    ResearchCancellation,
    evidence_followup,
    evidence_payload,
    previous_observations,
)
from ciel.screen.service import ScreenService, screen_scope
from ciel.tool_progress import ToolProgress

logger = get_logger("chat.session")
REQUEST_TIMEOUT_SECONDS = 60.0
_OVERRIDE = re.compile(r"^/(?:fast|strong|deep)\s+", re.I)
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
    "/fast, /strong, /deep — use your existing AI routing choices\n\n"
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
        self.store = store or MemoryStore(db_path=settings.memory.db_path)
        self.router = router if router is not None else get_model_router()
        self.conversation = ConversationManager(
            store=self.store, session_id=session_id,
            max_turns=settings.memory.max_short_term_turns,
            max_context_chars=settings.memory.max_context_chars,
            context_resolver=lambda: conversation_context(settings, session_id),
        )
        self.memory = MemoryService(
            self.store, learning_mode=settings.memory.learning_mode,
            recall_enabled=settings.memory.recall_enabled,
            confirm_corrections=settings.memory.confirm_corrections,
        )
        self.context = DialogueContext(settings, self.store, self.conversation, self.memory)
        self.internet = InternetService(settings.internet)
        self.progress: Callable[[str], None] = lambda _: None
        self._search_paused = False
        self._lock = threading.RLock()
        self.screen = ScreenService(settings, self.conversation)
        self.tool_progress = ToolProgress(self.router)
        self.tool_progress.warm(local_only=self.conversation.local_only_context())

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
                return (
                    "This chat's history was cleared. Voice history and saved facts are kept. "
                    "I can still remember what we discussed in voice."
                )
            if text in {"/start", "/help"}:
                self.memory.cancel_proposal()
                return _HELP
            if text == "/cancel":
                self.memory.cancel_proposal()
                return "The pending memory change was cancelled."
            if text.startswith("/") and text != "/memory" and not _OVERRIDE.match(text):
                self.memory.cancel_proposal()
                return "I don't recognize that chat command. Use /help to see the available ones."
            query = "What do you remember about me?" if text == "/memory" else text
            local_text = _OVERRIDE.sub("", query)
            dialogue = self.conversation.get_gate_messages(limit=8)
            self.conversation.add_turn("user", query)
            control = conversation_control(local_text)
            if control is not None:
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
            local = self.context.memory_reply(local_text, recent_messages=dialogue)
            if local is not None:
                return self._finish(local, model="memory")
            clock = answer_clock_query(local_text)
            if clock is not None:
                return self._finish(clock, model="clock")
            if _REMINDER.match(local_text):
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
                    turn_id = self.conversation.add_turn(
                        "assistant", result.text, provider=result.provider, model=result.model,
                    )
                    self.screen.mark(turn_id)
                    if result.image is not None:
                        from ciel.chat.telegram import ChatReply, OutgoingPhoto

                        return ChatReply(result.text, photos=(OutgoingPhoto(
                            data=result.image, expires_at=result.expires_at,
                            valid=lambda: self.screen.image_valid(result),
                        ),))
                    return result.text
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

    def _generate(self, query: str, canceled: threading.Event) -> str:
        scope = self.conversation.context_session_ids()
        version = screen_scope(self.conversation)
        self._search_paused = self.context.web_paused()
        recalled = self.context.recall(query)
        prompt = self.context.system_prompt(recalled, generate=_text_prompt)
        prompt += (
            f"\nCurrent channel: {self.channel} text chat. This channel supports conversation, "
            "clock questions, saved memory, and public web retrieval when enabled. "
            "No desktop action, reminder, scheduled delivery, voice playback, or file access "
            "was performed. Those features are unavailable in this channel. Do not claim "
            "success or promise future work. The latest user message was typed intentionally."
            " No screenshot was obtained for this turn. Earlier screen discussions are "
            "historical text, not current visual access."
        )
        messages = self.conversation.get_active_messages(system_prompt=prompt)
        if scope != self.conversation.context_session_ids():
            return ""
        private = ModelRouter.requires_local(messages)
        previous = previous_observations(query, self.conversation)
        research_cancel = ResearchCancellation(canceled, lambda: (
            (previous is not None or not self.context.web_paused())
            and scope == self.conversation.context_session_ids()
            and version == screen_scope(self.conversation)
        ))
        announced = False

        def announce(tool: str, topic: str) -> bool:
            nonlocal announced
            announced = self.tool_progress.announce(
                tool, topic, self.progress, cancel_event=research_cancel,
                current=lambda: not research_cancel.refresh(), local_only=private,
            )
            return announced

        lookup = previous if previous is not None else (
            InternetResult() if self._search_paused else self.internet.prepare(
                query, recent_messages=messages[1:], memories=recalled,
                cancel_event=research_cancel, local_only=private,
                before_lookup=announce,
            )
        )
        if canceled.is_set():
            raise TimeoutError("Chat request cancelled")
        if version != screen_scope(self.conversation):
            return ""
        if lookup.attempted and research_cancel.refresh():
            return ""
        if lookup.reply is not None:
            return self._finish(lookup.reply, model="internet",
                                web_evidence=evidence_payload(lookup))
        if lookup.attempted:
            addresses = public_urls(lookup.query)
            initial = ControllerDecision(
                action="read_page" if addresses else "search", confidence=1,
                query=addresses[0] if addresses else lookup.query or query,
                research=True,
            )
            report = AIController(self.router, self.settings.controller).run_web(
                initial, query, messages, self.internet, self.progress,
                lambda: not research_cancel.refresh(), cancel_event=research_cancel,
                initial_lookup=lookup, channel="text", evidence_only=previous is not None,
                announced=announced,
            )
            if report.canceled or research_cancel.refresh():
                return ""
            sources = normalize_sources([source.to_dict() for source in report.sources])
            self._finish(report.text, provider=report.provider, model=report.model,
                         sources=sources, web_evidence=evidence_payload(report.lookup))
            self.conversation.schedule_summary(router_or_provider=self.router)
            return report.text + ("\n\nSources:\n" + "\n".join(
                f"{index}. {source['title']}: {source['url']}"
                for index, source in enumerate(sources, 1)
            ) if sources else "")
        messages[0].content += (
            "\nApplication internet status for this turn: "
            + ("paused at the user's request. Do not search unless explicitly asked again."
               if self._search_paused else
               "blocked by the local-only conversation policy." if private else
               "disabled in settings." if not self.settings.internet.enabled else
               "retrieved evidence is supplied below." if lookup.attempted else
               "no lookup was performed. Do not claim a search ran.")
        )
        model_cancel = _RequestCancellation(canceled)
        try:
            response = _bounded(
                lambda: self.router.send(messages, temperature=0.7, max_tokens=800,
                                         cancel_event=model_cancel),
                time.monotonic() + REQUEST_TIMEOUT_SECONDS, canceled,
            )
        finally:
            model_cancel.set()
        if canceled.is_set():
            raise TimeoutError("Chat request cancelled")
        if scope != self.conversation.context_session_ids():
            return ""
        reply = strip_internal_reply_notes(response.content).strip()
        if not reply:
            raise ValueError("Provider returned no chat text")
        sources = normalize_sources([source.to_dict() for source in lookup.sources])
        self._finish(reply, provider=response.provider, model=response.model, sources=sources)
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
        self.store.close()
