"""Conservative local fact recognition, keyed updates, and explicit forgetting."""

import hashlib
import re
import time
from datetime import datetime
from typing import Literal

from ciel.memory.models import MemoryItem, MemoryType
from ciel.memory.projects import parse_project_entry
from ciel.memory.recall import rank_memories, terms
from ciel.memory.store import MemoryConflictError, MemoryStore

MONTHS = {
    name: number
    for number, names in enumerate(
        (
            "jan january",
            "feb february",
            "mar march",
            "apr april",
            "may",
            "jun june",
            "jul july",
            "aug august",
            "sep sept september",
            "oct october",
            "nov november",
            "dec december",
        ),
        1,
    )
    for name in names.split()
}
PROFILE_TOPICS = {
    "current project": "current_project", "active project": "current_project",
    "goal": "goal", "occupation": "occupation", "job": "occupation",
    "home city": "home_city", "timezone": "timezone", "time zone": "timezone",
    "preferred language": "preferred_language", "response style": "response_style",
}
PROFILE_STATEMENTS = {
    "user:current_project": "my current project is",
    "user:goal": "my goal is", "user:occupation": "my occupation is",
    "user:home_city": "my home city is", "user:timezone": "my timezone is",
    "user:preferred_language": "my preferred language is",
    "user:response_style": "my response style is",
}

RECALL_PAUSED = "Saved-memory recall is paused. Say 'resume memory recall' to use saved facts."
_FAVORITE = re.compile(
    r"my\s+favou?rite\s+(game|food|color|colour|drink|movie|language)\s+"
    r"(?:is(?: now)?|has changed to)\s+(.+)", re.I,
)
_NAME = re.compile(r"(?:my name is|call me|i prefer to be called)\s+(.+)", re.I)
_QUESTION_OR_REPORT = re.compile(
    r"^(?:what|why|when|where|who|whom|whose|which|how|do|does|did|is|are|am|was|were|"
    r"can|could|would|should|will|have|has|had|may|might|if|maybe|she|he)\b", re.I,
)


def _clean_statement(text: str) -> str:
    """Remove a discourse correction prefix without interpreting its subject."""
    corrected = re.sub(
        r"^(?:(?:no|actually|correction)[,.\s]+|i meant\s+)+", "", text, flags=re.I,
    ).strip()
    if corrected != text.strip():
        corrected = re.sub(r"^like[, ]+(?=my\b)", "", corrected, flags=re.I)
    corrected = re.sub(r"^(?:my\s+){2,}", "my ", corrected, flags=re.I)
    return corrected.rstrip(".!")



def _is_question_or_quotation(text: str) -> bool:
    """Treat unpunctuated questions and quoted speech as conversational data."""
    return bool("?" in text or text.startswith(('"', "'", "“", "‘"))
                or _QUESTION_OR_REPORT.match(text))


def _clear_automatic_statement(text: str, key: str = "") -> bool:
    """Avoid combining another clause, uncertainty, or a command into one profile value."""
    if re.search(r"\b(?:maybe|probably|perhaps|possibly|if|unless|or|but)\b", text, re.I):
        return False
    if re.search(
        r"\b(?:and|then)\s+(?:(?:also|then)\s+)?"
        r"(?:i|we|you|my|our|it|they|he|she|can|could|would|will|please|open|"
        r"remember|forget|tell|show|search)\b|"
        r"(?:[,;]\s*|\.\s+)(?:i|we|you|my|our|it|they|he|she)\b", text, re.I,
    ):
        return False
    if key == "user:home_city" and re.search(
        r"\b(?:until|while|when|during)\b", text, re.I,
    ):
        return False
    return True


def _valid_name(value: str) -> bool:
    """Keep schedule requests and speculative values out of the conversational name slot."""
    return bool(value and len(value.split()) <= 4 and not re.match(
        r"^(?:later|tomorrow|back|when|if|at|after|on|not|unknown|unsure|maybe)\b", value, re.I,
    ))


def parse_profile_fact(text: str) -> tuple[str, str, str, MemoryType] | None:
    """Recognize bounded first-person profile statements without guessing intent."""
    patterns = (
        (r"my (?:current|active) project (?:is(?: now)?|has changed to) (.+)",
         "current_project", "Your current project is", MemoryType.PROJECT),
        (r"my (?:current )?goal is (.+)", "goal", "Your goal is", MemoryType.PROJECT),
        (r"(?:i work as|my (?:job|occupation) is) (.+)",
         "occupation", "Your occupation is", MemoryType.FACT),
        (r"(?:i live in|my home city is) (.+)",
         "home_city", "Your home city is", MemoryType.FACT),
        (r"my time\s?zone is (.+)", "timezone", "Your timezone is", MemoryType.FACT),
        (r"my preferred language is (.+)",
         "preferred_language", "Your preferred language is", MemoryType.PREFERENCE),
        (r"(?:my response style is|i prefer) (.+? (?:answers|replies|responses))",
         "response_style", "Your response style is", MemoryType.PREFERENCE),
    )
    for pattern, key, description, kind in patterns:
        match = re.fullmatch(pattern, text, re.I)
        if match:
            value = match.group(1).strip()
            if not 2 <= len(value) <= 180 or re.match(
                r"^(?:not|unknown|unsure|maybe|probably)\b", value, re.I,
            ):
                return None
            return f"user:{key}", value, f"{description} {value}.", kind
    return None


def parse_project_date(text: str, now: datetime) -> str | None:
    """Resolve explicit month/day dates and ISO dates; never infer a date from 'day three'."""
    iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso:
        try:
            date = datetime.strptime(iso.group(1), "%Y-%m-%d").date()
            return date.isoformat() if date <= now.date() else None
        except ValueError:
            return None
    match = re.search(
        r"\b(" + "|".join(MONTHS) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?"
        r"(?:,?\s+(\d{4}))?\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    try:
        date = datetime(
            int(match.group(3) or now.year),
            MONTHS[match.group(1).lower()],
            int(match.group(2)),
        ).date()
        return date.isoformat() if date <= now.date() else None
    except ValueError:
        return None


def is_memory_statement(text: str) -> bool:
    """Recognize complete supported declarations and commands without authorizing a write.

    Controllers may use this to select the local handler. The handler still applies learning
    mode, confirmation requirements, and independent address/transcription authorization.
    Short contextual corrections are intentionally left to an existing pending exchange.
    """
    original = text.strip().replace("’", "'")
    if re.fullmatch(
        r"(?:please\s+)?(?:remember(?:\s+that)?|note\s+that)\s+.+|"
        r"(?:please\s+)?forget(?:\s+.+)?[.!?]*", original, re.I,
    ):
        return True
    clean = _clean_statement(original)
    if _is_question_or_quotation(clean):
        return False
    if parse_project_entry(clean) is not None:
        return True
    profile = parse_profile_fact(clean)
    if profile is not None:
        return _clear_automatic_statement(clean, profile[0])
    favorite = _FAVORITE.fullmatch(clean)
    if favorite:
        return bool(_clear_automatic_statement(clean) and not re.match(
            r"^(?:not|unknown|unsure|maybe)\b", favorite.group(2).strip(), re.I,
        ))
    name = _NAME.fullmatch(clean)
    if name:
        return _valid_name(name.group(1).strip()) and _clear_automatic_statement(clean)
    return bool(_clear_automatic_statement(clean) and re.match(
        r"^(?:(?:the|our|my)\s+)?(?:ciel(?:'s)?\s+)?"
        r"(?:project|development|first commit)\b.*\b"
        r"(?:started|began|begin|from|was)\b", clean, re.I,
    ))


class MemoryService:
    """Persist direct user statements; unknown or ambiguous speech stays conversational."""

    def __init__(
        self, store: MemoryStore, *,
        learning_mode: Literal["confirm", "automatic", "explicit"] = "confirm",
        recall_enabled: bool = True,
        confirm_corrections: bool = True,
    ) -> None:
        if learning_mode not in {"confirm", "automatic", "explicit"}:
            raise ValueError("Unknown memory learning mode")
        self.store = store
        self.learning_mode = learning_mode
        self.confirm_corrections = confirm_corrections
        self._default_recall_enabled = recall_enabled
        self.focus_key: str | None = None
        self._pending: tuple[str, MemoryItem, float, str | None] | None = None

    @property
    def recall_enabled(self) -> bool:
        """Use the persistent pause override, falling back to configuration."""
        return self.store.get_setting("recall_enabled", self._default_recall_enabled) is not False

    def cancel_proposal(self) -> None:
        """Discard a proposal when dialogue ends or another local task takes over."""
        self._pending = None
        self.focus_key = None

    @property
    def has_pending_confirmation(self) -> bool:
        """Expose whether a live proposal is waiting for an explicit user confirmation."""
        return self._pending is not None and time.monotonic() <= self._pending[2]

    def observe_topic(self, text: str) -> None:
        """Track an explicit topic for a subsequent short correction."""
        self.focus_key = None
        lowered = text.casefold()
        for topic, key in PROFILE_TOPICS.items():
            if re.search(r"\b" + topic + r"\b", lowered):
                self.focus_key = f"user:{key}"
                return
        if re.search(r"\b(?:project|development|developing)\b", lowered):
            self.focus_key = "ciel:project_start_date"
        if "first commit" in lowered:
            self.focus_key = "ciel:first_commit_date"
        match = re.search(
            r"\bfavou?rite\s+(game|food|color|colour|drink|movie|language)\b", lowered
        )
        if match:
            self.focus_key = "user:favorite_" + match.group(1).replace("colour", "color")

    def handle(
        self, text: str, now: datetime | None = None, *, authorized: bool = True,
    ) -> str | None:
        """Propose conversational facts; save explicit commands or timely confirmations.

        Callers must authorize writes independently of inferred dialogue intent.
        Pending proposals live only in RAM and survive neither restart nor topic changes.
        """
        if not authorized:
            self.cancel_proposal()
            return None
        now = now or datetime.now()
        original = text.strip().replace("’", "'")
        control = re.fullmatch(
            r"(?:please\s+)?(pause|disable|resume|enable)\s+(?:saved[- ]memory|memory)\s+"
            r"recall[.!?]*", original, re.I,
        )
        if control:
            self.cancel_proposal()
            enabled = control.group(1).casefold() in {"resume", "enable"}
            self.store.set_setting("recall_enabled", enabled)
            return ("Saved-memory recall has resumed." if enabled else
                    "Saved-memory recall is paused. Your saved facts are kept. "
                    "Recent conversation and learning remain active.")
        pending, self._pending = self._pending, None
        if pending is not None and time.monotonic() <= pending[2]:
            answer = original.casefold().strip(" .!?")
            if answer in {"yes", "yeah", "yes please", "save it", "remember that", "confirm"}:
                current = self.store.get_fact(pending[0])
                if (current.content if current else None) != pending[3]:
                    proposal = self._propose(
                        pending[0], pending[1],
                    )
                    return proposal + " The saved fact changed since the previous question."
                pending[1].source = "user_confirmed"
                return self._save(pending[0], pending[1], expected_content=pending[3])
            if answer in {"no", "no thanks", "don't save it", "do not save it", "cancel",
                          "never mind", "forget that"}:
                self.focus_key = None
                return "Okay, I won't save that fact."
        elif pending is not None:
            self.focus_key = None
        if re.fullmatch(
            r"(?:please\s+)?(?:remember(?:\s+(?:that|it))?|note\s+that)[.!?]*", original, re.I,
        ):
            return "What would you like me to remember? Please state the detail."
        explicit = re.match(
            r"^(?:please\s+)?(?:remember(?:\s+that)?|note\s+that)\s+(.+)$",
            original,
            re.IGNORECASE,
        )
        forget = re.match(
            r"^(?:please\s+)?forget(?:\s+that|\s+about)?\s+(.+)$",
            original,
            re.IGNORECASE,
        )
        if forget:
            return self._forget(forget.group(1))
        if re.fullmatch(r"(?:please\s+)?forget(?:\s+(?:that|it))?[.!]?", original, re.I):
            return self._forget("that")
        statement = explicit.group(1) if explicit else original
        correction = bool(
            re.match(r"^(?:no[,.\s]|actually\b|i meant\b|correction\b)", statement, re.I)
        )
        clean = _clean_statement(statement)
        # Questions, quotations and speculative statements aren't facts.
        if not explicit and _is_question_or_quotation(clean):
            self.observe_topic(original)
            profile_recall = re.fullmatch(
                r"(?:what(?:'s| is)|do you (?:remember|know))\s+my "
                r"(?:" + "|".join(PROFILE_TOPICS) + r")[?.!]*", original, re.I,
            )
            simple_recall = re.fullmatch(
                r"(?:what(?:'s| is)|which is|do you (?:remember|know))\s+"
                r"my favou?rite (?:game|food|color|colour|drink|movie|language)[?.!]*",
                original,
                re.I,
            )
            if simple_recall and self.focus_key:
                if not self.recall_enabled:
                    return RECALL_PAUSED
                remembered = self.store.get_fact(self.focus_key)
                if remembered is not None:
                    return self.store.redact_forgotten(remembered.content)
            if profile_recall and self.focus_key:
                if not self.recall_enabled:
                    return RECALL_PAUSED
                remembered = self.store.get_fact(self.focus_key)
                if remembered is not None:
                    return self.store.redact_forgotten(remembered.content)
            if re.fullmatch(
                r"(?:what(?:'s| is) my name|what do you call me)[?.!]*",
                original,
                re.I,
            ):
                if not self.recall_enabled:
                    return RECALL_PAUSED
                remembered = self.store.get_fact("user:preferred_name")
                if remembered is not None:
                    return self.store.redact_forgotten(remembered.content)
            return None
        key = None
        value = None
        kind = MemoryType.FACT
        metadata: dict[str, object] = {}
        inferred_subject = False
        favorite = _FAVORITE.fullmatch(clean)
        name = _NAME.fullmatch(clean)
        profile = parse_profile_fact(clean)
        project_entry = parse_project_entry(clean)
        if project_entry is not None:
            key, project_item = project_entry
            value = project_item.metadata.get("value")
            metadata = dict(project_item.metadata)
            kind, clean = project_item.memory_type, project_item.content
        elif favorite:
            topic = favorite.group(1).lower().replace("colour", "color")
            key, value = "user:favorite_" + topic, favorite.group(2).strip()
            clean = f"Your favorite {topic} is {value}."
            kind = MemoryType.PREFERENCE
        elif name:
            key, value = "user:preferred_name", name.group(1).strip()
            if not _valid_name(value):
                self.focus_key = None
                return None
            clean = f"Your preferred name is {value}."
            kind = MemoryType.PREFERENCE
        elif profile:
            key, value, clean, kind = profile
        elif re.search(
            r"\b(?:project|development|developing|building|first commit|start date)\b",
            clean,
            re.I,
        ) or self.focus_key in ("ciel:project_start_date", "ciel:first_commit_date"):
            value = parse_project_date(clean, now)
            if value and (
                re.search(r"\b(?:start|started|began|begin|from|commit|meant)\b", clean, re.I)
                or (correction and self.focus_key and self.focus_key.startswith("ciel:"))
            ):
                key = (
                    "ciel:first_commit_date"
                    if "first commit" in clean.lower()
                    or (correction and self.focus_key == "ciel:first_commit_date")
                    else "ciel:project_start_date"
                )
                kind = MemoryType.PROJECT
                inferred_subject = True
                clean = (
                    f"CIEL's first commit was on {value}."
                    if key.endswith("first_commit_date")
                    else f"CIEL project development began on {value}."
                )
        if key is None and correction and self.focus_key:
            short = re.fullmatch(r"(?:it(?:'s| is)\s+)?(.{1,80})", clean, re.I)
            if (
                short
                and (
                    self.focus_key.startswith("user:favorite_")
                    or self.focus_key in PROFILE_STATEMENTS
                )
                and not re.match(
                    r"^(?:i|you|that|this|wrong|incorrect|stop|wait|hold|what|huh|"
                    r"sorry|never|not|don'?t)\b",
                    short.group(1),
                    re.I,
                )
            ):
                inferred_subject = True
                key, value = self.focus_key, short.group(1).strip()
                if key.startswith("user:favorite_"):
                    topic = key.split("favorite_", 1)[1]
                    clean = f"Your favorite {topic} is {value}."
                    kind = MemoryType.PREFERENCE
                else:
                    profile = parse_profile_fact(f"{PROFILE_STATEMENTS[key]} {value}")
                    if profile is None:
                        return None
                    key, value, clean, kind = profile
        if key is None:
            if not explicit:
                self.observe_topic(original)
                return None
            digest = hashlib.sha256(re.sub(r"\W+", " ", clean.casefold()).encode()).hexdigest()[:20]
            key = "user:note_" + digest
        if (
            key.startswith(("user:favorite_", "user:preferred_name"))
            and value
            and re.match(
                r"^(?:not|unknown|unsure|maybe)\b",
                value,
                re.I,
            )
        ):
            return None
        item = MemoryItem(
            content=clean, memory_type=kind, source="user_explicit",
            metadata={**metadata, **({"value": value} if value is not None else {})},
        )
        if not explicit:
            if self.learning_mode == "explicit":
                self.observe_topic(original)
                return None
            # Automatic learning accepts one clear declaration, never a compound command.
            if self.learning_mode == "automatic":
                if project_entry is None and not _clear_automatic_statement(
                    _clean_statement(statement), key,
                ):
                    self.focus_key = None
                    return None
                item.source = "user_statement"
                old = self.store.get_fact(key)
                if old is not None and old.content == clean:
                    self.focus_key = key
                    return f"I already remember that. {clean}"
                if inferred_subject or (old is not None and self.confirm_corrections):
                    return self._propose(key, item)
                return self._save(key, item, expected_content=old.content if old else None)
            old = self.store.get_fact(key)
            if old is not None and old.content == clean:
                self.focus_key = key
                return f"I already remember that. {clean}"
            return self._propose(key, item)
        old = self.store.get_fact(key)
        if old is not None and old.content != clean and self.confirm_corrections:
            return self._propose(key, item)
        return self._save(key, item, expected_content=old.content if old else None)

    def _propose(self, key: str, item: MemoryItem) -> str:
        """Ask before overwriting an existing fact or saving in confirmation mode."""
        self.focus_key = key
        current = self.store.get_fact(key)
        self._pending = (key, item, time.monotonic() + 60.0, current.content if current else None)
        return f"Should I remember this? {item.content} Say yes to save it or no to skip it."

    def _save(self, key: str, item: MemoryItem, *, expected_content: str | None) -> str:
        """Persist exactly the explicit or confirmed fact and report the result."""
        old = self.store.get_fact(key)
        try:
            self.store.upsert_fact(key, item, expected_content=expected_content)
        except MemoryConflictError:
            return self._propose(key, item) + " The saved fact changed before that update."
        self.focus_key = key
        action = "updated" if old and old.content != item.content else "saved"
        return f"I've {action} that. {item.content}"

    def _forget(self, topic: str) -> str:
        """Resolve specific saved topics, refusing an ambiguous broad deletion."""
        topic = topic.strip().rstrip(".!?")
        if topic.strip().casefold() not in ("that", "it", "the last fact"):
            self.focus_key = None
            self.observe_topic(topic)
            timeline_topic = re.fullmatch(
                r"(?:(?:my|your|our|the)\s+)?(?:ciel(?:'s)?\s+)?"
                r"(?:project (?:start date|start)|development (?:start date|date|day)|"
                r"first commit(?: date)?)", topic, re.I,
            )
            profile_project = re.fullmatch(r"my (?:current|active) project", topic, re.I)
            if (self.focus_key or "").startswith("ciel:") and not timeline_topic:
                self.focus_key = None
            if re.search(r"\bproject\b", topic, re.I) and not (
                timeline_topic or profile_project
            ):
                # A named project's goal or next step must not select the user's
                # general goal or CIEL's unrelated development-date anchor.
                self.focus_key = None
        item = self.store.get_fact(self.focus_key) if self.focus_key else None
        if self.focus_key and item is None:
            self.focus_key = None
            return "I couldn't find a saved fact matching that. Which fact should I forget?"
        matches = self.store.find_memories_containing(topic, limit=2)
        if item is not None:
            matches = [item]
        if not matches:
            ranked = rank_memories(self.store, topic, limit=2)
            query_terms = terms(topic)
            ranked = [
                pair for pair in ranked if query_terms and query_terms <= terms(pair[1].content)
            ]
            if ranked and (len(ranked) == 1 or ranked[0][0] > ranked[1][0] * 1.5):
                matches = [ranked[0][1]]
        if not matches:
            return "I couldn't find a saved fact matching that. Which fact should I forget?"
        if len(matches) > 1:
            return "That matches several saved facts. Could you name the specific one to forget?"
        deleted = self.store.forget_memories(matches)
        self.focus_key = None
        return "I've forgotten that saved fact." if deleted else "That fact was already removed."
