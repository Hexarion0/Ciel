"""Deterministic projections of the owner's current task across conversational turns.

The projection is question data for a responding model. It grants no permission to
browse, read screens, change memory, or resume an action that the owner canceled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

from ciel.providers.base import ChatMessage
from ciel.search_context import (
    has_search_request,
    replace_search_subject,
    search_request_body,
    subject_correction,
)

TaskKind = Literal[
    "new", "retry", "continue", "correction", "detail", "followup", "explanation", "wait", "stop",
    "acceptance",
]
TaskType = Literal["conversation", "web", "screen"]

_PREFIX = r"(?:(?:okay|ok|yes|yeah|now|please)[, ]+)*"
_RETRY = re.compile(
    r"^" + _PREFIX + r"(?:(?:can|could|would|will)\s+(?:you|u)\s+)?"
    r"(?:please\s+)?(?:again|retry|try\s+(?:it\s+)?again|"
    r"(?:search|check|look|read|look\s+(?:it\s+)?up)\s+(?:it\s+)?again|"
    r"double[- ]check(?:\s+(?:it|that))?)(?:\s+please)?[.!? ]*$", re.I,
)
_CONTINUE = re.compile(
    r"^" + _PREFIX + r"(?:(?:can|could|would|will)\s+(?:you|u)\s+)?"
    r"(?:please\s+)?(?:continue|go\s+on|carry\s+on|keep\s+going|"
    r"finish(?:\s+(?:that|your\s+(?:answer|explanation)))?|"
    r"tell\s+me\s+more)(?:\s+please)?[.!? ]*$", re.I,
)
_EXPLANATION = re.compile(
    r"^(?:why(?:\s+(?:not|is\s+that))?|how\s+come|what\s+went\s+wrong|"
    r"what\s+do\s+you\s+mean)[.!? ]*$", re.I,
)
_STOP = re.compile(
    r"^(?:please\s+)?(?:stop|cancel|never\s*mind|be\s+quiet|shut\s+up|"
    r"forget\s+(?:it|that))(?:\s+please)?[.!? ]*$", re.I,
)
_WAIT = re.compile(
    r"^(?:please\s+)?(?:wait|hold\s+on|hang\s+on|one\s+sec(?:ond)?|pause)"
    r"(?:\s+please)?[.!? ]*$", re.I,
)
_GREETING = re.compile(
    r"(?:(?:hi|hello|hey)(?:\s+ciel)?|how are you|are you there|thanks|thank you)[.!? ]*",
    re.I,
)
_ACCEPTANCE = re.compile(
    r"(?:(?:yes|yeah|yep|sure|okay|ok)[,!. ]+)?"
    r"(?:yes|yeah|yep|sure|okay|ok|go\s+ahead|do\s+it|please\s+do)"
    r"(?:[, ]+please)?[.! ]*", re.I,
)
_WEB_OFFER = re.compile(
    r"(?:want\s+me\s+to|(?:would|do)\s+you\s+(?:like|want)\s+me\s+to|"
    r"(?:shall|should|can)\s+i)\s+"
    r"(?:search(?:\s+(?:the\s+)?(?:web|internet))?(?:\s+for)?|"
    r"look\s+(?:(?:it|that|this)\s+up|up|online(?:\s+for)?))"
    r"(?P<target>[^?\n]{0,200})\?", re.I,
)
_OFFER_ALTERNATIVE = re.compile(
    r"\b(?:or|and|then|also)\b[^?]*\b(?:search|look|read|explain|install|open|save|"
    r"send|check|tell|browse|download)\b", re.I,
)
_OFFER_FILLER = set(
    "a an the for about of on in it its that this online please you some information details"
    .split()
)
_QUOTED_OR_REPORTED = re.compile(
    r"^[\"'“‘]|\b(?:she|he|they|someone|the\s+page)\s+(?:said|says|wrote|asked)\b",
    re.I,
)
_NEGATED_LOOKUP = re.compile(
    r"\b(?:don't|do\s+not|never|avoid|without|stop)\b[^.!?;\n]{0,80}"
    r"\b(?:search|searching|browse|browsing|check|look\s+(?:it\s+)?up|screen)\b|"
    r"\bno\s+(?:more\s+)?(?:search(?:es|ing)?|brows(?:e|ing)|lookups?)\b", re.I,
)
_SCREEN = re.compile(
    r"\b(?:check|look\s+at|see|read|describe|inspect)\s+(?:(?:my|the|this)\s+)?screen\b",
    re.I,
)
_DETAIL_QUESTION = re.compile(
    r"\b(?:do|did)\s+you\s+mean\b|\b(?:repeat|clarify|exact\s+name|missing\s+detail)\b|"
    r"\b(?:what|which)\b[^?]*\b(?:name|title|topic|series|season|part|episode|"
    r"chapter|volume|medium|language|platform|version)\b", re.I,
)
_DETAIL = re.compile(
    r"^(?:(?:actually|no|instead|and|also|the|a|an|on|in|for)[, ]+)*"
    r"(?:season|part|episode|chapter|volume|version|edition|light[ -]?novel|lightnovel|"
    r"manga|novel|anime|linux|windows|macos)\b", re.I,
)
_REFERENCE = re.compile(
    r"\b(?:it|its|that|this|those|them|there)\b|"
    r"^(?:and\s+)?(?:what|which)\s+(?:volume|chapter|season|part|episode)\b|"
    r"^(?:what|how)\s+about\b", re.I,
)
_QUESTION = re.compile(
    r"^(?:(?:and|also|so|then)[, ]+)?"
    r"(?:what|which|who|when|where|why|how|is|are|does|do|can|could|will|would)\b", re.I,
)
_QUESTION_WORDS = set(
    "what which who when where why how is are does do can could will would".split()
)
_INTERRUPTED = re.compile(
    r"\[(?:playback|reply)\s+was\s+interrupted\.?\]|"
    r"\[generation\s+ended\s+early\s+because\s+the\s+connection\s+failed\.?\]", re.I,
)
_GENERIC = set(
    "a an the please can could would will you me i no just actually mean meant about "
    "for of to from with on in where what which when how why who does do did is are "
    "was were and also so then it its that this those them there after before "
    "anime manga novel light lightnovel season episode part chapter volume continuation "
    "continue end ends ending release date price pricing version notes search check again "
    "weather current latest official guide information meaning much cost tell more "
    "budget duration length support platform requirements find like read reading pick up "
    "should start begin do i".split()
)
_LATEST_LOOKUP = re.compile(
    r"\b(?:latest|newest|most\s+recent|last)\s+(?:aired\s+)?episode\b", re.I,
)
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "first": "1", "second": "2", "third": "3", "fourth": "4",
    "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9",
    "tenth": "10",
}
_SCOPE = r"season|episode|part|chapter|volume|version|edition"


def _detail_amendment(text: str) -> str:
    """Drop a spoken false start only when an explicit repair replaces its scope."""
    text = re.sub(r"^(?:no[, ]+)?i\s+(?:mean|meant)[, ]+", "", text, flags=re.I)
    repair = re.fullmatch(
        rf"(?P<discarded>[^.!?;\n]{{0,100}}?)\b(?:no[, ]+)+"
        rf"(?:i\s+(?:mean|meant)[, ]+)?(?P<scope>(?:the\s+)?(?:{_SCOPE})\s+.+)",
        text, re.I,
    )
    if repair:
        discarded = re.findall(r"[\w]+", repair["discarded"].casefold())
        allowed = set(_SCOPE.split("|")) | {"uh", "um", "er", "the"} | set(_NUMBER_WORDS)
        if all(word in allowed or re.fullmatch(r"\d+(?:st|nd|rd|th)?", word)
               for word in discarded):
            return repair["scope"]
    return text


@dataclass(frozen=True)
class TaskContext:
    """A bounded user-task interpretation; raw turns remain the authorization source."""

    request: str
    kind: TaskKind
    corrected_subject: str = ""
    interrupted: bool = False
    anchor_index: int | None = None
    task_type: TaskType = "conversation"


def _bounded(text: str, limit: int = 2400) -> str:
    """Keep the opening task and latest correction within a predictable prompt budget."""
    if len(text) <= limit:
        return text
    marker = " […] "
    head = (limit - len(marker)) // 2
    return text[:head] + marker + text[-(limit - head - len(marker)):]


def _normalize_scope(text: str) -> str:
    text = re.sub(r"\blightnovel\b", "light novel", text, flags=re.I)
    words = "|".join(_NUMBER_WORDS)
    return re.sub(
        rf"\b(?P<label>{_SCOPE})\s+(?P<number>{words})\b",
        lambda match: match["label"] + " " + _NUMBER_WORDS[match["number"].casefold()],
        text, flags=re.I,
    )


def _merge_detail(previous: str, text: str) -> str:
    """Replace only explicit installment or medium amendments from the owner's words."""
    previous, text = _normalize_scope(previous), _normalize_scope(text)
    scopes: dict[str, set[str]] = {}
    for match in re.finditer(rf"\b({_SCOPE})\s+(\d+(?:\.\d+)*)\b", text, re.I):
        scopes.setdefault(match[1].casefold(), set()).add(match[2])
    for label, numbers in scopes.items():
        if len(numbers) == 1:
            number = next(iter(numbers))
            previous = re.sub(rf"\b{label}\s+\d+(?:\.\d+)*\b", f"{label} {number}",
                              previous, flags=re.I)
    if re.search(r"\blight[ -]?novel\b", text, re.I) and not re.search(r"\bmanga\b", text, re.I):
        previous = re.sub(r"\bmanga\b", "light novel", previous, flags=re.I)
        previous = re.sub(r"\blight node\b", "light novel", previous, flags=re.I)
    elif re.search(r"\bmanga\b", text, re.I) and not re.search(r"\blight[ -]?novel\b", text, re.I):
        previous = re.sub(r"\blight[ -]?novel\b", "manga", previous, flags=re.I)
    return _bounded(previous.rstrip() + " " + text.strip())


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.casefold())) - _GENERIC


def _request_body(text: str) -> str:
    """Remove conversational lead-ins from projected requests, preserving raw dialogue."""
    text = re.sub(
        r"^(?:(?:alright|okay|ok|thanks|thank you|it's okay)[,.! ]+)+", "", text, flags=re.I,
    )
    return re.sub(
        r"^(?:can|could|would|will)\s+(?:you|u)\s+(?:(?:please|like)\s+)*", "", text,
        flags=re.I,
    )


def _reading_followup(text: str, previous: TaskContext, assistant: str) -> str:
    """Replace a latest-episode outcome with a related source-material reading request."""
    if previous.task_type != "web" or not _LATEST_LOOKUP.search(previous.request):
        return ""
    body = _request_body(text)
    if (not re.search(r"\b(?:continuation|continue|read|reading|pick\s+up)\b", body, re.I)
            or not re.search(r"\b(?:light[ -]novel|light node|novel|manga)\b", body, re.I)):
        return ""
    # This is a query interpretation in an established reading/adaptation task.
    # It does not rewrite speech, teach STT a guessed name, or save a personal fact.
    body = re.sub(r"\blight node\b", "light novel", body, flags=re.I)
    if _terms(body) - _terms(previous.request) - _terms(assistant):
        return ""  # A newly named series is its own task, not this show's follow-up.
    topic = search_request_body(_request_body(previous.request))
    topic = _LATEST_LOOKUP.sub("", topic)
    topic = re.sub(r"\b(?:air|release)\s+date\b", "", topic, flags=re.I)
    topic = " ".join(topic.split()).strip(" .!?")
    topic = re.sub(r"^(?:(?:the|of|in|for)\s+)+", "", topic, flags=re.I)
    if not _terms(topic):
        return ""
    return _bounded(topic + ": " + body)


def _task_type(text: str) -> TaskType:
    if _QUOTED_OR_REPORTED.search(text) or _NEGATED_LOOKUP.search(text):
        return "conversation"
    if has_search_request(text):
        return "web"
    if _SCREEN.search(text):
        return "screen"
    return "conversation"


def _offered_web_lookup(assistant: str, previous: TaskContext) -> bool:
    """Bind one direct lookup offer to owner-provided scope, not assistant claims."""
    if (previous.task_type == "screen" or previous.kind in {"stop", "wait"}
            or _GREETING.fullmatch(previous.request.strip())
            or _ACCEPTANCE.fullmatch(previous.request.strip())
            or _QUOTED_OR_REPORTED.search(previous.request)
            or _NEGATED_LOOKUP.search(previous.request)
            or _QUOTED_OR_REPORTED.search(assistant) or assistant.count("?") != 1):
        return False
    # Only the last sentence can supply the pending offer. A reported offer or a
    # later question about another action must not accept the earlier search.
    clause = re.split(r"(?<=[.!])\s+|\n", assistant.strip())[-1].strip()
    offer = _WEB_OFFER.fullmatch(clause)
    if offer is None or _OFFER_ALTERNATIVE.search(offer["target"]):
        return False
    target = set(re.findall(r"[^\W_]+", offer["target"].casefold())) - _OFFER_FILLER
    owner = set(re.findall(r"[^\W_]+", previous.request.casefold()))
    return bool(owner - _GENERIC) and target <= owner


def _next_task(
    text: str, previous: TaskContext | None, assistant: str, index: int,
) -> TaskContext:
    clean = text.strip().replace("’", "'")
    if _STOP.fullmatch(clean):
        return TaskContext("", "stop")
    if _WAIT.fullmatch(clean):
        return replace(previous, kind="wait") if previous and previous.request else TaskContext(
            "", "wait",
        )
    if not previous or not previous.request or _GREETING.fullmatch(clean):
        return TaskContext(_bounded(text), "new", anchor_index=index, task_type=_task_type(clean))
    if _QUOTED_OR_REPORTED.search(clean) or _NEGATED_LOOKUP.search(clean):
        return TaskContext(_bounded(text), "new", anchor_index=index)
    if _ACCEPTANCE.fullmatch(clean) and _offered_web_lookup(assistant, previous):
        return replace(previous, kind="acceptance", task_type="web")
    if _RETRY.fullmatch(clean):
        return replace(previous, kind="retry")
    if _CONTINUE.fullmatch(clean):
        return replace(previous, kind="continue")
    if assistant and _EXPLANATION.fullmatch(clean):
        return replace(previous, kind="explanation")
    followup = _reading_followup(clean, previous, assistant)
    if followup:
        return replace(previous, request=followup, kind="followup")
    subject = subject_correction(clean, contextual=True)
    if subject and previous.task_type == "screen":
        return replace(previous, request=_merge_detail(previous.request, clean), kind="detail")
    if (subject and _terms(subject) and not _DETAIL.match(subject)
            and previous.task_type != "screen"):
        projected = replace_search_subject(previous.request, subject)
        if projected:
            return replace(previous, request=_bounded(projected), kind="correction",
                           corrected_subject=subject)
    fresh = _task_type(clean)
    if fresh != "conversation" and not _REFERENCE.search(clean):
        return TaskContext(_bounded(text), "new", anchor_index=index, task_type=fresh)
    amendment = _detail_amendment(clean)
    detail = bool(_DETAIL.match(amendment))
    answer = bool(len(clean.split()) <= 16 and "?" in assistant
                  and _DETAIL_QUESTION.search(assistant) and not _QUESTION.match(clean))
    reference = bool(_REFERENCE.search(clean))
    # An independently named question is a new task even if it includes "this" or "also".
    current_names = {
        word.casefold() for word in re.findall(r"[^\W_]+", clean)
        if (word[0].isupper() or any(char.isdigit() for char in word))
        and word.casefold() not in _GENERIC | _QUESTION_WORDS
    }
    named_question = bool(_QUESTION.match(clean) and current_names
                          and not (current_names & _terms(previous.request)))
    has_pronoun = re.search(r"\b(?:it|its|that|this|those|them)\b", clean, re.I)
    if (_QUESTION.match(clean) and not has_pronoun
            and _terms(clean) and not _terms(clean) & _terms(previous.request)):
        named_question = True
    if detail or answer or reference and not named_question:
        return replace(previous, request=_merge_detail(previous.request, amendment), kind="detail")
    return TaskContext(_bounded(text), "new", anchor_index=index, task_type=fresh)


def resolve_task_context(text: str, history: list[ChatMessage]) -> TaskContext:
    """Project the active user question from bounded, already owner-authorized history.

    The caller must perform scope/reset checks around loading history and before using
    this result. Assistant messages supply only clarification/interruption metadata;
    their claimed facts never become part of the projected user request.
    """
    indexed = list(enumerate(history[-64:], start=max(0, len(history) - 64)))
    if indexed and indexed[-1][1].role == "user" and indexed[-1][1].content.strip() == text.strip():
        indexed.pop()
    task = None
    assistant = ""
    for index, message in indexed:
        if message.role == "assistant":
            assistant = message.content
            if task is not None:
                interrupted = bool(_INTERRUPTED.search(message.content))
                task = replace(task, interrupted=interrupted or (
                    task.kind == "wait" and task.interrupted
                ))
        elif message.role == "user" and message.content.strip():
            task = _next_task(message.content, task, assistant, index)
            assistant = ""
    return _next_task(text, task, assistant, len(history))
