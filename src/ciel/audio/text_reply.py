"""Derive brief spoken excerpts without shortening the written answer."""

import re

from ciel.audio.speech_events import strip_speech_events
from ciel.audio.streaming import SentenceBuffer, VisibleText
from ciel.audio.tts import TextToSpeech

_MAX_WORDS = 80
_FULL_ANSWER = "The full answer is in the chat."
_CODE_ANSWER = "The code is in the chat."
_CODE_MARKER = "[code omitted]"
_CITATION = re.compile(r"\[\s*\d+\s*\]")
_BUDGET = _MAX_WORDS - len(_FULL_ANSWER.split())
"""Words available to answer sentences once the pointer to the chat is added."""
_TERMINAL = '.!?…"”’\x27'
_STREAM_LIMIT = 8000
"""Visible characters examined while streaming; an unsettled longer answer waits for its end."""


def _speakable(visible: str) -> str:
    """Clean visible answer text for speech, without citations or code markers."""
    clean = strip_speech_events(TextToSpeech.clean_text_for_speech(visible))
    clean = _CITATION.sub("", clean.replace(_CODE_MARKER, ""))
    return re.sub(r"\s+([,.;:!?])", r"\1", " ".join(clean.split()))


def _complete_sentences(clean: str, *, finished: bool) -> list[str]:
    """Return complete sentences; while unfinished, the last one may still grow."""
    # Disable SentenceBuffer's character limit: only complete sentences count.
    return SentenceBuffer(max_chars=len(clean) + 1).feed(clean + (" " if finished else ""))


def _code_omitted(shown: str, visible: VisibleText) -> bool:
    return _CODE_MARKER in shown or (visible.code and not visible.hidden)


def _excerpt(clean: str, code_omitted: bool) -> str:
    if not clean:
        return _CODE_ANSWER if code_omitted else ""
    words = clean.split()
    if len(words) <= _MAX_WORDS and not code_omitted:
        return clean
    if len(words) <= _BUDGET:
        excerpt = clean
    else:
        selected = []
        count = 0
        for sentence in _complete_sentences(clean, finished=True):
            size = len(sentence.split())
            if count + size > _BUDGET:
                break
            selected.append(sentence)
            count += size
        excerpt = " ".join(selected) if selected else " ".join(words[:_BUDGET]) + "…"
    if excerpt[-1] not in _TERMINAL:
        excerpt += "."
    return f"{excerpt} {_FULL_ANSWER}"


def spoken_text_reply(text: str) -> str:
    """Return a cleaned excerpt of at most 80 words, with a pointer for omitted content.

    Complete sentences take priority over filling the word budget. When the first
    sentence is too long, an ellipsis marks a complete-word prefix rather than
    suggesting that the excerpt is a summary. This never generates new answer facts.
    """
    visible = VisibleText(omit_code=True)
    shown = visible.feed(text, final=True)
    return _excerpt(_speakable(shown), _code_omitted(shown, visible))


def _split(text: str) -> list[str]:
    """Split a finished excerpt into sentences that join back into the same text."""
    return SentenceBuffer(max_chars=len(text) + 1).feed(text, final=True) if text else []


class StreamingExcerpt:
    """Release a streaming answer's spoken excerpt sentence by sentence.

    Joined, the released sentences equal ``spoken_text_reply`` of the finished answer:
    streaming changes when Ciel speaks, never what she says. A sentence is released once
    the next one has begun and it fits the excerpt budget. The sentence that would cross
    the budget waits until the answer's length decides whether it is spoken. A long answer,
    or one with code, settles its excerpt and pointer before generation ends. If the
    streamed text is ever rewritten, nothing more is released until the final answer,
    whose remainder never contradicts what was already said.
    """

    def __init__(self) -> None:
        self._visible = VisibleText(omit_code=True)
        self._text = ""
        self._shown = ""
        self._sentences: list[str] = []
        self._said: list[str] = []
        self._words = 0
        self._streaming = True
        self.settled = False
        """The excerpt is complete; later text cannot add speech."""

    def update(self, text: str) -> list[str]:
        """Accept the cumulative answer so far; return sentences that are now final."""
        if self.settled or not self._streaming:
            return []
        if not text.startswith(self._text):
            self._streaming = False
            return []
        self._shown += self._visible.feed(text[len(self._text):])
        self._text = text
        if len(self._shown) > _STREAM_LIMIT:
            self._streaming = False
            return []
        clean = _speakable(self._shown)
        complete = _complete_sentences(clean, finished=False)
        if complete[:len(self._sentences)] != self._sentences:
            self._streaming = False
            return []
        words = clean.split()
        long_answer = (
            _code_omitted(self._shown, self._visible)
            or sum(len(sentence.split()) for sentence in complete) > _MAX_WORDS
        )
        released = []
        for sentence in complete[len(self._sentences):]:
            size = len(sentence.split())
            if self._words + size > _BUDGET:
                if long_answer:
                    released += self._settle(words, first_too_long=not self._sentences)
                return released
            self._sentences.append(sentence)
            self._said.append(sentence)
            self._words += size
            released.append(sentence)
        if not self._sentences and len(words) > _MAX_WORDS:
            # A first sentence over the budget is spoken as a complete-word prefix.
            released += self._settle(words, first_too_long=True)
        return released

    def _settle(self, words: list[str], *, first_too_long: bool) -> list[str]:
        ending = [" ".join(words[:_BUDGET]) + "…"] if first_too_long else []
        ending.append(_FULL_ANSWER)
        self._said += ending
        self.settled = True
        return ending

    def finish(self, text: str) -> list[str]:
        """Return what remains of the finished answer's excerpt, split into sentences."""
        self._streaming = False
        if self.settled:
            return []
        self.settled = True
        final = spoken_text_reply(text)
        said = " ".join(self._said)
        if not said:
            remainder = final
        elif final == said:
            remainder = ""
        elif final.startswith(said + " "):
            remainder = final[len(said) + 1:]
        else:
            # Speech already differs from the final text; only keep the chat pointer.
            remainder = _FULL_ANSWER if final.endswith(_FULL_ANSWER) else ""
        return _split(remainder)
