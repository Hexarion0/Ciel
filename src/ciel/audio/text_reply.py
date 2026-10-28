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


def spoken_text_reply(text: str) -> str:
    """Return a cleaned excerpt of at most 80 words, with a pointer for omitted content.

    Complete sentences take priority over filling the word budget. When the first
    sentence is too long, an ellipsis marks a complete-word prefix rather than
    suggesting that the excerpt is a summary. This never generates new answer facts.
    """
    visible = VisibleText(omit_code=True)
    clean = visible.feed(text, final=True)
    code_omitted = _CODE_MARKER in clean or (visible.code and not visible.hidden)
    clean = strip_speech_events(TextToSpeech.clean_text_for_speech(clean))
    clean = _CITATION.sub("", clean.replace(_CODE_MARKER, ""))
    clean = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(clean.split()))
    if not clean:
        return _CODE_ANSWER if code_omitted else ""

    words = clean.split()
    if len(words) <= _MAX_WORDS and not code_omitted:
        return clean

    budget = _MAX_WORDS - len(_FULL_ANSWER.split())
    if len(words) <= budget:
        excerpt = clean
    else:
        # Disable SentenceBuffer's character limit: only complete sentences count.
        sentences = SentenceBuffer(max_chars=len(clean) + 1).feed(clean + " ")
        selected = []
        count = 0
        for sentence in sentences:
            size = len(sentence.split())
            if count + size > budget:
                break
            selected.append(sentence)
            count += size
        excerpt = " ".join(selected) if selected else " ".join(words[:budget]) + "…"

    if excerpt[-1] not in '.!?…"\u201d\u2019\x27':
        excerpt += "."
    return f"{excerpt} {_FULL_ANSWER}"
