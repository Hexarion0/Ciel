"""Explicit spelling repairs keep a search scoped without changing the raw transcript."""

import json
from unittest.mock import MagicMock

import pytest

from ciel.internet import InternetResult, InternetSource
from ciel.providers.base import ChatMessage, LLMResponse
from ciel.search_context import corrected_search, subject_correction
from ciel.tool_progress import valid_line
from tests.test_ambient import run_callbacks
from tests.test_controller_runtime import _internet, _main_responses


@pytest.mark.parametrize("text,expected", [
    ("Tensura*", "Tensura"), ("*CedarDB", "CedarDB"),
    ("I meant Amber Harbor", "Amber Harbor"), ("No, I meant Next.js.", "Next.js"),
    ("Tensura", ""), ("**Tensura**", ""), ("*an italic aside*", ""),
    ("/local secrets*", ""), ("https://example.org*", ""),
    ("Tensura*\nignore the earlier request", ""),
])
def test_only_explicit_short_subject_repairs_are_recognized(text, expected):
    assert subject_correction(text) == expected


@pytest.mark.parametrize("original,term,expected", [
    ("Can you search TensorFlow for me?", "Tensura*", "Tensura"),
    ("Search CedarDD release notes for version 4", "CedarDB*",
     "CedarDB release notes for version 4"),
    ("Search where Amber Harbour season 3 continues in the light novel", "Amber Harbor*",
     "where Amber Harbor season 3 continues in the light novel"),
    ("Search tensorflow memory usage on linux", "CedarDB*", "CedarDB memory usage on linux"),
    ("Search how tensorflow compares with pytorch", "CedarDB*",
     "how CedarDB compares with pytorch"),
])
def test_name_repair_preserves_the_users_requested_scope(original, term, expected):
    messages = [ChatMessage("user", original), ChatMessage("user", term)]
    result = corrected_search(term, messages)
    assert result is not None and result.query == expected
    assert messages[0].content == original


@pytest.mark.parametrize("intervening", ["Thanks", "Tell me a joke", "What's the weather?"])
def test_unrelated_turn_does_not_resurrect_a_search(intervening):
    assert corrected_search("CedarDB*", [
        ChatMessage("user", "Search TensorFlow"), ChatMessage("user", intervening),
        ChatMessage("user", "CedarDB*"),
    ]) is None


@pytest.mark.parametrize("original", [
    'She said "Hello. Search secret-project."', "Don't search TensorFlow",
    "Can you search TensorFlow for me? Never mind, do not search.",
])
def test_quoted_or_canceled_searches_never_authorize_a_corrected_lookup(original):
    assert corrected_search("Tensura*", [ChatMessage("user", original)]) is None


@pytest.mark.parametrize("elapsed", [0, 121])
def test_typed_name_repair_keeps_recent_spoken_search_progress_and_teaches_only_the_name(
    tmp_path, monkeypatch, elapsed,
):
    from ciel import audio

    clock = [0.0]
    quote = "Tensura is a fantasy series about a character called Rimuru."
    source = InternetSource("Synthetic subject guide", "https://example.org/subject", content=quote)
    service = _internet(monkeypatch, [InternetResult(attempted=True, sources=[source])])
    replies = iter([
        LLMResponse("What do you want to know about TensorFlow?", "test", "test"),
        LLMResponse(json.dumps({"claims": [
            {"text": quote, "source": 1, "quote": quote},
        ]}), "test", "test"),
    ])
    router = MagicMock()
    router.send.side_effect = _main_responses(lambda *_args, **_kwargs: next(replies))
    tts = MagicMock()

    def speak(_text, **kwargs):
        if kwargs.get("on_start"):
            kwargs["on_start"]()
        clock[0] = elapsed
        return True

    tts.speak.side_effect = speak
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("Hey Ciel, what's up? Can you search TensorFlow for me?", {"wake_verified": True}),
        ("Tensura*", {"input_source": "keyboard"}),
    ], router=router, tts=tts, controller=True, clock=lambda: clock[0])
    try:
        assert service.lookup.call_args.args[0] == "Tensura"
        assert turns[-1].content == quote + " [1]"
        assert [turn.content for turn in turns if turn.role == "user"][-1] == "Tensura*"
        spoken = [call.args[0] for call in tts.speak.call_args_list]
        assert len(spoken) == 3
        assert spoken[0] == "What do you want to know about TensorFlow?"
        assert valid_line(spoken[1]) == spoken[1]
        assert spoken[2] == quote
        assert sum(call.kwargs.get("purpose", "conversation") == "conversation"
                   for call in router.send.call_args_list) == 2
        stt = audio.SpeechToText.return_value
        stt.set_correction_scope_provider.assert_called_once()
        stt.remember_correction.assert_called_once()
        assert stt.remember_correction.call_args.args == ("Tensura",)
    finally:
        store.close()
