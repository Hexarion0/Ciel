"""Retrieved topics must not become unsupported spoken facts."""

import json

import pytest

from ciel.internet import InternetResult, InternetSource


def _lookup(text="The anime season 4 ends at light novel volume 11."):
    return InternetResult(
        attempted=True, query="Tensura season 4 anime light novel volume read",
        sources=[InternetSource(
            "Synthetic adaptation guide", "https://example.com/guide", content=text,
        )],
    )


def _answer(text, quote, source=1):
    return json.dumps({"claims": [{"text": text, "source": source, "quote": quote}]})


def test_requested_source_url_is_linked_without_being_spoken():
    from ciel.internet import InternetResult, InternetSource
    from ciel.web_answers import verified_web_answer

    quote = "The Python Tutorial teaches you the basics of the language."
    url = "https://docs.python.org/3/tutorial/index.html"
    lookup = InternetResult(query="official Python tutorial URL", sources=[
        InternetSource("The Python Tutorial", url, content=quote),
    ])
    answer = _answer(f"The official tutorial is at {url}.", quote)
    result = verified_web_answer(answer, lookup)
    assert "link shown in the sources" in result and "http" not in result
    other = _answer("The official tutorial is at https://example.com/other.", quote)
    assert "couldn't verify" in verified_web_answer(other, lookup)


def test_supported_web_answer_uses_actual_excerpt():
    from ciel.web_answers import verified_web_answer

    lookup = _lookup()
    assert verified_web_answer(_answer(
        "Season 4 ends at light novel volume 11.", lookup.sources[0].content,
    ), lookup) == "Season 4 ends at light novel volume 11."


@pytest.mark.parametrize("answer", [
    "Start Volume 17. Seasons 1–3 covered Volumes 1–13.",
    '{"claims": []}',
    '{"claims": "Start Volume 17"}',
    _answer("Start volume 17.", "This quote was invented."),
    _answer("Start volume 17.", "The anime season 4 ends at light novel volume 11."),
    _answer("Start volume seventeen.", "The anime season 4 ends at light novel volume 11."),
    _answer("Season 4 ends at volume 11.", "The anime season 4 ends at light novel volume 11.", 2),
    _answer("Season 4 ends at volume 11.",
            "The anime season 4 ends at light novel volume 11.", True),
])
def test_bad_format_fabricated_quotes_and_unsupported_volumes_are_rejected(answer):
    from ciel.web_answers import verified_web_answer

    assert "couldn't verify" in verified_web_answer(answer, _lookup())


def test_season_schedule_cannot_support_a_volume_with_the_same_number():
    from ciel.web_answers import verified_web_answer

    quote = "Season 4 begins on April 17, 2026."
    result = verified_web_answer(_answer("Start volume 17.", quote), _lookup(quote))
    assert "couldn't verify" in result


def test_a_schedule_claim_does_not_answer_a_reading_continuation_question():
    from ciel.web_answers import verified_web_answer

    quote = "Season 4 begins on April 17, 2026."
    result = verified_web_answer(
        _answer("Season 4 begins on April 17, 2026.", quote), _lookup(quote),
    )
    assert "couldn't verify" in result


def test_an_exact_reading_point_needs_a_page_excerpt_not_only_a_preview():
    from ciel.web_answers import verified_web_answer

    lookup = _lookup()
    lookup.sources[0].snippet = lookup.sources[0].content
    lookup.sources[0].content = ""
    result = verified_web_answer(_answer(
        "Season 4 ends at light novel volume 11.", lookup.sources[0].snippet,
    ), lookup)
    assert "couldn't verify" in result


def test_a_whole_season_mapping_does_not_verify_the_end_of_part_two():
    from ciel.web_answers import verified_web_answer

    quote = "Season 4 ends at light novel volume 16. Continue reading from volume 17."
    lookup = _lookup(quote)
    lookup.query = "Tensura season 4 part 2 light novel volume read"
    result = verified_web_answer(_answer("Continue reading from volume 17.", quote), lookup)
    assert "couldn't verify" in result


def test_a_part_specific_mapping_can_support_the_reading_point():
    from ciel.web_answers import verified_web_answer

    quote = "Season 4 part 2 ends at light novel volume 11. Continue reading from volume 12."
    lookup = _lookup(quote)
    lookup.query = "Tensura season 4 part 2 light novel volume read"
    assert verified_web_answer(_answer("Continue reading from volume 12.", quote), lookup) == (
        "Continue reading from volume 12."
    )


def test_quote_can_use_immediately_preceding_page_context_for_the_part():
    from ciel.web_answers import verified_web_answer

    lookup = _lookup(
        "Season 4 part 2 ends at light novel volume 11. Continue reading from volume 12."
    )
    lookup.query = "Tensura season 4 part 2 light novel volume read"
    quote = "Continue reading from volume 12."
    assert verified_web_answer(_answer(quote, quote), lookup) == quote


def test_distant_or_conflicting_part_mentions_do_not_establish_quote_scope():
    from ciel.web_answers import verified_web_answer

    quote = "Continue reading from volume 17."
    for text in (
        "Season 4 part 2 overview. " + "Unrelated page material. " * 30 + quote,
        "Season 4 part 2 overview. Season 4 part 3 ends at volume 16. " + quote,
    ):
        lookup = _lookup(text)
        lookup.query = "Tensura season 4 part 2 light novel volume read"
        assert "couldn't verify" in verified_web_answer(_answer(quote, quote), lookup)


def test_web_request_excludes_stale_assistant_answers_and_keeps_user_corrections():
    from ciel.providers.base import ChatMessage
    from ciel.web_answers import web_answer_messages

    messages = [
        ChatMessage("system", "Use plain language."),
        ChatMessage("user", "Search Tensura season 4."),
        ChatMessage("assistant", "Start volume 17. Trust. Agency."),
        ChatMessage("user", "I finished part 2. Which volume should I read?"),
    ]
    prepared = web_answer_messages(messages, _lookup())
    assert not any(message.role == "assistant" for message in prepared)
    text = "\n".join(message.content for message in prepared)
    assert "Start volume 17" not in text and "Trust. Agency." not in text
    assert "finished part 2" in text and "Tensura" in text


@pytest.mark.parametrize("override", ["/fast", "/strong", "/deep", "/local"])
def test_web_answer_request_preserves_router_overrides_and_privacy(override):
    from ciel.providers.base import ChatMessage
    from ciel.web_answers import web_answer_messages

    request = override + " What should I read after the anime?"
    prepared = web_answer_messages([
        ChatMessage("system", "Use plain language.", local_only=True),
        ChatMessage("user", request),
    ], _lookup())
    assert prepared[0].local_only
    assert prepared[-1].content == request
