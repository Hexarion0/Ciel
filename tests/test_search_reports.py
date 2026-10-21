"""Search reports retain useful evidence without inventing missing details."""

import json
from datetime import datetime, timezone

from ciel.internet import InternetResult, InternetSource
from ciel.web_answers import verified_web_answer, web_answer_messages


def test_text_news_briefing_can_include_five_individually_supported_stories():
    sources = [InternetSource(f"Story {i}", f"https://example.org/news/{i}",
                              content=f"Mongolia opened public library number {i} this week.",
                              published_at=datetime.now(timezone.utc).isoformat())
               for i in range(1, 6)]
    lookup = InternetResult(attempted=True, query="Mongolia news", sources=sources)
    answer = json.dumps({"claims": [
        {"text": source.content, "quote": source.content, "source": i}
        for i, source in enumerate(sources, 1)
    ]})
    report = verified_web_answer(answer, lookup, channel="text", partial=True)
    assert "library number 5" in report
    assert "[5]" in report


def test_one_bad_claim_does_not_erase_other_supported_findings():
    quote = "The example project supports text chat."
    lookup = InternetResult(query="example project", sources=[
        InternetSource("Example project", "https://example.org", content=quote),
    ])
    answer = json.dumps({"claims": [
        {"text": quote, "quote": quote, "source": 1},
        {"text": "It also has video chat.", "quote": "Invented evidence here.", "source": 1},
    ]})
    result = verified_web_answer(answer, lookup, channel="text", partial=True)
    assert "supports text chat" in result
    assert "video chat" not in result
    assert "couldn't verify" in result


def test_original_mongolian_quote_can_support_an_english_translation():
    quote = "Улаанбаатар хотод шинэ номын сан нээгдлээ."
    lookup = InternetResult(query="Mongolia news", sources=[
        InternetSource("Шинэ номын сан", "https://example.mn/news/library", content=quote),
    ])
    answer = json.dumps({"claims": [{
        "text": "A new library opened in Ulaanbaatar.", "quote": quote, "source": 1,
    }]})
    assert "couldn't verify" in verified_web_answer(answer, lookup, channel="text", partial=True)
    result = verified_web_answer(answer, lookup, channel="text", partial=True,
                                 translations={(quote, "A new library opened in Ulaanbaatar.", 1)})
    assert "new library opened" in result


def test_old_news_cannot_be_rephrased_as_happening_today():
    quote = "Mongolia opened a railway linking regional cities today."
    lookup = InternetResult(query="Mongolia news", sources=[
        InternetSource("Railway news", "https://example.mn/news/railway", content=quote,
                       published_at="1998-01-02", date_kind="published"),
    ])
    result = verified_web_answer(json.dumps({"claims": [{
        "text": quote, "quote": quote, "source": 1,
    }]}), lookup, channel="text", partial=True)
    assert "couldn't verify" in result


def test_original_quote_does_not_automatically_verify_an_unrelated_english_claim():
    quote = "Улаанбаатар хотод шинэ номын сан нээгдлээ."
    lookup = InternetResult(query="Mongolia news", sources=[
        InternetSource("Шинэ номын сан", "https://example.mn/news/library", content=quote),
    ])
    answer = json.dumps({"claims": [{
        "text": "NVIDIA announced new processors worldwide.", "quote": quote, "source": 1,
        "translation_verified": True,
    }]})
    assert "couldn't verify" in verified_web_answer(answer, lookup, channel="text", partial=True)


def test_claims_are_english_even_when_the_source_is_mongolian():
    quote = "Монгол Улсын шинэ төмөр зам ашиглалтад орлоо."
    lookup = InternetResult(query="Mongolia news", sources=[
        InternetSource("Railway", "https://example.mn/news/railway", content=quote),
    ])
    answer = json.dumps({"claims": [{"text": quote, "quote": quote, "source": 1}]})
    assert "couldn't verify" in verified_web_answer(answer, lookup, channel="text", partial=True)


def test_text_report_prompt_asks_for_english_and_uses_a_larger_news_budget():
    from ciel.providers.base import ChatMessage

    prompt = web_answer_messages([
        ChatMessage("system", "Be helpful."), ChatMessage("user", "Mongolia news"),
    ], InternetResult(query="Mongolia news"), channel="text")[0].content
    assert "English" in prompt
    assert "five" in prompt.lower() or "5" in prompt
