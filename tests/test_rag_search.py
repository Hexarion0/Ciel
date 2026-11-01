"""Tests for direct Context-Augmented Search (RAG) and flexible verification."""

import json
from unittest.mock import Mock

from ciel.internet import InternetResult, InternetSource
from ciel.search_planning import plan_search_queries
from ciel.web_answers import (
    clean_voice_reply,
    extract_citations,
    format_rag_evidence,
    rag_answer_instructions,
    verified_web_answer,
)
from tests.test_main_tools import call
from tests.test_main_tools import make_tools as make_tools  # noqa: F401, PLC0414


def test_format_rag_evidence_structure():
    sources = [
        InternetSource(
            "Xe Currency", "https://www.xe.com/currency/usd-mnt",
            snippet="1 USD = 3,520.50 MNT mid-market rate.",
            published_at="2026-11-01",
        ),
        InternetSource(
            "Wise Rates", "https://wise.com/rates/usd-mnt",
            content="Detailed Wise breakdown: 1 US Dollar is 3,518 Mongolian Tugrik.",
        ),
    ]
    lookup = InternetResult(attempted=True, query="20 usd to mnt", sources=sources)
    formatted = format_rag_evidence(lookup)

    assert "[1] Xe Currency" in formatted
    assert "https://www.xe.com/currency/usd-mnt" in formatted
    assert "1 USD = 3,520.50 MNT" in formatted
    assert "[2] Wise Rates" in formatted
    assert "Detailed Wise breakdown" in formatted


def test_format_rag_evidence_empty():
    lookup = InternetResult(attempted=True, query="unknown query", sources=[])
    formatted = format_rag_evidence(lookup)
    assert "No web pages or sources were found" in formatted


def test_rag_answer_instructions_modalities():
    voice_inst = rag_answer_instructions(channel="voice")
    assert "under 80 words" in voice_inst
    assert "[1]" in voice_inst
    assert "arithmetic" in voice_inst or "calculations" in voice_inst
    assert "markdown tables" in voice_inst  # cautioned not to use in voice

    text_inst = rag_answer_instructions(channel="text")
    assert "markdown" in text_inst
    assert "[1]" in text_inst


def test_extract_citations():
    s1 = InternetSource("Source 1", "https://example.com/1")
    s2 = InternetSource("Source 2", "https://example.com/2")
    s3 = InternetSource("Source 3", "https://example.com/3")
    sources = [s1, s2, s3]

    text = "Based on [2] and [1], the conversion is 70,400 MNT."
    cited = extract_citations(text, sources)
    assert cited == [s1, s2]

    # Out of range citation ignored, defaults to in-range
    text_out_of_range = "Refer to [5] and [1]."
    cited_oor = extract_citations(text_out_of_range, sources)
    assert cited_oor == [s1]

    # No citations returns first sources
    text_none = "No citations provided."
    assert extract_citations(text_none, sources) == [s1, s2, s3]


def test_clean_voice_reply():
    raw = "At the rate of 3,520 MNT [1], see [Xe](https://xe.com). **Result**: ~70,400 MNT."
    cleaned = clean_voice_reply(raw)
    assert "https://xe.com" not in cleaned
    assert "Xe" in cleaned
    assert "**" not in cleaned


def test_currency_calculation_claim_verification():
    quote = "1 USD = 3,520.00 MNT exchange rate today."
    lookup = InternetResult(
        attempted=True,
        query="20 usd to mnt",
        sources=[InternetSource("Xe Currency", "https://xe.com", content=quote)],
    )

    # Claim computes 20 USD = 70,400 MNT (70400 not literally in quote)
    claim = json.dumps({
        "claims": [{
            "text": "At 3,520 MNT per dollar, 20 USD equals 70,400 Mongolian Tugrik.",
            "source": 1,
            "quote": quote,
        }],
    })

    answer = verified_web_answer(claim, lookup, channel="voice")
    assert "70,400" in answer
    assert "couldn't verify" not in answer


def test_answer_filter_with_natural_rag_citations(make_tools):
    tools = make_tools("20 usd to mnt")
    tools.internet.lookup = Mock(return_value=InternetResult(
        attempted=True,
        query="20 usd to mnt",
        sources=[InternetSource(
            "Xe.com", "https://xe.com",
            content="1 USD = 3,520 MNT today.",
        )],
    ))
    tools.execute(call("web_search", query="USD to MNT exchange rate today"))

    # Model returns natural text with citation [1]
    natural_reply = (
        "At the current rate of around 3,520 MNT per USD, 20 USD is approximately 70,400 MNT [1]."
    )
    result = tools.answer_filter(natural_reply)

    assert "70,400" in result.terminal_text
    assert len(result.sources) == 1
    assert result.sources[0].url == "https://xe.com"


def test_search_planning_allows_recent_year_numbers():
    from datetime import datetime
    year = str(datetime.now().year)
    queries = plan_search_queries(
        "USD to MNT exchange rate today",
        [f"USD to MNT exchange rate {year}"],
    )
    assert len(queries) == 1
    assert year in queries[0]
