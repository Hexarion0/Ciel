"""Search results must address the user's topic before they become answer evidence."""

import re
import socket
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ciel.config import InternetConfig
from ciel.internet import InternetService
from ciel.providers.base import ChatMessage

TEENSURA_REQUEST = (
    "Can you find like the Teensura anime in last episode and when it ends on the "
    "light normal so I can read it from there?"
)
SEARCH_FOLLOWUP = "Can you search like the last episode? Like search the internet?"


@pytest.fixture
def evidence_service(monkeypatch):
    """Mock all network boundaries while retaining real query and evidence decisions."""
    monkeypatch.setattr(socket, "getaddrinfo", Mock(side_effect=AssertionError("Unexpected DNS")))
    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(
        DDGS=Mock(side_effect=AssertionError("Unexpected search backend")),
    ))
    service = InternetService(InternetConfig(max_pages=0, research_max_pages=0, max_results=10))
    service.search = Mock(return_value=[])
    monkeypatch.setattr(service, "_search", service.search)
    monkeypatch.setattr(service, "_read_page", Mock(
        side_effect=AssertionError("Unexpected page read"),
    ))
    return service


def _result(title: str, slug: str, body: str) -> dict[str, str]:
    """Build a search hit whose topic and claims are explicit in the fixture."""
    return {"title": title, "href": f"https://example.com/{slug}", "body": body}


def _user_topic_history() -> list[ChatMessage]:
    """Retain real user wording alongside the assistant's unsupported earlier claims."""
    return [
        ChatMessage("user", "I'm still reading the T-0 light novel, but I'm..."),
        ChatMessage("assistant", "You are reading Tensura Vol12."),
        ChatMessage("user", "I don't know."),
        ChatMessage("user", TEENSURA_REQUEST),
        ChatMessage("assistant", "Season 2 finale ends in volume 8 chapter 54."),
    ]


def _assert_unverified(result) -> None:
    """Require an explicit failure or uncertainty signal instead of citation counting."""
    status = (result.reply or "") + " " + result.context
    assert re.search(
        r"(?:failed|insufficient|unverified|unsupported)|"
        r"not (?:answer )?verification|"
        r"(?:cannot|can't|could not|couldn't) (?:establish|verify)|"
        r"(?:not enough|no usable|no relevant)",
        status, re.I,
    ), status


def test_logged_episode_search_keeps_user_topic_and_rejects_search_tutorials(evidence_service):
    evidence_service.search.return_value = [
        _result("How to Search the Internet", "wikihow", "Learn how to search the internet."),
        _result("Web Search Tricks", "lifewire", "Use keywords to find better search results."),
        _result("Building Queries", "archive", "Build search queries for online archives."),
    ]
    result = evidence_service.prepare(SEARCH_FOLLOWUP, recent_messages=_user_topic_history())

    assert result.attempted
    query = result.query.casefold()
    assert "teensura" in query and "episode" in query and "light" in query
    assert "tensura" not in query  # An assistant's guessed title cannot replace the user's name.
    assert not re.search(r"\b(?:vol(?:ume)?\s*12|volume\s*8|chapter\s*54)\b", query)
    assert not result.sources
    _assert_unverified(result)


def test_user_season_correction_replaces_assistant_season_claim_in_search(evidence_service):
    evidence_service.search.return_value = [_result(
        "Tensura Season 4 Episode Guide", "tensura-season-4",
        "Tensura anime season 4 has an episode guide and light novel adaptation information.",
    )]
    correction = "Yo, there's already 4th season."
    result = evidence_service.prepare(correction, recent_messages=[
        *_user_topic_history(), ChatMessage("user", SEARCH_FOLLOWUP),
        ChatMessage("assistant", "The Season 2 finale corresponds to volume 8 chapter 54."),
    ])

    assert result.attempted and result.sources
    query = result.query.casefold()
    assert "teensura" in query and "season" in query
    assert re.search(r"\b4(?:th)?\b", query)
    assert not re.search(r"\b(?:vol(?:ume)?\s*12|volume\s*8|chapter\s*54|season\s*2)\b", query)


def test_nonempty_mortgage_results_do_not_verify_latest_python_version(evidence_service):
    evidence_service.search.return_value = [_result(
        "UK Mortgage Offers", "mortgages",
        "Savings and mortgage rates for customers in the United Kingdom.",
    )]
    result = evidence_service.prepare("What is the latest Python version?")

    assert result.attempted and evidence_service.search.called
    assert not result.sources
    _assert_unverified(result)


def test_mixed_search_results_only_retain_evidence_about_requested_software(evidence_service):
    evidence_service.search.return_value = [
        _result("UK Mortgage Offers", "mortgages", "Savings products for mortgage customers."),
        _result("Python Release Notes", "python-release",
                "Python 3.14 release notes describe the Python interpreter's new features."),
        _result("How to Search", "search-help", "Learn which search engine to use."),
    ]
    result = evidence_service.prepare("Search the latest Python release notes")

    assert result.attempted
    assert [source.url for source in result.sources] == ["https://example.com/python-release"]


def test_shared_anime_terms_do_not_make_a_different_series_relevant(evidence_service):
    evidence_service.search.return_value = [_result(
        "Naruto Anime and Light Novel Guide", "naruto-guide",
        "Naruto anime episode adaptations and light novel reading order.",
    )]
    result = evidence_service.prepare(
        "Search where the last Teensura anime episode ends on the light normal "
        "so I can read it from there"
    )

    assert result.attempted
    assert "teensura" in result.query.casefold()
    assert not result.sources
    _assert_unverified(result)


@pytest.mark.parametrize("source_count", [1, 8])
def test_relevant_novel_overviews_do_not_establish_exact_volume_or_chapter(
    evidence_service, source_count,
):
    evidence_service.search.return_value = [
        _result(f"Tensura adaptation overview {index}", f"tensura-overview-{index}",
                "Tensura is a fantasy light novel series with an anime adaptation and manga.")
        for index in range(source_count)
    ]
    result = evidence_service.prepare(
        "Search where the last Tensura anime episode ends in the light novel, "
        "which volume and chapter should I read next?"
    )

    assert result.attempted and result.sources
    assert all(not source.content for source in result.sources)
    _assert_unverified(result)
    assert "volume" in result.context.casefold() and "chapter" in result.context.casefold()


def test_generic_search_request_without_a_topic_asks_before_searching(evidence_service):
    result = evidence_service.prepare("Can you search the internet?")

    assert result.reply and "search" in result.reply.casefold()
    assert re.search(r"\b(?:what|which|topic)\b", result.reply, re.I)
    evidence_service.search.assert_not_called()


def test_software_search_followup_retains_user_project_name(evidence_service):
    evidence_service.search.return_value = [_result(
        "Krita 5.2 release notes", "krita-release",
        "Krita 5.2 release notes document new painting tools and compatibility changes.",
    )]
    result = evidence_service.prepare("Can you search for the latest release?", recent_messages=[
        ChatMessage("user", "I use Krita for drawing. I want to check its release notes."),
        ChatMessage("assistant", "You probably meant Photoshop 2025."),
    ])

    assert result.attempted and result.sources
    assert "krita" in result.query.casefold()
    assert "photoshop" not in result.query.casefold() and "2025" not in result.query


def test_music_search_followup_retains_user_artist_without_assistant_replacement(evidence_service):
    evidence_service.search.return_value = [_result(
        "AURORA Tour Dates", "aurora-tour", "AURORA concert tour dates and venue announcements.",
    )]
    result = evidence_service.prepare("Can you search their next concert dates?", recent_messages=[
        ChatMessage("user", "I have been listening to AURORA. I'd like to see a concert."),
        ChatMessage("assistant", "Taylor Swift is touring next month."),
    ])

    assert result.attempted and result.sources
    assert "aurora" in result.query.casefold()
    assert "taylor" not in result.query.casefold() and "swift" not in result.query.casefold()


def test_logged_reading_followups_keep_title_question_and_part(evidence_service):
    history = [ChatMessage("user", "fix the problem")]
    requests = [
        "nvm can u search the tensura season 4 where it ends",
        "so what volume should i read?",
        "research again finished the season 4 part 2 btw",
        "research",
    ]
    for index, request in enumerate(requests):
        result = evidence_service.prepare(request, recent_messages=history)
        assert result.attempted, request
        query = result.query.casefold()
        assert "tensura" in query and "season 4" in query
        assert not re.search(r"\b(?:fix|problem|nvm|u|again|finished|btw|17)\b", query)
        if index:
            assert "volume" in query and "read" in query
            assert "novel" in query and "continuation" in query
        if index >= 2:
            assert "part 2" in query
            assert query.count("season 4") == 1 and not re.search(r"season (?!4\b)", query)
        history += [
            ChatMessage("user", request),
            ChatMessage("assistant", "Start Volume 17. Season 4 adapts Volumes 14–16."),
        ]


def test_explicit_new_named_topic_does_not_inherit_an_unrelated_reference(evidence_service):
    result = evidence_service.prepare(
        "nvm can u search the tensura season 4 where it ends",
        recent_messages=[ChatMessage("user", "fix the microphone problem")],
    )
    assert result.attempted and "tensura" in result.query.casefold()
    assert "microphone" not in result.query.casefold()


def test_reading_followup_can_recover_title_beyond_four_recent_exchanges(evidence_service):
    history = [ChatMessage("user", "Search Tensura season 4 anime ending light novel")]
    for _ in range(5):
        history += [ChatMessage("assistant", "I couldn't verify it."), ChatMessage("user", "ok")]
    result = evidence_service.prepare("Which volume should I read?", recent_messages=history)
    assert result.attempted and "tensura" in result.query.casefold()
    assert "volume" in result.query.casefold()


def test_ordinary_contextual_followup_does_not_start_web_retrieval(evidence_service):
    result = evidence_service.prepare("What about that?", recent_messages=[
        ChatMessage("user", "Tell me about my keyboard"),
        ChatMessage("assistant", "We can compare the switches."),
    ])
    assert not result.attempted
    evidence_service.search.assert_not_called()


def test_unattributed_aggregate_answer_pages_are_not_used_as_sources(evidence_service):
    evidence_service.search.return_value = [{
        "title": "Tensura Season 4 novel reading point",
        "href": "https://www.facebook.com/fb-answers/tensura-season-4-novel/",
        "body": "Tensura season 4 ends at novel volume 17; read the novel continuation.",
    }]
    result = evidence_service.prepare("Search Tensura season 4 novel reading point")
    assert result.attempted and not result.sources
