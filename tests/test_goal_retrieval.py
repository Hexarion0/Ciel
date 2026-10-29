"""Goal-shaped evidence and bounded selective reads, using public synthetic fixtures."""

import threading
import time
from unittest.mock import Mock

import pytest

from ciel.config import InternetConfig
from ciel.internet import InternetService, _excerpt, _PageText


def hit(slug, title, content, *, domain="example.com", date=""):
    """Return one attributable provider excerpt without using live search."""
    return {"href": f"https://{domain}/{slug}", "title": title, "body": content,
            "content": content, "date": date}


def service_with(monkeypatch, rows, **settings):
    """Disable all actual web access while retaining real evidence selection."""
    service = InternetService(InternetConfig(provider="parallel", **settings))
    monkeypatch.setattr(service, "_search", Mock(return_value=rows))
    monkeypatch.setattr(service, "_read_page", Mock(side_effect=OSError("Unavailable")))
    return service


def test_explicit_reading_endpoint_outranks_keyword_heavy_episode_recaps(monkeypatch):
    service = service_with(monkeypatch, [
        hit("recap", "Astral anime season 2 final episode novel ending recap",
            "Astral season 2 final episode recap and ending explained. The anime has a finale."),
        hit("guide", "Adaptation notes", "Astral season 2 ends at volume 6 chapter 2 of "
            "the light novel. Continue reading from volume 6 chapter 3 after the anime."),
    ], max_pages=0, max_results=1)

    result = service.lookup("Astral season 2 finale novel", objective=(
        "Where does the Astral anime season 2 finale end in the light novel so I can read on?"
    ))

    assert result.sources[0].url.endswith("/guide")


def test_partial_mapping_excerpt_selectively_fetches_promising_guide(monkeypatch):
    service = service_with(monkeypatch, [
        hit("recap", "Astral episode discussion", "Astral season 2 finale recap and discussion."),
        hit("guide", "Astral light novel continuation guide", "Astral season 2 reading guide "
            "explains where the anime ends and where to continue in the light novel."),
    ], max_pages=2)
    fetch = Mock(return_value=[hit("guide", "Astral adaptation notes", "Astral season 2 "
        "ends at volume 6 chapter 2. Continue the light novel at volume 6 chapter 3.")])
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", fetch, raising=False)

    result = service.lookup("Astral season 2 light novel continuation", objective=(
        "Where should I continue reading the Astral light novel after the season 2 anime finale?"
    ))

    fetch.assert_called_once()
    assert fetch.call_args.args[0] == ["https://example.com/guide"]
    assert "volume 6 chapter 3" in result.sources[0].content
    assert result.sources[0].content_kind == "provider_excerpt"
    service._read_page.assert_not_called()


def test_complete_mapping_excerpt_avoids_redundant_fetch(monkeypatch):
    service = service_with(monkeypatch, [hit("guide", "Astral reading guide", "Astral season 2 "
        "ends at volume 6 chapter 2 of the light novel. Read on from volume 6 chapter 3.")],
        max_pages=2)
    fetch = Mock(side_effect=AssertionError("Complete evidence needs no second download"))
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", fetch, raising=False)

    result = service.lookup("Astral season 2 novel", objective=(
        "Where does the Astral anime season 2 end in the light novel?"
    ))

    assert result.sources
    fetch.assert_not_called()


def test_fetched_effect_is_not_discarded_for_having_fewer_query_keywords(monkeypatch):
    original = 'Quorum supports Linux offline task operation when one task fails.'
    effect = 'When a Quorum task fails, the remaining tasks are cancelled and then awaited.'
    service = service_with(monkeypatch, [hit('tasks', 'Quorum tasks', original)], max_pages=1)
    fetch = Mock(return_value=[hit('tasks', 'Quorum tasks', effect)])
    monkeypatch.setattr('ciel.parallel_search.fetch_parallel', fetch)
    result = service.lookup('Quorum task failure', objective=(
        'Search Quorum Linux offline tasks and explain what happens to other tasks when one fails.'
    ))
    assert original in result.sources[0].content
    assert effect in result.sources[0].content
    assert '[... omitted source text ...]' in result.sources[0].content
    service._read_page.assert_not_called()


def test_endpoint_in_another_season_section_does_not_suppress_targeted_fetch(monkeypatch):
    service = service_with(monkeypatch, [hit("guide", "Astral season 2 novel continuation", (
        "## Season 1\n\nSeason 1 ends at volume 3.\n\n## Season 2\n\n"
        "This season reading continuation is discussed below."
    ))], max_pages=1)
    fetch = Mock(return_value=[hit("guide", "Astral reading guide", (
        "## Astral season 2\n\n### Reading order\n\nThe anime ends at volume 6 chapter 2. "
        "Continue reading the light novel at volume 6 chapter 3."
    ))])
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", fetch)

    result = service.lookup("Astral season 2 light novel continuation", objective=(
        "Where should I continue reading after Astral anime season 2?"
    ))

    fetch.assert_called_once()
    assert "volume 6 chapter 3" in result.sources[0].content
    service._read_page.assert_not_called()


def test_compact_installment_scope_still_requires_the_requested_section(monkeypatch):
    service = service_with(monkeypatch, [hit("guide", "Astral S2 part2 novel continuation", (
        "## Season1 part2\n\nThe anime ends at volume 3 of the light novel.\n\n"
        "## Season2 part2\n\nAstral continuation details below."
    ))], max_pages=1)
    fetch = Mock(return_value=[hit("guide", "Astral guide", (
        "## Astral S2 part2\n\nThe anime ends at volume 6 of the light novel."
    ))])
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", fetch)

    result = service.lookup("Astral S2 part2 novel continuation", objective=(
        "Where should I continue reading the Astral light novel after S2 part2 anime?"
    ))

    fetch.assert_called_once()
    assert "volume 6" in result.sources[0].content


@pytest.mark.parametrize("wrong_section", [
    "## Season 1\n\nSeason 1 ends at volume 3.\n\n## Season 2\n\nDiscussion pending.",
    "## Season 2 part 1\n\nThe anime ends at volume 3.\n\n## Season 2 part 2\n\nPending.",
    "## Season 2 part 2\n\n[... omitted source text ...]\n\nThe anime ends at volume 3.",
    "## Astral season 2 part 2\n\n...\n\nThe anime ends at volume 3.",
    "## Astral season 2 part 2\n\n…\n\nThe anime ends at volume 3.",
])
def test_requested_section_mapping_outranks_wrong_or_unproved_scope(monkeypatch, wrong_section):
    service = service_with(monkeypatch, [
        hit("wrong", "Astral season 2 part 2 light novel reading guide",
            "Astral anime light novel part 2 continuation guide.\n\n" + wrong_section),
        hit("right", "Astral guide", "## Astral season 2 part 2\n\n### Reading order\n\n"
            "The anime ends at volume 6 chapter 2 of the light novel."),
    ], max_pages=0, max_results=1)

    result = service.lookup("Astral season 2 part 2 light novel", objective=(
        "Where does Astral season 2 part 2 anime end in the light novel?"
    ))

    assert result.sources[0].url.endswith("/right")


@pytest.mark.parametrize("missing", ["Linux", "offline"])
def test_compound_completeness_requires_explicit_operating_constraints(monkeypatch, missing):
    goal = "Find Quorum Linux offline streaming tool calling privacy and local storage requirements"
    partial = "Quorum supports Linux offline streaming, tool calling, privacy and local storage."
    partial = partial.replace(missing, "")
    service = service_with(monkeypatch, [
        hit("requirements", "Quorum client requirements", partial),
    ], max_pages=1)
    fetch = Mock(return_value=[hit("requirements", "Quorum client requirements", (
        "Quorum supports Linux offline streaming, tool calling, privacy and local storage."
    ))])
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", fetch)

    result = service.lookup("Quorum Linux offline streaming tool calling privacy local storage",
                            objective=goal)

    fetch.assert_called_once()
    assert missing in result.sources[0].content


def test_latest_release_evidence_outranks_upcoming_episode_announcement(monkeypatch):
    service = service_with(monkeypatch, [
        hit("future", "Astral latest episode 19", "Astral episode 19 will air next week; "
            "it is the upcoming episode.", date="2026-10-28"),
        hit("released", "Astral release notes", "As of October 28, the latest released "
            "Astral episode is episode 18, which aired October 27.", date="2026-10-27"),
    ], max_pages=0)

    result = service.lookup("Astral latest episode", objective="What is the latest released "
                            "episode of Astral right now?")

    assert result.sources[0].url.endswith("/released")


def test_overview_prefers_explanatory_content_to_repeated_title_terms(monkeypatch):
    service = service_with(monkeypatch, [
        hit("store", "Quorum database overview", "Buy Quorum database stickers and posters."),
        hit("docs", "Introduction", "Quorum is a distributed database that stores records "
            "across multiple servers and provides transactional queries."),
    ], max_pages=0)

    result = service.lookup("Quorum database", objective="What is the Quorum database?")

    assert result.sources[0].url.endswith("/docs")


def test_compound_requirements_use_content_and_retain_complementary_domains(monkeypatch):
    service = service_with(monkeypatch, [
        hit("promo", "Quorum streaming tools privacy documentation",
            "Quorum streaming tools privacy news and product announcement."),
        hit("api", "API guide", "Quorum supports streaming responses and tool calling; "
            "it provides local storage.", domain="docs.example.com"),
        hit("mirror", "API guide copy", "Quorum supports streaming responses and tool calling; "
            "it provides local storage.", domain="docs.example.com"),
        hit("privacy", "Privacy notes", "Quorum provides local storage for privacy and "
            "supports streaming responses.", domain="privacy.example.org"),
    ], max_pages=2, max_results=3)
    fetch = Mock(side_effect=AssertionError("Complementary excerpts already cover the goal"))
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", fetch)

    result = service.lookup("Quorum streaming tool calling privacy", objective=(
        "Find Quorum requirements for streaming responses, tool calling and local storage privacy."
    ))

    assert result.sources[0].url.endswith("/api")
    assert any(source.url.endswith("/privacy") for source in result.sources)
    assert not any(source.url.endswith("/promo") for source in result.sources)
    fetch.assert_not_called()


def test_provider_order_and_useful_video_evidence_survive_without_goal_signal(monkeypatch):
    service = service_with(monkeypatch, [
        hit("watch?v=123", "Workshop", "Quorum database presentation and demonstrations.",
            domain="youtube.com"),
        hit("guide", "Quorum database database database", "Quorum database notes."),
    ], max_pages=0)

    result = service.lookup("Quorum database")

    assert "youtube.com" in result.sources[0].url


def test_failed_targeted_fetch_keeps_original_excerpt_and_safe_local_fallback(monkeypatch):
    from ciel.search_providers import SearchProviderError

    original = "Astral season 2 reading guide explains where the anime ends in the light novel."
    service = service_with(monkeypatch, [hit("guide", "Astral novel continuation", original)],
                           max_pages=1)
    fetch = Mock(side_effect=SearchProviderError("Unavailable"))
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", fetch, raising=False)

    result = service.lookup("Astral season 2 light novel continuation", objective=(
        "Where should I continue reading after Astral anime season 2?"
    ))

    assert result.sources[0].content == original
    assert service._read_page.call_args.args[0] == "https://example.com/guide"


def test_targeted_fetch_rejects_unrequested_private_and_off_topic_rows(monkeypatch):
    original = "Astral season 2 reading guide explains where the anime ends in the light novel."
    service = service_with(monkeypatch, [hit("guide", "Astral novel continuation", original), {
        "href": "http://127.0.0.1/private", "title": "Astral novel", "content": original,
    }], max_pages=1)
    fetch = Mock(return_value=[
        {"href": "http://127.0.0.1/private", "content": "Astral ends at volume 9."},
        hit("other", "Astral guide", "Astral ends at volume 9."),
        hit("guide", "Unrelated", "A cooking recipe ends at step 4."),
    ])
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", fetch, raising=False)

    result = service.lookup("Astral season 2 light novel continuation", objective=(
        "Where should I continue reading after Astral anime season 2?"
    ))

    assert fetch.call_args.args[0] == ["https://example.com/guide"]
    assert result.sources[0].content == original


def test_cancellation_during_fetch_returns_no_partial_evidence(monkeypatch):
    canceled = threading.Event()
    service = service_with(monkeypatch, [hit("guide", "Astral novel continuation",
        "Astral season 2 reading guide explains where the anime ends in the light novel.")],
        max_pages=1)

    def cancel(*_args, **_kwargs):
        canceled.set()
        return []

    fetch = Mock(side_effect=cancel)
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", fetch, raising=False)

    result = service.lookup("Astral season 2 light novel continuation", cancel_event=canceled,
                            objective="Where should I read after Astral anime season 2?")

    fetch.assert_called_once()
    assert result.attempted and not result.sources and not result.context
    service._read_page.assert_not_called()


def test_targeted_fetch_spends_one_deadline_and_at_most_two_pages(monkeypatch):
    service = service_with(monkeypatch, [hit(f"guide-{index}", "Astral novel continuation",
        "Astral season 2 reading guide explains where the anime ends in the light novel.")
        for index in range(5)], max_pages=5, timeout_seconds=0.1)
    release = threading.Event()
    started = threading.Event()

    def delayed(*_args, **_kwargs):
        started.set()
        release.wait(1)
        return []

    fetch = Mock(side_effect=delayed)
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", fetch, raising=False)
    began = time.monotonic()
    try:
        result = service.lookup("Astral season 2 light novel continuation", objective=(
            "Where should I read after Astral anime season 2?"
        ))
        assert started.is_set() and time.monotonic() - began < 0.5
        assert len(fetch.call_args.args[0]) <= 2
        assert 0 < fetch.call_args.kwargs["timeout"] <= 0.1
        assert result.sources and result.sources[0].content
        service._read_page.assert_not_called()
    finally:
        release.set()


def test_optional_query_batch_is_cleaned_and_only_sent_to_parallel(monkeypatch):
    parallel = Mock(return_value=[])
    monkeypatch.setattr("ciel.parallel_search.search_parallel", parallel)
    service = InternetService(InternetConfig(provider="parallel", max_pages=0))

    service.lookup("Quorum database", search_queries=["Quorum streaming", "Quorum api_key=abc"])

    assert parallel.call_args.kwargs["search_queries"] == [
        "Quorum streaming", "Quorum [REDACTED_SECRET]",
    ]


def test_markdown_headings_retain_section_scope_and_omitted_ranges_are_marked(monkeypatch):
    text = "## Astral season 2\n\n### Reading order\n\nStart at volume 6 chapter 3."
    service = service_with(monkeypatch, [hit("guide", "Astral guide", text)], max_pages=0)

    result = service.lookup("Astral season 2 novel")

    assert result.sources[0].content == text
    excerpt = _excerpt("## Season 1\n\n" + "Background. " * 1000
                       + "\n\n## Astral season 2\n\nAstral season 2 novel volume 6 chapter 3.",
                       "Astral season 2 novel", limit=3000)
    assert "\n\n[... omitted source text ...]\n\n" in excerpt


def test_html_headings_are_preserved_without_hidden_content():
    parser = _PageText()
    parser.feed("<main><h2>Astral season 2</h2><h3>Reading order</h3>"
                "<p>Start at volume 6.</p><h2 hidden>Season 9</h2></main>")

    text = _excerpt(" ".join(parser.article), "Astral")

    assert text == "## Astral season 2\n\n### Reading order\n\nStart at volume 6."


def test_long_excerpt_prefers_answer_relation_over_repeated_question_keywords():
    intro = "Astral season 2 anime light novel finale ending discussion and recap. " * 180
    text = intro + "\n\n## Astral season 2\n\n### Adaptation endpoint\n\n" + (
        "The anime ends at volume 6 chapter 2. Continue the light novel at volume 6 chapter 3."
    )

    excerpt = _excerpt(text, "Where does Astral anime season 2 finale end in the light novel?",
                       limit=4000)

    assert "volume 6 chapter 3" in excerpt
    assert "[... omitted source text ...]" in excerpt


def test_excerpt_enrichment_and_local_reads_share_the_page_allowance(monkeypatch):
    service = service_with(monkeypatch, [
        hit("guide", "Astral novel continuation", "Astral season 2 reading guide explains "
            "where the anime ends in the light novel."),
        {"href": "https://example.com/snippet", "title": "Astral season 2", "body":
         "Astral season 2 information."},
    ], max_pages=1)
    fetch = Mock(return_value=[hit("guide", "Astral guide", "Astral season 2 ends at "
        "volume 6 chapter 2. Read the light novel from volume 6 chapter 3.")])
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", fetch)

    result = service.lookup("Astral season 2", objective=(
        "Where should I continue reading the light novel after Astral season 2 anime?"
    ))

    assert result.sources
    fetch.assert_called_once()
    service._read_page.assert_not_called()


def test_promising_incomplete_guide_can_be_read_before_result_limit_discards_it(monkeypatch):
    service = service_with(monkeypatch, [
        hit("recap", "Astral season 2 finale", "Astral season 2 finale recap."),
        hit("guide", "Astral novel continuation", "Astral season 2 reading guide explains "
            "where the anime ends in the light novel."),
    ], max_pages=1, max_results=1)
    fetch = Mock(return_value=[hit("guide", "Astral guide", "Astral season 2 ends at "
        "volume 6 chapter 2. Read the light novel from volume 6 chapter 3.")])
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", fetch)

    result = service.lookup("Astral season 2", objective=(
        "Where should I continue reading the light novel after Astral season 2 anime?"
    ))

    assert result.sources[0].url.endswith("/guide")
    assert fetch.call_args.args[0] == ["https://example.com/guide"]


@pytest.mark.parametrize("provider", ["ddgs", "brave", "tavily"])
def test_nonparallel_query_batch_preserves_selected_provider(monkeypatch, provider):
    service = InternetService(InternetConfig(provider=provider, max_pages=0))
    search = Mock(return_value=[])
    monkeypatch.setattr(service, "_search", search)

    service.lookup("Quorum database", search_queries=["Quorum streaming", "Quorum privacy"])

    assert search.call_count == 1
    assert search.call_args.args[0] == "Quorum database"
    assert "search_queries" not in search.call_args.kwargs


@pytest.mark.parametrize("heading", ["", "# Astral\n\n"])
def test_another_series_mapping_cannot_outrank_the_requested_heading(monkeypatch, heading):
    service = service_with(monkeypatch, [
        hit("related", "Astral anime light novel news", heading + (
            "Astral anime and light novel news.\n\n## Related series: Boreal season 2\n\n"
            "Boreal season 2 anime ends at volume 3 of the light novel. "
            "Continue reading from volume 4."
        )),
        hit("guide", "Astral guide", (
            "# Astral\n\n## Season 2\n\n### Reading order\n\n"
            "The anime ends at volume 6 of the light novel."
        )),
    ], max_results=1, max_pages=0)

    result = service.lookup("Astral season 2 light novel", objective=(
        "Where should I continue reading the Astral light novel after season 2 anime?"
    ))

    assert result.sources[0].url.endswith("/guide")


def test_another_series_mapping_cannot_suppress_fetching_the_requested_guide(monkeypatch):
    service = service_with(monkeypatch, [
        hit("guide", "Astral light novel continuation guide", (
            "Astral light novel anime adaptation and reading continuation guide. "
            "This novel reading guide explains where the anime ends and where to continue."
        )),
        hit("related", "Astral anime light novel news", (
            "Astral anime and light novel news.\n\n## Related series: Boreal\n\n"
            "Boreal anime ends at volume 3 of the light novel. Continue reading from volume 4."
        )),
    ], max_results=1, max_pages=1)
    fetch = Mock(return_value=[hit("guide", "Astral guide", (
        "# Astral\n\n## Reading order\n\nThe anime ends at volume 6 of the light novel."
    ))])
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", fetch)

    result = service.lookup("Astral light novel continuation", objective=(
        "Where should I continue reading the Astral light novel after the anime?"
    ))

    fetch.assert_called_once()
    assert fetch.call_args.args[0] == ["https://example.com/guide"]
    assert result.sources[0].url.endswith("/guide")
    assert "volume 6" in result.sources[0].content
    service._read_page.assert_not_called()


@pytest.mark.parametrize("requested,section", [("four", "4"), ("4", "four")])
def test_written_installment_numbers_keep_mapping_in_the_requested_section(
    monkeypatch, requested, section,
):
    service = service_with(monkeypatch, [
        hit("wrong", "Astral season 4 light novel guide", (
            "## Astral season 1\n\nThe anime ends at volume 3 of the light novel.\n\n"
            "## Season 4\n\nContinuation details follow."
        )),
        hit("guide", "Astral season 4 guide", (
            f"## Astral season {section}\n\n### Reading order\n\n"
            "The anime ends at volume 6 of the light novel."
        )),
    ], max_results=1, max_pages=0)

    result = service.lookup("Astral season 4 light novel", objective=(
        f"Where should I continue reading after Astral anime season {requested}?"
    ))

    assert result.sources[0].url.endswith("/guide")


def test_consequence_question_fetches_the_missing_effect_as_full_content(monkeypatch):
    partial = ("Python asyncio TaskGroup is a structured concurrency feature.\n\n"
               "The first time any task fails with an exception other than CancelledError,"
               "\n\n...\n\nThe context manager waits for child tasks.")
    full = ("Python asyncio TaskGroup is a structured concurrency feature.\n\n"
            "The first time any task fails with an exception other than CancelledError,\n\n"
            "the remaining tasks in the group are cancelled and awaited.")
    service = service_with(monkeypatch, [hit("taskgroup", "Python asyncio TaskGroup", partial)],
                           max_pages=1)
    fetch = Mock(return_value=[hit("taskgroup", "Python asyncio TaskGroup", full)])
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", fetch)

    result = service.lookup("Python asyncio TaskGroup", objective=(
        "Search Python asyncio TaskGroup and explain what happens to the other tasks "
        "when one fails"
    ))

    fetch.assert_called_once()
    assert fetch.call_args.args[0] == ["https://example.com/taskgroup"]
    assert fetch.call_args.kwargs["full_content"] is True
    assert "remaining tasks in the group are cancelled" in result.sources[0].content
    service._read_page.assert_not_called()


def test_broad_explain_request_keeps_a_complete_overview_without_fetching(monkeypatch):
    service = service_with(monkeypatch, [hit("taskgroup", "Python asyncio TaskGroup", (
        "Python asyncio TaskGroup is a structured concurrency feature that manages child tasks."
    ))], max_pages=1)
    fetch = Mock(side_effect=AssertionError("A complete overview needs no second download"))
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", fetch)

    result = service.lookup("Python asyncio TaskGroup", objective=(
        "Explain Python asyncio TaskGroup"
    ))

    assert result.sources
    fetch.assert_not_called()
