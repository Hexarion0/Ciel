"""Source-owned evidence stays contiguous and keeps proven section boundaries."""

import json
from unittest.mock import Mock

import pytest

from ciel.internet import InternetResult, InternetSource
from ciel.web_answers import verified_web_answer
from tests.test_main_tools import make_tools as make_tools


def _lookup(content, *, query="Amber Wings anime season 4 part 2 light novel read"):
    return InternetResult(query=query, sources=[
        InternetSource("Synthetic adaptation guide", "https://example.org/guide", content=content),
    ])


def _report(text, quote):
    return json.dumps({"claims": [{"text": text, "quote": quote, "source": 1}]})


@pytest.mark.parametrize("slashes", [1, 3])
def test_extra_quote_escaping_is_matched_to_the_same_source_passage(slashes):
    quote = 'Season 4 part 2 ends with "The Hero Awakens" at light novel volume 11.'
    copied = quote.replace('"', "\\" * slashes + '"')
    text = "Season 4 part 2 ends at light novel volume 11."

    assert verified_web_answer(_report(text, copied), _lookup(quote)) == text


@pytest.mark.parametrize("copy", [
    "Season 4 part 2 ends at volume 11... Continue reading from volume 12.",
    "Season 4 part 2 ends at volume 11. Continue reading from volume 12.",
    "Season 4 part 2 ends at volume 11. Continue reading from volume 13.",
])
def test_quote_repair_never_stitches_or_changes_source_passages(copy):
    content = (
        "Season 4 part 2 ends at volume 11. An unrelated intervening passage appears here. "
        "Continue reading from volume 12."
    )
    assert "couldn't verify" in verified_web_answer(
        _report("Continue reading from volume 12.", copy), _lookup(content),
    )


def test_title_and_snippet_cannot_be_stitched_into_a_quote():
    lookup = _lookup("", query="Amber Wings announcement")
    lookup.sources[0].title = "Amber Wings has a new announcement."
    lookup.sources[0].snippet = "Its next episode airs on Friday."
    stitched = lookup.sources[0].title + " " + lookup.sources[0].snippet

    assert "couldn't verify" in verified_web_answer(_report(stitched, stitched), lookup)


def test_a_quote_cannot_shorten_a_source_number_at_its_boundary():
    quote = "Season 4 part 2 ends at light novel volume 1"
    content = quote + "2."
    assert "couldn't verify" in verified_web_answer(
        _report("Season 4 part 2 ends at light novel volume 1.", quote), _lookup(content),
    )


def test_distant_structured_heading_proves_scope_inside_its_section():
    quote = "Continue reading from light novel volume 12."
    content = (
        "# Amber Wings adaptation guide\n\n## Season 4 part 2\n\n"
        + "This section discusses the adaptation and reading order. " * 12
        + "\n\n" + quote
    )

    assert verified_web_answer(_report(quote, quote), _lookup(content)) == quote


@pytest.mark.parametrize("heading", ["## Season 4 part 3", "## Other adaptations"])
def test_new_section_ends_the_previous_headings_scope(heading):
    quote = "Continue reading from light novel volume 12."
    content = (
        "# Amber Wings adaptation guide\n\n## Season 4 part 2\n\n"
        + "This section discusses the adaptation and reading order. " * 12
        + "\n\n" + heading + "\n\n" + quote
    )

    assert "couldn't verify" in verified_web_answer(_report(quote, quote), _lookup(content))


def test_nested_heading_keeps_parent_scope_without_accepting_conflicting_labels():
    quote = "Continue reading from light novel volume 12."
    content = (
        "## Season 4 part 2\n\n" + "This passage explains the adaptation. " * 12
        + "\n\n### Light novel reading order\n\n" + quote
    )
    assert verified_web_answer(_report(quote, quote), _lookup(content)) == quote

    conflict = content.replace("### Light novel reading order", "### Season 4 part 3")
    assert "couldn't verify" in verified_web_answer(_report(quote, quote), _lookup(conflict))


def test_omitted_source_ranges_reset_inherited_scope():
    quote = "Continue reading from light novel volume 12."
    content = (
        "## Season 4 part 2\n\n" + "This passage explains the adaptation. " * 12
        + "\n\n[... omitted source text ...]\n\n" + quote
    )
    assert "couldn't verify" in verified_web_answer(_report(quote, quote), _lookup(content))


def test_application_gap_marker_is_not_a_source_passage_or_valid_quotation():
    from ciel.web_answers import web_evidence_context

    marker = "[... omitted source text ...]"
    content = "Cedar supports Linux desktop systems.\n\n" + marker + "\n\nCedar runs offline."
    lookup = _lookup(content, query="Cedar Linux offline support")
    evidence = json.loads(web_evidence_context(lookup).split("\n", 1)[1])
    assert all(marker not in passage["text"]
               for source in evidence["sources"] for passage in source["passages"])
    assert "couldn't verify" in verified_web_answer(
        _report("The source text is omitted.", marker), lookup,
    )
    assert "couldn't verify" in verified_web_answer(
        _report("Cedar supports Linux desktop systems and runs offline.", content), lookup,
    )


def test_hard_wrapped_paragraph_stays_one_contiguous_evidence_passage():
    from ciel.web_answers import web_evidence_context

    content = ("Cedar supports multiple readers. Only one writer can proceed at a time;\n"
               "another writer must wait or fail with a busy error.")
    lookup = _lookup(content, query="Cedar concurrent readers and two writers")
    evidence = json.loads(web_evidence_context(lookup).split("\n", 1)[1])
    assert any(p["text"] == content for p in evidence["sources"][0]["passages"])


def test_latest_aired_episode_accepts_standard_markdown_release_table_row():
    row = r"|\#7 |English Sub |Fri, Sep 25, 2026 Actual release |Aired |"
    lookup = _lookup("|Episode |Release |Date |Status |\n| --- | --- | --- | --- |\n" + row,
                     query="Search the latest aired episode of Cedar")
    text = "Episode 7 aired on September 25, 2026."
    assert verified_web_answer(_report(text, row), lookup) == text


def test_release_version_in_active_heading_and_date_in_quote_share_source_attribution():
    quote = '**Release date:** October 25, 2026'
    lookup = _lookup('# Cedar 3.15.2\n\n' + quote, query='Search latest Cedar release')
    text = 'Cedar 3.15.2 was released on October 25, 2026.'
    assert verified_web_answer(_report(text, quote), lookup) == text
    lookup.sources[0].content = '# Cedar 3.15.2\n\n# Cedar 3.14.1\n\n' + quote
    assert "couldn't verify" in verified_web_answer(_report(text, quote), lookup)


def test_release_row_can_use_its_active_season_heading_but_not_a_conflicting_section():
    row = r'|\#7 |English Sub |Fri, Sep 25, 2026 Actual release |Aired |'
    content = '# Cedar season 4\n\n|Episode |Release |Date |Status |\n' + row
    lookup = _lookup(content, query='Search the latest aired episode of Cedar season 4')
    text = 'Cedar season 4 episode 7 aired on September 25, 2026.'
    assert verified_web_answer(_report(text, row), lookup) == text
    lookup.sources[0].content = content.replace('|Episode', '## Season 3\n\n|Episode')
    assert "couldn't verify" in verified_web_answer(_report(text, row), lookup)


def test_markdown_escaped_version_numbers_keep_their_value_without_changing_the_quote():
    quote = r'Download the latest Cedar version 4\.7.2 stable build.'
    lookup = _lookup(quote, query='Search latest stable Cedar release')
    text = 'The latest stable Cedar version is 4.7.2.'
    assert verified_web_answer(_report(text, quote), lookup) == text
    assert "couldn't verify" in verified_web_answer(
        _report(text.replace('4.7.2', '4.7.20'), quote), lookup,
    )


def test_passage_id_selects_an_application_supplied_source_span():
    from ciel.providers.base import ChatMessage
    from ciel.web_answers import normalize_web_report, web_answer_messages

    quote = "Season 4 part 2 ends at light novel volume 11."
    lookup = _lookup(quote)
    messages = web_answer_messages([ChatMessage("system", "Helpful")], lookup)
    evidence = json.loads(messages[1].content.split("\n", 1)[1])
    passage = next(item for item in evidence["sources"][0]["passages"]
                   if item["text"] == quote)
    report = json.dumps({"claims": [{"text": quote, "source": 1, "passage": passage["id"]}]})

    assert verified_web_answer(report, lookup) == quote
    normalized = json.loads(normalize_web_report(report, lookup))
    assert normalized["claims"][0]["quote"] == quote


@pytest.mark.parametrize("passage", ["content:-1:50", "content:0:1", "snippet:0:52", 1, None])
def test_unknown_or_invalid_passage_ids_do_not_validate_a_claim(passage):
    quote = "Season 4 part 2 ends at light novel volume 11."
    lookup = _lookup(quote)
    report = json.dumps({"claims": [{"text": quote, "source": 1, "passage": passage}]})
    assert "couldn't verify" in verified_web_answer(report, lookup)


def test_a_passage_id_and_a_conflicting_quote_are_rejected():
    from ciel.providers.base import ChatMessage
    from ciel.web_answers import web_answer_messages

    quote = "Season 4 part 2 ends at light novel volume 11."
    lookup = _lookup(quote)
    messages = web_answer_messages([ChatMessage("system", "Helpful")], lookup)
    evidence = json.loads(messages[1].content.split("\n", 1)[1])
    passage = next(item for item in evidence["sources"][0]["passages"]
                   if item["text"] == quote)
    report = json.dumps({"claims": [{
        "text": quote, "source": 1, "passage": passage["id"],
        "quote": "Season 4 part 2 ends at light novel volume 12.",
    }]})
    assert "couldn't verify" in verified_web_answer(report, lookup)


def test_report_diagnostics_distinguish_missing_mapping_and_scope_from_format():
    from ciel.web_answers import web_report_issues

    plot = "Season 4 part 2 follows the heroes on a journey."
    assert "missing_reading_mapping" in web_report_issues(_report(plot, plot), _lookup(plot))
    mapping = "Continue reading from light novel volume 12."
    issues = web_report_issues(_report(mapping, mapping), _lookup(mapping))
    assert "missing_requested_scope" in issues
    assert web_report_issues("not JSON", _lookup("")) == ("invalid_report",)


def test_a_quote_spanning_a_new_section_cannot_inherit_the_old_scope():
    quote = "## Season 4 part 2\n\n## Other adaptations\n\nRead light novel volume 12."
    assert "couldn't verify" in verified_web_answer(
        _report("Read light novel volume 12.", quote), _lookup(quote),
    )


def test_passage_catalog_is_bounded_and_selects_late_relevant_content():
    from ciel.web_answers import web_evidence_context

    mapping = "Season 4 part 2 ends at light novel volume 11."
    content = "\n\n".join([
        "A separate paragraph describes scenery and character costumes. " * 20,
    ] * 20 + [mapping])
    record = json.loads(web_evidence_context(_lookup(content)).split("\n", 1)[1])["sources"][0]

    assert record["page_excerpt"] == ""
    assert 1 <= len(record["passages"]) <= 12
    assert any(item["text"] == mapping for item in record["passages"])
    for item in record["passages"]:
        assert 4 <= len(item["text"].split()) <= 100
        assert len(item["text"]) <= 4000
        field, start, end = item["id"].split(":")
        assert getattr(_lookup(content).sources[0], field)[int(start):int(end)] == item["text"]


def test_passage_evidence_reaches_relevance_check_with_proven_section_context():
    from ciel.providers.base import ChatMessage, LLMResponse
    from ciel.web_answers import relevant_web_claims, web_evidence_context

    quote = "Continue reading from light novel volume 12."
    content = ("## Season 4 part 2\n\n" + "Other explanatory text appears here. " * 30
               + "\n\n" + quote)
    lookup = _lookup(content)
    record = json.loads(web_evidence_context(lookup).split("\n", 1)[1])["sources"][0]
    passage = next(item for item in record["passages"] if item["text"] == quote)
    report = json.dumps({"claims": [{"text": quote, "source": 1, "passage": passage["id"]}]})
    router = Mock()
    router.send.return_value = LLMResponse('{"relevant_claims":[1]}', "model", "provider")

    checked = relevant_web_claims(report, lookup, [ChatMessage("user", lookup.query)], router)

    payload = json.loads(router.send.call_args.args[0][-1].content)
    assert payload["claims"][0]["quote"] == quote
    assert payload["claims"][0]["source_section"] == "Season 4 part 2"
    assert verified_web_answer(checked, lookup) == quote


def test_repeated_quote_uses_its_matching_section_in_the_relevance_check():
    from ciel.providers.base import ChatMessage, LLMResponse
    from ciel.web_answers import relevant_web_claims

    quote = "Continue reading from light novel volume 12."
    lookup = _lookup("## Season 4 part 1\n\n" + quote + "\n\n## Season 4 part 2\n\n" + quote)
    router = Mock()
    router.send.return_value = LLMResponse('{"relevant_claims":[1]}', "model", "provider")
    relevant_web_claims(_report(quote, quote), lookup, [ChatMessage("user", lookup.query)], router)

    payload = json.loads(router.send.call_args.args[0][-1].content)
    assert payload["claims"][0]["source_section"] == "Season 4 part 2"


def test_passage_translations_keep_the_independent_translation_gate():
    from ciel.web_answers import translation_candidates, web_evidence_context

    quote = "Улаанбаатар хотод шинэ номын сан нээгдлээ."
    text = "A new library opened in Ulaanbaatar."
    lookup = _lookup(quote, query="Ulaanbaatar library news report")
    record = json.loads(web_evidence_context(lookup).split("\n", 1)[1])["sources"][0]
    passage = next(item for item in record["passages"] if item["text"] == quote)
    report = json.dumps({"claims": [{"text": text, "source": 1, "passage": passage["id"]}]})

    assert translation_candidates(report, lookup) == [{"text": text, "quote": quote, "source": 1}]
    assert "couldn't verify" in verified_web_answer(report, lookup)
    assert verified_web_answer(report, lookup, translations={(quote, text, 1)}) == text


def test_old_passage_remains_valid_after_new_content_evicts_it_from_the_catalog():
    from ciel.web_answers import normalize_web_report, web_evidence_context

    quote = "The group cancels remaining child tasks."
    lookup = _lookup(quote, query="TaskGroup cancel exception siblings structured concurrency")
    previous = json.loads(web_evidence_context(lookup).split("\n", 1)[1])
    passage = previous["sources"][0]["passages"][0]
    report = json.dumps({"claims": [{"text": quote, "source": 1, "passage": passage["id"]}]})
    added = "\n\n".join(
        f"TaskGroup cancel exception siblings structured concurrency details appear here {i}."
        for i in range(15)
    )
    lookup.sources[0].content += "\n\n[... omitted source text ...]\n\n" + added
    current = json.loads(web_evidence_context(lookup).split("\n", 1)[1])

    assert passage["id"] not in {p["id"] for p in current["sources"][0]["passages"]}
    assert verified_web_answer(report, lookup) == quote
    assert json.loads(normalize_web_report(report, lookup))["claims"][0]["quote"] == quote


@pytest.mark.parametrize("kind", ["partial", "gap", "out_of_range", "extra_field"])
def test_offsets_must_identify_a_complete_application_passage(kind):
    quote = "Cedar supports Linux and runs without a network connection."
    lookup = _lookup(quote, query="Cedar Linux offline support")
    if kind == "partial":
        passage = f"content:6:{len(quote)}"
    elif kind == "gap":
        lookup.sources[0].content += "\n\n[... omitted source text ...]\n\n" + quote
        passage = f"content:0:{len(lookup.sources[0].content)}"
    elif kind == "out_of_range":
        passage = f"content:0:{len(quote) + 1}"
    else:
        passage = f"content:0:{len(quote)}:extra"
    report = json.dumps({"claims": [{"text": quote, "source": 1, "passage": passage}]})

    assert "couldn't verify" in verified_web_answer(report, lookup)


def test_many_sources_fit_the_native_tool_result_and_keep_late_relevant_evidence(make_tools):
    from ciel.web_answers import web_evidence_context

    goal = "Search Cedar TaskGroup cancellation exception siblings and structured concurrency"
    lookup = InternetResult(attempted=True, query=goal, retrieved_at="2026-10-29T12:00:00Z")
    for index in range(20):
        paragraph = (
            "Cedar TaskGroup cancellation has documented behavior for child tasks. "
            + "Detailed source text includes quoted \"notes\" and backslashes \\ here. " * 10
        )
        content = "\n\n".join([paragraph] * 12)
        if index == 19:
            content += ("\n\nCedar TaskGroup cancels siblings when an exception interrupts "
                        "structured concurrency; this newest page supplies the requested details.")
        lookup.sources.append(InternetSource(
            f"Cedar documentation {index}", f"https://example.org/docs/{index}",
            snippet=paragraph[:800], content=content, publisher="Example documentation",
            fetched_at="2026-10-29T12:00:00Z", published_at="2026-10-01", date_kind="published",
        ))
    context = web_evidence_context(lookup)
    prefix, data = context.split("\n", 1)
    evidence = json.loads(data)
    tools = make_tools(goal)
    tools.lookup = lookup
    report = tools._web_report()

    assert len(json.dumps(context, ensure_ascii=False)) <= 36_000
    assert len(json.dumps(report.content, ensure_ascii=False)) <= 64_000
    assert "previously retrieved observations" in prefix and lookup.retrieved_at in prefix
    assert {record["source"] for record in evidence["sources"]} == set(range(1, 21))
    assert all(record["passages"] for record in evidence["sources"])
    assert all(not record["page_excerpt"] and not record["search_snippet"]
               for record in evidence["sources"])
    assert any("newest page" in p["text"] for p in evidence["sources"][-1]["passages"])
    assert evidence["sources"][-1]["url"] == lookup.sources[-1].url
    assert evidence["sources"][-1]["published_at"] == "2026-10-01"


def test_long_source_metadata_is_budgeted_without_truncating_citation_urls():
    from ciel.web_answers import web_evidence_context

    lookup = InternetResult(query="Cedar TaskGroup structured concurrency cancellation")
    for index in range(20):
        content = "Cedar has documented behavior for child tasks."
        if index == 19:
            content = "Cedar TaskGroup structured concurrency provides cancellation of child tasks."
        lookup.sources.append(InternetSource(
            f"Cedar documentation {index}",
            f"https://example.org/{'reference-' * 190}/{index}", content=content,
            publisher="Example documentation", published_at="2026-10-01",
        ))

    context = web_evidence_context(lookup)
    records = json.loads(context.split("\n", 1)[1])["sources"]

    assert len(json.dumps(context, ensure_ascii=False)) <= 36_000
    assert any(record["source"] == 20 and record["passages"] for record in records)
    assert all(record["url"] == lookup.sources[record["source"] - 1].url for record in records)
    assert all(record["published_at"] == "2026-10-01" for record in records)


def test_adjacent_paragraphs_export_a_condition_and_consequence_in_one_exact_passage():
    from ciel.web_answers import web_evidence_context

    content = ("If a child task fails with an exception,\n\n"
               "the remaining child tasks are cancelled and awaited.")
    lookup = _lookup(content, query="child task exception remaining tasks cancelled awaited")
    record = json.loads(web_evidence_context(lookup).split("\n", 1)[1])["sources"][0]
    passage = next(p for p in record["passages"] if p["text"] == content)
    text = ("Remaining child tasks are cancelled and awaited "
            "if a child task fails with an exception.")
    report = json.dumps({"claims": [{"text": text, "source": 1, "passage": passage["id"]}]})

    assert verified_web_answer(report, lookup) == text


@pytest.mark.parametrize("separator", ["\n", "\n\n"])
def test_short_release_heading_and_adjacent_date_export_as_one_exact_passage(separator):
    from ciel.web_answers import web_evidence_context

    content = "# Download" + separator + (
        "Cedar version 4.7.2 stable\n\n**Release date:** October 25, 2026"
    )
    lookup = _lookup(content, query="latest stable Cedar version and release date")
    record = json.loads(web_evidence_context(lookup).split("\n", 1)[1])["sources"][0]
    passage = next(p for p in record["passages"] if p["text"] == content)
    text = "Cedar version 4.7.2 was released on October 25, 2026."
    report = json.dumps({"claims": [{"text": text, "source": 1, "passage": passage["id"]}]})

    assert verified_web_answer(report, lookup) == text


@pytest.mark.parametrize("boundary", [
    "[... omitted source text ...]", "## Other documented behavior",
])
def test_overlapping_windows_cannot_connect_facts_across_a_gap_or_new_section(boundary):
    from ciel.web_answers import _passage_spans

    condition = "If a child task fails with an exception,"
    effect = "the remaining child tasks are cancelled and awaited."
    content = condition + "\n\n" + boundary + "\n\n" + effect
    lookup = _lookup(content, query="child task exception remaining tasks cancelled awaited")
    spans = _passage_spans(content)
    report = json.dumps({"claims": [{
        "text": "Remaining child tasks are cancelled if a child task fails with an exception.",
        "source": 1, "passage": f"content:0:{len(content)}",
    }]})

    assert not any(condition in content[start:end] and effect in content[start:end]
                   for start, end in spans)
    assert "couldn't verify" in verified_web_answer(report, lookup)


def test_overlapping_windows_preserve_base_spans_and_bound_complete_source_bytes():
    from ciel.web_answers import _passage_spans

    first = "A documented condition applies to each task. " * 8
    second = "A documented consequence applies to the remaining tasks. " * 8
    content = first.strip() + "\n\n" + second.strip()
    spans = _passage_spans(content)
    first_end = len(first.strip())
    second_start = first_end + 2

    assert (0, first_end) in spans
    assert (second_start, len(content)) in spans
    assert any(start == 0 and end > second_start for start, end in spans)
    assert any(start < second_start and end < len(content) for start, end in spans)
    assert all(4 <= len(content[start:end].split()) <= 100 and end - start <= 4000
               for start, end in spans)


@pytest.mark.parametrize("omission", ["...", "…"])
@pytest.mark.parametrize("separator", ["\n", "\n\n"])
def test_standalone_provider_omissions_stop_windows_and_legacy_quotations(omission, separator):
    from ciel.web_answers import _passage_spans

    condition = "If a child task fails with an exception,"
    effect = "the remaining child tasks are cancelled and awaited."
    content = condition + separator + omission + separator + effect
    lookup = _lookup(content, query="child task exception remaining tasks cancelled awaited")

    assert not any(condition in content[start:end] and effect in content[start:end]
                   for start, end in _passage_spans(content))
    assert "couldn't verify" in verified_web_answer(
        _report("Remaining child tasks are cancelled if a child task fails.", content), lookup,
    )


@pytest.mark.parametrize("omission", ["...", "…"])
def test_standalone_provider_omissions_reset_inherited_reading_scope(omission):
    quote = "Continue reading from light novel volume 12."
    content = "## Season 4 part 2\n\n" + omission + "\n\n" + quote

    assert "couldn't verify" in verified_web_answer(_report(quote, quote), _lookup(content))


@pytest.mark.parametrize("ellipsis", ["...", "…"])
def test_inline_ellipses_keep_contiguous_evidence_and_section_scope(ellipsis):
    from ciel.web_answers import _passage_spans

    quote = f"Continue reading {ellipsis} from light novel volume 12."
    content = "## Season 4 part 2\n\n" + quote
    lookup = _lookup(content)

    assert any(content[start:end] == quote for start, end in _passage_spans(content))
    assert verified_web_answer(_report("Continue reading from light novel volume 12.", quote),
                               lookup) == "Continue reading from light novel volume 12."
