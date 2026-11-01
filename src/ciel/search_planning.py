"""Admit bounded model-written retrieval queries without rewriting their wording."""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence

from ciel.internet import _scope_numbers

_EXCLUSIVE_SCOPE = ({'linux', 'windows', 'macos', 'android', 'ios'}, {'manga', 'novel'})
_NON_QUERY = set(
    'a an the i you me my your we our can could would will please do does did is are '
    'what which who when where why how tell find search look up for about it its that '
    'this yes yeah yep sure okay ok again retry continue go ahead'.split()
)


def _words(text: str) -> list[str]:
    return re.findall(r"(?<!\w)[.-]?[^\W_]+(?:[.+#'’-][^\W_]+)*(?:\+\+|#)?", text)


def _scope(text: str) -> set[tuple[str, str]]:
    """Compare explicit numeric scope using the same shorthand rules as retrieval."""
    return set(_scope_numbers(text))


def plan_search_queries(
    goal: str, proposed: Sequence[str] = (), *, superseded_terms: Collection[str] = (),
) -> list[str]:
    """Keep up to three AI queries, rejecting acknowledgements and invented scope.

    Queries may paraphrase, use aliases, and pursue different parts of the task.
    The caller retains the full objective for evidence checks. No transcript,
    keywords, or generic suffixes are added when the model omits a usable query.
    """
    queries: list[str] = []
    seen: set[str] = set()
    goal_words = {word.casefold() for word in _words(goal)}
    allowed = _scope(goal)
    from datetime import datetime
    current_year = str(datetime.now().year)
    recent_years = {current_year, str(int(current_year) - 1), str(int(current_year) + 1)}
    scope_nums = {number for _label, number in allowed}
    numbers = set(re.findall(r'\d+(?:\.\d+)*', goal)) | scope_nums | recent_years
    superseded = {word.casefold() for word in superseded_terms}
    for probe in proposed[:6]:
        if not isinstance(probe, str):
            continue
        query = ' '.join(probe.split())
        words = {word.casefold() for word in _words(query)}
        if (not query or len(query) > 200 or not words - _NON_QUERY
                or words & superseded or not _scope(query) <= allowed
                or not set(re.findall(r'\d+(?:\.\d+)*', query)) <= numbers):
            continue
        # Reject a conflicting explicitly requested platform/medium, while letting
        # the AI translate wording such as "how much" into "price" or "LN" into
        # "novel". Keywords are retrieval leads, not new factual claims or goals.
        if any(goal_words & group and words & group and not goal_words & words & group
               for group in _EXCLUSIVE_SCOPE):
            continue
        key = query.casefold()
        if key not in seen:
            queries.append(query)
            seen.add(key)
        if len(queries) == 3:
            break
    return queries
