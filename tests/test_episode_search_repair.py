"""A failed latest-episode lookup must not acquire a novel-continuation task."""

import json
from unittest.mock import Mock

import pytest

from ciel.chat.session import ChatSession
from ciel.internet import InternetResult
from ciel.providers.base import LLMResponse
from ciel.web_answers import verified_web_answer, web_answer_fallback
from tests.test_controller_runtime import _internet, _tool_call
from tests.test_conversation_search_replays import (
    _report,
    _router,
    _seed,
    _settings,
    _source,
)
from tests.test_screen_integration import run_callbacks

HISTORY = (
    ('user', 'Can you search the latest episode of Tensura?'),
    ('assistant', "I found related pages, but I couldn't verify the answer from their text."),
    ('user', 'Why?'),
    ('assistant', 'Which season of Tensura are you looking for?'),
)
REPAIR = '3rd chapter, episode, no, no, season 4.'


@pytest.mark.parametrize('query', [
    REPAIR, 'Tensura season 4 latest episode',
    'Compare the episode count and chapter count of Amber Harbor.',
])
def test_episode_or_comparison_fallback_does_not_invent_light_novel_intent(query):
    assert 'light novel' not in web_answer_fallback(InternetResult(query=query))


@pytest.mark.parametrize('entrypoint', ['desktop', 'chat'])
def test_season_repair_after_why_retains_latest_episode_task(tmp_path, monkeypatch, entrypoint):
    fact = 'Tensura season 4 episode 7 is the latest aired episode.'
    source = _source(fact)
    internet = _internet(monkeypatch, [InternetResult(attempted=True, sources=[source])])
    router = _router([_tool_call('web_search', query='Tensura season 4 latest episode'),
                      _report(fact)])
    if entrypoint == 'desktop':
        _, _, _, store, turns = run_callbacks(
            tmp_path, monkeypatch, [(REPAIR, {'wake_verified': True})],
            router=router, controller=True,
            initialize_conversation=lambda store, session: _seed(store, session, HISTORY),
        )
        close = store.close
    else:
        session = ChatSession(_settings(tmp_path), channel='telegram', session_id='repair-owner',
                              router=router)
        for role, text in HISTORY:
            session.conversation.add_turn(role, text)
        session.internet = internet
        session.reply(REPAIR)
        turns = session.conversation.get_recent_turns()
        close = session.close
    try:
        goal = internet.lookup.call_args.kwargs['objective']
        assert 'Tensura' in goal and 'latest episode' in goal and 'season 4' in goal
        assert 'chapter' not in goal and 'novel' not in goal and 'Why?' not in goal
        assert fact in turns[-1].content and turns[-1].sources == [source.to_dict()]
        assert [turn.content for turn in turns if turn.role == 'user'][-1] == REPAIR
    finally:
        close()


def test_why_explains_failure_without_starting_another_search(tmp_path):
    router = _router([_tool_call('web_search', query='Tensura latest episode'),
                      LLMResponse('I could not verify the answer from the retrieved evidence.',
                                  'fixture', 'synthetic')])
    session = ChatSession(_settings(tmp_path), channel='telegram', session_id='repair-owner',
                          router=router)
    for role, text in HISTORY[:2]:
        session.conversation.add_turn(role, text)
    session.internet.lookup = Mock(side_effect=AssertionError('Why did not request a new lookup'))
    try:
        answer = session.reply('Why?')
        session.internet.lookup.assert_not_called()
        assert 'validate an answer against the retrieved evidence' in answer
        assert not [call for call in router.send.call_args_list
                    if call.kwargs.get('purpose', 'conversation') == 'conversation']
    finally:
        session.close()


def test_latest_episode_claim_does_not_require_a_novel_volume():
    fact = 'Tensura season 4 episode 7 is the latest aired episode.'
    lookup = InternetResult(query='Tensura latest episode season 4', sources=[_source(fact)])
    assert verified_web_answer(_report(fact).content, lookup) == fact


@pytest.mark.parametrize('fact', [
    'Season 3 anime premieres on January 10, 2027.',
    'The first English-dubbed episode hits the service on April 19.',
])
def test_premiere_or_dub_announcement_cannot_answer_latest_aired_episode(fact):
    lookup = InternetResult(query='Tensura latest episode', sources=[_source(fact)])
    assert 'couldn\'t verify' in verified_web_answer(_report(fact).content, lookup)


@pytest.mark.parametrize('query, medium', [
    ('Which chapter follows anime episode 12?', 'source material'),
    ('Where does season 4 end in the manga?', 'manga'),
])
def test_reading_fallback_respects_requested_medium(query, medium):
    fallback = web_answer_fallback(InternetResult(query=query))
    assert medium in fallback
    assert 'light novel' not in fallback


def test_episode_schedule_table_can_support_its_hash_numbered_release_row():
    from ciel.internet import InternetSource

    quote = r'\#7 |English (Sub) |Fri, Oct 16, 2026 15:05 UTC Actual release |Aired'
    page = ('Episode schedule. Released episodes, newest episode numbers first. '
            '|Episode |Release |Date & time |Status | | --- | --- | --- | --- | |'
            + quote + ' |')
    fact = 'The latest listed released episode is episode 7, aired on October 16, 2026.'
    lookup = InternetResult(query='latest episode of Amber Harbor', sources=[
        InternetSource('Amber Harbor episode schedule', 'https://example.org/episodes',
                       content=page),
    ])
    report = json.dumps({'claims': [{'text': fact, 'source': 1, 'quote': quote}]})
    assert verified_web_answer(report, lookup) == fact
    lookup.sources[0].content = '|Issue |Project |Date |Status | |' + quote + ' |'
    lookup.sources[0].title = 'Issue tracker'
    assert 'couldn\'t verify' in verified_web_answer(report, lookup)


def test_final_episode_ordinal_with_modifier_keeps_its_numeric_scope():
    from ciel.internet import InternetSource

    quote = 'The 24th and final episode of Amber Harbor Season 4 aired on Friday.'
    fact = 'The 24th episode was the final episode of Season 4.'
    lookup = InternetResult(query='Amber Harbor season 4 latest episode', sources=[
        InternetSource('Amber Harbor episode guide', 'https://example.org/guide', content=quote),
    ])
    report = json.dumps({'claims': [{'text': fact, 'quote': quote, 'source': 1}]})
    assert verified_web_answer(report, lookup) == fact
    wrong = report.replace('24th episode was', '25th episode was')
    assert 'couldn\'t verify' in verified_web_answer(wrong, lookup)
