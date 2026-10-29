"""Search keywords can vary while the application retains the owner's actual goal."""

import pytest

from ciel.search_planning import plan_search_queries
from tests.test_main_tools import call
from tests.test_main_tools import make_tools as make_tools


def test_contextual_reading_lookup_batches_distinct_answer_seeking_queries(make_tools):
    goal = 'Tensura season 4: find the continuation from the Hero Awakens in the light novel?'
    tools = make_tools('Find the continuation.', objective=goal, contextual_web=True)
    queries = ['Tensura season 4 Hero Awakens light novel continuation',
               'Tensei Shitara Slime Datta Ken season 4 novel volume chapter']
    tools.execute(call('web_search', search_queries=queries))
    request = tools.internet.lookup.call_args
    assert request.args[0] == queries[0]
    assert request.kwargs['search_queries'] == queries
    assert request.kwargs['objective'] == goal


def test_invented_episode_requires_the_model_to_write_a_corrected_query(make_tools):
    goal = 'Tensura season 4: find the continuation from the Hero Awakens in the light novel?'
    tools = make_tools('Find the continuation.', objective=goal, contextual_web=True)

    rejected = tools.execute(call(
        'web_search', query='Tensura season 4 episode 24 novel volume chapter',
    ))

    assert rejected.is_error and rejected.content['task'] == goal
    tools.internet.lookup.assert_not_called()
    query = 'Tensura season 4 Hero Awakens light novel continuation'
    tools.execute(call('web_search', query=query))
    assert tools.internet.lookup.call_args.args[0] == query
    assert tools.internet.lookup.call_args.kwargs['objective'] == goal


def test_corrected_subject_does_not_leak_stale_topic_into_keyword_batch(make_tools):
    from ciel.providers.base import ChatMessage

    tools = make_tools('CedarDB*', messages=[
        ChatMessage('user', 'Search CedarDD Linux offline support'),
        ChatMessage('assistant', 'Could you spell the name?'),
        ChatMessage('user', 'CedarDB*'),
    ])
    for query in ('CedarDD Linux offline support', 'CedarDB season 9 release date'):
        rejected = tools.execute(call('web_search', query=query))
        assert rejected.is_error
        assert 'CedarDB' in rejected.content['task']
        assert 'CedarDD' not in rejected.content['task']
        tools.internet.lookup.assert_not_called()
    queries = ['CedarDB Linux platform compatibility', 'CedarDB airgapped deployment']
    tools.execute(call('web_search', search_queries=queries))
    request = tools.internet.lookup.call_args
    assert request.args[0] == queries[0]
    assert request.kwargs['search_queries'] == queries
    assert 'Linux' in request.kwargs['objective'] and 'offline' in request.kwargs['objective']


def test_native_batch_without_single_query_uses_the_first_model_query(make_tools):
    from ciel.agent import _validate_schema

    goal = 'Search what Tensura means.'
    tools = make_tools(goal)
    schema = next(t['function']['parameters'] for t in tools.schemas
                  if t['function']['name'] == 'web_search')
    arguments = {'search_queries': ['Tensura meaning Japanese', 'Tensura abbreviation']}
    _validate_schema(arguments, schema)
    tools.execute(call('web_search', **arguments))
    request = tools.internet.lookup.call_args
    assert request.kwargs['objective'] == goal
    assert request.args[0] == arguments['search_queries'][0]
    assert request.kwargs['search_queries'] == arguments['search_queries']


def test_distinct_admitted_batches_are_new_leads_but_reordered_batches_are_cached(make_tools):
    tools = make_tools('Search CedarDB Linux support and offline mode')
    first = ['CedarDB Linux platform compatibility', 'CedarDB Linux installation guide']
    second = ['CedarDB offline mode airgapped deployment', 'CedarDB offline configuration']
    tools.execute(call('web_search', search_queries=first))
    tools.execute(call('web_search', search_queries=second))
    assert tools.internet.lookup.call_count == 2
    repeated = tools.execute(call('web_search', search_queries=list(reversed(second))))
    assert tools.internet.lookup.call_count == 2
    assert repeated.metadata['finalize']


@pytest.mark.parametrize('goal', [
    'Search Dune novel reviews', 'Search current weather in Ulaanbaatar',
    'Search current Bitcoin price', 'Search restaurants in Tokyo without reservations',
])
def test_missing_model_queries_do_not_generate_a_transcript_or_category_suffix(goal):
    assert plan_search_queries(goal) == []


@pytest.mark.parametrize('topic', ['C++', 'C#', '.NET', 'Node.js'])
def test_search_topic_punctuation_survives_keyword_planning(topic):
    query = f'{topic} Linux offline support'
    assert plan_search_queries(f'Search {topic} offline support on Linux', [query]) == [query]


@pytest.mark.parametrize('goal,probe', [
    ('Search CedarDB Linux offline support', 'CedarDB Android online support'),
    ('Search Python 3.12 security fixes', 'Python 4 security fixes'),
])
def test_model_queries_cannot_introduce_platform_or_numeric_scope(goal, probe):
    assert plan_search_queries(goal, [probe]) == []


@pytest.mark.parametrize('goal,probes', [
    ('Search CedarDB Linux offline support',
     ['CedarDB Linux compatibility', 'CedarDB airgapped deployment']),
    ('Search CedarDB without cloud sync', ['CedarDB cloud sync privacy']),
    ('Search flights from Paris to London', ['Paris London flights']),
    ('Search AuroraClient Linux or Windows offline support',
     ['AuroraClient Linux support', 'AuroraClient Windows offline guide']),
])
def test_model_queries_can_cover_parts_while_the_full_task_remains_separate(
    make_tools, goal, probes,
):
    tools = make_tools(goal, objective=goal)
    tools.execute(call('web_search', search_queries=probes))
    request = tools.internet.lookup.call_args
    assert request.args[0] == probes[0]
    assert request.kwargs['search_queries'] == probes
    assert request.kwargs['objective'] == goal


def test_reading_queries_reject_invented_scope_without_rewriting_the_query():
    goal = ('Search Amber Wings season four light novel continuation after The Hero Awakens '
            'without spoilers')
    assert plan_search_queries(goal, ['Amber Wings season 4 episode 24 manga chapter 90']) == []
    query = 'Amber Wings season 4 The Hero Awakens light novel continuation without spoilers'
    assert plan_search_queries(goal, [query]) == [query]


def test_numeric_scope_is_rejected_without_leaving_a_malformed_query_fragment():
    assert plan_search_queries(
        'Search Amber Wings overview', ['Amber Wings episode 24 finale'],
    ) == []


def test_excluded_search_terms_retain_their_query_operator():
    query = 'CedarDB Linux support -Windows -cloud'
    assert plan_search_queries('Search CedarDB Linux support -Windows -cloud', [query]) == [query]


@pytest.mark.parametrize('goal,query', [
    ('How much is this RTX5090?', 'RTX5090 price'),
    ('When does Amber Wings come out?', 'Amber Wings release date'),
    ('Can CedarDB run without an internet connection?', 'CedarDB offline support'),
    ('What episodes does Astral have?', 'Astral episode list'),
    ('Where does Astral season 2 continue in the LN?', 'Astral season 2 novel volume chapter'),
])
def test_model_can_translate_conversational_intent_into_search_vocabulary(goal, query):
    assert plan_search_queries(goal, [query]) == [query]


@pytest.mark.parametrize('goal,query', [
    ('Search Astral S2 light novel continuation', 'Astral season 2 light novel continuation'),
    ('Search Astral S2E24 continuation', 'Astral season 2 episode 24 continuation'),
    ('Search Astral ep. 24 continuation', 'Astral episode 24 continuation'),
    ('Search Astral volume eleven', 'Astral volume 11'),
    ('Search Astral part twelve', 'Astral part 12'),
])
def test_ai_can_expand_the_owners_explicit_numeric_shorthand(make_tools, goal, query):
    tools = make_tools(goal, objective=goal)
    result = tools.execute(call('web_search', query=query))
    assert not result.is_error
    assert tools.internet.lookup.call_args.args[0] == query
    assert tools.internet.lookup.call_args.kwargs['objective'] == goal


@pytest.mark.parametrize('goal,query', [
    ('Search Astral S2 light novel continuation', 'Astral season 3 light novel continuation'),
    ('Search Astral ep. 24 continuation', 'Astral season 24 continuation'),
    ('Search Astral volume eleven', 'Astral volume 12'),
])
def test_shorthand_normalization_does_not_authorize_different_numeric_scope(goal, query):
    assert plan_search_queries(goal, [query]) == []
