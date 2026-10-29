"""A reading-point request replaces a completed episode lookup's outcome."""

import pytest

from ciel.chat.session import ChatSession
from ciel.internet import InternetResult
from ciel.providers.base import ChatMessage
from ciel.task_context import resolve_task_context
from ciel.web_answers import web_answer_fallback
from tests.test_controller_runtime import _internet, _tool_call
from tests.test_conversation_search_replays import _report, _router, _seed, _settings, _source
from tests.test_screen_integration import run_callbacks

EPISODE_REQUEST = 'can you search the latest episode in Tensura season 4?'
EPISODE_ANSWER = (
    'The season is listed as finished airing with 24 episodes. '
    'Episode 24, The Hero Awakens, aired September 25, 2026. '
    "I couldn't complete the additional lookup."
)
FOLLOWUP = ("Alright, it's okay. Can you like find the continuation "
            'from the Hero Awakens on the light node?')
HISTORY = (('user', EPISODE_REQUEST), ('assistant', EPISODE_ANSWER))


def test_reading_followup_replaces_episode_outcome_and_preserves_user_referent():
    history = [ChatMessage(role, text) for role, text in HISTORY]
    task = resolve_task_context(FOLLOWUP, history)
    assert task.kind == 'followup' and task.task_type == 'web'
    assert 'Tensura' in task.request and 'season 4' in task.request
    assert 'Hero Awakens' in task.request and 'light novel' in task.request
    assert 'latest episode' not in task.request and 'light node' not in task.request
    assert '24' not in task.request and '2026' not in task.request
    assert history[-1].content == EPISODE_ANSWER
    retry = resolve_task_context('Again.', [*history, ChatMessage('user', FOLLOWUP)])
    assert retry.request == task.request


@pytest.mark.parametrize('current, medium', [
    ('Where do I continue after that in the light novel?', 'light novel'),
    ('Where should I start reading the light novel after that?', 'light novel'),
    ('Find the manga continuation from The Hero Awakens.', 'manga'),
])
def test_reading_medium_does_not_restore_the_old_requested_outcome(current, medium):
    task = resolve_task_context(current, [ChatMessage(role, text) for role, text in HISTORY])
    assert task.kind == 'followup' and medium in task.request
    assert 'Tensura' in task.request and 'latest episode' not in task.request


@pytest.mark.parametrize('current', [
    'Find the continuation from Dawn Realm in the light novel.',
    'Search LightNode cloud hosting prices.',
])
def test_named_new_topic_does_not_borrow_the_previous_series(current):
    task = resolve_task_context(current, [ChatMessage(role, text) for role, text in HISTORY])
    assert 'Tensura' not in task.request
    assert task.kind == 'new'
    assert 'light node' not in current.casefold() or 'light novel' not in task.request


def test_unanchored_light_node_request_does_not_become_a_confirmed_novel_name():
    assert resolve_task_context(FOLLOWUP, []).request == FOLLOWUP


def test_greeting_and_search_leadin_do_not_become_part_of_the_reading_subject():
    history = [ChatMessage('user', 'Hey Ciel. Can you search the latest episode in Tensura '
                           'season 4 for me?'), ChatMessage('assistant', EPISODE_ANSWER)]
    task = resolve_task_context(FOLLOWUP, history)
    assert task.request.startswith('Tensura season 4:')
    assert 'Hey Ciel' not in task.request and 'for me' not in task.request


def test_cancelled_lookup_does_not_supply_a_series_to_the_reading_request():
    history = [ChatMessage(role, text) for role, text in HISTORY]
    history.append(ChatMessage('user', 'Cancel.'))
    task = resolve_task_context(FOLLOWUP, history)
    assert task.kind == 'new' and task.request == FOLLOWUP


@pytest.mark.parametrize('query', [
    'Tensura season 4: continuation from the Hero Awakens on the light novel?',
    'Where to read the light novel after Tensura episode 24?',
    'Where does the anime continue in the light novel after "the hero awakens"?',
])
def test_unverified_mapping_does_not_ask_the_user_to_repeat_their_episode(query):
    reply = web_answer_fallback(InternetResult(query=query))
    assert 'light novel' in reply and "couldn't verify" in reply
    assert '?' not in reply


def test_reading_request_without_an_episode_can_still_ask_for_it():
    reply = web_answer_fallback(InternetResult(
        query='Where does Tensura season 4 part 2 continue in the light novel?',
    ))
    assert 'What was the last episode' in reply


@pytest.mark.parametrize('entrypoint', ['desktop_voice', 'desktop_text', 'chat'])
def test_reading_followup_reaches_lookup_with_new_outcome(tmp_path, monkeypatch, entrypoint):
    fact = ("Tensura season 4 episode The Hero Awakens ends at light novel volume 11. "
            'Continue reading from light novel volume 12.')
    source = _source(fact)
    internet = _internet(monkeypatch, [InternetResult(attempted=True, sources=[source])])
    router = _router([_tool_call('web_search', query='Tensura season 4 light novel continuation'),
                      _report(fact)])
    if entrypoint.startswith('desktop'):
        info = {'wake_verified': True} if entrypoint == 'desktop_voice' else {
            'input_source': 'keyboard',
        }
        _, tts, _, store, turns = run_callbacks(
            tmp_path, monkeypatch, [(FOLLOWUP, info)], router=router, controller=True,
            initialize_conversation=lambda store, session: _seed(store, session, HISTORY),
        )
        close = store.close
        if entrypoint == 'desktop_text':
            assert tts.speak.call_count == 2
            assert fact in tts.speak.call_args.args[0]
            tts.begin_stream.assert_not_called()
    else:
        session = ChatSession(_settings(tmp_path), channel='telegram', session_id='followup-owner',
                              router=router)
        for role, text in HISTORY:
            session.conversation.add_turn(role, text)
        session.internet = internet
        session.reply(FOLLOWUP)
        turns = session.conversation.get_recent_turns()
        close = session.close
    try:
        internet.lookup.assert_called_once()
        actual = internet.lookup.call_args
        for text in [actual.args[0], actual.kwargs['objective']]:
            assert 'Tensura' in text and 'season 4' in text and 'light novel' in text
            assert 'latest episode' not in text and 'light node' not in text
        assert fact in turns[-1].content and turns[-1].sources == [source.to_dict()]
        assert [turn.content for turn in turns if turn.role == 'user'][-1] == FOLLOWUP
    finally:
        close()
