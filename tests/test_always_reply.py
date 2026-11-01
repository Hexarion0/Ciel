"""The owner's always-reply preference survives silence and canceled conversations."""

from unittest.mock import MagicMock

import pytest

from ciel.audio.ambient import AmbientConversation
from ciel.providers.base import LLMResponse
from tests.test_ambient import run_callbacks
from tests.test_controller_runtime import _internet


@pytest.mark.parametrize('transcript', [
    'Never mind.', 'Ciel is the name of the assistant.',
    'A completely new topic.', 'Mom, what is for dinner?',
])
def test_always_policy_replies_without_a_wake_or_recent_dialogue(transcript):
    ambient = AmbientConversation(followup_policy='always')
    router = MagicMock()
    decision = ambient.decide(transcript, router, [])
    assert decision.addressed and not decision.explicit
    assert decision.reason == 'always_reply'
    router.send.assert_not_called()
    ambient.reset()
    assert ambient.decide(transcript, router, []).addressed
    assert not ambient.decide('   ', router, []).addressed


@pytest.mark.parametrize('controller', [True, False])
def test_never_mind_and_next_statement_both_receive_spoken_replies(
    tmp_path, monkeypatch, controller,
):
    monkeypatch.setenv('AMBIENT_FOLLOWUP_POLICY', 'always')
    internet = _internet(monkeypatch, [])
    router = MagicMock()
    router.send.return_value = LLMResponse("Yes, I'm Ciel, your assistant.", 'fixture', 'fixture')
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [
            ('Never mind.', {}), ('Ciel is the name of the assistant.', {}),
        ], router=router, controller=controller,
    )
    try:
        assert [call.args[0] for call in tts.speak.call_args_list] == [
            'Okay, canceled.', "Yes, I'm Ciel, your assistant.",
        ]
        assert [turn.content for turn in turns if turn.role == 'user'] == [
            'Never mind.', 'Ciel is the name of the assistant.',
        ]
        assert not any(call.kwargs.get('purpose') == 'speech_gate'
                       for call in router.send.call_args_list)
        internet.lookup.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize('controller', [True, False])
def test_always_reply_asks_for_repeat_for_unclear_speech(tmp_path, monkeypatch, controller):
    monkeypatch.setenv('AMBIENT_FOLLOWUP_POLICY', 'always')
    _internet(monkeypatch, [])
    router = MagicMock()
    _, tts, _, store, _turns = run_callbacks(
        tmp_path, monkeypatch, [
            ('A doubtful transcript', {'stt_needs_repeat': True, 'stt_confidence': .1}),
        ], router=router, controller=controller,
    )
    try:
        assert tts.speak.call_args.args[0] == "I didn't catch that clearly. Could you say it again?"
        router.send.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize('controller', [True, False])
def test_reply_preference_does_not_treat_unverified_speech_as_memory_authority(
    tmp_path, monkeypatch, controller,
):
    monkeypatch.setenv('AMBIENT_FOLLOWUP_POLICY', 'always')
    _internet(monkeypatch, [])
    router = MagicMock()
    router.send.return_value = LLMResponse('Please tell me directly what to remember.',
                                         'fixture', 'fixture')
    _, tts, _, store, _turns = run_callbacks(
        tmp_path, monkeypatch, [('Remember my favorite color is violet.', {})],
        router=router, controller=controller,
    )
    try:
        assert tts.speak.call_count == 1
        assert not any('violet' in item.content.casefold() for item in store.list_memories())
    finally:
        store.close()
