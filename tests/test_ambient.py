"""Ambient reply decisions, local VAD framing, and persistence boundaries."""

import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ciel.audio.ambient import AmbientConversation
from ciel.memory.bridge import start_desktop_conversation as _start_desktop_conversation
from ciel.providers.base import ChatMessage, LLMResponse


@pytest.mark.parametrize(
    "text", [
        "Hey, Ciel.", "Ciel, help me.", "Can you help me, Ciel?",
        "What's up Ciel?", "What's up, Ciell?", "Hi Siel!", "Hello, Ciel.",
        "So what's good Ciel?", "So Ciel, what's on your mind?",
        "Well, hey Ciel, what's the time?", "Uh, what's up, Siel?",
    ]
)
def test_direct_address_needs_no_cloud_judgment(text):
    router = MagicMock()
    decision = AmbientConversation().decide(text, router, [])
    assert decision.addressed and decision.explicit
    router.send.assert_not_called()


def test_unaddressed_speech_is_temporary_and_expires(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("ciel.audio.ambient.time.monotonic", lambda: clock[0])
    ambient = AmbientConversation()
    router = MagicMock()
    assert not ambient.decide("I hate Ciel.", router, []).addressed
    assert not ambient.decide("Ciel is an assistant.", router, []).addressed
    assert not ambient.decide("What do you think about Ciel?", router, []).addressed
    assert not ambient.decide("Who is Ciel?", router, []).addressed
    assert not ambient.decide("Dad, what's for dinner?", router, []).addressed
    assert "Dad" in ambient.context_note()
    router.send.assert_not_called()
    clock[0] += 91
    assert ambient.context_note() == ""


@pytest.mark.parametrize("text", [
    "So Ciel is an assistant.", "Well, Ciel said hello.",
    "So what do you think about Ciel?", "Um, who is Ciel?",
    "So I hate Ciel.", "So what's good?", "Well, Mom, what's for dinner?",
])
def test_fillers_do_not_turn_background_mentions_into_addresses(text):
    router = MagicMock()
    assert not AmbientConversation().decide(text, router, []).addressed
    router.send.assert_not_called()


def test_independent_wake_confirms_address_even_when_stt_misses_name():
    router = MagicMock()
    decision = AmbientConversation().decide("What's up?", router, [], verified_wake=True)
    assert decision.addressed and decision.explicit and decision.reason == "verified_wake"
    router.send.assert_not_called()


@pytest.mark.parametrize(
    "result",
    [
        "not json", "[]", '{"addressed": true, "confidence": 0.7}',
        '{"addressed": true, "confidence": "0.99"}',
        '{"addressed": true, "confidence": true}',
        '{"addressed": true, "confidence": 2}',
        '{"addressed": false, "confidence": 0.99}',
    ],
)
def test_uncertain_or_malformed_judgment_means_silence(result):
    ambient = AmbientConversation()
    ambient.replied()
    router = MagicMock()
    router.send.return_value = LLMResponse(result, "test", "test")
    deadline = ambient.deadline
    assert not ambient.decide("What do you mean?", router, []).addressed
    if result == '{"addressed": false, "confidence": 0.99}':
        assert ambient.deadline == 0
    else:
        assert ambient.deadline == deadline


def test_followup_can_supply_hint_without_replacing_original():
    ambient = AmbientConversation()
    ambient.replied()
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"addressed": true, "confidence": 0.95, "interpretation": "A new feature"}',
        "test", "test",
    )
    original = "What future should we add?"
    decision = ambient.decide(original, router, [ChatMessage("assistant", "Let's plan features.")])
    assert decision.addressed and not decision.explicit
    assert decision.interpretation == "A new feature"
    request = router.send.call_args
    assert request.kwargs["purpose"] == "speech_gate"
    assert json.loads(request.args[0][1].content)["transcript"] == original


def test_fenced_json_judgment_is_accepted():
    ambient = AmbientConversation()
    ambient.replied()
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '```json\n{"addressed": true, "confidence": 0.95}\n```', "test", "test"
    )
    decision = ambient.decide("And what does that mean?", router, [])
    assert decision.addressed and decision.reason == "clear_followup"


def test_silent_decisions_explain_the_reason():
    ambient = AmbientConversation()
    router = MagicMock()
    assert ambient.decide("Dinner is ready.", router, []).reason == "outside_followup_window"
    assert ambient.decide("", router, []).reason == "no_transcript"
    assert ambient.decide("Mom, pass that.", router, []).reason == "addressed_to_someone_else"


def test_provider_failure_defaults_to_silence():
    ambient = AmbientConversation()
    ambient.replied()
    router = MagicMock()
    router.send.side_effect = RuntimeError("offline")
    assert not ambient.decide("Continue?", router, []).addressed
    ambient.reset()
    assert ambient.context_note() == ""


def test_long_followup_is_judged_using_when_speech_started(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("ciel.audio.ambient.time.monotonic", lambda: clock[0])
    ambient = AmbientConversation(followup_seconds=20)
    ambient.replied()
    clock[0] = 130.0  # Speech began during the window but took time to finish.
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"addressed": true, "confidence": 0.95}', "test", "test"
    )
    assert ambient.decide("Explain that a little more.", router, [], started_at=110).addressed


def test_long_pause_followup_uses_recent_dialogue_but_requires_context_match(monkeypatch):
    from ciel.providers.base import ChatMessage

    clock = [100.0]
    monkeypatch.setattr('ciel.audio.ambient.time.monotonic', lambda: clock[0])
    ambient = AmbientConversation(followup_seconds=20)
    ambient.record_addressed('user', 'What are we working on next?')
    ambient.record_addressed('assistant', 'We could improve the roadmap and memory system.')
    clock[0] += 90

    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"listener":"assistant","confidence":0.96,"interpretation":""}',
        'test', 'test',
    )
    turns = [
        ChatMessage('user', 'What are we working on next?'),
        ChatMessage('assistant', 'We could improve the roadmap and memory system.'),
    ]
    decision = ambient.decide('Which roadmap version are we on?', router, turns)
    assert decision.addressed and decision.reason == 'clear_followup'
    assert ambient.deadline == 0.0  # The short direct-follow-up window stays expired.
    gate_input = router.send.call_args.args[0][-1].content
    assert 'roadmap version' in gate_input
    assert 'improve the roadmap' in gate_input


def test_unrelated_speech_after_active_window_is_context_checked_not_accepted(monkeypatch):
    from ciel.providers.base import ChatMessage

    clock = [100.0]
    monkeypatch.setattr('ciel.audio.ambient.time.monotonic', lambda: clock[0])
    ambient = AmbientConversation(followup_seconds=20)
    ambient.record_addressed('assistant', 'The roadmap is ready.')
    clock[0] += 90
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"listener":"other","confidence":0.97,"interpretation":""}',
        'test', 'test',
    )
    turns = [ChatMessage('assistant', 'The roadmap is ready.')]
    decision = ambient.decide('Can somebody bring the groceries in?', router, turns)
    assert not decision.addressed and decision.reason == 'other_listener'
    router.send.assert_called_once()


def test_interrupted_followup_gate_receives_persistent_and_inflight_context():
    from ciel.providers.base import ChatMessage

    ambient = AmbientConversation()
    ambient.record_addressed('user', 'Help me understand the roadmap.')
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"listener":"assistant","confidence":0.96,"interpretation":""}',
        'test', 'test',
    )
    previous_turns = [
        ChatMessage('user', 'What should we improve?'),
        ChatMessage('assistant', 'We could work on the roadmap.'),
    ]

    decision = ambient.decide(
        'Could you explain the first part?', router, previous_turns, during_reply=True,
    )

    assert decision.addressed
    recent = json.loads(router.send.call_args.args[0][-1].content)['recent_dialogue']
    assert [item['content'] for item in recent] == [
        'What should we improve?', 'We could work on the roadmap.',
        'Help me understand the roadmap.',
    ]


def test_vad_preserves_partial_chunks_and_converts_integer_audio(monkeypatch):
    from ciel.audio.activity import SpeechActivity

    model = MagicMock()
    model.chunk_samples.return_value = 512
    model.process_array.side_effect = [0.1, 0.8]
    monkeypatch.setattr("pysilero_vad.SileroVoiceActivityDetector", lambda: model)
    vad = SpeechActivity()
    assert not vad(np.full(256, 16384, dtype=np.int16))
    assert vad(np.full(768, 16384, dtype=np.int16))
    assert model.process_array.call_count == 2
    np.testing.assert_allclose(model.process_array.call_args.args[0], 0.5)
    assert vad.pending.size == 0
    vad.reset()
    model.reset.assert_called_once()


def run_callbacks(
    tmp_path, monkeypatch, utterances, *, ambient=True, router=None, observed=None,
    tts=None, interrupted=None, persona_file=None,
    streaming=False, clock=None, ai_transcripts=False, cli_args=(), tts_enabled=True,
    controller=False, observed_after=None, initialize_conversation=None,
):
    """Run real CLI callbacks using an isolated DB and no audio hardware."""
    import sys

    from ciel import __main__, audio, config, platform, providers
    from ciel.memory import MemoryStore, bridge

    launched_session = []

    def start_conversation(store):
        session_id = _start_desktop_conversation(store)
        launched_session.append(session_id)
        if initialize_conversation is not None:
            initialize_conversation(store, session_id)
        return session_id

    monkeypatch.setattr(bridge, "start_desktop_conversation", start_conversation)

    settings = config.Settings(
        _env_file=None, memory_db_path=str(tmp_path / "memory.db"), tts_streaming=streaming,
        ambient_listening=ambient,
        show_ai_transcripts=ai_transcripts,
        tts_enabled=tts_enabled,
        controller_enabled=controller,
        ciel_persona_file=str(persona_file) if persona_file else "persona.txt",
    )
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(platform, "get_audio_backend", MagicMock())
    router = router or MagicMock()
    monkeypatch.setattr(providers, "get_model_router", lambda: router)
    tts = tts or MagicMock()
    for name in ("WakeWordDetector", "SpeechToText", "VoiceRecorder"):
        monkeypatch.setattr(audio, name, MagicMock())
    monkeypatch.setattr(audio, "TextToSpeech", MagicMock(return_value=tts))
    loop = SimpleNamespace(is_running=True, stop=MagicMock(), set_ambient=MagicMock())

    def listener(**kwargs):
        loop.submit_text = lambda text: kwargs["on_transcription"](
            text, {"input_source": "keyboard", "cancel_event": Event()}, None,
        )

        def start():
            if interrupted:
                kwargs['on_interruption'](interrupted)
            for text, info in observed or []:
                kwargs["on_transcript_observed"](text, info)
            for index, (text, info) in enumerate(utterances):
                kwargs["on_transcription"](text, info, None)
                for observed_text, observed_info in (observed_after or {}).get(index, []):
                    kwargs["on_transcript_observed"](observed_text, observed_info)

        loop.start = start
        loop.start_text_only = start
        return loop

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(audio, "WakeListenerLoop", listener)
    # Patch only the CLI clock: background provider threads use real time.sleep.
    monkeypatch.setattr(__main__, "time", SimpleNamespace(
        sleep=interrupt, monotonic=clock or __main__.time.monotonic,
    ))
    monkeypatch.setattr(sys, "argv", [
        "ciel", "--ambient" if ambient else "--listen", *cli_args,
    ])
    assert __main__.main() == 0
    store = MemoryStore(settings.memory.db_path)
    turns = store.get_recent_turns(launched_session[-1])
    return router, tts, loop, store, turns


def test_keyboard_message_is_explicit_and_not_reinterpreted_as_speech(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.return_value = LLMResponse("A profile describes someone.", "test", "test")
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("What is a profile?", {"input_source": "keyboard"})],
        router=router,
    )
    try:
        router.send.assert_called_once()
        prompt = router.send.call_args.args[0][0].content
        assert "latest message was typed on the keyboard" in prompt
        assert "message was transcribed from speech" not in prompt
        assert [turn.content for turn in turns if turn.role == "user"] == ["What is a profile?"]
    finally:
        store.close()


def test_keyboard_memory_confirmation_needs_no_wake_phrase(tmp_path, monkeypatch):
    _router, _tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch, [
            ("My favorite game is CS2.", {"input_source": "keyboard"}),
            ("yes", {"input_source": "keyboard"}),
        ],
    )
    try:
        assert store.get_fact("user:favorite_game").metadata["value"] == "CS2"
    finally:
        store.close()


@pytest.mark.parametrize('configured, arguments, enabled', [
    (True, (), True), (False, (), False),
    (True, ('--no-show-ai-transcripts',), False),
    (False, ('--show-ai-transcripts',), True),
])
def test_live_ai_transcript_config_and_cli_override_at_local_playback(
    tmp_path, monkeypatch, capsys, configured, arguments, enabled,
):
    from ciel.audio.tts import TextToSpeech

    tts = MagicMock()
    tts.clean_text_for_speech.side_effect = TextToSpeech.clean_text_for_speech

    def speak(text, **controls):
        lines = capsys.readouterr().out.splitlines()
        assert not any(line.startswith('CIEL: ') for line in lines)
        if enabled:
            assert "Hey! What's on your mind?" not in '\n'.join(lines)
        callback = controls.get('on_start')
        assert bool(callback) is enabled
        assert bool(controls.get('on_progress')) is enabled
        if callback:
            callback()
            assert capsys.readouterr().out == ''  # Starting audio does not reveal its ending.
            controls['on_progress']('Hey', False, False)
            assert capsys.readouterr().out == ''  # Non-TTY output waits for a safe snapshot.
            controls['on_progress'](TextToSpeech.clean_text_for_speech(text), True, False)
        return True

    tts.speak.side_effect = speak
    _router, _tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch, [('Hey Ciel.', {})], tts=tts,
        ai_transcripts=configured, cli_args=arguments,
    )
    try:
        output = capsys.readouterr().out
        assert ("CIEL: Hey! What's on your mind?" in output) is enabled
        from ciel import audio

        assert audio.TextToSpeech.call_args.kwargs.get('include_alignments', False) is enabled
    finally:
        store.close()


@pytest.mark.parametrize('streaming', [False, True])
def test_live_ai_transcripts_show_clean_speech_and_persist_one_reply(
    tmp_path, monkeypatch, capsys, streaming,
):
    from ciel.audio.tts import TextToSpeech
    from ciel.providers.base import LLMStreamChunk

    router, tts = MagicMock(), MagicMock()
    text = '**First** sentence. Second sentence.'
    router.send.return_value = LLMResponse(text, 'test', 'test')
    router.stream.side_effect = lambda *args, **kwargs: iter([
        LLMStreamChunk(text, 'test', 'test'),
    ])
    tts.clean_text_for_speech.side_effect = TextToSpeech.clean_text_for_speech

    def speak(_text, **controls):
        controls['on_start']()
        cleaned = TextToSpeech.clean_text_for_speech(_text)
        controls['on_progress'](cleaned[:4], False, False)
        controls['on_progress'](cleaned, True, False)
        return True

    tts.speak.side_effect = speak
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [('Ciel, explain this.', {})],
        router=router, tts=tts, streaming=streaming, ai_transcripts=True,
    )
    try:
        output = capsys.readouterr().out
        live_lines = [line for line in output.splitlines() if line.startswith('CIEL: ')]
        expected = [
            'First sentence. Second sentence.',
        ]
        assert [line.removeprefix('CIEL: ') for line in live_lines] == expected
        assert [turn.content for turn in turns if turn.role == 'assistant'] == [text]
    finally:
        store.close()


def test_failed_local_playback_has_no_live_ai_transcript(tmp_path, monkeypatch, capsys):
    tts = MagicMock()
    tts.speak.return_value = False  # No on_start when synthesis/playback fails.
    _router, _tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch, [('Hey Ciel.', {})], tts=tts, ai_transcripts=True,
    )
    try:
        lines = capsys.readouterr().out.splitlines()
        assert not any(line.startswith('CIEL: ') for line in lines)
        assert any('CIEL (text): "Hey! What\'s on your mind?"' in line for line in lines)
    finally:
        store.close()


def test_cancellation_before_batch_playback_callback_has_no_live_ai_transcript(
    tmp_path, monkeypatch, capsys,
):
    router, tts = MagicMock(), MagicMock()
    router.send.return_value = LLMResponse('Canceled reply.', 'test', 'test')

    def speak(_text, **controls):
        controls['cancel_event'].set()
        controls['on_start']()
        return False

    tts.speak.side_effect = speak
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [('Ciel, explain this.', {'cancel_event': Event()})],
        router=router, tts=tts, ai_transcripts=True,
    )
    try:
        lines = capsys.readouterr().out.splitlines()
        assert not any(line.startswith('CIEL: ') for line in lines)
        assert [turn.content for turn in turns if turn.role == 'assistant'] == ['Canceled reply.']
    finally:
        store.close()


def test_caption_logging_uses_root_handler_and_restores_on_shutdown(
    tmp_path, monkeypatch, capsys,
):
    import logging

    from ciel.audio import captions
    from ciel.audio.tts import TextToSpeech

    installer = captions.install_caption_logging
    restored = []

    def install(renderer, logger=None):
        assert logger is None  # CIEL's console handler belongs to the root logger.
        root = logging.getLogger()
        originals = root.handlers[:]
        restore = installer(renderer)
        assert any(isinstance(handler, captions.CaptionLoggingHandler) for handler in root.handlers)

        def restore_and_check():
            restore()
            assert root.handlers == originals
            restored.append(True)

        return restore_and_check

    monkeypatch.setattr(captions, 'install_caption_logging', install)
    tts = MagicMock()
    tts.clean_text_for_speech.side_effect = TextToSpeech.clean_text_for_speech

    def speak(_text, **controls):
        controls['on_start']()
        controls['on_progress']('Hey', False, False)
        # Leave a partial line open so shutdown must finish it before restoring logs.
        return True

    tts.speak.side_effect = speak
    _router, _tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch, [('Hey Ciel.', {})], tts=tts, ai_transcripts=True,
    )
    try:
        assert restored == [True]
        assert 'CIEL: Hey [interrupted]' in capsys.readouterr().out
    finally:
        store.close()


@pytest.mark.parametrize('tty', [False, True])
@pytest.mark.parametrize('kind', ['greeting', 'clock', 'batch'])
def test_full_response_is_not_revealed_before_playback_progress(
    tmp_path, monkeypatch, capsys, tty, kind,
):
    import io

    from ciel.audio import captions
    from ciel.audio.tts import TextToSpeech

    class CaptionStream(io.StringIO):
        def isatty(self):
            return tty

    output = CaptionStream()
    renderer = captions.TerminalCaptions(output, width=120)
    monkeypatch.setattr(captions, 'TerminalCaptions', lambda: renderer)
    router, tts = MagicMock(), MagicMock()
    router.send.return_value = LLMResponse('The full answer ends right here.', 'test', 'test')
    tts.clean_text_for_speech.side_effect = TextToSpeech.clean_text_for_speech
    spoken = []

    def speak(text, **controls):
        full = TextToSpeech.clean_text_for_speech(text)
        assert full not in capsys.readouterr().out + output.getvalue()
        controls['on_start']()
        assert full not in capsys.readouterr().out + output.getvalue()
        controls['on_progress'](full[:max(1, len(full) // 3)], False, False)
        assert full not in capsys.readouterr().out + output.getvalue()
        controls['on_progress'](full, True, False)
        spoken.append(full)
        return True

    tts.speak.side_effect = speak
    text = {
        'greeting': 'Hey Ciel.', 'clock': 'Ciel, what time is it?',
        'batch': 'Ciel, explain this.',
    }[kind]
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [(text, {})], router=router, tts=tts, ai_transcripts=True,
    )
    try:
        assert len(spoken) == 1 and spoken[0] in output.getvalue()
        assert '\x1b' not in output.getvalue()
        if kind != 'greeting':
            assert [turn.content for turn in turns if turn.role == 'assistant'] == spoken
    finally:
        store.close()


@pytest.mark.parametrize('failure', ['returns_false', 'raises'])
@pytest.mark.parametrize('kind', ['greeting', 'batch'])
def test_audio_failure_keeps_full_reply_available_as_text(
    tmp_path, monkeypatch, capsys, failure, kind,
):
    router, tts = MagicMock(), MagicMock()
    full = "Hey! What's on your mind?" if kind == 'greeting' else 'The complete intended answer.'
    router.send.return_value = LLMResponse(full, 'test', 'test')

    def speak(_text, **controls):
        assert full not in capsys.readouterr().out
        if failure == 'raises':
            raise RuntimeError('The audio device is unavailable')
        return False

    tts.speak.side_effect = speak
    text = 'Hey Ciel.' if kind == 'greeting' else 'Ciel, explain this.'
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [(text, {})], router=router, tts=tts, ai_transcripts=True,
    )
    try:
        assert f'CIEL (text): "{full}"' in capsys.readouterr().out
        if kind == 'batch':
            assert [turn.content for turn in turns if turn.role == 'assistant'] == [full]
    finally:
        store.close()


@pytest.mark.parametrize('captions_enabled, tts_enabled', [(False, True), (True, False)])
def test_regular_reply_logs_remain_when_playback_captions_are_inactive(
    tmp_path, monkeypatch, capsys, captions_enabled, tts_enabled,
):
    tts = MagicMock()
    full = "Hey! What's on your mind?"

    def speak(_text, **controls):
        assert f'CIEL: "{full}"' in capsys.readouterr().out
        return tts_enabled

    tts.speak.side_effect = speak
    _router, _tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch, [('Hey Ciel.', {})], tts=tts,
        ai_transcripts=captions_enabled, tts_enabled=tts_enabled,
    )
    store.close()


def test_generated_stream_without_audio_retains_full_text_answer(tmp_path, monkeypatch, capsys):
    from ciel.providers.base import LLMStreamChunk

    full = 'This generated answer is still available without audio.'
    router, tts = MagicMock(), MagicMock()
    router.stream.side_effect = lambda *args, **kwargs: iter([LLMStreamChunk(full, 'test', 'test')])
    tts.speak.return_value = False
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [('Ciel, explain this.', {})], router=router, tts=tts,
        streaming=True, ai_transcripts=True,
    )
    try:
        assert f'CIEL (text): "{full}"' in capsys.readouterr().out
        assert [turn.content for turn in turns if turn.role == 'assistant'] == [full]
    finally:
        store.close()


def test_followup_window_begins_after_long_playback(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('ciel.audio.ambient.time.monotonic', lambda: clock[0])
    tts = MagicMock()

    def speak(*_args, **_kwargs):
        clock[0] += 30  # Longer than the default 20-second follow-up window.
        return True

    tts.speak.side_effect = speak
    router = MagicMock()
    router.send.side_effect = [
        LLMResponse('{"addressed": true, "confidence": 0.95}', 'test', 'test'),
        LLMResponse('Let us continue.', 'test', 'test'),
    ]
    _router, tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch, [('Hey Ciel.', {}), ('Continue?', {})],
        router=router, tts=tts,
    )
    try:
        assert tts.speak.call_count == 2
        assert router.send.call_count == 2
    finally:
        store.close()


def test_live_stream_speaks_sentences_once_and_persists_one_assistant_turn(tmp_path, monkeypatch):
    from ciel.providers.base import LLMStreamChunk

    router, tts = MagicMock(), MagicMock()
    router.stream.side_effect = lambda *args, **kwargs: iter([
        LLMStreamChunk("First sentence. ", "stream-model", "nim"),
        LLMStreamChunk("Second sentence.", "stream-model", "nim"),
    ])

    def speak(text, **kwargs):
        kwargs["on_start"]()
        return True

    tts.speak.side_effect = speak
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Ciel, explain streaming.", {})],
        router=router, tts=tts, streaming=True,
    )
    try:
        router.send.assert_not_called()
        assert [call.args[0] for call in tts.speak.call_args_list] == [
            "First sentence.", "Second sentence.",
        ]
        assert [(turn.role, turn.content) for turn in turns] == [
            ("user", "explain streaming."), ("assistant", "First sentence. Second sentence."),
        ]
        assert turns[-1].model == "stream-model"
    finally:
        store.close()


def test_canceled_stream_and_new_addition_preserve_combined_request(tmp_path, monkeypatch):
    from ciel.providers.base import LLMStreamChunk

    canceled, replacement = Event(), Event()
    router = MagicMock()

    def generate(messages, **kwargs):
        if router.stream.call_count == 1:
            canceled.set()
            yield LLMStreamChunk("Discard this.", "stream-model", "nim")
        else:
            yield LLMStreamChunk("A combined answer.", "stream-model", "nim")

    router.stream.side_effect = generate
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Ciel, explain streaming.", {"cancel_event": canceled}),
         ("And include interruptions.", {
             "cancel_event": replacement, "supersedes_cancel_event": canceled,
         })], router=router, streaming=True,
    )
    try:
        router.send.assert_not_called()
        assert [turn.role for turn in turns] == ["user", "user", "assistant"]
        messages = router.stream.call_args.args[0]
        assert [message.content for message in messages if message.role == "user"] == [
            "explain streaming.\nAnd include interruptions.",
        ]
        tts.speak.assert_called_once()
    finally:
        store.close()


def test_live_stream_interrupted_after_audio_starts_archives_partial_reply(tmp_path, monkeypatch):
    from ciel.providers.base import LLMStreamChunk

    cancel = Event()
    router, tts = MagicMock(), MagicMock()
    router.stream.side_effect = lambda *args, **kwargs: iter([
        LLMStreamChunk("First sentence. Second sentence. ", "stream-model", "nim"),
    ])

    def speak(text, **kwargs):
        kwargs["on_start"]()
        cancel.set()
        return False

    tts.speak.side_effect = speak
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Ciel, explain streaming.", {"cancel_event": cancel}),
         ("", {"supersedes_cancel_event": cancel})],
        router=router, tts=tts, streaming=True,
    )
    try:
        assert [turn.role for turn in turns] == ["user", "assistant"]
        router.stream.assert_called_once()  # Empty noise must not restart partial playback.
        assert turns[-1].content.endswith("[Playback was interrupted.]")
        assert "First sentence." in turns[-1].content
        assert [call.args[0] for call in tts.speak.call_args_list] == ["First sentence."]
    finally:
        store.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("answer", ["", "Here is the answer."])
def test_reply_cannot_speak_internal_interruption_note(
    tmp_path, monkeypatch, capsys, streaming, answer,
):
    from ciel.providers.base import LLMStreamChunk

    router = MagicMock()
    router.send.return_value = LLMResponse(
        "[Playback was interrupted.]" + answer, "test-model", "nim",
    )
    router.stream.return_value = iter([
        LLMStreamChunk("[Playback was ", "test-model", "nim"),
        LLMStreamChunk("interrupted.]" + answer, "test-model", "nim"),
    ])
    tts = MagicMock()
    tts.supports_sentence_pipeline = False
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Ciel, tell me a joke.", {})], router=router,
        streaming=streaming, tts=tts,
    )
    try:
        assert all("[Playback was interrupted.]" not in call.args[0]
                   for call in tts.speak.call_args_list)
        assert "[Playback was interrupted.]" not in capsys.readouterr().out
        assert [turn.role for turn in turns] == (["user", "assistant"] if answer else ["user"])
        if answer:
            assert turns[-1].content == answer
    finally:
        store.close()


def test_live_partial_stream_failure_speaks_notice_without_repeating_answer(tmp_path, monkeypatch):
    from ciel.providers.base import LLMStreamChunk

    router, tts = MagicMock(), MagicMock()

    def generate(*args, **kwargs):
        yield LLMStreamChunk("First sentence. ", "stream-model", "nim")
        raise RuntimeError("connection lost")

    def speak(text, **kwargs):
        if "on_start" in kwargs:
            kwargs["on_start"]()
        return True

    router.stream.side_effect = generate
    tts.speak.side_effect = speak
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Ciel, explain streaming.", {})],
        router=router, tts=tts, streaming=True,
    )
    try:
        assert [call.args[0] for call in tts.speak.call_args_list] == [
            "First sentence.", "My connection cut out before I finished. Ask me to continue.",
        ]
        assert turns[-1].content.endswith("[Generation ended early because the connection failed.]")
        router.send.assert_not_called()
    finally:
        store.close()


def test_interruption_during_long_reply_can_be_judged_after_deadline(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('ciel.audio.ambient.time.monotonic', lambda: clock[0])
    ambient = AmbientConversation()
    ambient.record_addressed('assistant', 'Let me explain the next steps.')
    clock[0] = 130.0
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"addressed": true, "confidence": 0.95}', 'test', 'test'
    )
    assert ambient.decide('Wait, also include this.', router, [], during_reply=True).addressed
    router.send.assert_called_once()
    ambient.reset()
    router.reset_mock()
    assert not ambient.decide('Dinner is ready.', router, [], during_reply=True).addressed
    router.send.assert_not_called()


def test_superseded_request_and_added_words_reach_one_response(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.side_effect = [
        LLMResponse('{"addressed": true, "confidence": 0.95}', 'test', 'test'),
        LLMResponse('Here is the explanation, with headphones in mind.', 'test', 'test'),
    ]
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [('And include headphone support.', {})], router=router,
        observed=[('Ciel, explain always listening.', {'superseded': True})],
    )
    try:
        messages = router.send.call_args.args[0]
        users = [message.content for message in messages if message.role == 'user']
        assert users == ['explain always listening.\nAnd include headphone support.']
        assert 'same request' in messages[0].content
        tts.speak.assert_called_once()
        assert [turn.role for turn in turns] == ['user', 'user', 'assistant']
    finally:
        store.close()


def test_live_canceled_request_and_short_addition_are_merged(tmp_path, monkeypatch):
    canceled, replacement = Event(), Event()
    router = MagicMock()

    def respond(messages, **kwargs):
        assert kwargs.get('purpose', 'conversation') == 'conversation'
        if router.send.call_count == 1:
            canceled.set()  # Speech resumed while the first route was in flight.
            return LLMResponse('Discard this stale response.', 'test', 'test')
        return LLMResponse('Here is what I am good at.', 'test', 'test')

    router.send.side_effect = respond
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [
            ('Hey Ciel, what is it that you are special?', {'cancel_event': canceled}),
            ('And good.', {
                'cancel_event': replacement, 'supersedes_cancel_event': canceled,
                'response_pending': True, 'during_reply': True, 'stt_confidence': 0.54,
            }),
        ], router=router,
    )
    try:
        assert router.send.call_count == 2
        user_messages = [
            message.content for message in router.send.call_args.args[0] if message.role == 'user'
        ]
        assert user_messages == ['what is it that you are special?\nAnd good.']
        assert [turn.content for turn in turns if turn.role == 'user'] == [
            'what is it that you are special?', 'And good.',
        ]
        assert [turn.content for turn in turns if turn.role == 'assistant'] == [
            'Here is what I am good at.'
        ]
        tts.speak.assert_called_once_with(
            'Here is what I am good at.', block=True, cancel_event=replacement,
        )
    finally:
        store.close()


def test_linked_superseded_stt_request_is_merged_before_reply_gate(tmp_path, monkeypatch):
    canceled, replacement = Event(), Event()
    canceled.set()
    router = MagicMock()
    router.send.return_value = LLMResponse('Together, this is the full answer.', 'test', 'test')
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [('Also with headphones.', {
            'cancel_event': replacement, 'supersedes_cancel_event': canceled,
            'response_pending': True,
        })],
        router=router,
        observed=[('Ciel, explain always listening.', {
            'superseded': True, 'cancel_event': canceled,
        })],
    )
    try:
        router.send.assert_called_once()
        assert router.send.call_args.args[0][-1].content == (
            'explain always listening.\nAlso with headphones.'
        )
        assert [turn.role for turn in turns] == ['user', 'user', 'assistant']
        tts.speak.assert_called_once()
    finally:
        store.close()


def test_repeated_additions_keep_the_original_request(tmp_path, monkeypatch):
    first, second, third = Event(), Event(), Event()
    router = MagicMock()

    def respond(_messages, **kwargs):
        assert kwargs.get('purpose', 'conversation') == 'conversation'
        if router.send.call_count == 1:
            first.set()
        elif router.send.call_count == 2:
            second.set()
        return LLMResponse('Here is the combined answer.', 'test', 'test')

    router.send.side_effect = respond
    _router, tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch,
        [('Ciel, explain this in five steps.', {'cancel_event': first}),
         ('And include examples.', {'cancel_event': second, 'supersedes_cancel_event': first}),
         ('No, three steps, not five.', {
             'cancel_event': third, 'supersedes_cancel_event': second,
         })], router=router,
    )
    try:
        assert router.send.call_count == 3
        users = [
            message.content for message in router.send.call_args.args[0] if message.role == 'user'
        ]
        assert users == [
            'explain this in five steps.\nAnd include examples.\nNo, three steps, not five.'
        ]
        tts.speak.assert_called_once()
    finally:
        store.close()


def test_unrelated_recording_cannot_inherit_canceled_request(tmp_path, monkeypatch):
    canceled, other_token = Event(), Event()
    router = MagicMock()

    def respond(_messages, **kwargs):
        if kwargs.get('purpose') == 'speech_gate':
            return LLMResponse('{"addressed": false, "confidence": 0.95}', 'test', 'test')
        canceled.set()
        return LLMResponse('Discard this.', 'test', 'test')

    router.send.side_effect = respond
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [('Ciel, explain this.', {'cancel_event': canceled}), ('And good.', {
            'supersedes_cancel_event': other_token, 'response_pending': True,
        })], router=router,
    )
    try:
        assert router.send.call_args.kwargs['purpose'] == 'speech_gate'
        assert [turn.role for turn in turns] == ['user']
        tts.speak.assert_not_called()
    finally:
        store.close()


def test_merge_permission_does_not_override_other_listener_or_save_facts(tmp_path, monkeypatch):
    canceled = Event()
    canceled.set()
    router = MagicMock()
    router.send.return_value = LLMResponse('I can take that into account.', 'test', 'test')
    _router, tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch,
        [('Actually, Mom, pass me that.', {'supersedes_cancel_event': canceled}),
         ('Actually my favorite game is CS2.', {'supersedes_cancel_event': canceled})],
        router=router, observed=[('Ciel, explain this.', {
            'superseded': True, 'cancel_event': canceled,
        })],
    )
    try:
        assert store.get_fact('user:favorite_game') is None
        tts.speak.assert_called_once()
        router.send.assert_called_once()
    finally:
        store.close()


def test_continue_receives_interrupted_playback_context(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.return_value = LLMResponse('And the second step is to test it.', 'test', 'test')
    context = {
        'full_text': 'First, configure it. Second, test it.',
        'estimated_spoken_text': 'First, configure it.',
        'remaining_text': 'Second, test it.',
        'played_seconds': 2.0, 'duration_seconds': 4.0,
    }
    _router, tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch, [('Ciel, continue.', {})], router=router, interrupted=context,
    )
    try:
        system = router.send.call_args.args[0][0].content
        assert 'previous reply was interrupted' in system
        assert 'Second, test it.' in system
        assert 'word boundaries are estimated' in system
        tts.speak.assert_called_once()
    finally:
        store.close()


def test_overheard_facts_are_not_saved_or_sent_to_provider(tmp_path, monkeypatch):
    router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("My favorite game is CS2.", {"ambient": True})]
    )
    try:
        assert not turns
        assert store.get_fact("user:favorite_game") is None
        router.send.assert_not_called()
        tts.speak.assert_not_called()
    finally:
        store.close()


def test_voice_mode_controls_and_background_silence(tmp_path, monkeypatch):
    router, tts, loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [
            ("Ciel, listen continuously.", {}),
            ("Mom, pass me that.", {"ambient": True}),
            ("Ciel, stop listening.", {"ambient": True}),
        ],
        ambient=False,
    )
    try:
        assert [call.args[0] for call in loop.set_ambient.call_args_list] == [True, False]
        assert tts.speak.call_count == 2
        router.send.assert_not_called()
        assert not turns
    finally:
        store.close()


def test_uncertain_recognition_does_not_auto_save_a_fact(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.return_value = LLMResponse("Did you say CS2?", "test", "test")
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("My favorite game is CS2.", {"stt_confidence": 0.45})],
        ambient=False, router=router,
    )
    try:
        assert store.get_fact("user:favorite_game") is None
        assert turns[0].content == "My favorite game is CS2."
        prompt = router.send.call_args.args[0][0].content
        assert "transcribed from speech" in prompt
        assert "contradictory, nonsensical" in prompt
        assert "ask one brief, specific question" in prompt
        assert "guessed correction as a confirmed fact" in prompt
    finally:
        store.close()


def test_resumed_speech_suppresses_old_answer_and_assistant_history(tmp_path, monkeypatch):
    cancelled = Event()
    router = MagicMock()

    def interrupted(*_args, **_kwargs):
        cancelled.set()
        return LLMResponse("An old answer", "test", "test")

    router.send.side_effect = interrupted
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Explain that idea.", {"cancel_event": cancelled})],
        ambient=False, router=router,
    )
    try:
        tts.speak.assert_not_called()
        assert len(turns) == 1
        assert turns[0].role == "user"
    finally:
        store.close()


def test_ambient_local_greeting_opens_followup_with_actual_context(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.side_effect = [
        LLMResponse('{"addressed": true, "confidence": 0.95}', "test", "test"),
        LLMResponse("I can help you build things.", "test", "test"),
    ]
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Ciel.", {}), ("What are you good at?", {"ambient": True})],
        router=router,
    )
    try:
        assert tts.speak.call_count == 2
        gate_request = router.send.call_args_list[0]
        payload = json.loads(gate_request.args[0][1].content)
        assert payload["recent_dialogue"][0]["content"] == "Ciel."
        assert payload["recent_dialogue"][1]["role"] == "assistant"
        assert [turn.role for turn in turns] == ["user", "assistant"]
    finally:
        store.close()


def test_explicit_persona_request_updates_file_without_calling_provider(tmp_path, monkeypatch):
    persona_file = tmp_path / "persona.txt"
    persona_file.write_text("Keep a warm tone.\n", encoding="utf-8")
    router = MagicMock()
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Ciel, be more playful.", {})],
        router=router, persona_file=persona_file,
    )
    try:
        contents = persona_file.read_text(encoding="utf-8")
        assert "Keep a warm tone." in contents
        assert "more playful" in contents
        assert turns[-1].provider == "local" and turns[-1].model == "persona"
        router.send.assert_not_called()
        tts.speak.assert_called_once_with(
            "Got it. I'll use that style from now on.", block=True,
        )
    finally:
        store.close()


def test_persona_change_question_collects_and_saves_next_followup(tmp_path, monkeypatch):
    persona_file = tmp_path / "persona.txt"
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"listener":"assistant","confidence":0.98,"interpretation":""}',
        "test", "test",
    )
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [
            ("Ciel, can you update your persona?", {}),
            ("Curious, and ask more questions when you don't know something.", {}),
        ],
        router=router,
        persona_file=persona_file,
    )
    try:
        assert "ask more questions" in persona_file.read_text(encoding="utf-8")
        assert router.send.call_count == 1  # The follow-up is gated; persona saving is local.
        assert [turn.model for turn in turns if turn.role == "assistant"] == [
            "persona", "persona",
        ]
        assert tts.speak.call_args_list[-1].args[0] == "Got it. I'll use that style from now on."
    finally:
        store.close()


def test_inferred_ambient_followup_never_auto_saves_personal_facts(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.side_effect = [
        LLMResponse('{"addressed": true, "confidence": 0.99}', "test", "test"),
        LLMResponse("You can tell me your preference directly.", "test", "test"),
    ]
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Ciel.", {}), ("My favorite game is CS2.", {"ambient": True})],
        router=router,
    )
    try:
        assert store.get_fact("user:favorite_game") is None
        assert turns[0].content == "My favorite game is CS2."
    finally:
        store.close()


def test_superseded_direct_address_opens_temporary_followup_without_saving(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.side_effect = [
        LLMResponse('{"addressed": true, "confidence": 0.99}', "test", "test"),
        LLMResponse("I'm listening.", "test", "test"),
    ]
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("What are you good at?", {"ambient": True})],
        router=router, observed=[("Hey Ciel.", {"superseded": True})],
    )
    try:
        gate = json.loads(router.send.call_args_list[0].args[0][1].content)
        assert gate["recent_dialogue"][0]["content"] == "Hey Ciel."
        assert len(turns) == 2
        assert turns[0].content == "What are you good at?"
    finally:
        store.close()


def test_unclear_direct_speech_gets_repeat_without_saving(tmp_path, monkeypatch):
    router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("", {"stt_raw_text": "Hey Ciel.", "stt_needs_repeat": True})],
    )
    try:
        router.send.assert_not_called()
        assert not turns
        tts.speak.assert_called_once_with(
            "I didn't catch that clearly. Could you say it again?", block=True
        )
    finally:
        store.close()


def test_verified_wake_with_unclear_command_asks_repeat_without_guessing(tmp_path, monkeypatch):
    router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("", {"wake_verified": True, "stt_raw_text": "file", "stt_needs_repeat": True})],
    )
    try:
        router.send.assert_not_called()
        assert not turns
        tts.speak.assert_called_once_with(
            "I didn't catch that clearly. Could you say it again?", block=True
        )
    finally:
        store.close()


def test_verified_wake_allows_local_time_without_name_in_stt(tmp_path, monkeypatch):
    router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("What's the time?", {"wake_verified": True})],
    )
    try:
        router.send.assert_not_called()
        tts.speak.assert_called_once()
        assert ":" in tts.speak.call_args.args[0]
        assert [turn.role for turn in turns] == ["user", "assistant"]
        assert turns[0].content == "What's the time?"
    finally:
        store.close()


@pytest.mark.parametrize("text, query", [
    ("So what's good Ciel?", "So what's good Ciel?"),
    ("So Ciel, what's on your mind?", "what's on your mind?"),
])
def test_live_log_addresses_reply_without_speech_gate(tmp_path, monkeypatch, text, query):
    router = MagicMock()
    router.send.return_value = LLMResponse("Hey hexarion! How's your day going?", "test", "test")
    router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [(text, {"stt_confidence": 0.66})], router=router,
    )
    try:
        assert router.send.call_count == 1
        assert router.send.call_args.kwargs.get("purpose", "conversation") == "conversation"
        assert router.send.call_args.args[0][-1].content == query
        tts.speak.assert_called_once()
        assert turns[0].content == query
    finally:
        store.close()


@pytest.mark.parametrize("ambient", [True, False])
def test_empty_interruption_restores_superseded_question(tmp_path, monkeypatch, ambient):
    canceled, replacement = Event(), Event()
    canceled.set()
    router = MagicMock()
    router.send.return_value = LLMResponse('Here is what I can do.', 'test', 'test')
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [('', {'cancel_event': replacement, 'supersedes_cancel_event': canceled})],
        ambient=ambient, router=router,
        observed=[('Ciel, what are you good at?', {
            'superseded': True, 'cancel_event': canceled,
        })],
    )
    try:
        router.send.assert_called_once()
        assert router.send.call_args.args[0][-1].content == 'what are you good at?'
        assert [(turn.role, turn.content) for turn in turns] == [
            ('user', 'what are you good at?'), ('assistant', 'Here is what I can do.'),
        ]
        tts.speak.assert_called_once()
    finally:
        store.close()


def test_empty_interruption_retries_without_duplicate_history(tmp_path, monkeypatch):
    canceled, replacement = Event(), Event()
    router = MagicMock()

    def respond(messages, **kwargs):
        if router.send.call_count == 1:
            canceled.set()
            return LLMResponse('Discard me.', 'test', 'test')
        return LLMResponse('The answer.', 'test', 'test')

    router.send.side_effect = respond
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [('Ciel, explain this.', {'cancel_event': canceled}),
         ('', {'cancel_event': replacement, 'supersedes_cancel_event': canceled})],
        router=router,
    )
    try:
        assert router.send.call_count == 2
        assert [m.content for m in router.send.call_args.args[0] if m.role == 'user'] == [
            'explain this.',
        ]
        assert [(turn.role, turn.content) for turn in turns] == [
            ('user', 'explain this.'), ('assistant', 'The answer.'),
        ]
        tts.speak.assert_called_once()
    finally:
        store.close()


@pytest.mark.parametrize('case', ['unlinked', 'expired', 'unclear'])
def test_empty_audio_does_not_restore_unrelated_or_unclear_requests(tmp_path, monkeypatch, case):
    canceled = Event()
    canceled.set()
    info = {'supersedes_cancel_event': canceled}
    clock = None
    if case == 'unlinked':
        info['supersedes_cancel_event'] = Event()
    elif case == 'unclear':
        info.update(stt_raw_text='possibly stop', stt_needs_repeat=True)
    else:
        # observe_transcript records at time zero; restoration occurs after expiry.
        times = iter([0.0, 1000.0])
        def clock():
            return next(times)
    router = MagicMock()
    router.send.return_value = LLMResponse('{"addressed": false}', 'test', 'test')
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [('', info)], router=router, clock=clock,
        observed=[('Ciel, explain this.', {'superseded': True, 'cancel_event': canceled})],
    )
    try:
        assert all(
            call.kwargs.get('purpose') == 'speech_gate' for call in router.send.call_args_list
        )
        assert turns == []
    finally:
        store.close()


@pytest.mark.parametrize('ambient', [True, False])
def test_empty_recovery_cannot_execute_a_memory_write(tmp_path, monkeypatch, ambient):
    canceled = Event()
    canceled.set()
    handler = MagicMock(return_value=None)
    monkeypatch.setattr('ciel.memory.service.MemoryService.handle', handler)
    router = MagicMock()
    router.send.return_value = LLMResponse('I heard your request.', 'test', 'test')
    _router, _tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch, [('', {'supersedes_cancel_event': canceled})],
        ambient=ambient, router=router,
        observed=[('Ciel, remember my name is Alex.', {
            'superseded': True, 'cancel_event': canceled,
        })],
    )
    try:
        handler.assert_not_called()
        assert store.get_fact('user:preferred_name') is None
    finally:
        store.close()


@pytest.mark.parametrize('text', [
    'Why are you talking so fast right now?', 'You are speaking too fast.',
    "You're talking too fast!", 'Could you speak more slowly please?',
    'Please speak a little slower.', 'Your voice sounds very robotic.',
    'Can you talk louder?',
])
def test_recent_speech_feedback_bypasses_cloud_judgment(text):
    ambient = AmbientConversation()
    ambient.record_addressed('assistant', 'Here is what I can do.')
    router = MagicMock()
    decision = ambient.decide(text, router, [])
    assert decision.addressed and not decision.explicit
    assert decision.reason == 'speech_feedback'
    router.send.assert_not_called()


@pytest.mark.parametrize('text', [
    'Mom, why are you talking so fast right now?',
    'Why are you talking so fast right now, Dad?',
    'She asked why are you talking so fast right now.',
    'You are talking to my brother.',
])
def test_speech_feedback_does_not_bypass_other_listener_or_quoted_speech(text):
    ambient = AmbientConversation()
    ambient.record_addressed('assistant', 'Here is what I can do.')
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"addressed": false, "confidence": 0.99}', 'test', 'test',
    )
    assert not ambient.decide(text, router, []).addressed


def test_feedback_requires_recent_assistant_speech(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('ciel.audio.ambient.time.monotonic', lambda: clock[0])
    ambient = AmbientConversation()
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"addressed": false, "confidence": 0.99}', 'test', 'test',
    )
    feedback = 'Why are you talking so fast right now?'
    assert not ambient.decide(feedback, router, []).addressed
    ambient.record_addressed('assistant', 'Hello.')
    clock[0] += 21
    assert not ambient.decide(feedback, router, []).addressed
    ambient.record_addressed('user', 'Ciel?')
    assert not ambient.decide(feedback, router, []).addressed


def test_uncertain_fragment_keeps_original_followup_deadline(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('ciel.audio.ambient.time.monotonic', lambda: clock[0])
    ambient = AmbientConversation(followup_policy='strict')
    ambient.record_addressed('assistant', 'Here is what I can do.')
    original_deadline = ambient.deadline
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"addressed": true, "confidence": 0.7}', 'test', 'test',
    )
    clock[0] += 5
    assert not ambient.decide('Something unclear.', router, []).addressed
    assert ambient.deadline == original_deadline
    assert ambient.decide('Why are you talking so fast right now?', router, []).addressed
    clock[0] = original_deadline + 1
    assert not ambient.decide('Why are you talking so fast right now?', router, []).addressed


def test_live_voice_feedback_after_reply_reaches_conversation(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.return_value = LLMResponse('Here is what I can do.', 'test', 'test')
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [('Ciel, what are you good at?', {}),
         ('Why are you talking so fast right now?', {})], router=router,
    )
    try:
        assert router.send.call_count == 2
        assert all('purpose' not in call.kwargs for call in router.send.call_args_list)
        assert tts.speak.call_count == 2
        assert [turn.content for turn in turns if turn.role == 'user'] == [
            'what are you good at?', 'Why are you talking so fast right now?',
        ]
    finally:
        store.close()


@pytest.mark.parametrize('original_question', [
    'What time is Ciel?', "What's the time Ciel?", 'What time is it Siel?',
])
def test_live_clock_recovery_then_robot_feedback_get_two_replies(
    tmp_path, monkeypatch, original_question,
):
    router = MagicMock()
    router.send.return_value = LLMResponse('I hear you. The voice needs work.', 'test', 'test')
    original_feedback = 'You sound like a robot, do you know?'
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [(original_question, {'stt_confidence': 0.70}),
         (original_feedback, {'stt_confidence': 0.77})], router=router,
    )
    try:
        router.send.assert_called_once()  # Clock is local; feedback needs one answer, no gate.
        assert 'purpose' not in router.send.call_args.kwargs
        assert tts.speak.call_count == 2
        assert [(turn.role, turn.content) for turn in turns if turn.role == 'user'] == [
            ('user', original_question), ('user', original_feedback),
        ]
        assistant_turns = [turn for turn in turns if turn.role == 'assistant']
        assert assistant_turns[0].provider == 'local' and assistant_turns[0].model == 'clock'
        assert assistant_turns[0].content.startswith("It's ")
        assert assistant_turns[1].content == 'I hear you. The voice needs work.'
        assert router.send.call_args.args[0][-1].content == original_feedback
    finally:
        store.close()


@pytest.mark.parametrize('followup', [
    'What do you want to talk about?',
    'I think we should work on your memory next.',
    'Can you suggest something fun?',
])
def test_live_active_conversation_accepts_uncertain_followup(tmp_path, monkeypatch, followup):
    router = MagicMock()
    router.send.side_effect = [
        LLMResponse("What's something you'd like to work on today?", 'test', 'test'),
        LLMResponse('{"addressed": false, "confidence": 0.6}', 'test', 'test'),
        LLMResponse('We could improve the memory together.', 'test', 'test'),
    ]
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [('Ciel, what are you good at?', {}), (followup, {})], router=router,
    )
    try:
        assert router.send.call_count == 3
        assert router.send.call_args_list[1].kwargs['purpose'] == 'speech_gate'
        assert router.send.call_args.args[0][-1].content == followup
        assert [turn.content for turn in turns if turn.role == 'user'] == [
            'what are you good at?', followup,
        ]
        assert tts.speak.call_count == 2
    finally:
        store.close()


@pytest.mark.parametrize('answer, saved', [('Ciel, yes.', True), ('Ciel, no.', False)])
def test_memory_proposal_confirmation_in_real_voice_handler(tmp_path, monkeypatch, answer, saved):
    router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [('Ciel, my favorite game is CS2.', {}), (answer, {})],
    )
    try:
        assert (store.get_fact('user:favorite_game') is not None) is saved
        assert 'Should I remember' in tts.speak.call_args_list[0].args[0]
        assert 'hey ciel, yes' in tts.speak.call_args_list[0].args[0]
        assert len(turns) == 4
        router.send.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize('interruption', [
    ('Ciel, cancel.', {}), ('Ciel, what time is it?', {}),
    ('Ciel, yes.', {'stt_confidence': 0.45}),
])
def test_memory_proposal_does_not_survive_cancel_topic_change_or_uncertainty(
    tmp_path, monkeypatch, interruption,
):
    router = MagicMock()
    router.send.return_value = LLMResponse('Okay.', 'test', 'test')
    _router, _tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch,
        [('Ciel, my favorite game is CS2.', {}), interruption, ('Ciel, yes.', {})],
        router=router,
    )
    try:
        assert store.get_fact('user:favorite_game') is None
    finally:
        store.close()


def test_local_action_voice_handler_uses_telemetry_without_provider(tmp_path, monkeypatch):
    from ciel.platform.system_info import SystemSnapshot

    monkeypatch.setattr('ciel.actions.system_info.get_system_snapshot',
                        lambda **kwargs: SystemSnapshot(ram_used_gb=8, ram_total_gb=16))
    router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [('Ciel, how much RAM am I using?', {})],
    )
    try:
        router.send.assert_not_called()
        assert '8 of 16 GB' in tts.speak.call_args.args[0]
        assert [turn.role for turn in turns] == ['user', 'assistant']
        assert turns[-1].model == 'action'
    finally:
        store.close()


@pytest.mark.parametrize('direct', [True, False])
def test_application_launch_requires_direct_address_in_voice_handler(tmp_path, monkeypatch, direct):
    launch = MagicMock()
    monkeypatch.setattr('ciel.actions.open_app.subprocess.Popen', launch)
    monkeypatch.setattr('ciel.actions.open_app.sys.platform', 'linux')
    monkeypatch.setattr('ciel.actions.open_app.shutil.which', lambda name: '/usr/bin/discord')
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"listener": "assistant", "confidence": 0.99}', 'test', 'test',
    )
    _router, tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch,
        [('Ciel.', {}), ('Ciel, open Discord.' if direct else 'Open Discord.', {})],
        router=router,
    )
    try:
        assert launch.call_count == int(direct)
        assert ('sent the launch request' if direct else 'address Ciel directly') in (
            tts.speak.call_args.args[0]
        )
    finally:
        store.close()


@pytest.mark.parametrize('ambient', [True, False])
def test_natural_discord_request_uses_local_launcher(tmp_path, monkeypatch, ambient):
    launch = MagicMock()
    monkeypatch.setattr('ciel.actions.open_app.subprocess',
                        SimpleNamespace(Popen=launch, DEVNULL=-3))
    monkeypatch.setattr('ciel.actions.open_app.sys.platform', 'linux')
    monkeypatch.setattr('ciel.actions.open_app.shutil.which',
                        lambda name: '/usr/bin/discord' if name == 'discord' else None)
    router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [('Ciel, can you open Discord for me?', {})],
        ambient=ambient,
    )
    try:
        launch.assert_called_once()
        router.send.assert_not_called()
        assert 'sent the launch request' in tts.speak.call_args.args[0]
        assert turns[-1].model == 'action'
    finally:
        store.close()


@pytest.mark.parametrize('interrupted', [False, True])
def test_corrected_spelling_then_open_it_reaches_launcher(tmp_path, monkeypatch, interrupted):
    launch = MagicMock()
    monkeypatch.setattr('ciel.actions.open_app.subprocess',
                        SimpleNamespace(Popen=launch, DEVNULL=-3))
    monkeypatch.setattr('ciel.actions.open_app.sys.platform', 'linux')
    monkeypatch.setattr('ciel.actions.open_app.shutil.which',
                        lambda name: '/usr/bin/discord' if name == 'discord' else None)
    router = MagicMock()
    router.send.return_value = LLMResponse('You mean Discord.', 'test', 'test')
    canceled = Event()
    canceled.set()
    correction = 'No, no, no, I meant D I S C O R D'
    utterances = [
        (correction if interrupted else 'Ciel, ' + correction,
         {'supersedes_cancel_event': canceled} if interrupted else {}),
        ('Ciel, can you open it for me?', {}),
    ]
    observed = [('Ciel, can you open this quarter for me?', {
        'superseded': True, 'cancel_event': canceled,
    })] if interrupted else None
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, utterances, router=router, observed=observed,
    )
    try:
        launch.assert_called_once()
        assert launch.call_args.args[0] == ['/usr/bin/discord']
        assert router.send.call_count == 1  # Only the correction goes to the model.
        assert 'sent the launch request' in tts.speak.call_args.args[0]
        assert turns[-1].model == 'action'
    finally:
        store.close()


def test_uncertain_app_name_cannot_supply_later_launch_context(tmp_path, monkeypatch):
    launch = MagicMock()
    monkeypatch.setattr('ciel.actions.open_app.subprocess',
                        SimpleNamespace(Popen=launch, DEVNULL=-3))
    router = MagicMock()
    router.send.return_value = LLMResponse('Which app did you mean?', 'test', 'test')
    _router, tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch,
        [('Ciel, I meant Discord', {'stt_confidence': 0.45}),
         ('Ciel, can you open it for me?', {})], router=router,
    )
    try:
        launch.assert_not_called()
        assert 'Which application' in tts.speak.call_args.args[0]
    finally:
        store.close()
