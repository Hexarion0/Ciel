"""Reply captions must never hide a completed text answer when voice playback fails."""

from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.audio.tts import TextToSpeech
from ciel.providers.base import LLMResponse, LLMStreamChunk
from tests.test_ambient import run_callbacks


@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("tts_enabled", [False, True])
def test_typed_streamed_reply_stays_visible_once_when_audio_is_unavailable(
    tmp_path, monkeypatch, capsys, controller, tts_enabled,
):
    monkeypatch.setenv("INTERNET_ENABLED", "false")
    answer = "A complete answer remains readable without audio."
    router, tts = MagicMock(), MagicMock()
    router.send.return_value = LLMResponse(
        '{"action":"answer","confidence":0.99}', "synthetic", "nim",
    )
    router.stream.return_value = iter([LLMStreamChunk(answer, "synthetic", "nim")])
    tts.engine = "piper"
    tts.enabled = tts_enabled
    tts.supports_sentence_pipeline = False
    tts.speak.return_value = False
    tts.clean_text_for_speech.side_effect = TextToSpeech.clean_text_for_speech

    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Explain this.", {"input_source": "keyboard"})],
        router=router, tts=tts, controller=controller, streaming=True,
        ai_transcripts=True, tts_enabled=tts_enabled,
    )
    try:
        assert capsys.readouterr().out.count(answer) == 1
        assert [turn.content for turn in turns if turn.role == "assistant"] == [answer]
        router.stream.assert_called_once()
        if tts_enabled:
            tts.speak.assert_called_once()
            assert tts.speak.call_args.args[0] == answer
        else:
            tts.speak.assert_not_called()
        tts.begin_stream.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_canceled_stream_does_not_publish_stale_text_fallback(
    tmp_path, monkeypatch, capsys, controller,
):
    monkeypatch.setenv("INTERNET_ENABLED", "false")
    canceled = Event()
    answer = "This canceled answer must not become visible."
    router, tts = MagicMock(), MagicMock()
    router.send.return_value = LLMResponse(
        '{"action":"answer","confidence":0.99}', "synthetic", "nim",
    )

    def generate(*_args, **_kwargs):
        canceled.set()
        yield LLMStreamChunk(answer, "synthetic", "nim")

    router.stream.side_effect = generate
    tts.engine = "piper"
    tts.enabled = True
    tts.supports_sentence_pipeline = False
    tts.speak.return_value = False
    tts.clean_text_for_speech.side_effect = TextToSpeech.clean_text_for_speech
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Explain this.", {"input_source": "keyboard", "cancel_event": canceled})],
        router=router, tts=tts, controller=controller, streaming=True, ai_transcripts=True,
    )
    try:
        assert answer not in capsys.readouterr().out
        assert not [turn for turn in turns if turn.role == "assistant"]
        tts.speak.assert_not_called()
        tts.begin_stream.assert_not_called()
    finally:
        store.close()
