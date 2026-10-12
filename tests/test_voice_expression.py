"""Expression cues reach native speech without leaking into conversation state."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from raphael.audio.speech_events import (
    SpeechEventLimiter,
    speech_event_instruction,
    split_speech_event,
    strip_speech_events,
    voice_delivery_instruction,
)
from raphael.audio.streaming import stream_reply
from raphael.audio.tts import TextToSpeech
from raphael.config import AudioConfig, Settings
from raphael.providers.base import LLMResponse, LLMStreamChunk


@pytest.mark.parametrize("mode", ["expressive", "natural", "off"])
def test_expression_setting_is_forwarded_and_validated(monkeypatch, mode):
    monkeypatch.setenv("TTS_EXPRESSIVENESS", mode)
    assert Settings(_env_file=None).audio.tts_expressiveness == mode
    with pytest.raises(ValidationError):
        AudioConfig(tts_expressiveness="extreme")


def test_expressive_prompt_gives_context_and_off_disables_events():
    prompt = speech_event_instruction(enabled=True, expressiveness="expressive")
    assert "relief" in prompt and "amusement" in prompt
    assert "two" in prompt and "serious" in prompt
    assert speech_event_instruction(enabled=True, expressiveness="off") == ""
    assert voice_delivery_instruction(enabled=False) == ""


def test_unknown_leading_tag_does_not_discard_a_valid_reaction():
    assert split_speech_event("[whisper] [chuckle] Good one.") == ("[chuckle]", "Good one.")


@pytest.mark.parametrize("direction", [
    "[soft inhale — steady, focused]",
    "[brief pause — quiet, not dismissive, just present]",
    "[speaking softly]",
    "[smiles warmly]",
])
def test_stage_directions_are_removed_from_speech_and_history(direction):
    text = f"{direction} Alright. {direction} I can check."
    assert TextToSpeech.clean_text_for_speech(text) == "Alright. I can check."
    assert " ".join(strip_speech_events(text).split()) == "Alright. I can check."
    assert split_speech_event(f"{direction} [chuckle] Good one.") == (
        "[chuckle]", "Good one.",
    )


@pytest.mark.parametrize("engine", ["piper", "chatterbox_turbo"])
def test_stage_directions_never_reach_either_synthesizer(monkeypatch, engine):
    tts = TextToSpeech(engine=engine, enabled=False)
    tts.enabled = True
    synthesizer = MagicMock(return_value=None)
    target = "_synthesize_chatterbox" if engine == "chatterbox_turbo" else "_synthesize_piper"
    monkeypatch.setattr(tts, target, synthesizer)
    tts.synthesize("[soft inhale — steady, focused] Alright.")
    synthesizer.assert_called_once_with("Alright.")
    synthesizer.reset_mock()
    tts.synthesize("[brief pause — quiet, not dismissive, just present]")
    synthesizer.assert_not_called()


def test_bracketed_data_and_link_labels_survive_speech_cleaning():
    text = "Use [array] and [1, 2, 3]. Read [pause](https://example.com)."
    assert TextToSpeech.clean_text_for_speech(text) == (
        "Use [array] and [1, 2, 3]. Read pause."
    )
    assert strip_speech_events("Use [array] and [1, 2, 3].") == (
        "Use [array] and [1, 2, 3]."
    )
    assert split_speech_event("[array] is a placeholder.") == (
        "", "[array] is a placeholder.",
    )


@pytest.mark.parametrize("mode,count", [("expressive", 2), ("natural", 1), ("off", 0)])
def test_direct_synthesis_budget_collapses_stacks_and_preserves_words(mode, count):
    text = "[chuckle] [laugh] Good one. [sigh] Okay. [moan] Again."
    cleaned = SpeechEventLimiter(mode).apply(text)
    assert cleaned.count("[") == count
    assert "[laugh]" not in cleaned and "[moan]" not in cleaned
    assert " ".join(strip_speech_events(cleaned).split()) == "Good one. Okay. Again."


def test_tts_uses_configured_expression_and_accepts_an_override(monkeypatch):
    monkeypatch.setenv("TTS_EXPRESSIVENESS", "natural")
    monkeypatch.setattr("raphael.audio.tts.get_settings", lambda: Settings(_env_file=None))
    assert TextToSpeech(enabled=False).expressiveness == "natural"
    assert TextToSpeech(enabled=False, expressiveness="off").expressiveness == "off"
    with pytest.raises(ValueError):
        TextToSpeech(enabled=False, expressiveness="extreme")


@pytest.mark.parametrize("cue,native", [("[moan]", "[groan]"), ("[GIGGLE]", "[chuckle]")])
def test_aliases_and_inline_events_reach_native_worker(monkeypatch, cue, native):
    tts = TextToSpeech(engine="chatterbox_turbo", enabled=True)
    worker = MagicMock()
    worker.synthesize.return_value = (np.zeros(10, dtype=np.float32), 24000)
    monkeypatch.setattr(tts, "_load_chatterbox_worker", lambda: worker)
    tts.synthesize(f"Oh! {cue} That's funny.")
    assert worker.synthesize.call_args.args[0] == f"Oh! {native} That's funny."
    assert strip_speech_events(f"{cue} Hello [array]") == "Hello [array]"


def test_off_removes_all_events_and_standalone_event_is_supported(monkeypatch):
    tts = TextToSpeech(engine="chatterbox_turbo", enabled=True, expressiveness="off")
    worker = MagicMock()
    monkeypatch.setattr(tts, "_load_chatterbox_worker", lambda: worker)
    tts.synthesize("[sigh] All right. [moan]")
    assert worker.synthesize.call_args.args[0].strip() == "All right."
    tts.expressiveness = "expressive"
    tts.synthesize("[sigh]")
    assert worker.synthesize.call_args.args[0] == "[sigh]"


def test_piper_fallback_never_speaks_inline_aliases(monkeypatch):
    tts = TextToSpeech(engine="chatterbox_turbo", enabled=True)
    tts._chatterbox_failed = True
    fallback = MagicMock()
    monkeypatch.setattr(tts, "_synthesize_piper_fallback", fallback)
    tts.synthesize("Oh! [moan] Again? [chuckle]")
    assert "[" not in fallback.call_args.args[0]


@pytest.mark.parametrize("mode,count", [("expressive", 2), ("natural", 1), ("off", 0)])
@pytest.mark.parametrize("pipeline", [False, True])
def test_stream_limits_events_per_reply_across_chunks(mode, count, pipeline):
    played = []
    prepared = []

    def speak(text, **controls):
        played.append(text)
        controls["on_start"]()
        return True

    tts = SimpleNamespace(
        expressiveness=mode, engine="chatterbox_turbo", audio_queue_size=2,
        supports_sentence_pipeline=pipeline,
        begin_stream=lambda: 1, update_stream=lambda *_args: None,
        end_stream=lambda *_args: None, stop=lambda: None, speak=speak,
    )

    def prepare(text, *_args):
        prepared.append(text)
        return (np.zeros(10, dtype=np.float32), 24000, None)

    tts.prepare_sentence = prepare

    def generate(*_args, **_kwargs):
        for delta in ["[ch", "uckle] Hi. ", "[sigh] Okay. ", "[moan] Again."]:
            yield LLMStreamChunk(delta, "fake", "fake")

    router = SimpleNamespace(stream=generate)
    for _ in range(2):  # Budget resets on the next reply, including prefetch.
        played.clear()
        prepared.clear()
        result = stream_reply(router, [], tts, lambda: True)
        assert sum(text.count("[") for text in played) == count
        assert "[" not in result.response.content
        assert "Hi." in result.response.content and "Again." in result.response.content
        if pipeline:
            assert prepared == played


def test_batch_history_is_plain_but_voice_retains_cues(tmp_path, monkeypatch):
    from tests.test_ambient import run_callbacks

    router, tts = MagicMock(), MagicMock()
    tts.engine = "chatterbox_turbo"
    tts.enabled = True
    router.send.return_value = LLMResponse("[chuckle] You got me.", "fake", "fake")
    _, _, _, memory, turns = run_callbacks(
        tmp_path, monkeypatch, [("Tell me a joke", {"input_source": "keyboard"})],
        ambient=False, router=router, tts=tts, persona_file=tmp_path / "persona.txt",
    )
    try:
        prompt = router.send.call_args.args[0][0].content
        assert "Voice delivery:" in prompt and "relief" in prompt
        assert [t.content for t in turns if t.role == "assistant"] == ["You got me."]
        assert any(call.args[0] == "[chuckle] You got me." for call in tts.speak.call_args_list)
    finally:
        memory.close()


def test_batch_stage_directions_do_not_reach_voice_or_saved_history(tmp_path, monkeypatch):
    from tests.test_ambient import run_callbacks

    router, tts = MagicMock(), MagicMock()
    tts.engine = "chatterbox_turbo"
    tts.enabled = True
    router.send.return_value = LLMResponse(
        "[soft inhale — steady, focused] Alright. "
        "[brief pause — quiet, not dismissive, just present] I can check.", "fake", "fake",
    )
    _, _, _, memory, turns = run_callbacks(
        tmp_path, monkeypatch, [("Explain that", {"input_source": "keyboard"})],
        ambient=False, router=router, tts=tts, persona_file=tmp_path / "persona.txt",
    )
    try:
        assert [t.content for t in turns if t.role == "assistant"] == ["Alright. I can check."]
        tts.speak.assert_called_once_with("Alright. I can check.", block=True)
    finally:
        memory.close()
