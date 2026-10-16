"""Faint native cues use separate style conditioning without changing normal speech."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest


def test_cues_are_rendered_separately_and_speech_is_not_amplified():
    from ciel.audio.vocal_cues import VocalCueRenderer

    calls = []
    words = np.full(1000, 0.1, dtype=np.float32)
    cue = np.full(1000, 0.01, dtype=np.float32)

    def generate(text):
        calls.append(("speech", text))
        return words

    def generate_event(text):
        calls.append(("event", text))
        return cue

    renderer = VocalCueRenderer(generate, generate_event, 24000)
    result = renderer.render("[sigh] That's a relief.")
    assert calls == [("event", "[sigh]"), ("speech", "That's a relief.")]
    assert np.sqrt(np.mean(result[:1000] ** 2)) == pytest.approx(0.025)
    np.testing.assert_array_equal(result[-1000:], words)


def test_inline_positions_and_cached_events_are_preserved():
    from ciel.audio.vocal_cues import VocalCueRenderer

    speech, events = [], []

    def generate(text):
        speech.append(text)
        return np.ones(100, dtype=np.float32) * 0.1

    def event(text):
        events.append(text)
        return np.ones(100, dtype=np.float32) * 0.01

    renderer = VocalCueRenderer(generate, event, 24000)
    renderer.render("Oh! [groan] Not again.")
    renderer.render("[groan] Another try.")
    assert speech == ["Oh!", "Not again.", "Another try."]
    assert events == ["[groan]"]


@pytest.mark.parametrize("text", ["Hello.", "[chuckle] Good one."])
def test_other_speech_keeps_the_original_single_generate_path(text):
    from ciel.audio.vocal_cues import VocalCueRenderer

    expected = np.zeros(100, dtype=np.float32)
    calls = []

    def generate(value):
        calls.append(value)
        return expected

    renderer = VocalCueRenderer(generate, lambda _text: pytest.fail("unexpected cue"), 24000)
    assert renderer.render(text) is expected and calls == [text]


@pytest.mark.parametrize("wave", [np.zeros(100), np.full(100, np.nan),
                                  np.ones(100) * 0.0004, np.array([])])
def test_bad_or_faint_cues_do_not_become_amplified_noise(wave):
    from ciel.audio.vocal_cues import VocalCueRenderer

    words = np.ones(100, dtype=np.float32) * 0.1
    calls = []

    def event(text):
        calls.append(text)
        return wave

    renderer = VocalCueRenderer(lambda _text: words, event, 24000)
    np.testing.assert_array_equal(renderer.render("[sigh] Hello."), words)
    np.testing.assert_array_equal(renderer.render("[sigh] Again."), words)
    assert calls == ["[sigh]"]


def test_cue_gain_is_bounded_and_cannot_clip():
    from ciel.audio.vocal_cues import balance_vocal_cue

    wave = np.ones(10000, dtype=np.float32) * 0.003
    wave[50] = 0.4
    result = balance_vocal_cue(wave)
    assert np.max(np.abs(result)) <= 0.5
    assert np.max(result / wave) <= 3


def test_effect_generation_failure_preserves_normal_speech():
    from ciel.audio.vocal_cues import VocalCueRenderer

    words = np.ones(100, dtype=np.float32) * 0.1

    def event(_text):
        raise RuntimeError("effect failed")

    renderer = VocalCueRenderer(lambda _text: words, event, 24000)
    np.testing.assert_array_equal(renderer.render("[sigh] Hello."), words)


@dataclass
class FakeT3:
    speaker_emb: object
    cond_prompt_speech_tokens: object
    cond_prompt_speech_emb: object = None


@dataclass
class FakeConditionals:
    t3: FakeT3
    gen: object


@pytest.mark.parametrize("raises", [False, True])
def test_effect_conditioning_keeps_identity_and_is_restored_even_on_failure(raises):
    from ciel.audio.vocal_cues import generate_styled_cue

    original = FakeConditionals(FakeT3("ciel", "original", "embedded"), "ciel_decoder")
    style = FakeConditionals(FakeT3("demo", "lively"), "demo_decoder")

    def generate(_text):
        assert model.conds.t3.speaker_emb == "ciel"
        assert model.conds.t3.cond_prompt_speech_tokens == "lively"
        assert model.conds.t3.cond_prompt_speech_emb is None
        assert model.conds.gen == "ciel_decoder"
        if raises:
            raise RuntimeError("synthesis failed")
        return "audio"

    model = SimpleNamespace(conds=original, generate=generate)
    if raises:
        with pytest.raises(RuntimeError):
            generate_styled_cue(model, style, "[sigh]")
    else:
        assert generate_styled_cue(model, style, "[sigh]") == "audio"
    assert model.conds is original


def test_manifest_effect_reference_reaches_worker_client(tmp_path):
    from ciel.audio.speech_events import SpeechEvent
    from ciel.audio.tts import TextToSpeech

    profile = tmp_path / "ciel"
    profile.mkdir()
    reference = tmp_path / "reference.wav"
    transcript = tmp_path / "reference.txt"
    effect = tmp_path / "effects.wav"
    transcript.write_text("A test reference.", encoding="utf-8")
    (profile / "voice.json").write_text(json.dumps({
        "name": "ciel", "engine": "chatterbox_turbo",
        "reference_audio": str(reference), "reference_transcript": str(transcript),
        "event_reference_audio": str(effect),
        "event_audio": {"sigh": str(tmp_path / "sigh.wav")},
        "supported_events": [event.value for event in SpeechEvent],
    }), encoding="utf-8")
    tts = TextToSpeech(voice_name="ciel", engine="chatterbox_turbo", enabled=False,
                       voice_profiles_dir=tmp_path)
    worker = tts._load_chatterbox_worker()
    assert worker.event_reference == effect.resolve()
    assert worker.reference == reference.resolve()
    assert worker.cue_audio == {"sigh": tmp_path / "sigh.wav"}
    assert not worker.is_running


def test_local_cue_loading_validates_rate_and_handles_stereo(tmp_path):
    import soundfile as sf

    from ciel.audio.vocal_cues import load_cue_audio

    path = tmp_path / "cue.wav"
    sf.write(path, np.full((100, 2), 0.1, dtype=np.float32), 24000, subtype="FLOAT")
    loaded = load_cue_audio(path, 24000)
    assert loaded.shape == (100,) and loaded.dtype == np.float32
    with pytest.raises(ValueError, match="sample rate"):
        load_cue_audio(path, 16000)


def test_selected_local_clip_is_used_without_re_generation_or_style_reference(tmp_path):
    import soundfile as sf

    from ciel.audio.vocal_cues import create_turbo_renderer

    cue_path = tmp_path / "sigh.wav"
    cue = np.full(100, 0.025, dtype=np.float32)
    sf.write(cue_path, cue, 24000, subtype="FLOAT")
    calls = []
    original = object()

    def generate(text):
        calls.append(text)
        assert model.conds is original
        return np.full(100, 0.1, dtype=np.float32)

    model = SimpleNamespace(conds=original, sr=24000, generate=generate)
    renderer = create_turbo_renderer(model, tmp_path / "missing.wav", {"sigh": cue_path})
    result = renderer.render("[sigh] Hello.")
    np.testing.assert_allclose(result[:100], cue)
    assert calls == ["Hello."] and model.conds is original
