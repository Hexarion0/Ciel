"""Concurrency regressions for first-sentence playback and canceled replies."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ciel.audio.streaming import SentenceBuffer, VisibleText, _SentencePipeline, stream_reply
from ciel.audio.tts import TextToSpeech
from ciel.providers.base import ChatMessage, LLMStreamChunk


@pytest.fixture
def speech(monkeypatch):
    tts = TextToSpeech(enabled=False)
    tts.enabled = True
    played = []
    monkeypatch.setattr(tts, "synthesize", lambda text: (
        np.ones(160, dtype=np.float32), 16000,
    ))
    monkeypatch.setattr("ciel.audio.tts.sd", SimpleNamespace(
        play=lambda *args, **kwargs: played.append(1),
        get_stream=lambda: SimpleNamespace(active=False), stop=lambda: None,
    ))
    return tts, played


def chunk(text, final=False):
    return LLMStreamChunk(text, "test-model", "test-provider", final)


def test_delimiters_split_across_chunks_never_speak_thinking_or_code():
    visible = VisibleText()
    pieces = ["<thi", "nk>private. <reflection>secret</ref", "lection></think>Answer. ",
              "``", "`python\nprint('private')\n`", "`` Done."]
    output = "".join(visible.feed(piece) for piece in pieces) + visible.feed("", final=True)
    assert output == "Answer.  [code omitted]  Done."
    assert "private" not in output and "secret" not in output
    hidden = VisibleText()
    assert hidden.feed("<reasoning>unclosed secret.", final=True) == ""


@pytest.mark.parametrize("omit_code", [True, False])
def test_internal_playback_notes_are_hidden_across_every_chunk_boundary(omit_code):
    for note in (
        "[Playback was interrupted.]",
        "[Generation ended early because the connection failed.]",
    ):
        for boundary in range(len(note) + 1):
            visible = VisibleText(omit_code=omit_code)
            output = visible.feed(note[:boundary])
            output += visible.feed(note[boundary:] + "Let's continue.", final=True)
            assert output == "Let's continue."
    visible = VisibleText(omit_code=omit_code)
    assert visible.feed("[chuckle] Hello.", final=True) == "[chuckle] Hello."


@pytest.mark.parametrize("omit_code", [True, False])
def test_stage_directions_are_hidden_across_every_chunk_boundary(omit_code):
    text = "[soft inhale — steady, focused] Alright. [brief pause — quiet, present] I can check."
    for boundary in range(len(text) + 1):
        visible = VisibleText(omit_code=omit_code)
        output = visible.feed(text[:boundary])
        output += visible.feed(text[boundary:], final=True)
        assert " ".join(output.split()) == "Alright. I can check."
    visible = VisibleText(omit_code=omit_code)
    assert visible.feed("Use [array] and [1, 2, 3]. [chuckle] Hello.", final=True) == (
        "Use [array] and [1, 2, 3]. [chuckle] Hello."
    )
    assert visible.feed("[soft inhale", final=True) == ""
    assert visible.feed("[label <think>private</think>]", final=True) == "[label ]"


@pytest.mark.parametrize("pipeline", [False, True])
def test_stream_filters_long_stage_directions_before_sentence_splitting(pipeline):
    played, prepared = [], []

    def speak(text, **controls):
        played.append(text)
        controls["on_start"]()
        return True

    def prepare(text, *_args):
        prepared.append(text)
        return (np.zeros(10, dtype=np.float32), 24000, None)

    tts = SimpleNamespace(
        expressiveness="expressive", audio_queue_size=2, supports_sentence_pipeline=pipeline,
        begin_stream=lambda: 1, update_stream=lambda *_args: None,
        end_stream=lambda *_args: None, stop=lambda: None, speak=speak,
        prepare_sentence=prepare,
    )
    text = "[soft inhale — " + "steady, focused. " * 30 + "] Alright. I can check."

    def generate(*_args, **_kwargs):
        for index in range(0, len(text), 7):
            yield chunk(text[index:index + 7])

    result = stream_reply(SimpleNamespace(stream=generate), [], tts, lambda: True)
    assert result.spoken
    assert result.response.content.strip() == "Alright. I can check."
    assert played == ["Alright.", "I can check."]
    if pipeline:
        assert prepared == played


def test_sentence_boundaries_keep_decimal_and_abbreviation():
    buffer = SentenceBuffer()
    assert buffer.feed("Dr. Smith measured 3.") == []
    assert buffer.feed("14 volts. Next") == ["Dr. Smith measured 3.14 volts."]
    assert buffer.feed(" sentence", final=True) == ["Next sentence"]
    assert buffer.feed("1. First item\n2. Second item", final=True) == [
        "1. First item", "2. Second item",
    ]


def test_long_unpunctuated_speech_is_bounded_without_losing_words():
    text = " ".join(f"word{index}" for index in range(100))
    parts = SentenceBuffer(max_chars=80).feed(text, final=True)
    assert len(parts) > 1 and max(map(len, parts)) <= 80
    assert " ".join(parts) == text


def test_first_sentence_plays_before_provider_finishes(speech, monkeypatch):
    tts, played = speech
    first_audio, generation_finished = threading.Event(), threading.Event()

    def play(*args, **kwargs):
        assert not generation_finished.is_set() if not played else True
        played.append(1)
        first_audio.set()

    monkeypatch.setattr("ciel.audio.tts.sd.play", play)

    def generate(*args, **kwargs):
        yield chunk("First sentence. ")
        assert first_audio.wait(1), "Playback waited for full generation"
        generation_finished.set()
        yield chunk("Second sentence.")
        yield chunk("", final=True)

    result = stream_reply(SimpleNamespace(stream=generate), [ChatMessage("user", "Explain.")],
                          tts, lambda: True)
    assert result.response.content == "First sentence. Second sentence."
    assert result.response.provider == "test-provider"
    assert result.spoken and not result.canceled and len(played) == 2
    assert result.first_token_seconds is not None and result.first_audio_seconds is not None
    assert tts.stop() is None


def test_turbo_pipeline_synthesizes_next_sentence_during_playback():
    first_started = threading.Event()
    second_prepared = threading.Event()
    finished = threading.Event()
    played = []

    class TurboPipelineStub:
        supports_sentence_pipeline = True
        audio_queue_size = 1

        def begin_stream(self):
            return 3

        def update_stream(self, _generation, _text):
            pass

        def prepare_sentence(self, text, cancel_event, _generation):
            if cancel_event.is_set():
                return None
            if text == "Second sentence.":
                second_prepared.set()
            return (text, 24000, None)

        def speak(self, text, *, on_start, _prepared, **_kwargs):
            assert _prepared[0] == text
            on_start()
            played.append(text)
            if text == "First sentence.":
                first_started.set()
                assert second_prepared.wait(1), "Next sentence waited for playback to end"
                time.sleep(0.04)
            return True

        def end_stream(self, _generation):
            finished.set()

    tts = TurboPipelineStub()

    def generate(*_args, **_kwargs):
        yield chunk("First sentence. ")
        assert first_started.wait(1)
        yield chunk("Second sentence.")

    result = stream_reply(SimpleNamespace(stream=generate), [], tts, lambda: True)
    assert result.spoken and not result.canceled
    assert played == ["First sentence.", "Second sentence."]
    assert result.first_audio_seconds is not None
    assert result.speech_pipeline["sentences"][1]["synthesis_finished_seconds"] < (
        result.speech_pipeline["sentences"][0]["playback_started_seconds"] + 0.04
    )
    assert finished.is_set()


def test_barge_in_while_bounded_sentence_queue_is_full_does_not_deadlock():
    preparing, interrupted = threading.Event(), threading.Event()

    class BlockedPipelineStub:
        audio_queue_size = 1

        def prepare_sentence(self, _text, cancel_event, _generation):
            preparing.set()
            cancel_event.wait(1)
            return None

    pipeline = _SentencePipeline(
        BlockedPipelineStub(), 1, lambda: not interrupted.is_set(), None,
        time.monotonic(), lambda _text: None, None,
    )
    assert pipeline.submit("First.", time.monotonic())
    assert preparing.wait(1)
    assert pipeline.submit("Second.", time.monotonic())
    finisher = threading.Thread(target=lambda: pipeline.finish(cancel=False))
    finisher.start()
    time.sleep(0.05)
    interrupted.set()
    finisher.join(timeout=1)
    assert not finisher.is_alive(), "queue sentinel wait ignored barge-in"


def test_sentence_callback_arrives_at_playback_before_generation_finishes(speech):
    tts, played = speech
    first_sentence, generation_finished = threading.Event(), threading.Event()
    observed = []

    def sentence_started(text):
        assert len(played) == len(observed) + 1, "Transcript preceded audio playback"
        if not observed:
            assert not generation_finished.is_set(), "Transcript waited for full generation"
        observed.append(text)
        first_sentence.set()

    def generate(*args, **kwargs):
        yield chunk("First sentence. ")
        assert first_sentence.wait(1), "No first-sentence transcript during generation"
        generation_finished.set()
        yield chunk("Second sentence.")
        yield chunk("", final=True)

    result = stream_reply(
        SimpleNamespace(stream=generate), [], tts, lambda: True,
        on_sentence_start=sentence_started,
    )
    assert observed == ["First sentence.", "Second sentence."]
    assert result.spoken and not result.canceled
    assert generation_finished.is_set()


def test_stream_forwards_playback_prefixes_without_revealing_sentence_end():
    tts = MagicMock()
    starts, progress = [], []

    def speak(text, **controls):
        controls['on_start']()
        assert starts[-1] == text
        callback = controls['on_progress']
        callback(text[:1], False, False)
        callback(text[:5], False, False)
        callback(text, True, False)
        return True

    tts.speak.side_effect = speak
    router = SimpleNamespace(stream=lambda *a, **k: iter([chunk('First sentence. Second. ')]))
    result = stream_reply(
        router, [], tts, lambda: True, on_sentence_start=starts.append,
        on_progress=lambda text, done, stopped: progress.append((text, done, stopped)),
    )
    assert starts == ['First sentence.', 'Second.']
    assert progress == [
        ('F', False, False), ('First', False, False), ('First sentence.', True, False),
        ('S', False, False), ('Secon', False, False), ('Second.', True, False),
    ]
    assert result.spoken


def test_stream_finalizes_interrupted_prefix_and_suppresses_stale_progress():
    tts = MagicMock()
    cancel = threading.Event()
    progress = []

    def speak(_text, **controls):
        controls['on_start']()
        controls['on_progress']('First', False, False)
        cancel.set()
        controls['on_progress']('First sentence.', False, False)
        controls['on_progress']('First', True, True)
        return False

    tts.speak.side_effect = speak
    router = SimpleNamespace(stream=lambda *a, **k: iter([chunk('First sentence. Second. ')]))
    result = stream_reply(
        router, [], tts, lambda: not cancel.is_set(), cancel_event=cancel,
        on_progress=lambda text, done, stopped: progress.append((text, done, stopped)),
    )
    assert result.canceled
    assert progress == [('First', False, False), ('First', True, True)]
    tts.speak.assert_called_once()


def test_stream_does_not_install_progress_callback_when_captions_are_disabled():
    tts = MagicMock()

    def speak(_text, **controls):
        assert 'on_progress' not in controls
        controls['on_start']()
        return True

    tts.speak.side_effect = speak
    router = SimpleNamespace(stream=lambda *a, **k: iter([chunk('Hello.')]))
    assert stream_reply(router, [], tts, lambda: True).spoken


@pytest.mark.parametrize("cancel_sentence", ["First.", "Second."])
def test_canceled_synthesis_does_not_report_canceled_or_queued_sentences(
    speech, monkeypatch, cancel_sentence,
):
    tts, played = speech
    entered, release, cancel, finished, synthesized = (threading.Event() for _ in range(5))
    first_started = threading.Event()
    observed, results = [], []

    def synthesize(text):
        if text == cancel_sentence:
            entered.set()
            try:
                assert release.wait(2)
            finally:
                synthesized.set()
        return np.ones(160, dtype=np.float32), 16000

    monkeypatch.setattr(tts, "synthesize", synthesize)
    router = SimpleNamespace(stream=lambda *a, **k: iter([chunk("First. Second. Third. ")]))

    def on_sentence_start(text):
        observed.append(text)
        if text == "First.":
            first_started.set()

    def run():
        results.append(stream_reply(
            router, [], tts, lambda: not cancel.is_set(), cancel_event=cancel,
            on_sentence_start=on_sentence_start,
        ))
        finished.set()

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(1)
        if cancel_sentence == "Second.":
            assert first_started.wait(1)
        cancel.set()
        tts.stop()
        assert finished.wait(0.5), "Canceled synthesis delayed the new utterance"
    finally:
        release.set()
        worker.join(2)
        assert synthesized.wait(1)
    assert results[0].canceled
    assert observed == ([] if cancel_sentence == "First." else ["First."])
    assert len(played) == len(observed)


def test_cancel_during_blocked_provider_returns_without_waiting(speech):
    tts, played = speech
    entered, release, closed, done, cancel = (threading.Event() for _ in range(5))

    def generate(*args, **kwargs):
        entered.set()
        try:
            assert release.wait(2)
            yield chunk("This stale reply must not play.")
        finally:
            closed.set()

    results = []

    def run():
        results.append(stream_reply(SimpleNamespace(stream=generate), [], tts,
                                    lambda: not cancel.is_set(), cancel_event=cancel))
        done.set()

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(1)
        cancel.set()
        assert done.wait(0.5), "Canceled network read blocked the next utterance"
        assert results[0].canceled and not results[0].response.content
        assert played == []
    finally:
        release.set()
        worker.join(2)
        assert closed.wait(1)


def test_cancel_during_synthesis_discards_queued_sentences(speech, monkeypatch):
    tts, played = speech
    entered, release, cancel, finished = (threading.Event() for _ in range(4))

    def synthesize(text):
        entered.set()
        assert release.wait(2)
        return np.ones(160), 16000

    monkeypatch.setattr(tts, "synthesize", synthesize)
    results = []
    router = SimpleNamespace(stream=lambda *a, **k: iter([chunk("First. Second. ")]))
    def run():
        results.append(stream_reply(
            router, [], tts, lambda: not cancel.is_set(), cancel_event=cancel,
        ))
        finished.set()

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(1)
        cancel.set()
        tts.stop()
        assert finished.wait(0.5), "Canceled synthesis blocked the next utterance"
    finally:
        release.set()
        worker.join(2)
    assert results[0].canceled and played == []


def test_cancel_during_playback_stops_rest_of_reply(speech, monkeypatch):
    tts, played = speech
    cancel = threading.Event()

    def play(*args, **kwargs):
        played.append(1)
        cancel.set()

    monkeypatch.setattr("ciel.audio.tts.sd.play", play)
    monkeypatch.setattr("ciel.audio.tts.sd.get_stream", lambda: SimpleNamespace(active=True))
    router = SimpleNamespace(stream=lambda *a, **k: iter([chunk("First. Second. ")]))
    result = stream_reply(router, [], tts, lambda: not cancel.is_set(), cancel_event=cancel)
    assert result.canceled and len(played) == 1
    assert result.first_audio_seconds is not None
    assert not tts.is_speaking()


def test_partial_failure_keeps_answer_and_does_not_restart(speech):
    tts, played = speech

    def generate(*args, **kwargs):
        yield chunk("First sentence. ")
        raise RuntimeError("Connection dropped")

    result = stream_reply(SimpleNamespace(stream=generate), [], tts, lambda: True)
    assert result.response.content == "First sentence."
    assert isinstance(result.error, RuntimeError) and len(played) == 1


def test_code_stays_in_history_but_is_omitted_from_spoken_reply(speech, monkeypatch):
    tts, _played = speech
    synthesized = []
    original = tts.synthesize

    def synthesize(text):
        synthesized.append(text)
        return original(text)

    monkeypatch.setattr(tts, "synthesize", synthesize)
    text = "Here is the code.\n```python\nx = 1\n```\nThat is all."
    router = SimpleNamespace(stream=lambda *a, **k: iter([chunk(text)]))
    result = stream_reply(router, [], tts, lambda: True)
    assert result.response.content == text
    spoken = " ".join(synthesized)
    assert "x = 1" not in spoken and "python" not in spoken
    assert "[code omitted]" in spoken and "That is all." in spoken


def test_sentence_transcripts_filter_reasoning_and_code_across_chunk_boundaries(speech):
    tts, _played = speech
    observed = []
    pieces = [
        "<thi", "nk>Private reasoning. <reflection>Secret.</ref",
        "lection></think>Here is the code.\n", "``", "`python\nprivate_value = 1\n`",
        "``\nThat is all.",
    ]
    router = SimpleNamespace(stream=lambda *a, **k: (chunk(text) for text in pieces))
    result = stream_reply(router, [], tts, lambda: True, on_sentence_start=observed.append)
    assert observed == ["Here is the code.", "[code omitted]", "That is all."]
    transcripts = " ".join(observed)
    assert "Private" not in transcripts and "Secret" not in transcripts
    assert "private_value" not in transcripts and "python" not in transcripts
    assert "private_value = 1" in result.response.content
    assert "Private" not in result.response.content and "Secret" not in result.response.content


def test_empty_or_reasoning_only_reply_never_plays(speech):
    tts, played = speech
    router = SimpleNamespace(stream=lambda *a, **k: iter([chunk("<think>Secret.</think>")]))
    with pytest.raises(RuntimeError, match="without a visible reply"):
        stream_reply(router, [], tts, lambda: True)
    assert played == []


def test_stream_interruption_reports_completed_and_current_sentences(speech, monkeypatch):
    tts, _played = speech
    clock = [100.0]
    monkeypatch.setattr("ciel.audio.tts.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(tts, "synthesize", lambda text: (np.ones(100), 10))
    generation = tts.begin_stream()
    tts.update_stream(generation, "First complete sentence. One two three four. Last sentence.")
    assert tts.speak("First complete sentence.", generation=generation)
    monkeypatch.setattr("ciel.audio.tts.sd.get_stream", lambda: SimpleNamespace(active=True))
    assert tts.speak("One two three four.", generation=generation, block=False)
    clock[0] += 5
    interrupted = tts.stop()
    assert interrupted["estimated_spoken_text"] == "First complete sentence. One two"
    assert interrupted["remaining_text"] == "three four. Last sentence."
    assert interrupted["played_seconds"] == 15
    assert tts.stop() is None
    assert not tts.speak("Last sentence.", generation=generation)


def test_disabled_speech_still_consumes_and_returns_reply():
    tts = TextToSpeech(enabled=False)
    router = SimpleNamespace(stream=lambda *a, **k: iter([chunk("A text-only reply.")]))
    result = stream_reply(router, [], tts, lambda: True)
    assert result.response.content == "A text-only reply." and not result.spoken


def test_interruption_between_sentences_does_not_count_previous_audio_twice(speech, monkeypatch):
    tts, _played = speech
    clock = [100.0]
    monkeypatch.setattr("ciel.audio.tts.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(tts, "synthesize", lambda text: (np.ones(100), 10))
    generation = tts.begin_stream()
    tts.update_stream(generation, "First sentence. Second sentence.")
    assert tts.speak("First sentence.", generation=generation)
    clock[0] += 15  # Finished audio plus a pause while the provider generates more.
    interrupted = tts.stop()
    assert interrupted["estimated_spoken_text"] == "First sentence."
    assert interrupted["remaining_text"] == "Second sentence."
    assert interrupted["played_seconds"] == interrupted["duration_seconds"] == 10
