"""Caption timing uses real word spans and exposes fallback estimates honestly."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ciel.audio.alignment import (
    CharacterTimeline,
    character_timeline,
    estimated_timeline,
)


def chunk(alignments, *, sample_rate=100, audio_samples=None):
    """Build a Piper-shaped synthesis result without loading speech models."""
    counts = [SimpleNamespace(phoneme=p, num_samples=n) for p, n in alignments]
    samples = sum(n for _, n in alignments) if audio_samples is None else audio_samples
    return SimpleNamespace(
        phoneme_alignments=counts, sample_rate=sample_rate, audio_float_array=[0] * samples,
    )


def test_native_word_times_preserve_initial_silence_and_between_word_pauses():
    audio = chunk([('^', 10), ('h', 10), ('i', 10), (' ', 30), ('m', 10), ('e', 10), ('$', 20)])
    timeline = character_timeline('Hi me!', [audio], 100, 100)

    assert timeline.source == 'phoneme'
    assert timeline.character_times == pytest.approx((0.2, 0.3, 0.3, 0.7, 0.8, 0.8))
    assert timeline.prefix(0.1) == ''
    assert timeline.prefix(0.4) == 'Hi '
    assert timeline.prefix(0.75) == 'Hi m'
    assert timeline.prefix(1) == 'Hi me!'


def test_native_word_span_includes_stress_marker_audio_and_excludes_punctuation_pause():
    audio = chunk([('^', 10), ('ˈ', 20), ('h', 10), ('i', 10), (',', 30), ('$', 20)])
    timeline = character_timeline('Hi,', [audio], 100, 100)

    assert timeline.source == 'phoneme'
    assert timeline.character_times == pytest.approx((0.3, 0.5, 0.5))


def test_sentence_offsets_include_complete_previous_waveform():
    first = chunk([('^', 10), ('h', 10), ('i', 10), ('.', 20), ('$', 50)])
    second = chunk([('^', 20), ('b', 10), ('a', 10), ('ɪ', 10), ('!', 20), ('$', 30)])
    timeline = character_timeline('Hi. Bye!', [first, second], 100, 200)

    assert timeline.source == 'phoneme'
    assert timeline.character_times == pytest.approx((0.2, 0.3, 0.3, 0.3, 1.3, 1.4, 1.5, 1.5))
    assert timeline.prefix(1.1) == 'Hi. '


def test_contractions_remain_one_written_word_and_preserve_apostrophe():
    audio = chunk([('^', 5), ('w', 5), ('ʌ', 5), ('t', 5), ('s', 5), ('?', 5), ('$', 5)])
    timeline = character_timeline("What's?", [audio], 100, 35)

    assert timeline.source == 'phoneme'
    assert timeline.text == "What's?"
    assert timeline.prefix(1) == "What's?"


def test_combining_mark_reveals_with_its_base_character():
    audio = chunk([('^', 10), ('e', 20), ('$', 10)])
    timeline = character_timeline('e\u0301', [audio], 100, 40)

    assert timeline.source == 'phoneme'
    assert timeline.character_times[0] == timeline.character_times[1]
    assert timeline.prefix(0.31) == 'e\u0301'


@pytest.mark.parametrize('case', [
    'no_alignment', 'mismatched_words', 'mismatched_audio', 'mismatched_rate',
    'bad_count', 'wrong_total', 'zero_word_duration', 'digits',
])
def test_uncertain_mapping_falls_back_to_labelled_duration_estimate(case):
    text = 'Hi'
    audio = chunk([('^', 10), ('h', 20), ('$', 10)])
    samples = 40
    if case == 'no_alignment':
        audio.phoneme_alignments = None
    elif case == 'mismatched_words':
        text = 'Hi there'
    elif case == 'mismatched_audio':
        samples = 50
    elif case == 'mismatched_rate':
        audio.sample_rate = 200
    elif case == 'bad_count':
        audio.phoneme_alignments[1].num_samples = -1
    elif case == 'wrong_total':
        audio.audio_float_array += [0]
    elif case == 'zero_word_duration':
        audio.phoneme_alignments[1].num_samples = 0
    else:
        text = '5'

    timeline = character_timeline(text, [audio], 100, samples)

    assert timeline.source == 'estimated'
    assert timeline.character_times == estimated_timeline(text, samples / 100).character_times
    assert timeline.text == text


def test_estimate_preserves_original_text_and_whitespace_without_extra_delay():
    timeline = estimated_timeline('Hi me!', 1)

    assert timeline.source == 'estimated'
    assert timeline.character_times == pytest.approx((0.2, 0.4, 0.4, 0.6, 0.8, 1))
    assert timeline.prefix(0.4) == 'Hi '
    assert timeline.prefix(1) == 'Hi me!'


@pytest.mark.parametrize('duration', [0, -1, float('nan'), float('inf')])
def test_invalid_or_zero_estimated_duration_is_finite(duration):
    timeline = estimated_timeline('Hi', duration)

    assert timeline.character_times == (0, 0)


def test_visible_count_handles_boundaries_and_invalid_cursors():
    timeline = CharacterTimeline('Hi!', (0.1, 0.2, 0.2), 'phoneme')

    assert timeline.visible_count(-1) == 0
    assert timeline.visible_count(0) == 0
    assert timeline.visible_count(float('nan')) == 0
    assert timeline.prefix(0.1) == 'H'
    assert timeline.prefix(0.2) == 'Hi!'
    assert timeline.visible_count(float('inf')) == 3


def test_empty_text_is_valid_for_every_cursor():
    timeline = character_timeline('', [], 100, 0)

    assert timeline.text == ''
    assert timeline.character_times == ()
    assert timeline.visible_count(10) == 0


def test_voice_phonemizer_maps_expanded_time_to_original_digits_without_audio_inference():
    audio = chunk([
        ('^', 10), ('ɪ', 10), (' ', 10), ('z', 10), (' ', 10),
        ('f', 10), (' ', 10), ('θ', 10), (' ', 10), ('t', 10), (' ', 10),
        ('a', 10), ('m', 10), ('.', 10), ('$', 10),
    ])
    phonemize = MagicMock(side_effect=[
        [['ɪ']], [['z']], [['f', ' ', 'θ', ' ', 't']], [['a', 'm']],
    ])

    timeline = character_timeline('It is 5:32 AM.', [audio], 100, 150, phonemize=phonemize)

    assert timeline.source == 'phoneme'
    assert timeline.text == 'It is 5:32 AM.'
    assert [call.args[0] for call in phonemize.call_args_list] == ['It', 'is', '5:32', 'AM']
    assert timeline.character_times[6:10] == pytest.approx((0.625, 0.75, 0.875, 1))
    assert timeline.prefix(0.76) == 'It is 5:'
    assert timeline.prefix(1.5) == 'It is 5:32 AM.'


@pytest.mark.parametrize('result', [
    [['f', ' ', 'θ']], [['f', ' ', 'θ', ' ', 't', ' ', 'a']], [],
    'f θ t', ['f θ t'], [[None]],
])
def test_uncertain_token_expansion_does_not_claim_native_timing(result):
    audio = chunk([('f', 10), (' ', 10), ('θ', 10), (' ', 10), ('t', 10)])
    phonemize = MagicMock(return_value=result)

    timeline = character_timeline('5:32', [audio], 100, 50, phonemize=phonemize)

    assert timeline.source == 'estimated'


def test_failed_phonemizer_falls_back_without_breaking_caption_or_audio():
    audio = chunk([('f', 10)])
    phonemize = MagicMock(side_effect=RuntimeError('unavailable'))

    timeline = character_timeline('5', [audio], 100, 10, phonemize=phonemize)

    assert timeline.source == 'estimated'
    assert timeline.prefix(0.1) == '5'
