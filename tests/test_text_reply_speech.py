"""Written replies retain detail while their spoken excerpts stay brief."""

import pytest

from ciel.audio.text_reply import spoken_text_reply

_POINTER = "The full answer is in the chat."


def test_short_prose_is_preserved_after_speech_cleanup():
    text = "[chuckle] **Yes.**\nYou can use `pip install` and [the guide](https://example.com)."
    assert spoken_text_reply(text) == "Yes. You can use pip install and the guide."


def test_long_reply_uses_complete_sentences_and_preserves_written_answer():
    sentence = "This is one complete sentence with ten simple ordinary words."
    original = " ".join([sentence] * 12)
    written = original

    spoken = spoken_text_reply(original)

    assert spoken == " ".join([sentence] * 7) + " " + _POINTER
    assert len(spoken.split()) <= 80
    assert original == written


def test_long_first_sentence_uses_a_complete_word_prefix_and_ellipsis():
    words = [f"word{number}" for number in range(120)]
    spoken = spoken_text_reply(" ".join(words))
    budget = 80 - len(_POINTER.split())

    assert spoken == " ".join(words[:budget]) + "… " + _POINTER
    assert len(spoken.split()) == 80
    assert f"word{budget}" not in spoken


def test_numeric_citations_are_not_spoken():
    assert spoken_text_reply("The value is 3.14 [1]. More detail follows.[12][3]") == (
        "The value is 3.14. More detail follows."
    )


@pytest.mark.parametrize("code", ["```python\nanswer = 42\n```", "```python\nanswer = 42"])
def test_code_only_reply_gets_a_brief_acknowledgement(code):
    assert spoken_text_reply(code) == "The code is in the chat."


@pytest.mark.parametrize("fence", ["```", ""])
def test_code_is_hidden_and_omission_is_acknowledged(fence):
    text = "Here is the example.\n```python\nprivate_value = 42\n" + fence
    assert spoken_text_reply(text) == "Here is the example. " + _POINTER


def test_code_marker_is_not_read_aloud():
    assert spoken_text_reply("The example follows. [code omitted]") == (
        "The example follows. " + _POINTER
    )


@pytest.mark.parametrize("tag", ["think", "thought", "reasoning", "reflection"])
def test_hidden_reasoning_is_never_in_the_spoken_excerpt(tag):
    assert spoken_text_reply(f"<{tag}>Private thoughts.</{tag}>Visible answer.") == (
        "Visible answer."
    )
    assert spoken_text_reply(f"Visible answer. <{tag}>Unfinished private thoughts.") == (
        "Visible answer."
    )
    assert spoken_text_reply(f"<{tag}>```python\nprivate_value = 42") == ""


def test_sentence_boundary_keeps_decimals_abbreviations_and_quotation_marks():
    first = 'Dr. Smith measured 3.14 volts. "That is correct!"'
    long_tail = " ".join(["continuation"] * 100)
    assert spoken_text_reply(first + " " + long_tail) == first + " " + _POINTER


@pytest.mark.parametrize("text", ["", " \n\t", "<think>Private.</think>", "[sigh]"])
def test_empty_or_entirely_hidden_reply_is_silent(text):
    assert spoken_text_reply(text) == ""


@pytest.mark.parametrize("word_count", [73, 74, 80, 81, 200])
@pytest.mark.parametrize("has_code", [False, True])
def test_output_word_count_includes_any_pointer(word_count, has_code):
    text = " ".join(["word"] * word_count) + "."
    if has_code:
        text += "\n```python\nexample = 1\n```"
    spoken = spoken_text_reply(text)
    assert len(spoken.split()) <= 80
    assert spoken.endswith(_POINTER) == (has_code or word_count > 80)


_SENTENCE = "This is one complete sentence with ten simple ordinary words."
_STREAMED_ANSWERS = [
    "Sure. Here is a short answer.",
    "Yes! It works. The answer is 42",
    " ".join([_SENTENCE] * 12),
    " ".join([_SENTENCE] * 7) + " Four more words here.",
    " ".join([_SENTENCE] * 7) + " Here are ten more words to finish it.",
    " ".join([_SENTENCE] * 8) + " One.",
    "First, an overview.\n\n- **Bold point** one.\n- A [linked point](https://example.com).\n\n"
    "1. Step one.\n2. Step two.",
    "Here is the example.\n```python\nprivate_value = 42\n```\nThat prints it. It is simple.",
    "Here is the example.\n```python\nprivate_value = 42",
    "```python\nanswer = 42\n```",
    "The value is 3.14 [1]. More detail follows.[12][3] Dr. Smith agrees.",
    'Dr. Smith measured 3.14 volts. "That is correct!" ' + " ".join(["continuation"] * 100),
    " ".join(f"word{number}" for number in range(120)),
    " ".join(f"word{number}" for number in range(120)) + ". Then more. And more.",
    "<think>Private thoughts.</think>Visible answer. Another sentence.",
    "Visible answer. <think>Unfinished private thoughts.",
    "[chuckle] **Yes.**\nYou can use `pip install` and [the guide](https://example.com).",
    "",
]

_ANSWER_IDS = [f"answer{index}" for index in range(len(_STREAMED_ANSWERS))]

def _stream_excerpt(text, cuts):
    from ciel.audio.text_reply import StreamingExcerpt

    excerpt, spoken, start = StreamingExcerpt(), [], 0
    for cut in [*cuts, len(text)]:
        spoken += excerpt.update(text[:cut])
        start = cut
    assert start == len(text)
    return spoken + excerpt.finish(text)


@pytest.mark.parametrize("text", _STREAMED_ANSWERS, ids=_ANSWER_IDS)
@pytest.mark.parametrize("seed", range(6))
def test_streamed_excerpt_speaks_exactly_the_one_shot_excerpt(text, seed):
    """Streaming changes when Ciel speaks, never what she says."""
    import random

    rng = random.Random(seed)
    cuts = sorted(rng.sample(range(len(text) + 1), min(len(text) + 1, rng.randint(1, 40))))

    assert " ".join(_stream_excerpt(text, cuts)) == spoken_text_reply(text)


@pytest.mark.parametrize("text", _STREAMED_ANSWERS, ids=_ANSWER_IDS)
def test_streamed_excerpt_matches_when_every_character_arrives_separately(text):
    assert " ".join(_stream_excerpt(text, range(len(text)))) == spoken_text_reply(text)


def test_first_sentence_is_released_as_soon_as_the_next_one_begins():
    from ciel.audio.text_reply import StreamingExcerpt

    excerpt = StreamingExcerpt()
    assert excerpt.update("Sure thing.") == []
    assert excerpt.update("Sure thing. Here") == ["Sure thing."]
    assert excerpt.update("Sure thing. Here is more") == []
    assert excerpt.finish("Sure thing. Here is more.") == ["Here is more."]


def test_long_answer_settles_its_excerpt_before_generation_ends():
    from ciel.audio.text_reply import StreamingExcerpt

    excerpt = StreamingExcerpt()
    spoken = excerpt.update(" ".join([_SENTENCE] * 9) + " More")

    assert spoken == [_SENTENCE] * 7 + [_POINTER]
    assert excerpt.settled
    assert excerpt.update(" ".join([_SENTENCE] * 12)) == []
    assert excerpt.finish(" ".join([_SENTENCE] * 12)) == []


def test_sentence_crossing_the_budget_waits_for_the_final_length():
    from ciel.audio.text_reply import StreamingExcerpt

    short = " ".join([_SENTENCE] * 7) + " Five more words are here."
    excerpt = StreamingExcerpt()

    assert excerpt.update(short + " ") == [_SENTENCE] * 7
    assert excerpt.update(short + " And") == []
    assert not excerpt.settled
    assert excerpt.finish(short) == ["Five more words are here."]


def test_code_settles_the_pointer_without_waiting_for_more_words():
    from ciel.audio.text_reply import StreamingExcerpt

    text = " ".join([_SENTENCE] * 8) + "\n```python\nanswer = 42\n```\nDone."
    excerpt = StreamingExcerpt()

    assert excerpt.update(text) == [_SENTENCE] * 7 + [_POINTER]
    assert excerpt.finish(text) == []


def test_rewritten_stream_text_stops_releasing_and_never_contradicts_speech():
    from ciel.audio.text_reply import StreamingExcerpt

    excerpt = StreamingExcerpt()
    assert excerpt.update("Hello there. General") == ["Hello there."]
    assert excerpt.update("Goodbye now. Something else") == []
    assert excerpt.finish("Goodbye now. Something else.") == []


def test_rewritten_long_answer_still_ends_with_the_pointer():
    from ciel.audio.text_reply import StreamingExcerpt

    excerpt = StreamingExcerpt()
    assert excerpt.update("Hello there. General") == ["Hello there."]
    assert excerpt.update("Goodbye now.") == []
    assert excerpt.finish(" ".join([_SENTENCE] * 12)) == [_POINTER]


_FUZZ_WORDS = "the a value system answer quick ciel phone voice model sentence simple".split()
_FUZZ_PIECES = (
    lambda r: " ".join(r.choice(_FUZZ_WORDS) for _ in range(r.randint(1, 18))).capitalize()
    + r.choice(".!?"),
    lambda r: "**" + r.choice(_FUZZ_WORDS) + " " + r.choice(_FUZZ_WORDS) + "**",
    lambda r: "`" + r.choice(_FUZZ_WORDS) + "_" + r.choice(_FUZZ_WORDS) + "`",
    lambda r: "[" + r.choice(_FUZZ_WORDS) + " guide](https://example.com/a)",
    lambda r: "[" + str(r.randint(1, 12)) + "]",
    lambda r: "\n- " + r.choice(_FUZZ_WORDS) + " item.",
    lambda r: "\n" + str(r.randint(1, 9)) + ". Step " + r.choice(_FUZZ_WORDS) + ".",
    lambda r: "\n```python\nx = 1\nprint(x). y\n```\n",
    lambda r: "Dr. " + r.choice(_FUZZ_WORDS).capitalize() + " measured 3.14 volts.",
    lambda r: '"' + r.choice(_FUZZ_WORDS).capitalize() + ' is right!"',
    lambda r: "\n\n## " + r.choice(_FUZZ_WORDS).capitalize() + "\n",
    lambda r: "<think>" + r.choice(_FUZZ_WORDS) + ". hidden.</think>",
    lambda r: " e.g. " + r.choice(_FUZZ_WORDS) + ",",
    lambda r: "[sigh] ",
    lambda r: " \U0001f389 ",
)


@pytest.mark.parametrize("seed", range(150))
def test_streamed_excerpt_matches_random_markdown_answers(seed):
    import random

    rng = random.Random(seed)
    text = " ".join(rng.choice(_FUZZ_PIECES)(rng) for _ in range(rng.randint(1, 60)))
    cuts = sorted(rng.sample(range(len(text) + 1), min(len(text) + 1, rng.randint(1, 80))))

    assert " ".join(_stream_excerpt(text, cuts)) == spoken_text_reply(text)
