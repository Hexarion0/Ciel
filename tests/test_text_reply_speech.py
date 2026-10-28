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
