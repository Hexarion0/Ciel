import pytest

from ciel.conversation import is_farewell


@pytest.mark.parametrize(
    "text",
    [
        "bye",
        "Goodbye!",
        "good night",
        "thanks, bye Ciel!",
        "See you tomorrow.",
        "take care",
        "bye-bye",
    ],
)
def test_deliberate_sign_off(text):
    assert is_farewell(text)


@pytest.mark.parametrize(
    "text",
    [
        "How do you say goodbye in Spanish?",
        "Explain Goodbye Yellow Brick Road",
        "Say goodbye",
        "Don't say goodbye yet",
        "What does take care mean?",
        'The answer is "goodbye".',
        "Good night shifts pay more",
    ],
)
def test_farewell_words_in_questions_do_not_end_session(text):
    assert not is_farewell(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hey Ciel.", ""),
        ("Hey Ciel!", ""),
        ("Ciel?", ""),
        ("Hey Ciel, are you good?", "are you good?"),
        ("hey ciel. Hey Ciel, what's the time?", "what's the time?"),
        ("Cielite art", "Cielite art"),
        ("Tell me about Ciel.", "Tell me about Ciel."),
        ("So Ciel, what's on your mind?", "what's on your mind?"),
        ("Well, um, hey Ciel, what's the time?", "what's the time?"),
        ("So what's good Ciel?", "So what's good Ciel?"),
        ("So what's good?", "So what's good?"),
        ("So I hate Ciel.", "So I hate Ciel."),
    ],
)
def test_wake_phrase_is_removed_with_its_punctuation(text, expected):
    from ciel.conversation import strip_wake_phrase

    assert strip_wake_phrase(text) == expected


def test_custom_wake_phrase_is_removed():
    from ciel.conversation import strip_wake_phrase

    assert strip_wake_phrase("Hey Jarvis. Open the browser.", "hey jarvis") == "Open the browser."
    assert strip_wake_phrase("So computer, open the browser.", "so computer") == "open the browser."


@pytest.mark.parametrize('text', [
    'Ciel is the name of the assistant.', 'Well, Ciel said hello.',
    'Ciel has a new interface.',
])
def test_assistant_name_as_sentence_subject_is_not_stripped_as_a_wake_phrase(text):
    from ciel.conversation import strip_wake_phrase

    assert strip_wake_phrase(text) == text
