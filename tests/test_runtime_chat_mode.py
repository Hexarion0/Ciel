"""Desktop typed chat and spoken requests use separate delivery contracts."""

from unittest.mock import MagicMock

import pytest

from ciel.providers.base import LLMResponse, LLMStreamChunk
from tests.test_ambient import run_callbacks


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("controller", [False, True])
def test_typed_chat_has_room_for_code_while_voice_stays_brief(
    tmp_path, monkeypatch, capsys, streaming, typed, controller,
):
    monkeypatch.setenv("INTERNET_ENABLED", "false")
    monkeypatch.setenv("MEMORY_CONSOLIDATION_ENABLED", "false")
    monkeypatch.setenv("MEMORY_FLUSH_ENABLED", "false")
    answer = ("Start with a small function.\n\n```python\ndef example():\n    return 42\n```"
              if typed else "Start with a small function.")
    router = MagicMock()
    router.send.return_value = LLMResponse(answer, "synthetic-550b", "nim")
    router.stream.return_value = iter([LLMStreamChunk(answer, "synthetic-550b", "nim")])
    info = {"input_source": "keyboard"} if typed else {"wake_verified": True}
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Explain a Python function with an example.", info)],
        router=router, streaming=streaming, controller=controller,
    )
    try:
        request = router.stream.call_args if streaming else router.send.call_args
        assert request.kwargs["max_tokens"] == (4096 if typed else 400)
        prompt = request.args[0][0].content
        if typed:
            assert "Text output:" in prompt and "Spoken output:" not in prompt
            assert "one to three short spoken sentences" not in prompt
            assert "```python\ndef example():\n    return 42\n```" in turns[-1].content
            tts.speak.assert_called_once()
            assert "Start with a small function." in tts.speak.call_args.args[0]
            assert "def example" not in tts.speak.call_args.args[0]
            tts.begin_stream.assert_not_called()
            assert capsys.readouterr().out.count("def example():") == 1
        else:
            assert "Usually use one to three short spoken sentences" in prompt
            assert "Spoken output:" in prompt
            assert tts.speak.called
        assert turns[-1].content == answer
        assert turns[-1].model == "synthetic-550b"
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_failed_typed_playback_keeps_the_answer_without_a_second_reply(
    tmp_path, monkeypatch, capsys, controller,
):
    monkeypatch.setenv("INTERNET_ENABLED", "false")
    monkeypatch.setenv("MEMORY_CONSOLIDATION_ENABLED", "false")
    monkeypatch.setenv("MEMORY_FLUSH_ENABLED", "false")
    answer = "The written answer remains available when playback fails."
    router = MagicMock()
    router.send.return_value = LLMResponse(answer, "synthetic", "nim")
    tts = MagicMock()
    tts.speak.side_effect = RuntimeError("Synthetic output device unavailable")
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Explain this.", {"input_source": "keyboard"})],
        router=router, tts=tts, controller=controller,
    )
    try:
        assert [turn.content for turn in turns if turn.role == "assistant"] == [answer]
        assert capsys.readouterr().out.count(answer) == 1
        tts.speak.assert_called_once()
        router.send.assert_called_once()
    finally:
        store.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("controller", [False, True])
def test_long_typed_answer_speaks_a_brief_excerpt_and_retains_full_text(
    tmp_path, monkeypatch, streaming, controller,
):
    monkeypatch.setenv("INTERNET_ENABLED", "false")
    monkeypatch.setenv("MEMORY_CONSOLIDATION_ENABLED", "false")
    monkeypatch.setenv("MEMORY_FLUSH_ENABLED", "false")
    answer = ("This is the first supported point. "
              + "This explanation continues in chat. " * 50).strip()
    router = MagicMock()
    router.send.return_value = LLMResponse(answer, "synthetic", "nim")
    router.stream.return_value = iter([LLMStreamChunk(answer, "synthetic", "nim")])
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Explain this in detail.", {"input_source": "keyboard"})],
        router=router, streaming=streaming, controller=controller,
    )
    try:
        assert turns[-1].content == answer
        tts.speak.assert_called_once()
        spoken = tts.speak.call_args.args[0]
        assert "first supported point" in spoken
        assert "full answer is in the chat" in spoken
        assert len(spoken.split()) <= 80
        request = router.stream.call_args if streaming else router.send.call_args
        assert request.kwargs["max_tokens"] == 4096
        assert router.stream.call_count + router.send.call_count == 1
    finally:
        store.close()
