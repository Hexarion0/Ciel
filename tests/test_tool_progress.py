"""Tool lead-ins describe intended work and cannot authorize or fabricate results."""

import json
import threading
import time
from unittest.mock import Mock

from ciel.providers.base import LLMResponse
from ciel.providers.router import ModelRouter
from ciel.tool_progress import ToolProgress, valid_line


def router_with(lines):
    router = Mock()
    router.send.return_value = LLMResponse(json.dumps({
        "screen_anime": lines, "screen": lines,
    }), "small", "nim")
    return router


def test_ai_variants_fit_intent_hide_original_query_and_do_not_repeat():
    router = router_with(["Lemme see which anime is on your screen.",
                          "Let me look at what you're watching.",
                          "I'll check the anime on your screen."])
    progress = ToolProgress(router)
    progress.warm()
    assert progress._jobs[False][1].wait(0.5)
    first = progress.line("screen", "Identify the anime; private-name-should-not-leave")
    second = progress.line("screen", "What anime am I watching?")
    assert first != second
    assert router.send.call_count == 1
    sent = router.send.call_args.args[0]
    assert "private-name-should-not-leave" not in str(sent)
    assert "anime" in sent[-1].content
    assert router.send.call_args.kwargs["purpose"] == "tool_progress"
    progress.close()


def test_proposed_ai_search_speech_reused_without_an_extra_request():
    router = Mock()
    progress = ToolProgress(router)
    assert progress.line("search", "News?", proposed="I'll check the latest reporting.") == (
        "I'll check the latest reporting.")
    router.send.assert_not_called()


def test_private_intent_still_requires_local_routing():
    router = router_with(["Lemme see.", "Let me check your screen."])
    progress = ToolProgress(router)
    progress.warm(local_only=True)
    assert progress._jobs[True][1].wait(0.5)
    progress.line("screen", "Look at screen", local_only=True)
    assert router.send.call_args.args[0][-1].local_only
    progress.close()


def test_canceled_progress_neither_calls_model_nor_emits():
    canceled = threading.Event()
    canceled.set()
    router, emit = Mock(), Mock()
    assert not ToolProgress(router).announce("screen", "Look", emit, cancel_event=canceled)
    router.send.assert_not_called()
    emit.assert_not_called()


def test_cancel_during_generation_suppresses_emission_and_tool_start():
    canceled = threading.Event()
    router, emit = Mock(), Mock()
    def generate(*_, **__):
        canceled.set()
        return LLMResponse('{"screen":["Lemme see.","Let me look."]}', "small", "nim")
    router.send.side_effect = generate
    progress = ToolProgress(router)
    progress.warm()
    assert progress._jobs[False][1].wait(0.5)
    assert not progress.announce("screen", "Look", emit, cancel_event=canceled)
    emit.assert_not_called()
    progress.close()


def test_slow_generator_is_bounded_and_fallback_still_fits_context():
    router = Mock()
    ended = threading.Event()
    def generate(*_, cancel_event=None, **__):
        while not cancel_event.is_set():
            time.sleep(0.005)
        ended.set()
        raise TimeoutError
    router.send.side_effect = generate
    began = time.monotonic()
    progress = ToolProgress(router, budget=0.03)
    line = progress.line("screen", "What anime am I watching?")
    assert time.monotonic() - began < 0.3
    assert "screen" in line and ("anime" in line or "watching" in line)
    assert ended.wait(0.3)
    progress.close()


def test_completed_work_claims_are_not_valid_progress():
    assert not valid_line("I've found the anime.")
    assert not valid_line("I'll say I've captured your screen.")
    assert not valid_line("I'll open a browser and delete files.")
    assert valid_line("Lemme take a look.")


def test_background_wording_yields_to_foreground_conversation():
    manager = Mock()
    for name in ("nim", "groq", "openrouter", "ollama"):
        getattr(manager, name).is_configured.return_value = name == "nim"
    started = threading.Event()
    def stream(*_, cancel_event=None, **__):
        started.set()
        while not cancel_event.is_set():
            time.sleep(0.005)
        yield from ()
    manager.stream_with_fallback.side_effect = stream
    router = ModelRouter(manager)
    progress = ToolProgress(router)
    progress.warm()
    try:
        assert started.wait(0.5)
        with router.priority.foreground():
            assert progress._jobs[False][1].wait(0.5)
        assert not progress._variants
    finally:
        progress.close()
