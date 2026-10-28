"""A typed medium correction completes an admitted spoken research request."""

import json
from unittest.mock import MagicMock

from ciel.internet import InternetResult, InternetSource
from ciel.providers.base import LLMResponse
from ciel.tool_progress import valid_line
from tests.test_ambient import run_callbacks
from tests.test_controller_runtime import _internet, _main_responses, _tool_call


def test_typed_lightnovel_correction_keeps_spoken_title_scope_and_requested_answer(
    tmp_path, monkeypatch,
):
    quote = "Amber Harbor anime episode 12 ends at volume 3. Continue reading from volume 4."
    source = InternetSource("Adaptation guide", "https://example.org/guide", content=quote)
    service = _internet(monkeypatch, [InternetResult(attempted=True, sources=[source])])
    router = MagicMock()
    responses = iter([
        LLMResponse("Did you mean the light novel or the manga?", "test", "test"),
        _tool_call("web_search", query="Amber Harbor episode 12 light novel continuation"),
        LLMResponse(json.dumps({"claims": [{
            "text": "Continue reading from volume 4.", "source": 1, "quote": quote,
        }]}), "test", "test"),
    ])
    router.send.side_effect = _main_responses(lambda *_args, **_kwargs: next(responses))

    _, tts, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("Ciel, search Amber Harbor episode 12 anime continuation on the light node.",
         {"wake_verified": True}),
        ("lightnovel contunation", {"input_source": "keyboard"}),
    ], router=router, controller=True)
    try:
        service.lookup.assert_called_once()
        objective = service.lookup.call_args.kwargs["objective"]
        assert "Amber Harbor" in objective and "12" in objective and "light novel" in objective
        assistant = [turn for turn in turns if turn.role == "assistant"]
        assert assistant[-1].content == "Continue reading from volume 4. [1]"
        assert assistant[-1].sources == [source.to_dict()]
        assert any(quote in message.content for call in router.send.call_args_list
                   for message in call.args[0])
        assert len(assistant) == 2
        assert "exact name" not in assistant[-1].content
        spoken = [call.args[0] for call in tts.speak.call_args_list]
        assert len(spoken) == 3
        assert spoken[0] == assistant[0].content
        assert valid_line(spoken[1]) == spoken[1]
        assert spoken[2] == "Continue reading from volume 4."
        assert sum(call.kwargs.get("purpose", "conversation") == "conversation"
                   for call in router.send.call_args_list) == 3
    finally:
        store.close()
