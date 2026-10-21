"""The real voice/keyboard callback uses AI choices before speech, tools, or persistence."""

import json
from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.config import InternetConfig
from ciel.internet import InternetResult, InternetService, InternetSource
from ciel.providers.base import LLMResponse
from ciel.providers.router import ModelRouter
from tests.test_ambient import run_callbacks


def _decision(action, **fields):
    return LLMResponse(json.dumps({"action": action, "confidence": 0.99, **fields}), "light", "nim")


def _internet(monkeypatch, results):
    service = MagicMock()
    service.config = InternetConfig()
    service.lookup.side_effect = results
    factory = MagicMock(return_value=service)
    factory._context = InternetService._context
    monkeypatch.setattr("ciel.internet.InternetService", factory)
    return service


@pytest.mark.parametrize("typed", [False, True])
def test_ai_query_and_acknowledgment_precede_search_and_only_final_reply_is_saved(
    tmp_path, monkeypatch, typed,
):
    events = []
    query = "Amber Works season 4 part 2 light novel continuation"
    quote = "Amber Works season 4 part 2 ends at light novel volume 11."
    source = InternetSource("Synthetic adaptation page", "https://example.com/guide", content=quote)
    service = _internet(monkeypatch, [])

    def lookup(*args, **_kwargs):
        events.append("search")
        assert events == ["decision", "ack", "search"]
        assert args[0] == query
        return InternetResult(attempted=True, sources=[source], query=query)

    service.lookup.side_effect = lookup
    router = MagicMock()
    choices = [
        _decision("search", query=query, speech="I'll check that online."),
        LLMResponse(json.dumps({"claims": [{"text": "That part ends at light novel volume 11.",
                                            "source": 1, "quote": quote}]}), "light", "nim"),
    ]

    def send(*_args, **_kwargs):
        events.append("decision" if len(events) == 0 else "report")
        return choices.pop(0)

    router.send.side_effect = send
    tts = MagicMock()

    def speak(text, **kwargs):
        events.append("ack" if text == "I'll check that online." else "final")
        if kwargs.get("on_start"):
            kwargs["on_start"]()
        return True

    tts.speak.side_effect = speak
    info = {"input_source": "keyboard"} if typed else {"wake_verified": True}
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Where does Amber Works season 4 part 2 end?", info)],
        router=router, tts=tts, controller=True,
    )
    try:
        assert events == ["decision", "ack", "search", "report", "final"]
        service.prepare.assert_not_called()
        assistant = [turn for turn in turns if turn.role == "assistant"]
        assert len(assistant) == 1
        assert assistant[0].content == "That part ends at light novel volume 11."
        assert assistant[0].sources == [source.to_dict()]
    finally:
        store.close()


def test_background_is_rejected_before_models_speech_persistence_or_tools(tmp_path, monkeypatch):
    service = _internet(monkeypatch, [])
    router = MagicMock()
    router.send.return_value = _decision("silent")
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Mom, what's for dinner?", {})],
        router=router, controller=True,
    )
    try:
        router.send.assert_not_called()
        assert turns == []
        tts.speak.assert_not_called()
        service.lookup.assert_not_called()
    finally:
        store.close()


def test_active_spoken_followup_recovers_from_false_other_listener(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.side_effect = [
        _decision("answer", intent="greeting"),
        LLMResponse("Hi. What's on your mind?", "chat", "nim"),
        _decision("silent", intent="other_listener"),
        LLMResponse("Tell me what you want to check.", "chat", "nim"),
    ]
    _, tts, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("Hey Ciel.", {"wake_verified": True}),
        ("Can you help me with that?", {}),
    ], router=router, controller=True)
    try:
        assert [turn.content for turn in turns if turn.role == "user"] == [
            "Hey Ciel.", "Can you help me with that?",
        ]
        assert [turn.content for turn in turns if turn.role == "assistant"] == [
            "Hi. What's on your mind?", "Tell me what you want to check.",
        ]
        assert router.send.call_count == 4
        assert [call.kwargs.get("purpose", "conversation")
                for call in router.send.call_args_list] == [
            "controller", "conversation", "controller", "conversation",
        ]
        assert tts.speak.call_count == 2
    finally:
        store.close()


def test_capability_question_goes_to_ai_answer_without_searching_why_cant(tmp_path, monkeypatch):
    service = _internet(monkeypatch, [])
    router = MagicMock()
    router.send.side_effect = [
        _decision("answer"), LLMResponse("The earlier sources didn't establish that detail.",
                                        "chat", "nim"),
    ]
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Why can't you search it on the internet?", {"input_source": "keyboard"})],
        router=router, controller=True,
    )
    try:
        service.prepare.assert_not_called()
        service.lookup.assert_not_called()
        assert turns[-1].content == "The earlier sources didn't establish that detail."
    finally:
        store.close()


def test_canceling_initial_acknowledgment_prevents_all_network_work(tmp_path, monkeypatch):
    canceled = Event()
    service = _internet(monkeypatch, [])
    router = MagicMock()
    router.send.return_value = _decision("search", query="Tensura season 4")
    tts = MagicMock()
    tts.speak.side_effect = lambda *_args, **_kwargs: canceled.set()
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Search Tensura", {"input_source": "keyboard", "cancel_event": canceled})],
        router=router, tts=tts, controller=True,
    )
    try:
        service.lookup.assert_not_called()
        router.send.assert_called_once()
        assert not any(turn.role == "assistant" for turn in turns)
    finally:
        store.close()


def test_explicit_clock_executes_before_ai_selection(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.return_value = _decision("local")
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("What time is it?", {"input_source": "keyboard"})],
        router=router, controller=True,
    )
    try:
        router.send.assert_not_called()
        assert turns[-1].provider == "local" and ":" in turns[-1].content
        tts.speak.assert_called_once()
    finally:
        store.close()


@pytest.mark.parametrize("confirmation,expected", [
    ({"input_source": "keyboard"}, True), ({}, False),
])
def test_ai_cannot_authorize_an_inferred_memory_confirmation(
    tmp_path, monkeypatch, confirmation, expected,
):
    router = MagicMock()
    router.send.side_effect = [
        _decision("local"), _decision("local"),
        LLMResponse("I haven't saved that.", "chat", "nim"),
    ]
    _, _, _, store, _ = run_callbacks(tmp_path, monkeypatch, [
        ("My favorite game is CS2.", {"input_source": "keyboard"}),
        ("yes", confirmation),
    ], router=router, controller=True)
    try:
        assert (store.get_fact("user:favorite_game") is not None) is expected
        if expected:
            router.send.assert_not_called()
        else:
            from ciel.controller import _CONTEXT_MARKER

            last_decision = [call for call in router.send.call_args_list
                             if call.kwargs.get("purpose") == "controller"][-1]
            payload = json.loads(last_decision.args[0][0].content.split(_CONTEXT_MARKER)[1])
            assert payload["pending_local_confirmation"] is True
    finally:
        store.close()


def test_topic_change_clears_pending_memory_before_later_yes(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.side_effect = [
        _decision("answer"), LLMResponse("Hello.", "chat", "nim"), _decision("local"),
    ]
    _, _, _, store, _ = run_callbacks(tmp_path, monkeypatch, [
        ("My favorite game is CS2.", {"input_source": "keyboard"}),
        ("What's up?", {"input_source": "keyboard"}),
        ("yes", {"input_source": "keyboard"}),
    ], router=router, controller=True)
    try:
        assert store.get_fact("user:favorite_game") is None
    finally:
        store.close()


def test_local_session_stays_private_after_restart_and_cannot_search(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.return_value = _decision("local")
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("/local What time is it?", {"input_source": "keyboard"}),
    ], router=router, controller=True)
    assert ":" in turns[-1].content
    router.send.assert_not_called()
    store.close()
    service = _internet(monkeypatch, [])
    router.send.return_value = _decision("search", query="Tensura season 4")
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("Search Tensura season 4", {"input_source": "keyboard"}),
    ], router=router, controller=True)
    try:
        assert all(message.local_only for message in router.send.call_args.args[0])
        service.lookup.assert_not_called()
        assert "private" in turns[-1].content
    finally:
        store.close()


def test_superseded_background_speech_does_not_call_controller(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.return_value = _decision("silent")
    _, tts, _, store, turns = run_callbacks(tmp_path, monkeypatch, [],
        observed=[("Dad, come here", {"superseded": True})],
        router=router, controller=True,
    )
    try:
        router.send.assert_not_called()
        assert turns == []
        tts.speak.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("middle", ["And include examples.", ""])
def test_repeated_superseded_fragments_reach_one_final_controller_turn(
    tmp_path, monkeypatch, private, middle,
):
    first, second, latest = Event(), Event(), Event()
    first.set()
    second.set()
    initial = ("/local " if private else "") + "explain this setting."
    router = MagicMock()
    router.send.side_effect = [
        _decision("answer", intent="follow_up"),
        LLMResponse("Here is the complete explanation.", "chat", "nim"),
    ]
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Keep the answer short.", {
            "cancel_event": latest, "supersedes_cancel_event": second,
        })],
        observed=[
            ("Ciel, " + initial, {"superseded": True, "cancel_event": first}),
            (middle, {
                "superseded": True, "cancel_event": second,
                "supersedes_cancel_event": first,
            }),
        ],
        router=router, controller=True,
    )
    try:
        expected = [initial, *([middle] if middle else []), "Keep the answer short."]
        assert router.send.call_count == 2
        assert router.send.call_args_list[0].args[0][-1].content == " ".join(expected)
        assert router.send.call_args_list[0].kwargs["purpose"] == "controller"
        assert [turn.content for turn in turns if turn.role == "user"] == expected
        if private:
            assert all(
                ModelRouter.requires_local(request.args[0])
                for request in router.send.call_args_list
            )
        tts.speak.assert_called_once()
    finally:
        store.close()


def test_superseded_other_listener_cannot_join_an_addressed_fragment_chain(
    tmp_path, monkeypatch,
):
    first, second = Event(), Event()
    first.set()
    second.set()
    router = MagicMock()
    router.send.side_effect = [_decision("answer"), LLMResponse("A new answer.", "chat", "nim")]
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Ciel, explain something else.", {
            "cancel_event": Event(), "supersedes_cancel_event": second,
        })],
        observed=[
            ("Ciel, explain this setting.", {"superseded": True, "cancel_event": first}),
            ("Actually, Mom, bring the book.", {
                "superseded": True, "cancel_event": second,
                "supersedes_cancel_event": first,
            }),
        ],
        router=router, controller=True,
    )
    try:
        assert router.send.call_args_list[0].args[0][-1].content == "explain something else."
        assert [turn.content for turn in turns if turn.role == "user"] == [
            "explain something else.",
        ]
    finally:
        store.close()


@pytest.mark.parametrize("private", [False, True])
def test_active_nameless_superseded_followup_is_gated_as_one_complete_turn(
    tmp_path, monkeypatch, private,
):
    canceled = Event()
    canceled.set()
    greeting = "Ciel, " + ("/local " if private else "") + "hello."
    router = MagicMock()
    router.send.side_effect = [
        _decision("answer", intent="greeting"), LLMResponse("Hello.", "chat", "nim"),
        _decision("answer", intent="follow_up"), LLMResponse("A short explanation.", "chat", "nim"),
    ]
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [(greeting, {}), ("And keep it brief.", {
            "cancel_event": Event(), "supersedes_cancel_event": canceled,
        })],
        observed_after={0: [("Explain this setting.", {
            "superseded": True, "cancel_event": canceled,
        })]},
        router=router, controller=True,
    )
    try:
        request = router.send.call_args_list[2]
        assert request.kwargs["purpose"] == "controller"
        assert request.args[0][-1].content == "Explain this setting. And keep it brief."
        assert '"direct_address": false' in " ".join(message.content for message in request.args[0])
        assert [turn.content for turn in turns if turn.role == "user"] == [
            greeting.removeprefix("Ciel, "), "Explain this setting.", "And keep it brief.",
        ]
        if private:
            assert all(
                ModelRouter.requires_local(call.args[0]) for call in router.send.call_args_list
            )
        assert tts.speak.call_count == 2
    finally:
        store.close()


@pytest.mark.parametrize("case", ["expired", "other_listener", "unclear", "unlinked"])
def test_superseded_unverified_background_cannot_join_a_new_request(
    tmp_path, monkeypatch, case,
):
    from ciel.audio.ambient import AmbientConversation

    canceled = Event()
    canceled.set()
    if case == "expired":
        replied = AmbientConversation.replied

        def expire(self):
            replied(self)
            self.deadline = 0

        monkeypatch.setattr(AmbientConversation, "replied", expire)
    fragment = "Mom, bring the book." if case == "other_listener" else "Explain this setting."
    info = {"superseded": True, "cancel_event": canceled}
    if case == "unclear":
        info.update(stt_needs_repeat=True, stt_raw_text="uncertain speech")
    router = MagicMock()
    router.send.side_effect = [
        _decision("answer", intent="greeting"), LLMResponse("Hello.", "chat", "nim"),
        _decision("answer"), LLMResponse("A separate answer.", "chat", "nim"),
    ]
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Ciel, hello.", {}), ("Ciel, explain something else.", {
            "cancel_event": Event(),
            "supersedes_cancel_event": Event() if case == "unlinked" else canceled,
        })],
        observed_after={0: [(fragment, info)]}, router=router, controller=True,
    )
    try:
        assert router.send.call_args_list[2].args[0][-1].content == "explain something else."
        assert [turn.content for turn in turns if turn.role == "user"] == [
            "hello.", "explain something else.",
        ]
    finally:
        store.close()


def test_retained_nameless_followup_does_not_authorize_saving_memory(tmp_path, monkeypatch):
    canceled = Event()
    canceled.set()
    router = MagicMock()
    router.send.side_effect = [
        _decision("answer", intent="greeting"), LLMResponse("Hello.", "chat", "nim"),
        _decision("local"),
    ]
    _, tts, _, store, _turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Ciel, hello.", {}), ("Remember my favorite game is CS2.", {
            "cancel_event": Event(), "supersedes_cancel_event": canceled,
        })],
        observed_after={0: [("My favorite game changed.", {
            "superseded": True, "cancel_event": canceled,
        })]},
        router=router, controller=True,
    )
    try:
        assert store.get_fact("user:favorite_game") is None
        assert "I haven't saved that" in tts.speak.call_args.args[0]
        assert router.send.call_args.args[0][-1].content == (
            "My favorite game changed. Remember my favorite game is CS2."
        )
    finally:
        store.close()


def test_stopped_followup_fragments_cannot_be_revived_by_later_speech(tmp_path, monkeypatch):
    canceled, rejected = Event(), Event()
    canceled.set()
    router = MagicMock()
    router.send.side_effect = [
        _decision("answer", intent="greeting"), LLMResponse("Hello.", "chat", "nim"),
        _decision("answer"), LLMResponse("A separate answer.", "chat", "nim"),
    ]

    def utterances():
        yield "Ciel, hello.", {}
        yield "Stop.", {
            "cancel_event": rejected, "supersedes_cancel_event": canceled,
        }
        rejected.set()
        yield "Ciel, explain something else.", {
            "cancel_event": Event(), "supersedes_cancel_event": rejected,
        }

    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, utterances(),
        observed_after={0: [("Explain this setting.", {
            "superseded": True, "cancel_event": canceled,
        })]},
        router=router, controller=True,
    )
    try:
        assert router.send.call_args_list[2].args[0][-1].content == "explain something else."
        assert router.send.call_count == 4
        assert [turn.content for turn in turns if turn.role == "user"] == [
            "hello.", "explain something else.",
        ]
    finally:
        store.close()


def test_legacy_gate_does_not_inherit_unclassified_followup_fragments(tmp_path, monkeypatch):
    canceled = Event()
    canceled.set()
    router = MagicMock()
    router.send.return_value = LLMResponse("A separate answer.", "chat", "nim")
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Hey Ciel.", {}), ("Ciel, explain something else.", {
            "cancel_event": Event(), "supersedes_cancel_event": canceled,
        })],
        observed_after={0: [("Explain this setting.", {
            "superseded": True, "cancel_event": canceled,
        })]},
        router=router, controller=False,
    )
    try:
        router.send.assert_called_once()
        assert "Explain this setting." not in " ".join(
            message.content for message in router.send.call_args.args[0]
        )
        assert all(turn.content != "Explain this setting." for turn in turns)
    finally:
        store.close()


def test_controller_reply_streams_and_keeps_transcription_context(tmp_path, monkeypatch):
    from ciel.providers.base import LLMStreamChunk

    router, tts = MagicMock(), MagicMock()
    router.send.return_value = _decision("answer")
    router.stream.return_value = iter([LLMStreamChunk("A concise answer.", "chat", "nim")])
    tts.supports_sentence_pipeline = False
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("Ciel, explain it", {"wake_verified": True}),
    ], router=router, tts=tts, controller=True, streaming=True)
    try:
        router.stream.assert_called_once()
        assert "transcribed from speech" in router.stream.call_args.args[0][0].content
        assert turns[-1].content == "A concise answer."
        assert turns[-1].provider == "nim"
    finally:
        store.close()


def test_logged_garbled_request_then_greeting_never_executes_old_search(
    tmp_path, monkeypatch, capsys,
):
    # Local-only regression using the reported utterance; no real providers or database.
    garbled = ('can you search the Tensor Anime Assistant for finale continuation '
               'of On the Light normal?')
    service = _internet(monkeypatch, [])
    router = MagicMock()
    router.send.side_effect = [
        _decision('clarify', intent='unclear', speech='Could you repeat the exact title?'),
        _decision('search', intent='greeting', query='Tensor Anime Assistant'),
        LLMResponse('Hello! What can I help with?', 'chat', 'nim'),
        _decision('clarify', intent='unclear', speech='Should I retry, or explain the background?'),
        LLMResponse('Should I retry, or explain the background?', 'chat', 'nim'),
    ]
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        (garbled, {'wake_verified': True}),
        ('Hello Ciel.', {'wake_verified': True}),
        ('Okay, sure, what is it?', {'input_source': 'keyboard'}),
    ], router=router, controller=True)
    try:
        service.lookup.assert_not_called()
        replies = [turn.content for turn in turns if turn.role == 'assistant']
        assert len(replies) == 3 and 'repeat' in replies[0]
        assert replies[1] == 'Hello! What can I help with?'
        assert replies[2] == 'Should I retry, or explain the background?'
        assert garbled in capsys.readouterr().out
        prompt = router.send.call_args_list[2].args[0][0].content
        assert 'current message is a greeting' in prompt
    finally:
        store.close()


@pytest.mark.parametrize("failure", [
    RuntimeError("NIM is not configured"), TimeoutError("offline"),
])
@pytest.mark.parametrize("text", ["What time is it?", "/local What time is it?"])
def test_explicit_clock_needs_no_controller_when_provider_is_unavailable(
    tmp_path, monkeypatch, failure, text,
):
    router = MagicMock()
    router.send.side_effect = failure
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [(text, {"input_source": "keyboard"})],
        router=router, controller=True,
    )
    try:
        assert turns[-1].provider == "local" and ":" in turns[-1].content
        router.send.assert_not_called()
    finally:
        store.close()


def test_explicit_memory_and_confirmation_need_no_controller(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.side_effect = RuntimeError("NIM is not configured")
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("My favorite game is CS2.", {"input_source": "keyboard"}),
        ("yes", {"input_source": "keyboard"}),
        ("What's my favorite game?", {"input_source": "keyboard"}),
    ], router=router, controller=True)
    try:
        assert store.get_fact("user:favorite_game").metadata["value"] == "CS2"
        assert "CS2" in turns[-1].content
        router.send.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize("info", [
    {"input_source": "keyboard"}, {"wake_verified": True, "stt_confidence": 0.99},
])
def test_explicit_app_launch_uses_allowlist_without_controller(tmp_path, monkeypatch, info):
    from ciel.actions.open_app import OpenAppAction

    execute = MagicMock(return_value="I've sent the launch request for discord.")
    monkeypatch.setattr(OpenAppAction, "execute", execute)
    router = MagicMock()
    router.send.side_effect = TimeoutError("offline")
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Open Discord", info)], router=router, controller=True,
    )
    try:
        execute.assert_called_once()
        assert execute.call_args.args[0] == {"app": "discord"}
        assert "launch request" in turns[-1].content
        router.send.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize("info", [
    {}, {"wake_verified": True, "stt_confidence": 0.1},
    {"input_source": "keyboard", "cancel_event": Event()},
])
def test_local_preflight_preserves_address_confidence_and_cancellation(
    tmp_path, monkeypatch, info,
):
    from ciel.actions.open_app import OpenAppAction

    if "cancel_event" in info:
        info["cancel_event"].set()
    execute = MagicMock()
    monkeypatch.setattr(OpenAppAction, "execute", execute)
    router = MagicMock()
    router.send.return_value = _decision("local")
    _, _, _, store, _ = run_callbacks(
        tmp_path, monkeypatch, [("Open Discord", info)], router=router, controller=True,
    )
    try:
        execute.assert_not_called()
        if "cancel_event" in info or not info:
            router.send.assert_not_called()
        else:
            router.send.assert_called_once()
    finally:
        store.close()


def test_local_preflight_does_not_expand_app_allowlist(tmp_path, monkeypatch):
    from ciel.actions.open_app import OpenAppAction

    execute = MagicMock()
    monkeypatch.setattr(OpenAppAction, "execute", execute)
    router = MagicMock()
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Open /bin/sh --help", {"input_source": "keyboard"})],
        router=router, controller=True,
    )
    try:
        execute.assert_not_called()
        router.send.assert_not_called()
        assert "Please name one of those applications" in turns[-1].content
    finally:
        store.close()


def test_local_only_session_does_not_schedule_provider_summaries(tmp_path, monkeypatch):
    from ciel.memory import ConversationManager

    schedule = MagicMock(return_value=False)
    monkeypatch.setattr(ConversationManager, "schedule_summary", schedule)
    router = MagicMock()
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("What time is it?", {"input_source": "keyboard"}) for _ in range(10)],
        router=router, controller=True,
    )
    try:
        assert len(store.get_recent_turns(turns[-1].session_id, limit=100)) == 20
        router.send.assert_not_called()
        schedule.assert_not_called()
    finally:
        store.close()


def test_compound_clock_request_keeps_controller_routing(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.side_effect = [_decision("answer"), LLMResponse("Time zones vary.", "chat", "nim")]
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("What time is it and explain time zones?", {"input_source": "keyboard"})],
        router=router, controller=True,
    )
    try:
        assert router.send.call_args_list[0].kwargs["purpose"] == "controller"
        assert turns[-1].content == "Time zones vary."
    finally:
        store.close()


@pytest.mark.parametrize("topic", [
    "Do you have memory?", "What time is it?", "Can you change your style?",
    "Be more playful", "Listen continuously", "Open Discord",
])
def test_local_topic_change_clears_pending_memory(tmp_path, monkeypatch, topic):
    from ciel.actions.open_app import OpenAppAction

    monkeypatch.setattr(OpenAppAction, "execute", MagicMock(return_value="Launch requested."))
    router = MagicMock()
    router.send.return_value = _decision("local")
    _, _, _, store, _ = run_callbacks(tmp_path, monkeypatch, [
        ("My favorite game is CS2.", {"input_source": "keyboard"}),
        (topic, {"input_source": "keyboard"}),
        ("yes", {"input_source": "keyboard"}),
    ], router=router, controller=True, persona_file=tmp_path / "persona.txt")
    try:
        assert store.get_fact("user:favorite_game") is None
    finally:
        store.close()


@pytest.mark.parametrize("topic", [
    "Do you have memory?", "What time is it?", "My favorite game is CS2.",
    "Listen continuously", "Open Discord",
])
def test_local_topic_change_clears_pending_persona(tmp_path, monkeypatch, topic):
    from ciel.actions.open_app import OpenAppAction

    monkeypatch.setattr(OpenAppAction, "execute", MagicMock(return_value="Launch requested."))
    persona = tmp_path / "persona.txt"
    router = MagicMock()
    router.send.return_value = _decision("local")
    _, _, _, store, _ = run_callbacks(tmp_path, monkeypatch, [
        ("Can you change your style?", {"input_source": "keyboard"}),
        (topic, {"input_source": "keyboard"}),
        ("more playful", {"input_source": "keyboard"}),
    ], router=router, controller=True, persona_file=persona)
    try:
        assert not persona.exists()
    finally:
        store.close()


@pytest.mark.parametrize("topic", [
    "Do you have memory?", "What time is it?", "My favorite game is CS2.",
    "Can you change your style?", "Be more playful", "Listen continuously",
])
def test_local_topic_change_clears_pending_action(tmp_path, monkeypatch, topic):
    from ciel.actions import ActionRegistry
    from ciel.actions.open_app import OpenAppAction

    action = OpenAppAction()
    action.requires_confirmation = True
    execute = MagicMock(return_value="Launch requested.")
    monkeypatch.setattr(action, "execute", execute)
    monkeypatch.setattr(ActionRegistry, "discover", lambda: ActionRegistry([action]))
    router = MagicMock()
    router.send.return_value = _decision("local")
    _, _, _, store, _ = run_callbacks(tmp_path, monkeypatch, [
        ("Open Discord", {"input_source": "keyboard"}),
        (topic, {"input_source": "keyboard"}),
        ("yes", {"input_source": "keyboard"}),
    ], router=router, controller=True, persona_file=tmp_path / "persona.txt")
    try:
        execute.assert_not_called()
    finally:
        store.close()


def test_local_clock_turn_invalidates_recent_app_reference(tmp_path, monkeypatch):
    from ciel.actions.open_app import OpenAppAction

    execute = MagicMock(return_value="Launch requested.")
    monkeypatch.setattr(OpenAppAction, "execute", execute)
    router = MagicMock()
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("Open Discord", {"input_source": "keyboard"}),
        ("What time is it?", {"input_source": "keyboard"}),
        ("Open it", {"input_source": "keyboard"}),
    ], router=router, controller=True)
    try:
        execute.assert_called_once()
        assert "Which application" in turns[-1].content
        router.send.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize("text", ["Remember.", "Remember that.", "Remember it.", "Note that."])
def test_bare_remember_prompt_needs_no_controller(tmp_path, monkeypatch, text):
    router = MagicMock()
    router.send.side_effect = TimeoutError("offline")
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [(text, {"input_source": "keyboard"})],
        router=router, controller=True,
    )
    try:
        assert "What would you like me to remember" in turns[-1].content
        router.send.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize("replace_request,expected_app", [(False, "discord"), (True, "firefox")])
def test_local_action_confirmation_preserves_only_current_arguments(
    tmp_path, monkeypatch, replace_request, expected_app,
):
    from ciel.actions import ActionRegistry
    from ciel.actions.open_app import OpenAppAction

    action = OpenAppAction()
    action.requires_confirmation = True
    execute = MagicMock(return_value="Launch requested.")
    monkeypatch.setattr(action, "execute", execute)
    monkeypatch.setattr(ActionRegistry, "discover", lambda: ActionRegistry([action]))
    utterances = [("Open Discord", {"input_source": "keyboard"})]
    if replace_request:
        utterances.append(("Open Firefox", {"input_source": "keyboard"}))
    utterances.append(("yes", {"input_source": "keyboard"}))
    router = MagicMock()
    _, _, _, store, _ = run_callbacks(
        tmp_path, monkeypatch, utterances, router=router, controller=True,
    )
    try:
        execute.assert_called_once()
        assert execute.call_args.args[0] == {"app": expected_app}
        router.send.assert_not_called()
    finally:
        store.close()


def test_local_persona_followup_keeps_its_own_pending_prompt(tmp_path, monkeypatch):
    persona = tmp_path / "persona.txt"
    router = MagicMock()
    _, _, _, store, _ = run_callbacks(tmp_path, monkeypatch, [
        ("Can you change your style?", {"input_source": "keyboard"}),
        ("more playful", {"input_source": "keyboard"}),
    ], router=router, controller=True, persona_file=persona)
    try:
        assert "more playful" in persona.read_text()
        router.send.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize("correction", [
    "My favorite game is Standoff 2.", "No, like my my favorite game is Standoff 2.",
])
def test_active_memory_correction_reports_no_write_instead_of_model_promise(
    tmp_path, monkeypatch, correction,
):
    router = MagicMock()
    router.send.side_effect = [
        _decision("local"), LLMResponse("I'll update my memory.", "chat", "nim"),
    ]
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("What time is it?", {"input_source": "keyboard"}),
        (correction, {}),
    ], router=router, controller=True)
    try:
        assert store.get_fact("user:favorite_game") is None
        assert "haven't saved" in turns[-1].content
        assert "Ciel" in turns[-1].content
        assert router.send.call_count == 1
    finally:
        store.close()


def test_superseded_verified_speech_never_waits_for_provider_and_cannot_block_typing(
    tmp_path, monkeypatch,
):
    router = MagicMock()
    router.send.side_effect = TimeoutError("offline")
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("What time is it?", {"input_source": "keyboard"})],
        observed=[("Hey Ciel, remember my old request", {
            "superseded": True, "wake_verified": True, "cancel_event": Event(),
        })], router=router, controller=True,
    )
    try:
        router.send.assert_not_called()
        assert turns[-1].provider == "local" and ":" in turns[-1].content
        assert [turn.content for turn in turns if turn.role == "user"] == ["What time is it?"]
    finally:
        store.close()
