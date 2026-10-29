"""Conversation-shaped replays exercise real entry points and synthetic read boundaries.

The scripted model deliberately proposes underspecified or stale search keywords. The
application must retain the owner's task itself; a correct query in a mock is not evidence
of successful context resolution. These tests establish application behavior, not hosted
model comprehension (the opt-in evaluation script measures that separately).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock

import pytest

from ciel.chat.session import ChatSession
from ciel.config import Settings
from ciel.internet import InternetResult, InternetSource
from ciel.memory import ConversationManager, MemoryItem, MemoryType
from ciel.providers.base import LLMResponse
from ciel.research import evidence_payload
from tests.test_controller_runtime import _internet, _main_responses, _tool_call
from tests.test_screen_integration import install_screen, run_callbacks


@dataclass(frozen=True)
class SearchReplay:
    """Owner dialogue plus externally observable search and answer requirements."""

    name: str
    history: tuple[tuple[str, str], ...]
    current: str
    planned: str
    required: tuple[str, ...]
    forbidden: tuple[str, ...] = ()
    subject: str = "Amber Harbor"
    fact: str = ("Amber Harbor anime episode 12 ends at light novel volume 3. "
                 "Continue reading from light novel volume 4.")
    rewrite: str = ""


REPLAYS = (
    SearchReplay(
        "accepted_offer_keeps_corrected_minecraft_mod",
        (("user", "Hey Ciel, do you know about Tinsura anime mod in Minecraft?"),
         ("assistant", "I'm not familiar with that mod."),
         ("user", "I meant Tensura."),
         ("assistant", "I haven't heard of a Tensura mod for Minecraft either. "
          "Want me to look it up?")),
        "Yes, sure.", "Yes, sure.", ("Tensura", "mod", "Minecraft"),
        ("Tinsura", "Yes, sure"), "Tensura",
        "Tensura is a Minecraft mod inspired by the anime series.",
        rewrite="Tensura Minecraft mod CurseForge",
    ),
    SearchReplay(
        "tensorflow_to_tensura_overview",
        (("user", "Hey Ciel, what's up? Can you search TensorFlow for me?"),
         ("assistant", "What do you want to know about TensorFlow?")),
        "Tensura*", "TensorFlow machine learning library", ("Tensura",),
        ("TensorFlow", "season 3"), "Tensura", "Tensura is a fantasy anime series.",
        rewrite="Tensura overview",
    ),
    SearchReplay(
        "corrected_title_keeps_episode_and_medium",
        (("user", "Search where Amber Harbour episode 12 continues in the light novel."),
         ("assistant", "Could you spell the title?")),
        "Amber Harbor*", "Amber Harbour", ("Amber Harbor", "12", "light novel"),
        ("Amber Harbour",),
        rewrite="Amber Harbor episode 12 light novel continuation",
    ),
    SearchReplay(
        "light_node_to_light_novel",
        (("user", "Search Amber Harbor episode 12 continuation on the light node."),
         ("assistant", "Did you mean the light novel or the manga?")),
        "light novel continuation", "Amber Harbor anime",
        ("Amber Harbor", "12", "light novel"),
    ),
    SearchReplay(
        "latest_season_replaces_old_scope",
        (("user", "Search Amber Harbor season 4 anime to light novel continuation."),
         ("assistant", "Which season did you finish?")),
        "Actually season 5, the light novel.", "Amber Harbor anime",
        ("Amber Harbor", "season 5", "light novel"), ("season 4",),
        fact=("Amber Harbor anime season 5 ends at light novel volume 3. "
              "Continue reading from light novel volume 4."),
    ),
    SearchReplay(
        "title_answer_completes_pending_question",
        (("user", "Which novel volume should I read after anime episode 12?"),
         ("assistant", "Which series do you mean?")),
        "Amber Harbor", "Amber Harbor", ("Amber Harbor", "12", "novel"),
    ),
    SearchReplay(
        "retry_keeps_corrected_title",
        (("user", "Search Amber Harbour episode 12 light novel continuation."),
         ("assistant", "Could you spell the title?"), ("user", "Amber Harbor*"),
         ("assistant", "The lookup timed out. Please try again.")),
        "Try again.", "Amber Harbor", ("Amber Harbor", "12", "light novel"),
        ("Amber Harbour",),
    ),
    SearchReplay(
        "continue_keeps_medium_repair",
        (("user", "Search Amber Harbor episode 12 continuation on the light node."),
         ("assistant", "Did you mean the light novel or the manga?"),
         ("user", "The light novel."),
         ("assistant", "I was checking the adaptation guide. [Playback was interrupted.]")),
        "Continue.", "Amber Harbor", ("Amber Harbor", "12", "light novel"),
    ),
    SearchReplay(
        "assistant_invention_is_not_owner_scope",
        (("user", "Search Amber Harbor anime to light novel continuation."),
         ("assistant", "Violet Castle season 9 episode 92 is the last episode.")),
        "And where does it continue in the novel?", "Amber Harbor novel",
        ("Amber Harbor", "novel"), ("Violet Castle", "season 9", "92"),
    ),
    SearchReplay(
        "new_named_topic_replaces_old_task",
        (("user", "Search Amber Harbor anime to light novel continuation."),
         ("assistant", "The guide was unavailable.")),
        "Search CedarDB Linux support instead.", "CedarDB",
        ("CedarDB", "Linux"), ("Amber Harbor", "light novel"),
        "CedarDB", "CedarDB supports Linux desktop systems.",
    ),
    SearchReplay(
        "broad_request_gets_supported_overview",
        (), "Can you search CedarDB for me?", "CedarDB", ("CedarDB",),
        ("release date",), "CedarDB", "CedarDB is a database for analytical queries.",
    ),
)


def _settings(tmp_path) -> Settings:
    """Use no production settings, persona, workspace, or conversation database."""
    return Settings(
        _env_file=None, memory_db_path=str(tmp_path / "replays.db"),
        memory_workspace_path=str(tmp_path / "notes"), memory_workspace_enabled=False,
        memory_semantic_enabled=False, memory_consolidation_enabled=False,
        memory_recall_enabled=False, ciel_persona_file="", internet_enabled=True,
        chat_services="", controller_max_tool_steps=3,
    )


def _seed(store, session_id, history):
    conversation = ConversationManager(store, session_id=session_id)
    for role, text in history:
        conversation.add_turn(role, text, origin="owner" if role == "user" else "")
    conversation.close()


def _source(fact: str) -> InternetSource:
    return InternetSource("Synthetic reference guide", "https://example.org/reference",
                          content=fact)


def _report(fact: str) -> LLMResponse:
    return LLMResponse(json.dumps({"claims": [{
        "text": fact, "quote": fact, "source": 1,
    }]}), "synthetic-chat", "fixture")


def _router(responses):
    router = MagicMock()
    iterator = iter(responses)

    def respond(_messages, **kwargs):
        response = next(iterator)
        return response

    router.send.side_effect = _main_responses(respond)
    return router


@pytest.mark.parametrize("entrypoint", ["desktop", "chat"])
@pytest.mark.parametrize("case", REPLAYS, ids=lambda case: case.name)
def test_scoped_search_replay_reaches_real_boundary_and_supported_final(
    tmp_path, monkeypatch, entrypoint, case,
):
    source = _source(case.fact)
    internet = _internet(monkeypatch, [InternetResult(attempted=True, sources=[source])])
    router = _router([
        _tool_call("web_search", query=case.planned),
        *([_tool_call("web_search", query=case.rewrite)] if case.rewrite else []),
        _report(case.fact),
    ])
    if entrypoint == "desktop":
        _, tts, _, store, turns = run_callbacks(
            tmp_path, monkeypatch, [(case.current, {"input_source": "keyboard"})],
            router=router, controller=True,
            initialize_conversation=lambda store, session: _seed(store, session, case.history),
        )
        close = store.close
        assert tts.speak.call_count == 2
        assert case.fact in tts.speak.call_args.args[0]
        tts.begin_stream.assert_not_called()
    else:
        session = ChatSession(_settings(tmp_path), channel="telegram", session_id="replay-owner",
                              router=router)
        for role, text in case.history:
            session.conversation.add_turn(role, text)
        session.internet = internet
        session.reply(case.current)
        turns = session.conversation.get_recent_turns(limit=30)
        close = session.close
    try:
        internet.lookup.assert_called_once()
        goal = internet.lookup.call_args.kwargs["objective"].casefold()
        assert all(term.casefold() in goal for term in case.required), goal
        assert not any(term.casefold() in goal for term in case.forbidden), goal
        final = turns[-1]
        assert final.role == "assistant" and case.fact in final.content
        assert final.sources == [source.to_dict()]
        assert "couldn't verify" not in final.content and "tool limit" not in final.content
        assert [turn.content for turn in turns if turn.role == "user"] == [
            *(text for role, text in case.history if role == "user"), case.current,
        ]
        assert all(call.kwargs.get("purpose", "conversation") in {"conversation", "verification"}
                   for call in router.send.call_args_list)
    finally:
        close()


@pytest.mark.parametrize("entrypoint", ["voice", "keyboard", "chat"])
def test_correction_then_accepted_offer_searches_the_mod_in_a_running_conversation(
    tmp_path, monkeypatch, entrypoint,
):
    question = "Do you know about Tinsura anime mod in Minecraft?"
    correction = "I meant Tensura."
    acceptance = "Yes, sure."
    fact = "Tensura is a Minecraft mod inspired by the anime series."
    source = _source(fact)
    internet = _internet(monkeypatch, [InternetResult(attempted=True, sources=[source])])
    router = _router([
        LLMResponse("I'm not familiar with that mod.", "synthetic", "fixture"),
        LLMResponse("I haven't heard of a Tensura mod for Minecraft either. "
                    "Want me to look it up?", "synthetic", "fixture"),
        _tool_call("web_search", query=acceptance),
        _tool_call("web_search", query="Tensura Minecraft mod wiki"), _report(fact),
    ])
    # Accepting the offer is an explicit lookup even when automatic search is off.
    monkeypatch.setenv("INTERNET_AUTOMATIC", "false")
    if entrypoint == "chat":
        session = ChatSession(_settings(tmp_path), channel="telegram", session_id="mod-owner",
                              router=router)
        session.internet = internet
        for text in (question, correction, acceptance):
            session.reply(text)
        turns = session.conversation.get_recent_turns()
        close = session.close
    else:
        info = {"input_source": "keyboard"} if entrypoint == "keyboard" else {}
        _, tts, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
            (question, info or {"wake_verified": True}), (correction, info), (acceptance, info),
        ], router=router, controller=True)
        close = store.close
        assert fact in tts.speak.call_args.args[0]
    try:
        internet.lookup.assert_called_once()
        request = internet.lookup.call_args
        expected = question.replace("Tinsura", "Tensura")
        assert request.args[0] == "Tensura Minecraft mod wiki"
        assert request.kwargs["objective"] == expected
        assert fact in turns[-1].content and turns[-1].sources == [source.to_dict()]
        assert [turn.content for turn in turns if turn.role == "user"] == [
            question, correction, acceptance,
        ]
    finally:
        close()


@pytest.mark.parametrize("boundary", ["reset", "private", "cancel", "ambiguous"])
def test_a_search_offer_does_not_bypass_the_current_chat_scope(tmp_path, boundary):
    router = _router([LLMResponse("Which task should I check?", "synthetic", "fixture")])
    session = ChatSession(_settings(tmp_path), channel="telegram", session_id="offer-owner",
                          router=router)
    session.settings.internet_automatic = False
    session.conversation.add_turn("user", "Tensura mod for Minecraft?")
    session.conversation.add_turn("assistant", "Want me to look it up?" if boundary != "ambiguous"
                                  else "Want me to look it up or explain what I remember?")
    if boundary == "reset":
        session.reply("/reset")
    elif boundary == "cancel":
        session.reply("Don't search")
    elif boundary == "private":
        session.store.save_memory(MemoryItem(
            content="Synthetic private context.", memory_type=MemoryType.CONVERSATION,
            metadata={"session_id": session.conversation.session_id, "local_only": True},
        ))
    session.internet.lookup = Mock(side_effect=AssertionError("Unexpected public read"))
    try:
        session.reply("Yes, sure.")
        session.internet.lookup.assert_not_called()
        request = router.send.call_args
        assert "web_search" not in {tool["function"]["name"] for tool in request.kwargs["tools"]}
    finally:
        session.close()


@pytest.mark.parametrize("spoken_final", [False, True], ids=["typed_final", "spoken_final"])
def test_voice_keyboard_voice_repair_keeps_one_current_task(
    tmp_path, monkeypatch, spoken_final,
):
    fact = "Tensura anime continues in light novel volume 4."
    source = _source(fact)
    internet = _internet(monkeypatch, [
        InternetResult(attempted=True, sources=[source]),
        InternetResult(attempted=True, sources=[source]),
    ])
    router = _router([
        LLMResponse("Which medium do you want to continue in?", "synthetic", "fixture"),
        _tool_call("web_search", query="TensorFlow"),
        _tool_call("web_search", query="Tensura anime continuation"), _report(fact),
        _tool_call("web_search", query="Tensura"), _report(fact),
    ])
    _, tts, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("Hey Ciel, search TensorFlow anime continuation.", {"wake_verified": True}),
        ("Tensura*", {"input_source": "keyboard"}),
        ("The light novel.", {} if spoken_final else {"input_source": "keyboard"}),
    ], router=router, controller=True)
    try:
        assert internet.lookup.call_count == 2
        goals = [call.kwargs["objective"] for call in internet.lookup.call_args_list]
        assert all("Tensura" in goal and "TensorFlow" not in goal for goal in goals)
        assert "light novel" in goals[-1].casefold()
        assert fact in turns[-1].content and turns[-1].sources == [source.to_dict()]
        assert [turn.content for turn in turns if turn.role == "user"] == [
            "search TensorFlow anime continuation.", "Tensura*", "The light novel.",
        ]
        spoken = [call.args[0] for call in tts.speak.call_args_list]
        assert spoken[0] == "Which medium do you want to continue in?"
        assert spoken.count(fact) == 2
    finally:
        store.close()


@pytest.mark.parametrize("pattern", ["repeated_query", "last_round_new_lead"])
def test_bounded_search_replay_finishes_with_evidence_instead_of_tool_limit(
    tmp_path, pattern,
):
    fact = "CedarDB supports Linux desktop systems."
    queries = (["CedarDB Linux", "CedarDB Linux"] if pattern == "repeated_query" else
               ["CedarDB platform guide", "CedarDB installation guide", "CedarDB Linux"])
    router = _router([*(_tool_call("web_search", query=query) for query in queries),
                      _report(fact)])
    session = ChatSession(_settings(tmp_path), channel="telegram", session_id="replay-owner",
                          router=router)
    source = _source(fact)
    session.internet.lookup = Mock(return_value=InternetResult(attempted=True, sources=[source]))
    try:
        answer = session.reply("Search whether CedarDB supports Linux desktop systems.")
        assert fact in answer and "tool limit" not in answer
        assert session.internet.lookup.call_count == (1 if pattern == "repeated_query" else 3)
        assert session.conversation.get_recent_turns()[-1].sources == [source.to_dict()]
        # Exhausting the read budget still permits a final model answer without more tools.
        if pattern == "last_round_new_lead":
            call = next(call for call in reversed(router.send.call_args_list)
                        if call.kwargs.get("purpose", "conversation") == "conversation")
            assert not call.kwargs.get("tools")
    finally:
        session.close()


@pytest.mark.parametrize("boundary", ["reset", "private"])
def test_old_task_cannot_reappear_across_reset_or_escape_private_context(tmp_path, boundary):
    router = _router([LLMResponse("Which task should I check?", "synthetic", "fixture")])
    session = ChatSession(_settings(tmp_path), channel="telegram", session_id="replay-owner",
                          router=router)
    session.conversation.add_turn("user", "Search Amber Harbor light novel continuation.")
    session.conversation.add_turn("assistant", "Which episode did you finish?")
    if boundary == "reset":
        session.reply("/reset")
        query = "Try again."
    else:
        session.store.save_memory(MemoryItem(
            content="Synthetic private context.", memory_type=MemoryType.CONVERSATION,
            metadata={"session_id": session.conversation.session_id, "local_only": True},
        ))
        query = "Search it again."
    session.internet.lookup = Mock(side_effect=AssertionError("Unexpected public read"))
    try:
        session.reply(query)
        session.internet.lookup.assert_not_called()
        requests = [call for call in router.send.call_args_list
                    if call.kwargs.get("purpose", "conversation") == "conversation"]
        assert requests
        if boundary == "reset":
            assert "Amber Harbor" not in " ".join(
                message.content for message in requests[-1].args[0]
            )
        else:
            tools = requests[-1].kwargs["tools"]
            assert "web_search" not in {tool["function"]["name"] for tool in tools}
            assert any(message.local_only for message in requests[-1].args[0])
    finally:
        session.close()


@pytest.mark.parametrize("entrypoint", ["desktop", "chat"])
def test_screen_pause_resume_then_cancel_uses_two_fresh_observations_only(
    tmp_path, monkeypatch, entrypoint,
):
    from ciel.screen.service import ScreenResult

    observations = iter([
        ScreenResult("Fresh observation alpha.", "fixture", "vision", "frame-a"),
        ScreenResult("Fresh observation beta.", "fixture", "vision", "frame-b"),
    ])
    created = install_screen(monkeypatch, on_answer=lambda *_args, **_kwargs: next(observations))
    router = _router([LLMResponse("Which task should I repeat?", "synthetic", "fixture")])
    phrases = ["Look at my screen", "Wait.", "Again.", "Cancel.", "Again."]
    if entrypoint == "desktop":
        _internet(monkeypatch, [])
        _, tts, _, store, _turns = run_callbacks(
            tmp_path, monkeypatch, [(text, {"input_source": "keyboard"}) for text in phrases],
            router=router, controller=True,
        )
        close = store.close
        spoken = [call.args[0] for call in tts.speak.call_args_list]
        assert spoken.count("Fresh observation alpha.") == 1
        assert spoken.count("Fresh observation beta.") == 1
        assert spoken.count("Which task should I repeat?") == 1
        tts.begin_stream.assert_not_called()
    else:
        session = ChatSession(_settings(tmp_path), channel="telegram", session_id="replay-owner",
                              router=router)
        for text in phrases:
            session.reply(text)
        close = session.close
    try:
        requests = [call.args[0] for call in created[0].answer.call_args_list]
        assert len(requests) == 2
        assert all(not request.previous for request in requests)
        assert all("Look at my screen" in request.query for request in requests)
    finally:
        close()


@pytest.mark.parametrize("entrypoint", ["desktop", "chat"])
def test_main_reply_receives_non_search_task_after_interruption(tmp_path, monkeypatch, entrypoint):
    history = (("user", "Explain why rainbows form in simple terms."),
               ("assistant", "Sunlight enters water droplets. [Reply was interrupted.]"))
    router = _router([LLMResponse("The light bends and separates into colors.", "main", "fixture")])
    if entrypoint == "desktop":
        _, _, _, store, _ = run_callbacks(
            tmp_path, monkeypatch, [("Continue.", {"input_source": "keyboard"})],
            router=router, controller=True,
            initialize_conversation=lambda store, session: _seed(store, session, history),
        )
        close = store.close
    else:
        session = ChatSession(_settings(tmp_path), channel="telegram", session_id="replay-owner",
                              router=router)
        for role, text in history:
            session.conversation.add_turn(role, text)
        session.reply("Continue.")
        close = session.close
    try:
        request = next(call for call in router.send.call_args_list
                       if call.kwargs.get("purpose", "conversation") == "conversation")
        note = " ".join(message.content for message in request.args[0]
                        if message.role == "system")
        assert '"request": "Explain why rainbows form in simple terms."' in note
        assert '"relationship": "continue"' in note and '"interrupted": true' in note
        assert [message.content for message in request.args[0] if message.role == "user"][-1] == (
            "Continue."
        )
    finally:
        close()


@pytest.mark.parametrize("entrypoint", ["desktop", "chat"])
def test_continue_uses_completed_evidence_without_repeating_search(
    tmp_path, monkeypatch, entrypoint,
):
    fact = "CedarDB supports Linux desktop systems."
    source = _source(fact)
    lookup = InternetResult(attempted=True, query="Search CedarDB Linux support.", sources=[source])
    router = _router([_report(fact)])
    internet = _internet(monkeypatch, [])

    def seed(store, session_id):
        conversation = ConversationManager(store, session_id=session_id)
        conversation.add_turn("user", "Search CedarDB Linux support.")
        conversation.add_turn("assistant", "CedarDB supports [Playback was interrupted.]",
                              web_evidence=evidence_payload(lookup))
        conversation.close()

    if entrypoint == "desktop":
        _, _, _, store, turns = run_callbacks(
            tmp_path, monkeypatch, [("Continue.", {"input_source": "keyboard"})],
            router=router, controller=True, initialize_conversation=seed,
        )
        close = store.close
    else:
        session = ChatSession(_settings(tmp_path), channel="telegram", session_id="replay-owner",
                              router=router)
        seed(session.store, session.conversation.session_id)
        session.internet = internet
        session.reply("Continue.")
        turns = session.conversation.get_recent_turns()
        close = session.close
    try:
        internet.lookup.assert_not_called()
        assert fact in turns[-1].content
        assert turns[-1].sources == [source.to_dict()]
    finally:
        close()


@pytest.mark.parametrize("control", ["Wait.", "Cancel.", "Don't search."])
@pytest.mark.parametrize("entrypoint", ["desktop", "chat"])
def test_only_wait_retains_a_resumable_web_task(tmp_path, monkeypatch, control, entrypoint):
    fact = "CedarDB supports Linux desktop systems."
    router = _router([_tool_call("web_search", query="CedarDB Linux"), _report(fact)])
    history = (("user", "Search CedarDB Linux support."),
               ("assistant", "The lookup timed out."))
    internet = _internet(monkeypatch, [InternetResult(attempted=True, sources=[_source(fact)])])
    if entrypoint == "desktop":
        _, _, _, store, _ = run_callbacks(
            tmp_path, monkeypatch,
            [(control, {"input_source": "keyboard"}), ("Again.", {"input_source": "keyboard"})],
            router=router, controller=True,
            initialize_conversation=lambda store, session: _seed(store, session, history),
        )
        close = store.close
    else:
        session = ChatSession(_settings(tmp_path), channel="telegram", session_id="replay-owner",
                              router=router)
        session.settings.internet_automatic = False
        for role, text in history:
            session.conversation.add_turn(role, text)
        session.internet = internet
        session.reply(control)
        session.reply("Again.")
        close = session.close
    try:
        assert internet.lookup.call_count == (1 if control == "Wait." else 0)
    finally:
        close()


def test_interrupted_search_playback_retains_evidence_for_typed_continue(tmp_path, monkeypatch):
    from threading import Event
    from types import SimpleNamespace

    fact = "CedarDB supports Linux desktop systems."
    source = _source(fact)
    internet = _internet(monkeypatch, [InternetResult(attempted=True, sources=[source])])
    router = _router([_tool_call("web_search", query="CedarDB Linux"),
                      _report(fact), _report(fact)])
    cancellation = Event()

    def interrupted(agent, messages, _tts, _current, **kwargs):
        response = agent.send(messages, cancel_event=kwargs.get("cancel_event"))
        assert fact in response.content
        cancellation.set()
        return SimpleNamespace(response=response, spoken=False,
                               first_audio_seconds=0.01, error=None)

    monkeypatch.setattr("ciel.audio.streaming.stream_reply", interrupted)
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("Hey Ciel, search CedarDB Linux support.",
         {"wake_verified": True, "cancel_event": cancellation}),
        ("Continue.", {"input_source": "keyboard", "cancel_event": Event()}),
    ], router=router, streaming=True, controller=True)
    try:
        internet.lookup.assert_called_once()
        interrupted_turn = next(turn for turn in turns
                                if "Playback was interrupted" in turn.content)
        assert interrupted_turn.web_evidence["sources"][0]["content"] == fact
        assert fact in turns[-1].content and turns[-1].sources == [source.to_dict()]
    finally:
        store.close()
