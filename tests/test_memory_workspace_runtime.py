"""Actual desktop callbacks preserve owner authority and use real Markdown tool evidence."""

import json
from threading import Event
from types import SimpleNamespace

import pytest

from ciel import config
from ciel.memory.tools import MemoryTools
from ciel.providers.base import LLMResponse, LLMStreamChunk, ToolCall, ToolCallDelta
from tests.test_ambient import run_callbacks

RAW_PATH = "memory/cedar-rollout.md"
RAW_DECISION = "Decision: retain a signed offline recovery bundle."
RAW_NOTE = f"# Cedar rollout\n{RAW_DECISION}\nRetention: keep two validated copies.\n"
FACT = "Meadow Quest"


@pytest.fixture(autouse=True)
def isolated_runtime_configuration(tmp_path, monkeypatch):
    """Keep the existing callback harness independent of ambient environment and services."""
    settings_type = config.Settings

    def settings(**values):
        return settings_type(**{
            **values, "_env_file": None, "screen_enabled": False, "internet_enabled": False,
            "memory_workspace_path": "", "memory_workspace_enabled": True,
            "memory_semantic_enabled": False, "memory_learning_mode": "confirm",
            "memory_daily_capture_enabled": True, "memory_flush_enabled": True,
            "memory_recall_enabled": True, "ambient_followup_seconds": 20,
            "stt_retry_confidence": 0.65,
            "memory_consolidation_enabled": False, "chat_services": "",
            "chat_state_path": str(tmp_path / "chat.db"), "ciel_persona_file": "",
        })

    monkeypatch.setattr(config, "Settings", settings)
    monkeypatch.setattr("ciel.memory.manager.get_settings", lambda: config.get_settings())


def _initialize_desktop(seed=None):
    """Disable external services before the real runtime constructs them."""
    def initialize(store, session_id):
        settings = config.get_settings()
        settings.screen_enabled = False
        settings.internet_enabled = False
        settings.memory_semantic_enabled = False
        settings.memory_consolidation_enabled = False
        settings.chat_services = ""
        if seed is not None:
            seed(store, session_id)

    return initialize


def _seed_raw_note(store, _session_id):
    """Write an unmanaged source that cannot be supplied by structured fact recall."""
    target = store.workspace.root / RAW_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(RAW_NOTE, encoding="utf-8")


class DesktopModel:
    """Model transport double; all routing, persistence, and Markdown reads remain real."""

    def __init__(self, *, retrieve=False):
        self.retrieve = retrieve
        self.purposes = []
        self.native_inputs = []
        self.gates = []
        self.final_inputs = []

    @staticmethod
    def _response(text):
        return LLMResponse(text, "runtime-fixture", "fixture")

    def send(self, messages, **kwargs):
        purpose = kwargs.get("purpose")
        self.purposes.append(purpose)
        if purpose == "tool_progress":
            return self._response("{}")
        if purpose == "speech_gate":
            self.gates.append(list(messages))
            return self._response('{"listener":"assistant","confidence":0.99}')
        assert purpose != "controller", "Foreground routing must use the main response model"
        native = "tools" in kwargs
        if native:
            self.native_inputs.append(list(messages))
            assert purpose == "conversation"
            previous = [json.loads(message.content) for message in messages
                        if message.role == "tool"]
            if self.retrieve and not previous:
                assert {"memory_search", "memory_get"} <= {
                    tool["function"]["name"] for tool in kwargs["tools"]
                }
                return LLMResponse("", "runtime-fixture", "fixture", tool_calls=(
                    ToolCall("search_cedar", "memory_search", '{"query":"Cedar recovery"}'),
                ))
            if self.retrieve and len(previous) == 1:
                source = next(item for item in previous[0]["results"] if item["path"] == RAW_PATH)
                return LLMResponse("", "runtime-fixture", "fixture", tool_calls=(
                    ToolCall("read_cedar", "memory_get", json.dumps({
                        "path": source["path"], "from_line": 2, "lines": 1,
                    })),
                ))
        self.final_inputs.append(list(messages))
        if not self.retrieve:
            return self._response("Tell me which preference you want to discuss.")
        assert native
        requests = [call for message in messages for call in message.tool_calls]
        assert [call.name for call in requests] == ["memory_search", "memory_get"]
        results = [message for message in messages if message.role == "tool"]
        assert [message.tool_call_id for message in results] == [call.id for call in requests]
        exact = json.loads(results[-1].content)
        assert exact["path"] == RAW_PATH
        assert exact["from_line"] == exact["to_line"] == 2
        assert exact["text"] == RAW_DECISION
        return self._response("The saved note says: " + exact["text"])


def _workspace_text(store):
    return "\n".join(document.text for document in store.workspace.documents())


def _owner_observations(store):
    """Inspect persisted authority without substituting a fake memory manager."""
    with store._lock:
        return [row[0] for row in store._get_connection().execute(
            "SELECT content FROM memory_observations",
        )]


@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("typed", [False, True])
def test_authenticated_desktop_memory_save_updates_real_curated_markdown(
    tmp_path, monkeypatch, controller, typed,
):
    model = DesktopModel()
    statement = f"Remember that my favorite game is {FACT}."
    text = statement if typed else "Ciel, " + statement
    info = {"input_source": "keyboard"} if typed else {
        "wake_verified": True, "stt_confidence": 0.99,
    }
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [(text, info)], router=model, controller=controller,
        initialize_conversation=_initialize_desktop(),
    )
    try:
        saved = store.get_fact("user:favorite_game")
        assert saved is not None and saved.metadata["value"] == FACT
        assert FACT in (store.workspace.root / "MEMORY.md").read_text(encoding="utf-8")
        assert any(FACT in turn.content for turn in turns if turn.role == "user")
        assert any(turn.role == "assistant" for turn in turns)
        assert not model.final_inputs
    finally:
        store.close()


@pytest.mark.parametrize("typed", [False, True])
def test_model_selected_search_then_get_supplies_actual_note_to_desktop_final_reply(
    tmp_path, monkeypatch, typed,
):
    model = DesktopModel(retrieve=True)
    info = {"input_source": "keyboard"} if typed else {"wake_verified": True}
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Which recovery choice did we record for the Cedar rollout?", info)],
        router=model, controller=True, initialize_conversation=_initialize_desktop(_seed_raw_note),
    )
    try:
        assert model.final_inputs
        assert len(model.native_inputs) == 3
        assert model.purposes == ["conversation"] * 3
        assistant = [turn for turn in turns if turn.role == "assistant"]
        assert assistant and RAW_DECISION in assistant[-1].content
        if typed:
            tts.speak.assert_called_once()
            assert tts.speak.call_args.args[0] == assistant[-1].content
            tts.begin_stream.assert_not_called()
        else:
            assert any(RAW_DECISION in call.args[0] for call in tts.speak.call_args_list)
        assert store.count_memories() == 0
        assert (store.workspace.root / RAW_PATH).read_text(encoding="utf-8") == RAW_NOTE
    finally:
        store.close()


@pytest.mark.parametrize("query", [
    "Search my notes for Cedar rollout", "Read memory/cedar-rollout.md lines 2-2",
])
@pytest.mark.parametrize("typed", [False, True])
def test_controller_disabled_still_supports_exact_local_memory_tools(
    tmp_path, monkeypatch, capsys, query, typed,
):
    model = DesktopModel()
    info = {"input_source": "keyboard"} if typed else {"wake_verified": True}
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [(query, info)],
        router=model, controller=False, initialize_conversation=_initialize_desktop(_seed_raw_note),
    )
    try:
        assert not model.purposes and not model.final_inputs
        assert RAW_DECISION in turns[-1].content
        if typed:
            tts.speak.assert_called_once()
            assert RAW_DECISION in tts.speak.call_args.args[0]
            tts.begin_stream.assert_not_called()
            assert RAW_DECISION in capsys.readouterr().out
        else:
            assert any(RAW_DECISION in call.args[0] for call in tts.speak.call_args_list)
        assert (store.workspace.root / RAW_PATH).read_text(encoding="utf-8") == RAW_NOTE
    finally:
        store.close()


def test_controller_disabled_personal_history_cannot_trigger_public_search(tmp_path, monkeypatch):
    model = DesktopModel()

    def seed(store, session_id):
        _seed_raw_note(store, session_id)
        config.get_settings().internet_enabled = True

    def unexpected_public_search(*args, **kwargs):
        pytest.fail("A personal history request reached public search")

    monkeypatch.setattr("ciel.internet.InternetService.prepare", unexpected_public_search)
    _, _, _, store, _ = run_callbacks(
        tmp_path, monkeypatch,
        [("What did we decide for the Cedar rollout?", {"input_source": "keyboard"})],
        router=model, controller=False, initialize_conversation=_initialize_desktop(seed),
    )
    try:
        assert model.final_inputs
        assert RAW_DECISION in model.final_inputs[-1][0].content
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("admission", ["uncertain", "ambient"])
def test_model_admission_or_uncertain_speech_cannot_create_owner_evidence_or_facts(
    tmp_path, monkeypatch, controller, admission,
):
    model = DesktopModel()
    statement = f"Remember that my favorite game is {FACT}."
    utterances = [("Ciel, explain how preferences work.", {"wake_verified": True})]
    if admission == "uncertain":
        utterances = [("Ciel, " + statement, {"wake_verified": True, "stt_confidence": 0.45})]
    else:
        utterances.append((statement, {}))
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, utterances, router=model, controller=controller,
        initialize_conversation=_initialize_desktop(),
    )
    try:
        assert any(FACT in turn.content for turn in turns if turn.role == "user")
        assert store.get_fact("user:favorite_game") is None
        assert all(FACT not in content for content in _owner_observations(store))
        assert FACT not in _workspace_text(store)
        if admission == "ambient":
            if controller:
                assert len(model.native_inputs) == 1
                assert "controller" not in model.purposes and not model.gates
                assert any("address Ciel directly" in turn.content
                           for turn in turns if turn.role == "assistant")
            else:
                assert model.gates
    finally:
        store.close()


@pytest.mark.parametrize("interruption", ["cancel", "reset"])
def test_cancel_or_real_scope_reset_during_memory_read_blocks_late_speech_and_commits(
    tmp_path, monkeypatch, interruption,
):
    cancellation = Event()
    model = DesktopModel(retrieve=True)
    original = MemoryTools.get
    completed_reads = []

    def interrupted_read(self, path, **kwargs):
        result = original(self, path, **kwargs)
        completed_reads.append(result)
        assert result["text"] == RAW_DECISION
        if interruption == "cancel":
            cancellation.set()
        else:
            self.conversation.clear_session()
        return result

    monkeypatch.setattr(MemoryTools, "get", interrupted_read)
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Which recovery choice did we record for the Cedar rollout?", {
            "input_source": "keyboard", "cancel_event": cancellation,
        })],
        router=model, controller=True, initialize_conversation=_initialize_desktop(_seed_raw_note),
    )
    try:
        assert completed_reads
        assert not model.final_inputs
        assert len(model.native_inputs) == 2
        assert any(message.role == "tool" and message.tool_call_id == "search_cedar"
                   for message in model.native_inputs[-1])
        assert all(message.tool_call_id != "read_cedar"
                   for messages in model.native_inputs for message in messages)
        assert not [turn for turn in turns if turn.role == "assistant"]
        tts.speak.assert_not_called()
        assert store.count_memories() == 0
        assert RAW_DECISION not in "\n".join(_owner_observations(store))
        assert (store.workspace.root / RAW_PATH).read_text(encoding="utf-8") == RAW_NOTE
        assert not (store.workspace.root / "MEMORY.md").exists()
        if interruption == "reset":
            assert turns == [] and _owner_observations(store) == []
        else:
            assert any(turn.role == "user" for turn in turns)
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_reset_during_initial_bootstrap_blocks_models_and_user_turn_reinsertion(
    tmp_path, monkeypatch, controller,
):
    model = DesktopModel()
    curated = "Owner prefers muted green design layouts."
    original = MemoryTools.bootstrap
    completed_reads = []

    def reset_after_bootstrap(self, **kwargs):
        result = original(self, **kwargs)
        if result and not completed_reads:
            assert curated in "\n".join(result)
            completed_reads.append(result)
            self.conversation.clear_session()
        return result

    def seed(store, _session_id):
        store.workspace.append_curated(curated)

    monkeypatch.setattr(MemoryTools, "bootstrap", reset_after_bootstrap)
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("What design palette do you suggest?", {"input_source": "keyboard"})],
        router=model, controller=controller, initialize_conversation=_initialize_desktop(seed),
    )
    try:
        assert completed_reads
        assert not model.purposes and not model.final_inputs
        assert turns == [] and _owner_observations(store) == []
        tts.speak.assert_not_called()
        assert curated in (store.workspace.root / "MEMORY.md").read_text(encoding="utf-8")
    finally:
        store.close()


@pytest.mark.parametrize("interruption", ["cancel", "reset"])
@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("typed", [False, True])
def test_streamed_memory_reply_preserves_delivered_history_only_in_unchanged_session(
    tmp_path, monkeypatch, interruption, controller, typed,
):
    cancellation = Event()
    model = DesktopModel(retrieve=True)
    state = {}

    def seed(store, session_id):
        state.update(store=store, session_id=session_id)
        _seed_raw_note(store, session_id)

    def interrupt_stream(router, messages, tts, current, **kwargs):
        if controller:
            # The stream wrapper owns native planning now; finish its real search/get
            # loop before interrupting playback at this transport boundary.
            response = router.send(messages, cancel_event=kwargs.get("cancel_event"))
            assert RAW_DECISION in response.content
            assert len(model.native_inputs) == 3 and len(model.final_inputs) == 1
            assert any(message.role == "tool" and message.tool_call_id == "read_cedar"
                       for message in model.final_inputs[0])
        else:
            assert RAW_DECISION in messages[0].content
        if interruption == "reset":
            state["store"].clear_session(state["session_id"])
        else:
            cancellation.set()
        assert not current()
        return SimpleNamespace(
            response=LLMResponse("The note says: " + RAW_DECISION, "fixture", "fixture"),
            spoken=False, first_audio_seconds=0.1, error=None,
        )

    interrupted_chunks = []

    def interrupt_model_stream(messages, **kwargs):
        """Read real memory evidence, then revoke ownership after its first answer chunk."""
        if controller:
            response = model.send(messages, **kwargs)
        else:
            # The legacy path supplies deterministic local recall before generation.
            assert RAW_DECISION in messages[0].content
            response = LLMResponse("The note says: " + RAW_DECISION, "fixture", "fixture")
        yield LLMStreamChunk(
            response.content, response.model, response.provider,
            tool_calls=tuple(ToolCallDelta(index, call.id, call.name, call.arguments)
                             for index, call in enumerate(response.tool_calls)),
        )
        if response.content:
            assert RAW_DECISION in response.content
            if controller:
                assert len(model.native_inputs) == 3 and len(model.final_inputs) == 1
                assert any(message.role == "tool" and message.tool_call_id == "read_cedar"
                           for message in model.final_inputs[0])
            interrupted_chunks.append(response.content)
            if interruption == "reset":
                state["store"].clear_session(state["session_id"])
            else:
                cancellation.set()
            assert kwargs["cancel_event"].is_set()
            yield LLMStreamChunk("Stale continuation must not be saved.", "fixture", "fixture")

    if typed:
        monkeypatch.setattr(model, "stream", interrupt_model_stream, raising=False)
    monkeypatch.setattr("ciel.audio.streaming.stream_reply", interrupt_stream)
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("What did we decide for the Cedar rollout?", {
            **({"input_source": "keyboard"} if typed else {"wake_verified": True}),
            "cancel_event": cancellation,
        })],
        router=model, controller=controller, streaming=True,
        initialize_conversation=_initialize_desktop(seed),
    )
    try:
        if typed:
            assert interrupted_chunks
            tts.speak.assert_not_called()
            tts.begin_stream.assert_not_called()
        if interruption == "reset":
            assert turns == [] and _owner_observations(store) == []
        elif typed:
            # No terminal stream is attached in this harness, so no partial answer was shown.
            assert [turn.role for turn in turns] == ["user"]
        else:
            assert [turn.role for turn in turns] == ["user", "assistant"]
            assert "Playback was interrupted" in turns[-1].content
        assert (store.workspace.root / RAW_PATH).read_text(encoding="utf-8") == RAW_NOTE
    finally:
        store.close()
