"""Regression coverage for the repository review's reproduced failures."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ciel.audio.listener import ListenerState, WakeListenerLoop
from ciel.config import Settings
from ciel.memory import ConversationManager, MemoryStore
from ciel.memory.service import MemoryService
from ciel.platform.linux import LinuxAudioBackend
from ciel.providers.base import ChatMessage, LLMResponse, LLMStreamChunk
from ciel.providers.manager import ProviderManager
from ciel.providers.router import ModelRouter
from tests.test_ambient import run_callbacks


@pytest.fixture
def memory():
    store = MemoryStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def providers():
    manager = ProviderManager.__new__(ProviderManager)
    manager.fallback_chain = ["nim", "groq", "openrouter", "ollama"]
    manager.providers = {name: MagicMock() for name in manager.fallback_chain}
    for provider in manager.providers.values():
        provider.is_configured.return_value = True
    manager.ollama.send.side_effect = RuntimeError("local unavailable")
    manager.ollama.stream.side_effect = RuntimeError("local unavailable")
    manager.nim.send.return_value = LLMResponse("Cloud reply", "fake", "nim")
    manager.nim.stream.return_value = iter([LLMStreamChunk("Cloud reply", "fake", "nim")])
    return manager


@pytest.mark.parametrize("streaming", [False, True])
def test_local_failure_never_contacts_cloud(providers, streaming):
    router = ModelRouter(providers)
    messages = [ChatMessage("user", "/local private example")]
    with pytest.raises(RuntimeError):
        list(router.stream(messages)) if streaming else router.send(messages)
    for name in ("nim", "groq", "openrouter"):
        providers.providers[name].send.assert_not_called()
        providers.providers[name].stream.assert_not_called()


def test_forget_missing_structured_topic_preserves_other_fact(memory):
    service = MemoryService(memory)
    service.handle("remember my favorite game is CS2")
    assert "couldn't find" in service.handle("forget my favorite food")
    assert memory.get_fact("user:favorite_game").metadata["value"] == "CS2"


def test_forgotten_old_values_stay_suppressed_after_new_value_and_restart(tmp_path):
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    service = MemoryService(store)
    service.handle("remember my favorite game is CS2")
    service.handle("forget my favorite game")
    service.handle("remember my favorite game is Minecraft")
    store.close()
    reopened = MemoryStore(path)
    try:
        assert "CS2" not in reopened.redact_forgotten("Your favorite game is CS2.")
        assert reopened.redact_forgotten("Minecraft") == "Minecraft"
        service = MemoryService(reopened)
        service.handle("forget my favorite game")
        assert "CS2" not in reopened.redact_forgotten("CS2 and Minecraft")
        assert "Minecraft" not in reopened.redact_forgotten("CS2 and Minecraft")
    finally:
        reopened.close()


def test_active_stream_can_be_replaced(monkeypatch):
    backend = LinuxAudioBackend()
    old, replacement = MagicMock(), MagicMock()
    old.active = True
    backend._stream = old
    monkeypatch.setattr(backend, "resolve_device", lambda *args, **kwargs: 0)
    monkeypatch.setattr("ciel.platform.linux.sd", SimpleNamespace(
        InputStream=MagicMock(return_value=replacement),
    ))
    errors = []

    def restart():
        try:
            backend.start_stream(lambda *args: None)
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=restart, daemon=True)
    worker.start()
    worker.join(timeout=1)
    assert not worker.is_alive(), "stream replacement deadlocked"
    assert not errors
    old.stop.assert_called_once()
    old.close.assert_called_once()
    replacement.start.assert_called_once()
    backend.stop_stream()


def test_wake_mode_requires_wake_evidence_for_interruption():
    detector, tts = MagicMock(), MagicMock()
    detector.process_frame.return_value = None
    tts.is_speaking.return_value = True
    tts.stop.return_value = None
    loop = WakeListenerLoop(MagicMock(), detector=detector, tts=tts, barge_in_mode="wake")
    loop._running = True
    loop._state = ListenerState.PROCESSING
    previous = loop._response_cancel = threading.Event()
    loop._handle_frame(np.full(1280, .05, dtype=np.float32))
    assert not previous.is_set()
    tts.stop.assert_not_called()
    detector.process_frame.return_value = {"wake_phrase": "hey ciel"}
    loop._handle_frame(np.full(1280, .05, dtype=np.float32))
    assert previous.is_set()
    assert loop.state == ListenerState.RECORDING


@pytest.mark.parametrize("intervening", ["cancel", "goodbye", "what is 2 plus 2?"])
def test_persona_proposal_does_not_capture_later_clock_question(
    tmp_path, monkeypatch, intervening,
):
    persona = tmp_path / "persona.txt"
    router = MagicMock()
    router.send.return_value = LLMResponse("Four.", "fake", "fake")
    _, tts, _, store, _ = run_callbacks(
        tmp_path, monkeypatch,
        [("Ciel, can you update your persona?", {}),
         ("Ciel, " + intervening, {}), ("Ciel, what's the time?", {})],
        persona_file=persona, router=router,
    )
    try:
        assert not persona.exists()
        assert tts.speak.call_args.args[0].startswith("It's ")
    finally:
        store.close()


def test_wake_models_flow_from_environment_to_audio(monkeypatch):
    monkeypatch.setenv("WAKE_MODELS", '["models/custom/hey_ciel.onnx"]')
    assert Settings(_env_file=None).audio.wake_models == ["models/custom/hey_ciel.onnx"]


def test_failed_summary_does_not_skip_turns_on_recovery(memory):
    conversation = ConversationManager(memory, max_turns=2, summary_interval=1)
    for index in range(20):
        conversation.add_turn("user", f"unique detail number {index}")
    provider = MagicMock()
    provider.send.side_effect = RuntimeError("temporary outage")
    assert conversation.summarize_older_turns(provider) is None
    assert memory.get_session_summary("default") is None
    provider.send.side_effect = None
    provider.send.return_value = LLMResponse("Recovered summary", "fake", "fake")
    assert conversation.summarize_older_turns(provider) == "Recovered summary"
    assert "unique detail number 10" in provider.send.call_args.args[0][1].content


def test_local_summary_and_restart_keep_private_context_on_ollama(memory, providers):
    providers.ollama.stream.side_effect = None
    providers.ollama.stream.return_value = [LLMStreamChunk("Private summary", "fake", "ollama")]
    conversation = ConversationManager(memory, max_turns=2, auto_summarize_threshold=2)
    conversation.add_turn("user", "/local private example")
    for index in range(5):
        conversation.add_turn("user", f"follow-up {index}")
    assert conversation.summarize_older_turns(ModelRouter(providers)) == "Private summary"
    restarted = ConversationManager(memory, max_turns=2)
    messages = restarted.get_active_messages("Persona")
    assert ModelRouter.requires_local(messages)
    assert all("local_only" not in message.to_dict() for message in messages)
    providers.ollama.stream.side_effect = RuntimeError("local unavailable")
    with pytest.raises(RuntimeError):
        list(ModelRouter(providers).stream(messages))
    providers.nim.stream.assert_not_called()
    assert all(message.local_only for message in restarted.get_gate_messages())


def test_private_speech_gate_never_contacts_cloud(providers):
    from ciel.audio.ambient import AmbientConversation

    ambient = AmbientConversation()
    ambient.record_addressed("assistant", "How can I help?")
    dialogue = [ChatMessage("assistant", "Private reply", local_only=True)]
    decision = ambient.decide("Could you explain more?", ModelRouter(providers), dialogue)
    assert not decision.addressed
    providers.nim.stream.assert_not_called()


@pytest.mark.parametrize("source", ["transcript", "interaction", "unfinished"])
def test_private_gate_payload_sources_never_contact_cloud(providers, source):
    from ciel.audio.ambient import AmbientConversation

    ambient = AmbientConversation()
    if source == "interaction":
        ambient.record_addressed("user", "/local private question")
    ambient.record_addressed("assistant", "How can I help?")
    text = "/local private follow-up" if source == "transcript" else "Could you explain more?"
    unfinished = ["/local private question"] if source == "unfinished" else None
    decision = ambient.decide(text, ModelRouter(providers), [], unfinished_request=unfinished)
    assert not decision.addressed
    for name in ("nim", "groq", "openrouter"):
        providers.providers[name].stream.assert_not_called()


def test_private_summary_refuses_direct_cloud_provider(memory):
    conversation = ConversationManager(memory, max_turns=2, auto_summarize_threshold=2)
    conversation.add_turn("user", "/local private example")
    for _ in range(3):
        conversation.add_turn("user", "follow-up")
    provider = MagicMock(name="cloud")
    assert conversation.summarize_older_turns(provider) is None
    provider.send.assert_not_called()


def test_local_policy_survives_context_eviction_and_failed_summary(memory, providers):
    conversation = ConversationManager(memory, max_turns=2)
    conversation.add_turn("user", "/local confidential question")
    for index in range(100):
        conversation.add_turn("assistant", f"private reply {index}")
    assert conversation.summarize_older_turns(ModelRouter(providers)) is None
    restarted = ConversationManager(memory, max_turns=2)
    messages = restarted.get_active_messages("Persona")
    assert ModelRouter.requires_local(messages)
    assert all(message.local_only for message in restarted.get_gate_messages())
    with pytest.raises(RuntimeError):
        list(ModelRouter(providers).stream(messages))
    providers.nim.stream.assert_not_called()
    restarted.clear_session()
    assert not ModelRouter.requires_local(restarted.get_active_messages("Persona"))


def test_providerless_summary_advances_only_through_represented_turns(memory):
    conversation = ConversationManager(memory, max_turns=2)
    for index in range(20):
        conversation.add_turn("user", f"topic {index}")
    assert conversation.summarize_older_turns()
    assert memory.get_session_summary("default").metadata["last_turn_id"] == 3
    assert "topic 3" in conversation.summarize_older_turns()


def test_persona_proposal_expires(tmp_path, monkeypatch):
    ticks = iter(range(0, 10000, 120))
    persona = tmp_path / "persona.txt"
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"listener":"assistant","confidence":0.98}', "fake", "fake",
    )
    _, _, _, store, _ = run_callbacks(
        tmp_path, monkeypatch,
        [("Ciel, can you update your persona?", {}), ("Be more playful", {})],
        persona_file=persona, router=router, clock=lambda: next(ticks),
    )
    try:
        assert not persona.exists()
    finally:
        store.close()


@pytest.mark.parametrize("text", [
    "Tell me a joke", "Show my favorite food", "Explain quantum physics",
    "Remember my favorite game is CS2", "Open Firefox", "The weather is sunny today",
])
def test_pending_persona_requires_style_evidence(text):
    from ciel.persona import parse_persona_request

    assert parse_persona_request(text, pending=True) is None


def test_failed_stream_start_closes_device_and_allows_retry(monkeypatch):
    backend = LinuxAudioBackend()
    failed, replacement = MagicMock(), MagicMock()
    failed.start.side_effect = RuntimeError("device disappeared")
    monkeypatch.setattr(backend, "resolve_device", lambda *args, **kwargs: 0)
    monkeypatch.setattr("ciel.platform.linux.sd", SimpleNamespace(
        InputStream=MagicMock(side_effect=[failed, replacement]),
    ))
    with pytest.raises(RuntimeError, match="device disappeared"):
        backend.start_stream(lambda *args: None)
    failed.close.assert_called_once()
    assert backend._stream is None
    backend.start_stream(lambda *args: None)
    replacement.start.assert_called_once()
    backend.stop_stream()


def test_stt_failed_load_is_not_ready():
    from ciel.audio.stt import SpeechToText

    stt = SpeechToText.__new__(SpeechToText)
    stt._ready = threading.Event()
    stt._ready.set()
    stt.model = None
    stt._load_error = RuntimeError("load failed")
    assert not stt.is_ready()
    assert not stt.wait_ready(timeout=0)


def test_stt_loading_wait_has_a_deadline():
    from ciel.audio.stt import SpeechToText

    stt = SpeechToText.__new__(SpeechToText)
    stt._ready = threading.Event()
    stt.load_timeout = .01
    with pytest.raises(TimeoutError, match="STT"):
        stt.transcribe_detailed(np.ones(16000, dtype=np.float32))


def test_alive_but_unresponsive_turbo_worker_times_out(tmp_path, monkeypatch):
    from ciel.audio.chatterbox_worker import ChatterboxTurboWorker, ChatterboxWorkerError

    worker = ChatterboxTurboWorker(
        tmp_path / "python", tmp_path / "worker.py", tmp_path / "model",
        tmp_path / "reference.wav", "reference", synthesis_timeout=.05,
    )
    monkeypatch.setattr(worker, "_start", lambda: None)
    monkeypatch.setattr(worker, "close", MagicMock())
    worker._process = MagicMock()
    worker._process.poll.return_value = None
    with pytest.raises(ChatterboxWorkerError, match="Timed out"):
        worker.synthesize("Test speech")
    worker.close.assert_called_once_with(force=True)
    assert not worker._pending


@pytest.mark.parametrize("values", [
    {"audio_sample_rate": 44100}, {"audio_channels": 2},
    {"memory_max_short_term_turns": 0}, {"tts_speed": 0},
    {"wake_threshold": 1.1}, {"stt_startup_timeout_seconds": 0},
    {"tts_synthesis_timeout_seconds": float("inf")},
])
def test_unsupported_audio_and_invalid_limits_fail_at_configuration(values):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **values)
