"""Real chat and voice entry points admit visuals without capturing a real desktop."""

import threading
from unittest.mock import MagicMock, Mock

import pytest

from ciel.chat.session import ChatSession
from ciel.chat.state import ChatState
from ciel.chat.telegram import ChatReply
from ciel.config import Settings
from ciel.internet import InternetResult, InternetService
from ciel.memory import MemoryType
from ciel.memory.bridge import telegram_conversation_id
from ciel.providers.base import LLMResponse
from ciel.providers.vision import synthetic_test_image
from ciel.screen.service import ScreenResult, ScreenService, screen_scope
from tests.test_ambient import run_callbacks as _run_callbacks

TEXT = "Both monitors show the settings page; the cursor is over Apply."
RESULT = ScreenResult(TEXT, "selected-image-provider", "actual-vision-model", "synthetic-image")


def run_callbacks(tmp_path, monkeypatch, utterances, **kwargs):
    """Use synthetic preferences rather than the developer's local persona file."""
    persona = tmp_path / "synthetic-persona.txt"
    persona.write_text("", encoding="utf-8")
    return _run_callbacks(
        tmp_path, monkeypatch, utterances, persona_file=persona, **kwargs,
    )


def install_screen(monkeypatch, *, on_answer=None):
    """Keep the real resolver and turn marker, replacing all pixel/model operations."""
    created = []
    monkeypatch.setattr("ciel.screen.lifecycle.invalidate_screen_images", Mock())

    def factory(settings, conversation):
        service = ScreenService(settings, conversation, backend=Mock(), vision=Mock())
        service._cache = Mock()
        service._cache.contains.return_value = True
        service._scope = service._scope_key()
        service._epochs = screen_scope(conversation)
        service.answer = Mock(
            return_value=RESULT,
            side_effect=(lambda *args, **kwargs: on_answer(service, *args, **kwargs))
            if on_answer is not None else None,
        )
        service.mark = Mock(wraps=service.mark)
        created.append(service)
        return service

    monkeypatch.setattr("ciel.chat.session.ScreenService", factory)
    monkeypatch.setattr("ciel.screen.service.ScreenService", factory)
    return created


@pytest.fixture
def chat(tmp_path, monkeypatch):
    created = install_screen(monkeypatch)
    settings = Settings(
        _env_file=None, memory_db_path=str(tmp_path / "memory.db"),
        chat_state_path=str(tmp_path / "chat.db"), ciel_persona_file="",
        internet_enabled=False,
    )
    router = Mock()
    router.send.return_value = LLMResponse("An ordinary reply.", "text-model", "text-provider")
    session = ChatSession(settings, channel="telegram", session_id="synthetic-owner", router=router)
    session.internet.prepare = Mock(return_value=InternetResult())
    yield session, created[0]
    session.close()


def assert_visual_turn(turn):
    assert turn.role == "assistant" and turn.content == TEXT
    assert turn.provider == "selected-image-provider"
    assert turn.model == "actual-vision-model"
    assert turn.sources == []


def test_chat_visual_request_bypasses_text_router_and_internet_and_saves_observation(chat):
    session, screen = chat
    assert session.reply("Look at my screen") == TEXT
    screen.answer.assert_called_once()
    assert screen.answer.call_args.args[0].query == "Look at my screen"
    assert not screen.answer.call_args.args[0].previous
    session.router.send.assert_not_called()
    session.internet.prepare.assert_not_called()
    turns = session.conversation.get_recent_turns()
    assert_visual_turn(turns[-1])
    screen.mark.assert_called_once_with(turns[-1].id)
    assert session.store.list_memories(memory_type=MemoryType.FACT) == []


@pytest.mark.parametrize("followup", ["What about now?", "What should I do next?", "Look again"])
def test_chat_active_followup_requests_new_pixels_instead_of_reusing_old_capture(chat, followup):
    session, screen = chat
    assert session.reply("Look at my screen") == TEXT
    assert session.reply(followup) == TEXT
    requests = [call.args[0] for call in screen.answer.call_args_list]
    assert len(requests) == 2
    assert requests[-1].query == followup
    assert not requests[-1].previous


@pytest.mark.parametrize("query", [
    "What's up?", "What about now?", "Do not look at my screen",
    "Please don't take a screenshot", 'He said "look at my screen".',
    "I was wondering how screen capture works.",
])
def test_ordinary_negative_and_reported_chat_text_cannot_acquire_pixels(chat, query):
    session, screen = chat
    session.reply(query)
    screen.answer.assert_not_called()


def test_nonvisual_topic_ends_implicit_visual_followup(chat):
    session, screen = chat
    session.reply("Look at my screen")
    session.reply("Tell me a short story")
    session.reply("What about now?")
    screen.answer.assert_called_once()


@pytest.mark.parametrize("invalidation", ["cancel", "reset", "expired"])
def test_chat_stale_visual_results_are_never_replied_or_persisted(chat, invalidation):
    session, screen = chat
    canceled = threading.Event()

    def answer(*_args, **_kwargs):
        if invalidation == "cancel":
            canceled.set()
        elif invalidation == "reset":
            session.conversation.clear_session()
        else:
            screen._cache.contains.return_value = False
        return RESULT

    screen.answer.side_effect = answer
    assert session.reply("Look at my screen", canceled) == ""
    assert not any(turn.role == "assistant" for turn in session.conversation.get_recent_turns())
    screen.mark.assert_not_called()
    session.router.send.assert_not_called()


def test_reset_then_visual_error_cannot_append_fallback_to_cleared_chat(chat):
    session, screen = chat

    def answer(*_args, **_kwargs):
        session.conversation.clear_session()
        raise RuntimeError("Synthetic image failure after reset")

    screen.answer.side_effect = answer
    assert session.reply("Look at my screen") == ""
    assert session.conversation.get_recent_turns() == []
    screen.mark.assert_not_called()


def test_unpairing_during_visual_answer_revokes_cross_channel_scope(tmp_path, monkeypatch):
    created = install_screen(monkeypatch)
    settings = Settings(
        _env_file=None, memory_db_path=str(tmp_path / "memory.db"),
        chat_services="telegram", telegram_bot_token="123:synthetic-token",
        chat_state_path=str(tmp_path / "chat.db"), ciel_persona_file="", internet_enabled=False,
    )
    state = ChatState(settings.chat_state_path)
    state.bind(123, 42)
    session = ChatSession(
        settings, channel="telegram", session_id=telegram_conversation_id(123, 42), router=Mock(),
    )
    screen = created[0]

    def answer(*_args, **_kwargs):
        state.unpair()
        return RESULT

    screen.answer.side_effect = answer
    try:
        assert session.reply("Look at my screen") == ""
        assert not any(turn.role == "assistant" for turn in session.conversation.get_recent_turns())
        screen.mark.assert_not_called()
    finally:
        session.close()
        state.close()


def test_telegram_image_is_explicit_and_png_never_enters_persisted_history(chat):
    session, screen = chat
    data = synthetic_test_image().data
    screen.answer.side_effect = [RESULT, ScreenResult(
        "Here is the screenshot.", "local", "screen", "synthetic-image", 2_000_000_000, data,
    )]
    ordinary = session.reply("Look at my screen")
    assert type(ordinary) is str
    attachment = session.reply("Send me the screenshot")
    assert isinstance(attachment, ChatReply)
    assert attachment.photos[0].data == data
    assert attachment.photos[0].expires_at == 2_000_000_000
    assert attachment.photos[0].valid()
    screen._cache.contains.return_value = False
    assert not attachment.photos[0].valid()
    turns = session.conversation.get_recent_turns()
    assert [turn.content for turn in turns if turn.role == "assistant"] == [
        TEXT, "Here is the screenshot.",
    ]
    assert all(type(turn.content) is str for turn in turns)
    assert all("data:image" not in turn.content and "PNG" not in turn.content for turn in turns)


def test_history_reset_revokes_prepared_telegram_attachment_before_transport(chat):
    session, screen = chat
    screen.answer.return_value = ScreenResult(
        "Here is the screenshot.", "local", "screen", "synthetic-image", 2_000_000_000,
        synthetic_test_image().data,
    )
    attachment = session.reply("Send me a screenshot")
    assert attachment.photos[0].valid()
    session.conversation.clear_session()
    assert not attachment.photos[0].valid()


def stub_runtime_internet(monkeypatch):
    service = MagicMock()
    service.prepare.return_value = InternetResult()
    factory = MagicMock(return_value=service)
    factory._context = InternetService._context
    monkeypatch.setattr("ciel.internet.InternetService", factory)
    return service


@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("typed", [False, True])
def test_voice_and_keyboard_visual_requests_bypass_general_models(
    tmp_path, monkeypatch, controller, typed,
):
    created = install_screen(monkeypatch)
    internet = stub_runtime_internet(monkeypatch)
    router = MagicMock()
    info = {"input_source": "keyboard"} if typed else {"wake_verified": True}
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Look at my screen", info)], router=router, controller=controller,
    )
    try:
        screen = created[0]
        screen.answer.assert_called_once()
        router.send.assert_not_called()
        router.stream.assert_not_called()
        internet.prepare.assert_not_called()
        internet.lookup.assert_not_called()
        tts.speak.assert_called_once()
        assert tts.speak.call_args.args[0] == TEXT
        assert_visual_turn(turns[-1])
        screen.mark.assert_called_once_with(turns[-1].id)
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_background_visual_mention_does_not_capture_or_speak(tmp_path, monkeypatch, controller):
    created = install_screen(monkeypatch)
    router = MagicMock()
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Mom, look at my screen", {})],
        router=router, controller=controller,
    )
    try:
        assert created[0].answer.call_count == 0
        router.send.assert_not_called()
        tts.speak.assert_not_called()
        assert turns == []
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_canceled_runtime_visual_result_is_never_spoken_or_saved(
    tmp_path, monkeypatch, controller,
):
    canceled = threading.Event()

    def answer(_screen, *_args, **_kwargs):
        canceled.set()
        return RESULT

    created = install_screen(monkeypatch, on_answer=answer)
    router = MagicMock()
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Look at my screen", {"wake_verified": True, "cancel_event": canceled})],
        router=router, controller=controller,
    )
    try:
        created[0].answer.assert_called_once()
        created[0].mark.assert_not_called()
        tts.speak.assert_not_called()
        assert not any(turn.role == "assistant" for turn in turns)
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_runtime_screen_followup_obtains_a_fresh_capture(tmp_path, monkeypatch, controller):
    created = install_screen(monkeypatch)
    router = MagicMock()
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [
            ("Look at my screen", {"input_source": "keyboard"}),
            ("What should I do next?", {"input_source": "keyboard"}),
        ], router=router, controller=controller,
    )
    try:
        requests = [call.args[0] for call in created[0].answer.call_args_list]
        assert len(requests) == 2
        assert requests[1].query == "What should I do next?"
        assert not requests[1].previous
        assert len([turn for turn in turns if turn.role == "assistant"]) == 2
        router.send.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_low_confidence_speech_never_authorizes_capture(tmp_path, monkeypatch, controller):
    created = install_screen(monkeypatch)
    stub_runtime_internet(monkeypatch)
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"action":"answer","confidence":0.99}', "text-model", "text-provider",
    )
    _, _, _, store, _ = run_callbacks(
        tmp_path, monkeypatch,
        [("Look at my screen", {"wake_verified": True, "stt_confidence": 0.1})],
        router=router, controller=controller,
    )
    try:
        assert created[0].answer.call_count == 0
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_reset_while_runtime_visual_request_runs_suppresses_stale_answer(
    tmp_path, monkeypatch, controller,
):
    def answer(screen, *_args, **_kwargs):
        screen.conversation.clear_session()
        return RESULT

    created = install_screen(monkeypatch, on_answer=answer)
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Look at my screen", {"wake_verified": True})],
        router=MagicMock(), controller=controller,
    )
    try:
        created[0].answer.assert_called_once()
        created[0].mark.assert_not_called()
        tts.speak.assert_not_called()
        assert not any(turn.role == "assistant" for turn in turns)
    finally:
        store.close()


def test_reset_then_visual_error_cannot_speak_or_persist_controller_fallback(tmp_path, monkeypatch):
    def answer(screen, *_args, **_kwargs):
        screen.conversation.clear_session()
        raise RuntimeError("Synthetic image failure after reset")

    created = install_screen(monkeypatch, on_answer=answer)
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Look at my screen", {"wake_verified": True})],
        router=MagicMock(), controller=True,
    )
    try:
        created[0].answer.assert_called_once()
        created[0].mark.assert_not_called()
        tts.speak.assert_not_called()
        assert turns == []
    finally:
        store.close()
