"""Image requests preserve selected connection, privacy, cancellation, and text contracts."""

import base64
import threading
from contextlib import nullcontext
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError

from ciel import setup_wizard
from ciel.config import DEFAULT_SCREEN_MODEL, Settings
from ciel.providers import vision
from ciel.providers.base import ChatImage, ChatMessage
from ciel.providers.manager import ProviderManager
from ciel.providers.router import ModelRouter
from ciel.providers.vision import VisionCancelled, VisionClient, VisionError, synthetic_test_image


def configured(**overrides):
    """Read no local env file and never use real account credentials in tests."""
    return Settings(_env_file=None, nim_api_key="synthetic-nim-key", **overrides)


def response(content="LEFT=red; RIGHT=blue", *, status=200, **extra):
    return httpx.Response(
        status, request=httpx.Request("POST", "https://example.test/image"),
        json={"choices": [{"message": {"content": content}}], **extra},
    )


@pytest.fixture
def connection(monkeypatch):
    client = MagicMock()
    client.post.return_value = response()
    monkeypatch.setattr(vision, "streaming_client", lambda *_args: nullcontext(client))
    return client


def personal_message(**kwargs):
    return ChatMessage(
        "user", "Explain this screen.", images=(ChatImage(b"private-pixels"),), **kwargs,
    )


def test_text_serialization_is_unchanged_and_image_bytes_stay_out_of_repr():
    assert ChatMessage("user", "hello").to_dict() == {"role": "user", "content": "hello"}
    image = ChatImage(b"private-pixels")
    message = personal_message(local_only=True)
    assert "private-pixels" not in repr(image) + repr(message)
    assert message.content == "Explain this screen."
    assert "local_only" not in message.to_dict()
    encoded = base64.b64encode(b"private-pixels").decode()
    assert message.to_dict() == {
        "role": "user", "content": [
            {"type": "text", "text": message.content},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{encoded}", "detail": "auto",
            }},
        ],
    }
    assert message.to_ollama_dict() == {
        "role": "user", "content": message.content, "images": [encoded],
    }


@pytest.mark.parametrize("kwargs", [
    {"data": b""}, {"data": b"x", "mime_type": "image/svg+xml"},
    {"data": b"x", "detail": "arbitrary"}, {"data": "not-bytes"},
    {"data": b"x" * (8 * 1024 * 1024 + 1)},
])
def test_invalid_or_unbounded_attachment_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ChatImage(**kwargs)


def test_only_user_messages_accept_typed_tuple_images():
    with pytest.raises(ValueError):
        ChatMessage("system", "policy", images=(ChatImage(b"x"),))
    with pytest.raises(ValueError):
        ChatMessage("user", "question", images=[ChatImage(b"x")])


def test_selected_model_gets_synthetic_probe_before_any_personal_pixels(connection):
    connection.post.side_effect = [response(), response("Visible screen"), response("Follow-up")]
    client = VisionClient(configured())
    assert client.send([personal_message()]).content == "Visible screen"
    assert client.send([personal_message()]).content == "Follow-up"
    calls = connection.post.call_args_list
    assert len(calls) == 3  # Verify once, then two requested answers.
    assert {call.kwargs["json"]["model"] for call in calls} == {DEFAULT_SCREEN_MODEL}
    first = calls[0].kwargs["json"]["messages"][0]
    personal = calls[1].kwargs["json"]["messages"][0]
    assert first["content"][1]["image_url"]["url"] != personal["content"][1]["image_url"]["url"]
    assert "Explain this screen" not in str(first)
    assert calls[1].kwargs["headers"]["Authorization"] == "Bearer synthetic-nim-key"


def test_failed_synthetic_check_never_sends_screen_and_never_falls_back(connection):
    connection.post.return_value = response("I cannot read images")
    client = VisionClient(configured(groq_api_key="synthetic-spare-key"))
    with pytest.raises(VisionError, match="synthetic image check"):
        client.send([personal_message()])
    assert connection.post.call_count == 1
    assert connection.post.call_args.args[0].startswith("https://integrate.api.nvidia.com/")


def test_unverified_multi_image_requests_fail_before_any_provider_call(connection):
    with pytest.raises(VisionError, match="one image"):
        VisionClient(configured()).send([personal_message(), personal_message()])
    connection.post.assert_not_called()


@pytest.mark.parametrize("messages", [
    [personal_message(local_only=True)],
    [ChatMessage("system", "Protected summary", local_only=True), personal_message()],
    [ChatMessage("user", "/local confidential"), personal_message()],
])
def test_legacy_private_history_fails_closed_before_probe(connection, messages):
    with pytest.raises(VisionError, match="protected local history"):
        VisionClient(configured()).send(messages)
    connection.post.assert_not_called()


@pytest.mark.parametrize("override,match", [
    ({"screen_enabled": False}, "disabled"),
    ({"nim_api_key": None}, "API key"),
    ({"screen_model": " "}, "image-capable"),
])
def test_unusable_selected_connection_never_borrows_another_key(connection, override, match):
    values = {"nim_api_key": "synthetic-nim-key", "groq_api_key": "synthetic-spare", **override}
    with pytest.raises(VisionError, match=match):
        VisionClient(Settings(_env_file=None, **values)).send([personal_message()])
    connection.post.assert_not_called()


def test_ollama_images_use_native_wire_shape_and_preserve_private_policy(connection):
    connection.post.side_effect = [
        response(message={"content": "LEFT=red; RIGHT=blue"}),
        response(message={"content": "Local answer"}),
    ]
    client = VisionClient(configured(screen_provider="ollama", screen_model="installed-vision"))
    assert client.send([personal_message(local_only=True)]).content == "Local answer"
    call = connection.post.call_args
    assert call.args[0] == "http://localhost:11434/api/chat"
    assert "Authorization" not in call.kwargs["headers"]
    message = call.kwargs["json"]["messages"][0]
    assert message["images"] == [base64.b64encode(b"private-pixels").decode()]
    assert message["content"] == "Explain this screen."


@pytest.mark.parametrize("provider", ["groq", "openrouter"])
def test_other_explicit_connections_use_only_their_saved_key(connection, provider):
    settings = configured(**{
        "screen_provider": provider, "screen_model": "selected-image-model",
        f"{provider}_api_key": "synthetic-selected-key",
    })
    VisionClient(settings).send([personal_message()])
    for call in connection.post.call_args_list:
        assert provider in call.args[0]
        assert call.kwargs["headers"]["Authorization"] == "Bearer synthetic-selected-key"
        assert call.kwargs["json"]["model"] == "selected-image-model"


def test_failure_messages_do_not_contain_payload_or_credentials(connection):
    connection.post.return_value = response(status=404, error={"payload": "private-pixels"})
    with pytest.raises(VisionError, match="HTTP 404") as failure:
        VisionClient(configured()).send([personal_message()])
    assert "private-pixels" not in str(failure.value)
    assert "synthetic-nim-key" not in str(failure.value)
    assert connection.post.call_count == 1


def test_timeout_is_actionable_and_does_not_retry_or_send_personal_image(connection):
    connection.post.side_effect = httpx.ReadTimeout("sensitive request detail")
    with pytest.raises(VisionError, match="timed out") as failure:
        VisionClient(configured()).send([personal_message()])
    assert "sensitive" not in str(failure.value)
    assert connection.post.call_count == 1


def test_cancel_before_request_never_connects(connection):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(VisionCancelled):
        VisionClient(configured()).send([personal_message()], cancel)
    connection.post.assert_not_called()


def test_cancel_during_probe_suppresses_personal_request(connection):
    cancel = threading.Event()
    def probe(*_args, **_kwargs):
        cancel.set()
        return response()
    connection.post.side_effect = probe
    with pytest.raises(VisionCancelled):
        VisionClient(configured()).send([personal_message()], cancel)
    assert connection.post.call_count == 1


def test_cancel_aborts_blocked_http_client(monkeypatch):
    entered, closed, cancel, finished = (threading.Event() for _ in range(4))
    class BlockedClient:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
        def close(self):
            closed.set()
        def post(self, *args, **kwargs):
            entered.set()
            assert closed.wait(2)
            raise httpx.ReadError("cancelled read")
    monkeypatch.setattr("ciel.providers.streaming.httpx.Client", BlockedClient)
    errors = []
    def run():
        try:
            VisionClient(configured()).send([personal_message()], cancel)
        except VisionCancelled as error:
            errors.append(error)
        finally:
            finished.set()
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(1)
        cancel.set()
        assert finished.wait(1)
        assert errors
    finally:
        cancel.set()
        closed.set()
        worker.join(2)


def test_request_deadline_closes_a_stalled_connection(monkeypatch):
    closed = threading.Event()
    class StalledClient:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
        def close(self):
            closed.set()
        def post(self, *args, **kwargs):
            assert closed.wait(2), "The overall request deadline did not close the client"
            raise httpx.ReadError("stalled")
    monkeypatch.setattr("ciel.providers.streaming.httpx.Client", StalledClient)
    with pytest.raises(VisionError, match="timed out"):
        VisionClient(configured(screen_timeout_seconds=1)).send([personal_message()])
    assert closed.is_set()


def test_screen_timeout_settings_are_bounded():
    for timeout in (0, 121):
        with pytest.raises(ValidationError):
            configured(screen_timeout_seconds=timeout)
    assert configured().screen_timeout_seconds == 60
    assert synthetic_test_image().data.startswith(b"\x89PNG\r\n\x1a\n")


def test_setup_screen_uses_existing_provider_without_requesting_another_key():
    ui = MagicMock()
    ui.choose.side_effect = ["groq", "custom"]
    ui.prompt.return_value = "explicit-image-model"
    saved = setup_wizard._initial_configuration({"GROQ_API_KEY": "synthetic-existing"})
    setup_wizard._configure_screen(ui, saved)
    offered = dict(ui.choose.call_args_list[0].args[1])
    assert "groq" in offered and "nim" not in offered and "openrouter" not in offered
    assert saved["SCREEN_PROVIDER"] == "groq"
    assert saved["SCREEN_MODEL"] == "explicit-image-model"
    assert saved["GROQ_API_KEY"] == "synthetic-existing"
    assert saved["SCREEN_ENABLED"] == "true"


def test_setup_can_disable_screen_without_erasing_saved_connection():
    ui = MagicMock()
    ui.choose.return_value = "off"
    saved = {"SCREEN_MODEL": "saved-model", "SCREEN_PROVIDER": "nim", "NIM_API_KEY": "saved"}
    setup_wizard._configure_screen(ui, saved)
    assert saved["SCREEN_ENABLED"] == "false"
    assert saved["SCREEN_MODEL"] == "saved-model" and saved["NIM_API_KEY"] == "saved"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("protected", [False, True])
def test_text_fallback_manager_rejects_images_before_provider_access_or_logging(
    monkeypatch, stream, protected,
):
    manager = ProviderManager()
    provider = MagicMock()
    manager.providers = {"nim": provider, "ollama": provider}
    logger = MagicMock()
    monkeypatch.setattr("ciel.providers.manager.logger", logger)
    messages = [ChatMessage("user", "Earlier text"), personal_message(local_only=protected)]
    with pytest.raises(ValueError, match="VisionClient"):
        if stream:
            list(manager.stream_with_fallback(messages, preferred_provider="nim"))
        else:
            manager.send_with_fallback(messages, preferred_provider="nim")
    assert provider.mock_calls == []
    assert logger.mock_calls == []
    assert messages[-1].images  # Nothing silently stripped from the original input.


@pytest.mark.parametrize("stream,purpose", [
    (False, "conversation"), (True, "conversation"),
    (False, "controller"), (False, "summary"), (False, "speech_gate"),
])
def test_text_router_rejects_images_before_classification_logging_or_fallback(
    monkeypatch, stream, purpose,
):
    manager = MagicMock()
    router = ModelRouter(manager)
    manager.reset_mock()
    router.route = MagicMock()
    router.classify_complexity = MagicMock()
    logger = MagicMock()
    marker = MagicMock()
    monkeypatch.setattr("ciel.providers.router.logger", logger)
    monkeypatch.setattr("ciel.providers.router.mark", marker)
    # Even an attachment in older history must not disappear during prefix cleanup.
    messages = [personal_message(), ChatMessage("user", "/fast explain it")]
    with pytest.raises(ValueError, match="VisionClient"):
        if stream:
            list(router.stream(messages))
        else:
            router.send(messages, purpose=purpose)
    router.route.assert_not_called()
    router.classify_complexity.assert_not_called()
    assert manager.mock_calls == []
    assert logger.mock_calls == []
    marker.assert_not_called()
    assert messages[0].images
