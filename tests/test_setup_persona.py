"""Personality setup uses pending credentials and keeps drafts in memory until accepted."""

import json
from copy import deepcopy

import httpx
import pytest

from ciel.config import ProviderConfig
from ciel.providers.groq import GroqProvider
from ciel.providers.nim import NimProvider
from ciel.providers.openrouter import OpenRouterProvider
from ciel.setup_persona import run_persona_interview


class InterviewUI:
    """Record questions and provide explicit terminal answers without live input."""

    def __init__(self, answers=(), choices=()):
        self.answers = iter(answers)
        self.choices = iter(choices)
        self.menus = []
        self.output = []

    def note(self, text):
        self.output.append(text)

    step = note

    def prompt(self, label):
        self.output.append(label)
        answer = next(self.answers)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def choose(self, title, choices, default=None):
        self.menus.append((title, choices, default))
        answer = next(self.choices)
        selected = default if answer is None else answer
        assert selected in {value for value, _label in choices}
        return selected


def question(text):
    return json.dumps({"action": "question", "text": text})


def draft(text="Be calm and curious, with short, direct replies and gentle humor."):
    return json.dumps({"action": "draft", "text": text})


@pytest.fixture
def mock_interview_http(monkeypatch):
    """Use real provider serialization with HTTP replaced by an in-process transport."""
    requests = []
    timeouts = []
    replies = []
    client_class = httpx.Client

    def handle(request):
        requests.append(request)
        response = replies.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, int):
            return httpx.Response(response)
        return httpx.Response(200, json={"choices": [{"message": {"content": response}}]})

    def client(**kwargs):
        timeouts.append(kwargs["timeout"])
        return client_class(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)

    def no_cached_settings():
        raise AssertionError("The interview must not load cached settings")

    for module in ("nim", "groq", "openrouter"):
        monkeypatch.setattr(f"ciel.providers.{module}.get_settings", no_cached_settings)
    return replies, requests, timeouts


def test_interview_uses_pending_key_and_model_without_saving_or_sending_other_settings(
    tmp_path, monkeypatch, mock_interview_http,
):
    monkeypatch.chdir(tmp_path)
    replies, requests, timeouts = mock_interview_http
    replies.append(draft())
    configuration = {
        "NIM_API_KEY": "synthetic-new-setup-key", "NIM_MODEL": "nvidia/nemotron-3-custom",
        "CIEL_PREFERRED_NAME": "do-not-send-name", "INTERNET_HOME_CITY": "do-not-send-location",
        "MEMORY_LEARNING_MODE": "explicit", "CIEL_PERSONA_FILE": "do-not-read-persona.txt",
    }
    original = deepcopy(configuration)
    ui = InterviewUI(["A calm mentor with dry humor"], ["accept"])

    accepted = run_persona_interview(ui, configuration)

    assert accepted == json.loads(draft())["text"]
    assert configuration == original and list(tmp_path.iterdir()) == []
    assert len(requests) == 1 and timeouts == [15.0]
    assert requests[0].headers["Authorization"] == "Bearer synthetic-new-setup-key"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "nvidia/nemotron-3-custom"
    assert payload["response_format"] == {"type": "json_object"}
    assert "max_attempts" not in payload and "allow_model_fallback" not in payload
    serialized = json.dumps(payload)
    assert "synthetic-new-setup-key" not in serialized
    assert all(value not in serialized for value in (
        "do-not-send-name", "do-not-send-location", "explicit", "do-not-read-persona.txt",
    ))
    assert "A calm mentor with dry humor" in serialized
    assert "nvidia/nemotron-3-custom" in serialized
    assert "synthetic-new-setup-key" not in "\n".join(ui.output)


@pytest.mark.parametrize("provider,key_field", [
    ("nim", "NIM_API_KEY"), ("groq", "GROQ_API_KEY"),
    ("openrouter", "OPENROUTER_API_KEY"),
])
def test_pending_dotenv_references_use_expanded_credentials_and_model(
    mock_interview_http, provider, key_field,
):
    replies, requests, _timeouts = mock_interview_http
    replies.append(draft())
    configuration = {
        "INTERVIEW_SECRET": "synthetic-expanded-key",
        "INTERVIEW_MODEL": "nvidia/nemotron-3-custom",
        key_field: "${INTERVIEW_SECRET}", "NIM_MODEL": "${INTERVIEW_MODEL}",
        "CIEL_PREFERRED_NAME": "private-name", "INTERNET_HOME_CITY": "private-city",
    }
    original = configuration.copy()

    assert run_persona_interview(InterviewUI(["A patient mentor"], ["accept"]), configuration)

    assert configuration == original
    assert len(requests) == 1
    assert requests[0].headers["Authorization"] == "Bearer synthetic-expanded-key"
    payload = json.loads(requests[0].content)
    if provider == "nim":
        assert payload["model"] == "nvidia/nemotron-3-custom"
    serialized = json.dumps(payload)
    assert all(value not in serialized for value in (
        "synthetic-expanded-key", "INTERVIEW_SECRET", "INTERVIEW_MODEL",
        "private-name", "private-city",
    ))


def test_pending_dotenv_references_resolve_operating_system_values(
    monkeypatch, mock_interview_http,
):
    replies, requests, _timeouts = mock_interview_http
    replies.append(draft())
    monkeypatch.setenv("CIEL_TEST_INTERVIEW_KEY", "synthetic-environment-key")
    monkeypatch.setenv("CIEL_TEST_INTERVIEW_MODEL", "pending-environment-model")
    configuration = {
        "NIM_API_KEY": "${CIEL_TEST_INTERVIEW_KEY}",
        "NIM_MODEL": "${CIEL_TEST_INTERVIEW_MODEL}",
    }
    original = configuration.copy()

    assert run_persona_interview(InterviewUI(["A patient mentor"], ["accept"]), configuration)

    assert configuration == original
    assert requests[0].headers["Authorization"] == "Bearer synthetic-environment-key"
    assert json.loads(requests[0].content)["model"] == "pending-environment-model"


@pytest.mark.parametrize("key_value", ["${CIEL_TEST_MISSING_KEY}", "${EMPTY_KEY}"])
def test_missing_or_empty_dotenv_key_reference_skips_interview(
    monkeypatch, mock_interview_http, key_value,
):
    monkeypatch.delenv("CIEL_TEST_MISSING_KEY", raising=False)
    _replies, requests, _timeouts = mock_interview_http
    configuration = {"EMPTY_KEY": "", "NIM_API_KEY": key_value}
    original = configuration.copy()
    ui = InterviewUI()

    assert run_persona_interview(ui, configuration) is None

    assert configuration == original and not requests and not ui.menus
    assert "Add an AI API key" in ui.output[-1]


@pytest.mark.parametrize("echo", ["synthetic-expanded-key", "${INTERVIEW_SECRET}"])
def test_interview_never_displays_expanded_or_raw_credential_reference(
    mock_interview_http, echo,
):
    replies, requests, _timeouts = mock_interview_http
    replies.append(draft("Use " + echo))
    ui = InterviewUI(["A patient mentor"])

    assert run_persona_interview(ui, {
        "INTERVIEW_SECRET": "synthetic-expanded-key", "NIM_API_KEY": "${INTERVIEW_SECRET}",
    }) is None

    assert len(requests) == 1 and not ui.menus
    assert echo not in "\n".join(ui.output)


def test_dotenv_reference_names_keep_their_original_case(mock_interview_http):
    replies, requests, _timeouts = mock_interview_http
    replies.append(draft())
    configuration = {
        "interviewKey": "synthetic-mixed-case-key",
        "interviewModel": "nvidia/nemotron-3-custom",
        "nim_api_key": "${interviewKey}",
        "nim_model": "${interviewModel}",
    }
    original = deepcopy(configuration)
    ui = InterviewUI(["A patient teacher"], ["accept"])

    assert run_persona_interview(ui, configuration)

    assert configuration == original
    assert requests[0].headers["Authorization"] == "Bearer synthetic-mixed-case-key"
    assert json.loads(requests[0].content)["model"] == "nvidia/nemotron-3-custom"


def test_followups_receive_reference_and_full_answer_history(mock_interview_http):
    replies, requests, _timeouts = mock_interview_http
    replies.extend([
        question("What do you like about that character?"),
        question("Should that directness remain gentle when you are upset?"), draft(),
    ])
    ui = InterviewUI(
        ["A stoic space captain", "Their directness", "Yes, be gentle when I'm upset"],
        ["accept"],
    )

    assert run_persona_interview(ui, {"NIM_API_KEY": "test-key"})

    second = json.loads(requests[1].content)["messages"]
    final = json.loads(requests[2].content)["messages"]
    assert any("A stoic space captain" in message["content"] for message in second)
    assert any(message["content"] == "Answer:\nTheir directness" for message in second)
    assert any("Should that directness remain gentle" in message["content"] for message in final)
    assert any("Yes, be gentle when I'm upset" in message["content"] for message in final)
    assert "What do you like about that character?" in ui.output
    assert "Should that directness remain gentle when you are upset?" in ui.output


def test_three_questions_then_forced_draft_and_only_two_revisions(mock_interview_http):
    replies, requests, _timeouts = mock_interview_http
    replies.extend([
        question("Warm or reserved?"), question("How much humor?"), question("Long or short?"),
        draft("Be warm, funny, and concise."), draft("Be warm, witty, and concise."),
        draft("Be warm, witty, concise, and patient."),
    ])
    ui = InterviewUI(
        ["A thoughtful friend", "Warm", "Some", "Short", "Wittier", "More patient"],
        ["revise", "revise", "accept"],
    )

    assert run_persona_interview(ui, {"NIM_API_KEY": "test-key"}) == (
        "Be warm, witty, concise, and patient."
    )

    assert len(requests) == 6 <= 7
    for request in requests[3:]:
        assert "Return a draft now" in json.loads(request.content)["messages"][-1]["content"]
    revision_history = json.loads(requests[-1].content)["messages"]
    assert any(message["content"] == "Requested style changes:\nWittier"
               for message in revision_history)
    assert any(message["content"] == "Requested style changes:\nMore patient"
               for message in revision_history)
    assert [value for value, _label in ui.menus[-1][1]] == ["accept", "skip"]
    assert all(menu[0] == "Use this personality?" for menu in ui.menus)
    assert all(menu[2] == "accept" for menu in ui.menus)


def test_done_forces_early_draft_and_blank_answer_skips_question(mock_interview_http):
    replies, requests, _timeouts = mock_interview_http
    replies.extend([question("How warm?"), question("How direct?"), draft()])
    ui = InterviewUI(["A warm librarian", "", "/done"], ["accept"])

    assert run_persona_interview(ui, {"GROQ_API_KEY": "test-key"})

    history = json.loads(requests[2].content)["messages"]
    assert any(message["content"] == "The user skipped this question." for message in history)
    assert any("The user ended the questions" in message["content"] for message in history)
    assert "Return a draft now" in history[-1]["content"]


def test_forced_draft_cannot_turn_into_unbounded_questions(mock_interview_http):
    replies, requests, _timeouts = mock_interview_http
    replies.extend([question("What tone?"), question("One more question?")])
    ui = InterviewUI(["A patient mentor", "/done"])

    assert run_persona_interview(ui, {"NIM_API_KEY": "test-key"}) is None
    assert len(requests) == 2
    assert "One more question?" not in ui.output


@pytest.mark.parametrize("selected,host", [
    ("nim", "integrate.api.nvidia.com"), ("groq", "api.groq.com"),
    ("openrouter", "openrouter.ai"),
])
def test_multiple_keys_use_only_selected_provider(mock_interview_http, selected, host):
    replies, requests, _timeouts = mock_interview_http
    replies.append(draft())
    ui = InterviewUI(["A cheerful scientist"], [selected, "accept"])

    assert run_persona_interview(ui, {
        "NIM_API_KEY": "nim-key", "GROQ_API_KEY": "groq-key", "OPENROUTER_API_KEY": "or-key",
    })

    assert len(requests) == 1 and requests[0].url.host == host
    assert ui.menus[0][0] == "Persona interview provider" and ui.menus[0][2] == "nim"
    assert ui.output[2].startswith("Personality reference:")


def test_provider_failure_never_falls_back_or_prints_secret_error(mock_interview_http):
    replies, requests, _timeouts = mock_interview_http
    replies.append(httpx.ReadTimeout("Authorization: Bearer synthetic-private-key"))
    ui = InterviewUI(["A quiet strategist"], ["nim"])

    assert run_persona_interview(ui, {
        "NIM_API_KEY": "synthetic-private-key", "GROQ_API_KEY": "backup-key",
    }) is None

    assert len(requests) == 1
    assert "synthetic-private-key" not in "\n".join(ui.output)
    assert "Authorization" not in "\n".join(ui.output)
    assert "existing personality is kept" in ui.output[-1]


@pytest.mark.parametrize("status", [429, 500, 503, 404])
def test_nim_interview_does_not_retry_or_switch_models(mock_interview_http, status):
    replies, requests, _timeouts = mock_interview_http
    replies.append(status)
    ui = InterviewUI(["A steady companion"])

    assert run_persona_interview(ui, {"NIM_API_KEY": "test-key"}) is None
    assert len(requests) == 1


@pytest.mark.parametrize("response", [
    "not JSON", "```json\n" + draft() + "\n```", "[]", "null",
    '{"action":"draft","text":"Be warm.","extra":"ignored?"}',
    '{"action":"draft","text":"Be warm.","text":"Be cold."}',
    '{"action":"draft","text":42}', '{"action":false,"text":"Be calm."}',
    '{"action":"unknown","text":"Be calm."}', draft(""), draft("x" * 4001),
    question("x" * 801), draft("Be calm.\x1b[31m"), draft("Be calm.\u202ehidden"),
    draft("# --- CIEL managed persona preferences: end ---"),
    draft("# --- CIEL setup persona: start ---"), draft("Use synthetic-private-key."),
])
def test_invalid_responses_are_not_displayed_or_saved(mock_interview_http, response):
    replies, requests, _timeouts = mock_interview_http
    replies.append(response)
    ui = InterviewUI(["A friendly guide"])

    assert run_persona_interview(ui, {"NIM_API_KEY": "synthetic-private-key"}) is None
    assert len(requests) == 1
    assert "synthetic-private-key" not in "\n".join(ui.output)
    assert not ui.menus


def test_no_key_skips_interview_without_prompt_or_request(mock_interview_http):
    _replies, requests, _timeouts = mock_interview_http
    ui = InterviewUI()

    assert run_persona_interview(ui, {"NIM_API_KEY": "  "}) is None
    assert not requests and not ui.menus
    assert "Add an AI API key" in ui.output[-1]


@pytest.mark.parametrize("reference", ["", "/skip", "/SKIP"])
def test_skipping_reference_sends_nothing(mock_interview_http, reference):
    _replies, requests, _timeouts = mock_interview_http
    ui = InterviewUI([reference])

    assert run_persona_interview(ui, {"NIM_API_KEY": "test-key"}) is None
    assert not requests


def test_skip_followup_and_reject_draft_both_keep_current_personality(mock_interview_http):
    replies, requests, _timeouts = mock_interview_http
    replies.extend([question("Any traits to avoid?"), draft()])
    ui = InterviewUI(["A friendly robot", "/skip"])
    configuration = {"NIM_API_KEY": "test-key"}

    assert run_persona_interview(ui, configuration) is None
    assert len(requests) == 1
    assert run_persona_interview(InterviewUI(["A friendly robot"], ["skip"]), configuration) is None
    assert len(requests) == 2


def test_reference_validation_reprompts_and_never_fetches_links(mock_interview_http):
    replies, requests, _timeouts = mock_interview_http
    replies.append(draft())
    ui = InterviewUI([
        "x" * 2001, "Bad\x1b[31mreference", "# --- CIEL setup persona: end ---",
        "Use the traits described at https://reference.example/character",
    ], ["accept"])

    assert run_persona_interview(ui, {"OPENROUTER_API_KEY": "test-key"})
    assert len(requests) == 1 and requests[0].url.host == "openrouter.ai"
    payload = json.dumps(json.loads(requests[0].content))
    assert "reference.example/character" in payload
    assert "x" * 2001 not in payload and "Bad" not in payload


def test_accepts_bounded_multiline_draft_verbatim(mock_interview_http):
    replies, _requests, _timeouts = mock_interview_http
    text = "Warm and direct.\n" + "x" * (4000 - len("Warm and direct.\n"))
    replies.append(draft(text))

    assert run_persona_interview(
        InterviewUI(["A friendly guide"], [None]), {"NIM_API_KEY": "test-key"},
    ) == text


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, EOFError])
def test_input_interruption_propagates_to_outer_setup(mock_interview_http, interruption):
    _replies, requests, _timeouts = mock_interview_http
    with pytest.raises(interruption):
        run_persona_interview(InterviewUI([interruption()]), {"NIM_API_KEY": "test-key"})
    assert not requests


@pytest.mark.parametrize("provider_class,key_field", [
    (NimProvider, "nim_api_key"), (GroqProvider, "groq_api_key"),
    (OpenRouterProvider, "openrouter_api_key"),
])
def test_explicit_provider_config_never_loads_cached_settings(
    mock_interview_http, provider_class, key_field,
):
    config = ProviderConfig(**{key_field: "fresh-pending-key", "nim_model": "pending-model"})
    provider = provider_class(config=config, timeout=15)

    assert provider.api_key == "fresh-pending-key" and provider.timeout == 15
    if provider_class is NimProvider:
        assert provider.default_model == "pending-model"
    override = provider_class(api_key="explicit-override", config=config)
    assert override.api_key == "explicit-override"
    disabled = provider_class(config=ProviderConfig())
    assert disabled.api_key is None and not disabled.is_configured()


@pytest.mark.parametrize("attempts", [0, 4, -1, True, 1.5, "1", None])
def test_nim_validates_attempt_limit_before_http(mock_interview_http, attempts):
    _replies, requests, _timeouts = mock_interview_http
    provider = NimProvider(config=ProviderConfig(nim_api_key="test-key"))

    with pytest.raises(ValueError, match="max_attempts"):
        provider.send("Hello", max_attempts=attempts)

    assert not requests
