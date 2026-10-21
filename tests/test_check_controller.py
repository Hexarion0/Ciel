"""The live diagnostic must distinguish model responses from local safety fallbacks."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ciel.config import ControllerConfig
from ciel.controller import ControllerDecision
from ciel.providers.base import LLMResponse
from scripts import check_controller


@pytest.mark.parametrize("case", ["ambiguous acceptance", "memory"])
@pytest.mark.parametrize("failure", [
    TimeoutError("synthetic provider timeout"),
    LLMResponse("not json", "synthetic", "nim"),
    LLMResponse('{"action":"answer","confidence":0.1}', "synthetic", "nim"),
])
def test_local_override_cannot_hide_model_failure(case, failure, monkeypatch, capsys):
    router = MagicMock()
    if isinstance(failure, Exception):
        router.send.side_effect = failure
    else:
        router.send.return_value = failure
    monkeypatch.setattr(check_controller, "ModelRouter", lambda: router)
    monkeypatch.setattr(check_controller, "get_settings", lambda: SimpleNamespace(
        providers=SimpleNamespace(nim_api_key=object()),
        controller=ControllerConfig(model="synthetic", timeout_seconds=0.1),
    ))
    monkeypatch.setattr("sys.argv", ["check_controller.py", case])

    assert check_controller.main() == 1

    lines = capsys.readouterr().out.splitlines()
    report = json.loads(lines[0])
    assert report["case"] == case and report["passed"] is False
    assert report["model_decision_valid"] is False
    assert report["model_error"]
    assert "0/1 checks passed" in lines[-1]
    expected_calls = 2 if isinstance(failure, LLMResponse) and failure.content == "not json" else 1
    assert router.send.call_count == expected_calls


@pytest.mark.parametrize("case,action", [("ambiguous acceptance", "clarify"), ("memory", "local")])
def test_successful_model_decision_can_pass_selected_case(case, action, monkeypatch, capsys):
    router = MagicMock()
    router.send.return_value = LLMResponse(
        json.dumps({"action": action, "confidence": 0.98}), "synthetic", "nim",
    )
    monkeypatch.setattr(check_controller, "ModelRouter", lambda: router)
    monkeypatch.setattr(check_controller, "get_settings", lambda: SimpleNamespace(
        providers=SimpleNamespace(nim_api_key=object()),
        controller=ControllerConfig(model="synthetic", timeout_seconds=0.1),
    ))
    monkeypatch.setattr("sys.argv", ["check_controller.py", case])

    assert check_controller.main() == 0

    lines = capsys.readouterr().out.splitlines()
    report = json.loads(lines[0])
    assert report["passed"] is True and report["model_decision_valid"] is True
    assert report["model_error"] is None
    assert "1/1 checks passed" in lines[-1]
    router.send.assert_called_once()


def test_late_valid_reply_cannot_hide_controller_timeout(monkeypatch, capsys):
    router = MagicMock()

    def stalled(_messages, **kwargs):
        kwargs["cancel_event"].wait(1)
        return LLMResponse('{"action":"local","confidence":0.98}', "synthetic", "nim")

    router.send.side_effect = stalled
    monkeypatch.setattr(check_controller, "ModelRouter", lambda: router)
    monkeypatch.setattr(check_controller, "get_settings", lambda: SimpleNamespace(
        providers=SimpleNamespace(nim_api_key=object()),
        controller=ControllerConfig(model="synthetic", timeout_seconds=0.1),
    ))
    monkeypatch.setattr("sys.argv", ["check_controller.py", "memory"])

    assert check_controller.main() == 1

    report = json.loads(capsys.readouterr().out.splitlines()[0])
    assert report["passed"] is False and report["model_decision_valid"] is False
    assert report["model_error"]
    router.send.assert_called_once()


def test_full_check_still_validates_source_backed_report(monkeypatch, capsys):
    router = MagicMock()
    quote = "Amber Works season 4 part 2 ends at light novel volume 11."

    def send(_messages, **kwargs):
        content = (
            json.dumps({"claims": [{"text": "That part ends at light novel volume 11.",
                                    "source": 1, "quote": quote}]})
            if "WEB ANSWER FORMAT FOR THIS TURN" in _messages[0].content else
            '{"action":"answer","confidence":0.98}'
        )
        return LLMResponse(content, "synthetic", "nim")

    router.send.side_effect = send
    monkeypatch.setattr(check_controller, "ModelRouter", lambda: router)
    monkeypatch.setattr(check_controller, "get_settings", lambda: SimpleNamespace(
        providers=SimpleNamespace(nim_api_key=object()),
        controller=ControllerConfig(model="synthetic", timeout_seconds=0.1),
    ))
    monkeypatch.setattr("sys.argv", ["check_controller.py"])

    assert check_controller.main() == 1  # Deliberately incorrect routing replies fail their cases.

    lines = capsys.readouterr().out.splitlines()
    report = json.loads(lines[-2])
    assert report["case"] == "validated report" and report["passed"] is True
    assert "11" in report["text"]
    assert any("WEB ANSWER FORMAT FOR THIS TURN" in call.args[0][0].content
               for call in router.send.call_args_list)


def test_probe_does_not_prevent_controller_repair():
    response = LLMResponse("not json", "synthetic", "groq")
    router = MagicMock()
    router.send.return_value = response
    probe = check_controller._DecisionProbe(router, timeout=1)

    assert probe.send([]) is response
    assert probe.result()[0] is False
    assert probe.attempts()[0]["provider"] == "groq"
    assert probe.attempts()[0]["model"] == "synthetic"
    assert probe.attempts()[0]["valid"] is False


def test_successful_repair_keeps_failed_attempt_visible(monkeypatch, capsys):
    router = MagicMock()
    router.send.side_effect = [
        LLMResponse("not json", "first", "groq"),
        LLMResponse('{"action":"clarify","confidence":0.98}', "second", "openrouter"),
    ]

    monkeypatch.setattr(check_controller, "ModelRouter", lambda: router)
    monkeypatch.setattr(check_controller, "get_settings", lambda: SimpleNamespace(
        providers=SimpleNamespace(nim_api_key=None, groq_api_key=object()),
        controller=ControllerConfig(model="synthetic", timeout_seconds=1),
    ))
    monkeypatch.setattr("sys.argv", ["check_controller.py", "ambiguous acceptance"])

    assert check_controller.main() == 0

    report = json.loads(capsys.readouterr().out.splitlines()[0])
    assert report["model_decision_valid"] is True
    assert report["model_repaired"] is True
    assert report["decision_method"] == "model"
    assert [attempt["valid"] for attempt in report["model_attempts"]] == [False, True]
    assert [attempt["provider"] for attempt in report["model_attempts"]] == [
        "groq", "openrouter",
    ]
    assert report["model_attempts"][0]["error"]
    assert report["model_error"] is None


@pytest.mark.parametrize("case,allowed", [("background", True), ("ambiguous acceptance", False)])
def test_no_model_success_requires_an_intentionally_deterministic_case(
    case, allowed, monkeypatch, capsys,
):
    router = MagicMock()

    class DeterministicController:
        def __init__(self, _probe, _config):
            pass

        def decide(self, *_args, **_kwargs):
            return ControllerDecision(
                intent="other_listener", action="silent", confidence=1,
            )

    monkeypatch.setattr(check_controller, "AIController", DeterministicController)
    monkeypatch.setattr(check_controller, "ModelRouter", lambda: router)
    monkeypatch.setattr(check_controller, "get_settings", lambda: SimpleNamespace(
        providers=SimpleNamespace(nim_api_key=None),
        controller=ControllerConfig(model="synthetic", timeout_seconds=1),
    ))
    monkeypatch.setattr("sys.argv", ["check_controller.py", case])

    assert check_controller.main() == (0 if allowed else 1)

    report = json.loads(capsys.readouterr().out.splitlines()[0])
    assert report["passed"] is allowed
    assert report["model_decision_valid"] is None
    assert report["model_repaired"] is False
    assert report["decision_method"] == "deterministic"
    assert report["model_attempts"] == []
    router.send.assert_not_called()


def test_unfinished_final_attempt_cannot_use_earlier_valid_response():
    router = MagicMock()
    router.send.side_effect = [
        LLMResponse('{"action":"answer","confidence":0.98}', "synthetic", "nim"),
        TimeoutError("synthetic timeout"),
    ]
    probe = check_controller._DecisionProbe(router, timeout=1)
    probe.send([])
    with pytest.raises(TimeoutError):
        probe.send([])

    assert probe.result() == (False, "TimeoutError")
    assert [attempt["valid"] for attempt in probe.attempts()] == [True, False]
