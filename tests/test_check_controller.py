"""The live diagnostic must distinguish model responses from local safety fallbacks."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ciel.config import ControllerConfig
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
    router.send.assert_called_once()


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
