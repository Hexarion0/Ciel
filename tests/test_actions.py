"""Action discovery, authorization, validation, telemetry, and launch regressions."""

import threading
import time
from unittest.mock import MagicMock

import pytest

from ciel.actions import ActionRegistry
from ciel.actions.base import ActionContext, ActionError
from ciel.actions.open_app import OpenAppAction
from ciel.actions.system_info import SystemInfoAction
from ciel.platform.system_info import GpuInfo, SystemSnapshot


def test_discovery_loads_builtins_and_rejects_duplicate_names():
    registry = ActionRegistry.discover()
    assert {"system_info", "open_app"} <= registry.actions.keys()
    with pytest.raises(ValueError, match="Duplicate"):
        ActionRegistry([OpenAppAction(), OpenAppAction()])


@pytest.mark.parametrize("text", [
    "She said open Discord", "Could you explain how to open Discord?",
    "What if I open Discord", "How much RAM am I using and open Firefox",
])
def test_quoted_or_compound_requests_do_not_execute(text):
    assert ActionRegistry.discover().handle(text, authorized=True) is None


@pytest.mark.parametrize("arguments", [
    {}, {"app": "firefox", "args": "--private"}, {"app": "firefox; touch /tmp/injected"},
    {"app": "/bin/sh"}, {"app": "bash"}, {"app": 1}, None,
])
def test_launch_arguments_cannot_inject_shell_commands(monkeypatch, arguments):
    launch = MagicMock()
    monkeypatch.setattr("ciel.actions.open_app.subprocess.Popen", launch)
    ActionRegistry([OpenAppAction()]).execute("open_app", arguments, authorized=True)
    launch.assert_not_called()


def test_inferred_followup_cannot_launch_an_application(monkeypatch):
    launch = MagicMock()
    monkeypatch.setattr("ciel.actions.open_app.subprocess.Popen", launch)
    answer = ActionRegistry([OpenAppAction()]).handle("Open Discord", authorized=False)
    assert "address Ciel directly" in answer
    launch.assert_not_called()


def test_launch_uses_only_resolved_allowlisted_executable(monkeypatch):
    monkeypatch.setattr("ciel.actions.open_app.sys.platform", "linux")
    monkeypatch.setattr("ciel.actions.open_app.shutil.which",
                        lambda name: "/usr/bin/vesktop" if name == "vesktop" else None)
    launch = MagicMock()
    monkeypatch.setattr("ciel.actions.open_app.subprocess.Popen", launch)
    answer = ActionRegistry([OpenAppAction()]).handle("Please open Discord.", authorized=True)
    assert "sent the launch request" in answer
    assert launch.call_args.args == (["/usr/bin/vesktop"],)
    assert not launch.call_args.kwargs.get("shell", False)


def test_missing_app_and_permissions_are_reported(monkeypatch):
    monkeypatch.setattr("ciel.actions.open_app.sys.platform", "linux")
    monkeypatch.setattr("ciel.actions.open_app.shutil.which", lambda name: None)
    registry = ActionRegistry([OpenAppAction()])
    assert "couldn't find" in registry.handle("Open Firefox", authorized=True)
    monkeypatch.setattr("ciel.actions.open_app.shutil.which", lambda name: "/bin/firefox")
    monkeypatch.setattr("ciel.actions.open_app.subprocess.Popen",
                        MagicMock(side_effect=PermissionError))
    assert "permission" in registry.handle("Open Firefox", authorized=True)


def test_canceled_request_never_launches(monkeypatch):
    launch = MagicMock()
    monkeypatch.setattr("ciel.actions.open_app.subprocess.Popen", launch)
    cancel = threading.Event()
    cancel.set()
    answer = ActionRegistry([OpenAppAction()]).handle(
        "Open Discord", authorized=True, cancel_event=cancel,
    )
    assert "canceled" in answer
    launch.assert_not_called()


def test_expired_execution_budget_is_rejected():
    with pytest.raises(ActionError, match="timed out"):
        ActionContext(deadline=time.monotonic() - 1).remaining()


@pytest.mark.parametrize("text, expected", [
    ("How much RAM am I using?", "8 of 16 GB"),
    ("What's my GPU temperature?", "50 degrees Celsius"),
    ("How many CPU cores do I have?", "8 logical CPU cores"),
])
def test_telemetry_answers_are_local_and_allow_read_only_followups(monkeypatch, text, expected):
    snap = SystemSnapshot(ram_total_gb=16, ram_used_gb=8, ram_percent=50, cpu_count=8,
                          gpu=GpuInfo(name="Test GPU", temperature_c=50, available=True))
    snapshot = MagicMock(return_value=snap)
    monkeypatch.setattr("ciel.actions.system_info.get_system_snapshot", snapshot)
    answer = ActionRegistry([SystemInfoAction()]).handle(text, authorized=False)
    assert expected in answer
    assert 0 < snapshot.call_args.kwargs["gpu_timeout"] <= 1.5


def test_unavailable_metrics_are_not_reported_as_zero(monkeypatch):
    monkeypatch.setattr("ciel.actions.system_info.get_system_snapshot",
                        lambda **kwargs: SystemSnapshot())
    registry = ActionRegistry([SystemInfoAction()])
    assert "unavailable" in registry.handle("How much RAM am I using?", authorized=True)
    assert "unavailable" in registry.handle("How hot is my GPU?", authorized=True)


def test_confirmation_is_bound_to_validated_arguments_and_direct_address(monkeypatch):
    action = OpenAppAction()
    action.requires_confirmation = True
    execute = MagicMock(return_value="Done")
    monkeypatch.setattr(action, "execute", execute)
    registry = ActionRegistry([action])
    assert "Confirm open_app" in registry.handle("Open Firefox", authorized=True)
    execute.assert_not_called()
    assert registry.handle("yes", authorized=False) is None
    execute.assert_not_called()
    registry.handle("Open Discord", authorized=True)
    assert registry.handle("yes", authorized=True) == "Done"
    assert execute.call_args.args[0] == {"app": "discord"}
    assert registry.handle("yes", authorized=True) is None
    assert execute.call_count == 1


@pytest.mark.parametrize("reason", ["timeout", "unrelated", "cancel", "no"])
def test_stale_action_confirmation_is_cleared(monkeypatch, reason):
    action = OpenAppAction()
    action.requires_confirmation = True
    execute = MagicMock()
    monkeypatch.setattr(action, "execute", execute)
    clock = [100.0]
    monkeypatch.setattr("ciel.actions.time.monotonic", lambda: clock[0])
    registry = ActionRegistry([action])
    registry.handle("Open Discord", authorized=True)
    if reason == "timeout":
        clock[0] += 61
    elif reason == "cancel":
        registry.cancel_pending()
    else:
        registry.handle("What's the weather?" if reason == "unrelated" else "no", authorized=True)
    assert registry.handle("yes", authorized=True) is None
    execute.assert_not_called()


def test_ambiguous_match_does_not_execute_either_action(monkeypatch):
    first, second = OpenAppAction(), OpenAppAction()
    second.name = 'another_launcher'
    execute = MagicMock()
    monkeypatch.setattr(first, 'execute', execute)
    monkeypatch.setattr(second, 'execute', execute)
    reply = ActionRegistry([first, second]).handle('Open Discord', authorized=True)
    assert 'several actions' in reply
    execute.assert_not_called()


@pytest.mark.parametrize('error, expected', [
    (RuntimeError('private diagnostic details'), "couldn't complete"),
    (TimeoutError(), 'timed out'),
])
def test_handler_failures_are_local_and_do_not_leak_details(monkeypatch, error, expected):
    action = SystemInfoAction()
    monkeypatch.setattr(action, 'execute', MagicMock(side_effect=error))
    reply = ActionRegistry([action]).handle('How hot is my GPU?', authorized=True)
    assert expected in reply
    assert 'private diagnostic' not in reply


def test_discovery_finds_new_module_without_registry_changes(tmp_path, monkeypatch):
    import sys

    import ciel.actions as package

    (tmp_path / 'test_extension.py').write_text(
        'from ciel.actions.system_info import SystemInfoAction\n'
        'ACTION = SystemInfoAction()\nACTION.name = "test_extension"\n'
    )
    monkeypatch.setattr(package, '__path__', [str(tmp_path)])
    # Restore sys.modules after discovery so the fixture doesn't leak a plugin.
    monkeypatch.delitem(sys.modules, 'ciel.actions.test_extension', raising=False)
    try:
        assert set(ActionRegistry.discover().actions) == {'test_extension'}
    finally:
        sys.modules.pop('ciel.actions.test_extension', None)


def test_failed_matcher_does_not_prevent_other_actions(monkeypatch):
    broken, working = OpenAppAction(), SystemInfoAction()
    monkeypatch.setattr(broken, 'match', MagicMock(side_effect=RuntimeError))
    monkeypatch.setattr(working, 'execute', lambda *args: 'Hardware result')
    assert ActionRegistry([broken, working]).handle(
        'How hot is my GPU?', authorized=True,
    ) == 'Hardware result'


@pytest.mark.parametrize('text', [
    'Can you open Discord for me?', 'Could you please launch Discord?',
    'Please open Discord for me.', 'Open Discord, please.',
    'Would you start the Discord app for me please?', 'Open D I S C O R D.',
    'Can you open D-I-S-C-O-R-D for me?',
])
def test_natural_launch_requests_reach_the_action(monkeypatch, text):
    action = OpenAppAction()
    execute = MagicMock(return_value='Launch requested')
    monkeypatch.setattr(action, 'execute', execute)
    assert ActionRegistry([action]).handle(text, authorized=True) == 'Launch requested'
    assert execute.call_args.args[0] == {'app': 'discord'}


def test_spelled_correction_resolves_subsequent_open_it(monkeypatch):
    action = OpenAppAction()
    execute = MagicMock(return_value='Launch requested')
    monkeypatch.setattr(action, 'execute', execute)
    registry = ActionRegistry([action])
    assert registry.handle('No, no, no, I meant D I S C O R D', authorized=True) is None
    execute.assert_not_called()
    assert registry.handle('Can you open it for me?', authorized=True) == 'Launch requested'
    assert execute.call_args.args[0] == {'app': 'discord'}


@pytest.mark.parametrize('text', [
    'Disk Card', 'D-I-S-C-R-O-T', 'She said Discord', 'Discord or Firefox',
    'Do not open Discord', 'What is Discord?', '"Discord"',
])
def test_uncertain_or_nonaffirmative_app_reference_cannot_resolve_it(monkeypatch, text):
    action = OpenAppAction()
    execute = MagicMock()
    monkeypatch.setattr(action, 'execute', execute)
    registry = ActionRegistry([action])
    registry.handle(text, authorized=True)
    assert 'Which application' in registry.handle('Can you open it for me?', authorized=True)
    execute.assert_not_called()


@pytest.mark.parametrize('reason', ['expired', 'unrelated', 'canceled', 'unauthorized', 'restart'])
def test_context_does_not_authorize_stale_or_inferred_launches(monkeypatch, reason):
    action = OpenAppAction()
    execute = MagicMock(return_value='Launch requested')
    monkeypatch.setattr(action, 'execute', execute)
    clock = [100.0]
    monkeypatch.setattr('ciel.actions.time.monotonic', lambda: clock[0])
    registry = ActionRegistry([action])
    registry.handle('I meant Discord', authorized=True)
    if reason == 'expired':
        clock[0] += 61
    elif reason == 'unrelated':
        registry.handle('Tell me about Python', authorized=True)
    elif reason == 'canceled':
        registry.cancel_pending()
    elif reason == 'restart':
        registry = ActionRegistry([action])
    result = registry.handle('Can you open it for me?', authorized=reason != 'unauthorized')
    assert 'address Ciel directly' in result if reason == 'unauthorized' else (
        'Which application' in result
    )
    execute.assert_not_called()
