"""A stalled or failed native audio query stays isolated from inspection commands."""

import json
import subprocess
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from ciel import device_probe
from ciel.platform.base import AudioDeviceInfo


def test_device_metadata_survives_process_boundary(monkeypatch):
    device = AudioDeviceInfo(4, "Test microphone", "Test API", 1, 0, 48000, True)
    worker = SimpleNamespace(returncode=0, stdout=json.dumps([asdict(device)]))
    invocations = []

    def run(command, **kwargs):
        invocations.append((command, kwargs))
        return worker

    monkeypatch.setattr(device_probe.subprocess, "run", run)
    assert device_probe.list_devices() == [device]
    command, options = invocations[0]
    assert command[1:] == ["-m", "ciel.device_probe"]
    assert options["timeout"] == 5.0


def test_stalled_worker_returns_safe_timeout(monkeypatch):
    def stalled(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], stderr="private detail")

    monkeypatch.setattr(device_probe.subprocess, "run", stalled)
    with pytest.raises(TimeoutError, match="Audio device discovery timed out") as error:
        device_probe.list_devices(timeout=0.1)
    assert "private detail" not in str(error.value)


@pytest.mark.parametrize("returncode, stdout", [(1, ""), (0, "invalid"), (0, '[{"bad":1}]')])
def test_failed_or_invalid_worker_is_reported(monkeypatch, returncode, stdout):
    monkeypatch.setattr(device_probe.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=returncode, stdout=stdout, stderr="private diagnostic output",
    ))
    with pytest.raises(RuntimeError, match="Audio device discovery") as error:
        device_probe.list_devices()
    assert "private diagnostic output" not in str(error.value)
