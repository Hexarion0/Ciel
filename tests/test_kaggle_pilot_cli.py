"""Check pilot defaults without submitting a notebook or loading credentials."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def pilot(monkeypatch):
    """Import the standalone script without running its command handler."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("kaggle_pilot")


def test_pilot_uses_its_checkout_instead_of_a_machine_specific_path(pilot, monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    args = pilot.parse_args(["prepare", "--owner", "test-owner"])
    assert args.repo == tmp_path


def test_pilot_requires_an_explicit_account(pilot, monkeypatch, capsys):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    with pytest.raises(SystemExit) as error:
        pilot.parse_args(["run"])
    assert error.value.code == 2
    assert "--owner" in capsys.readouterr().err


def test_pilot_accepts_account_environment_and_explicit_override(pilot, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "environment-owner")
    assert pilot.parse_args(["prepare"]).owner == "environment-owner"
    args = pilot.parse_args(["prepare", "--owner", "explicit-owner", "--repo", "/tmp/checkout"])
    assert args.owner == "explicit-owner"
    assert args.repo == Path("/tmp/checkout")


def test_invalid_poll_interval_fails_before_any_run_directory_is_created(pilot, tmp_path):
    directory = tmp_path / "run"
    with pytest.raises(SystemExit) as error:
        pilot.parse_args([
            "run", "--owner", "test-owner", "--poll-seconds", "1",
            "--run-directory", str(directory),
        ])
    assert error.value.code == 2
    assert not directory.exists()


def test_forward_probe_requires_an_explicit_account_before_building(pilot, monkeypatch):
    probe = importlib.import_module("kaggle_forward_probe")
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    with pytest.raises(SystemExit) as error:
        probe.parse_args([])
    assert error.value.code == 2


def test_forward_probe_uses_explicit_cli_and_private_output(pilot, monkeypatch, tmp_path):
    probe = importlib.import_module("kaggle_forward_probe")
    monkeypatch.setenv("KAGGLE_USERNAME", "test-owner")
    monkeypatch.setattr(probe, "PILOT_DIRECTORY", tmp_path / "private-pilot")
    args = probe.parse_args(["--tag", "02", "--kaggle", "/tmp/kaggle"])
    assert args.owner == "test-owner" and args.kaggle == "/tmp/kaggle"
    assert args.output == tmp_path / "private-pilot/runs/forward-diagnostic-02"


def test_notebook_builder_has_no_implicit_personal_owner(pilot, monkeypatch, tmp_path):
    notebook = importlib.import_module("kaggle_pilot_notebook")
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    with pytest.raises(ValueError, match="owner"):
        notebook.build(tmp_path)


@pytest.mark.parametrize("name", ["kaggle_pilot", "kaggle_forward_probe"])
def test_kaggle_tools_support_script_and_module_help(name, tmp_path):
    root = Path(__file__).resolve().parents[1]
    for command, cwd in [
        ([sys.executable, "-I", str(root / "scripts" / f"{name}.py"), "--help"], tmp_path),
        ([sys.executable, "-m", f"scripts.{name}", "--help"], root),
    ]:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert "--owner" in result.stdout and "--kaggle" in result.stdout
