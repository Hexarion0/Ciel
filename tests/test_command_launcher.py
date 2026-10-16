"""The user command selects the checkout's runtime from any directory."""

import shutil
import subprocess
from pathlib import Path


def make_checkout(tmp_path: Path) -> Path:
    """Create an isolated checkout with a fake virtual-environment interpreter."""
    checkout = tmp_path / "project with spaces"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts"
    for name in ["install_ciel_command.sh", "launch_ciel_gpu.sh"]:
        shutil.copyfile(source / name, scripts / name)
    python = checkout / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        '#!/usr/bin/env bash\n'
        'if [[ "$1" == "-" ]]; then cat >/dev/null; exit 0; fi\n'
        'printf "%s\\n" "$PWD" "$0" "$@"\n'
    )
    python.chmod(0o755)
    return checkout


def test_installed_command_uses_venv_and_preserves_arguments(tmp_path):
    checkout = make_checkout(tmp_path)
    bin_dir = tmp_path / "user bin"
    installer = checkout / "scripts/install_ciel_command.sh"
    for _ in range(2):
        subprocess.run(
            ["bash", str(installer), "--bin-dir", str(bin_dir)], check=True,
            capture_output=True, text=True,
        )
    for arguments in [[], ["start", "dev", "--wake-word", "hey ciel"]]:
        result = subprocess.run(
            [str(bin_dir / "ciel"), *arguments], cwd=tmp_path, check=True,
            capture_output=True, text=True,
        )
        assert result.stdout.splitlines() == [
            str(checkout), str(checkout / ".venv/bin/python"), "-m", "ciel",
            *(arguments or ["--listen"]),
        ]


def test_installer_preserves_unrelated_command(tmp_path):
    checkout = make_checkout(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    command = bin_dir / "ciel"
    command.write_text("existing command\n")
    result = subprocess.run(
        ["bash", str(checkout / "scripts/install_ciel_command.sh"),
         "--bin-dir", str(bin_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "Refusing to replace" in result.stderr
    assert command.read_text() == "existing command\n"
