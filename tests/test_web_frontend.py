"""Run frontend interaction regressions without introducing browser dependencies."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script",
    [
        "web_ui.cjs",
        "orb_animation.cjs",
        "chat_render.cjs",
        "pcm_resample.cjs",
        "web_attachments.cjs", "browser_audio.cjs", "web_settings.cjs", "continuous_audio.cjs",
    ],
)
def test_web_frontend_regressions(script):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the optional frontend interaction checks")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [node, str(root / "tests/frontend" / script)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
