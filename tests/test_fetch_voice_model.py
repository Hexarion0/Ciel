"""Voice downloads must use the revision accepted by the production worker."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
REVISION = json.loads((ROOT / "voice_profiles/ciel/voice.json").read_text())["model_revision"]


@pytest.fixture
def downloader(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "fetch_voice_model", ROOT / "scripts/fetch_voice_model.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    api = MagicMock()
    api.model_info.return_value = SimpleNamespace(
        sha=REVISION,
        siblings=[SimpleNamespace(rfilename="ve.safetensors", size=10)],
    )
    download = MagicMock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda: api, hf_hub_download=download,
    ))
    monkeypatch.setattr(
        sys, "argv", ["fetch_voice_model.py", "chatterbox-turbo", "--root", str(tmp_path)],
    )
    return module, api, download


@pytest.mark.parametrize("cached_revision", [None, "newer-incompatible-revision", REVISION])
def test_download_uses_production_revision(downloader, tmp_path, cached_revision):
    module, api, download = downloader
    directory = tmp_path / "chatterbox-turbo"
    if cached_revision is not None:
        directory.mkdir()
        (directory / "inventory.json").write_text(json.dumps({
            "repository": "ResembleAI/chatterbox-turbo", "revision": cached_revision,
            "files": [{"path": "ve.safetensors", "bytes": 10}], "selected_bytes": 10,
        }))
    module.main()
    if cached_revision == REVISION:
        api.model_info.assert_not_called()
    else:
        api.model_info.assert_called_once_with(
            "ResembleAI/chatterbox-turbo", revision=REVISION, files_metadata=True,
        )
    assert download.call_args.kwargs["revision"] == REVISION
    assert json.loads((directory / "inventory.json").read_text())["revision"] == REVISION


def test_inventory_only_and_budget_do_not_download(downloader, monkeypatch):
    module, _api, download = downloader
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--inventory-only"])
    module.main()
    download.assert_not_called()
    monkeypatch.setattr(sys, "argv", [*sys.argv[:-1], "--max-bytes", "1"])
    with pytest.raises(SystemExit, match="2"):
        module.main()
    download.assert_not_called()
