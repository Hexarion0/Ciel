"""Large private outputs must stream rather than materialize response.content."""

import hashlib
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from scripts.kaggle_pilot_download import download, output_path


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "x\\y"])
def test_output_paths_stay_private(tmp_path, name):
    with pytest.raises(ValueError):
        output_path(tmp_path, name)


@pytest.mark.parametrize("partial_symlink", [False, True])
def test_download_streams_and_does_not_log_signed_url(tmp_path, monkeypatch, partial_symlink):
    payload = b"checkpoint-byte-fixture"
    url = "https://example.invalid/checkpoint?signature=never-log-this"

    class Response:
        headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        @property
        def content(self):
            raise AssertionError("Whole-file buffering is forbidden")

        def iter_content(self, chunk_size):
            assert chunk_size <= 4 * 1024**2
            yield payload[:5]
            yield payload[5:]

    def listing(request):
        assert not getattr(request, "version_label", None)
        return SimpleNamespace(files=[SimpleNamespace(file_name="native/checkpoint.pth", url=url)],
                               log="", next_page_token="")

    client = SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(
        list_kernel_session_output=listing)))
    api = SimpleNamespace(parse_kernel_string=lambda ref: ("hexarion0", "pilot", "5"),
                          get_config_value=lambda key: "hexarion0",
                          build_kaggle_client=lambda: nullcontext(client))
    monkeypatch.setitem(sys.modules, "kaggle", SimpleNamespace(api=api))
    monkeypatch.setitem(sys.modules, "kagglesdk.kernels.types.kernels_api_service",
                        SimpleNamespace(ApiListKernelSessionOutputRequest=SimpleNamespace))
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(
        get=lambda supplied, **kwargs: Response(), RequestException=ConnectionError))
    destination = tmp_path / "outputs"
    if partial_symlink:
        protected = tmp_path / "unrelated.txt"
        protected.write_text("preserve this file")
        (destination / "native").mkdir(parents=True)
        (destination / "native/checkpoint.pth.part").symlink_to(protected)
        with pytest.raises(ValueError, match="partial"):
            download("hexarion0/pilot/5", destination)
        assert protected.read_text() == "preserve this file"
        return
    download("hexarion0/pilot/5", destination)
    assert (destination / "native/checkpoint.pth").read_bytes() == payload
    inventory_path = tmp_path / "streamed-download-inventory.json"
    assert "never-log-this" not in inventory_path.read_text()
    inventory = json.loads(inventory_path.read_text())
    assert inventory["files"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert not list(destination.rglob("*.part"))
