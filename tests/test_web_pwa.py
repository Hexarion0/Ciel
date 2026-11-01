"""The browser app is installable: public manifest, icons, and a root-scoped service worker."""

import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image


def test_manifest_describes_a_standalone_app_with_packaged_icons():
    from ciel.webapp.http import load_web_asset

    body, content_type = load_web_asset("/assets/manifest.webmanifest")
    manifest = json.loads(body)

    assert content_type.startswith("application/manifest+json")
    assert manifest["name"] == "Ciel" and manifest["short_name"] == "Ciel"
    assert manifest["start_url"] == "/" and manifest["scope"] == "/" and manifest["id"] == "/"
    assert manifest["display"] == "standalone"
    sizes = {}
    for icon in manifest["icons"]:
        data, icon_type = load_web_asset(icon["src"])
        assert icon_type == "image/png" == icon["type"]
        width, height = Image.open(io.BytesIO(data)).size
        assert icon["sizes"] == f"{width}x{height}"
        sizes.setdefault(icon["sizes"], set()).add(icon.get("purpose", "any"))
    assert "any" in sizes["192x192"] and "any" in sizes["512x512"]
    assert "maskable" in sizes["512x512"]


def test_service_worker_is_served_from_the_root_scope():
    from ciel.webapp.http import load_web_asset

    body, content_type = load_web_asset("/sw.js")

    assert content_type.startswith("text/javascript")
    assert b"addEventListener('fetch'" in body


def test_index_links_the_manifest_and_icon():
    from importlib.resources import files

    page = files("ciel").joinpath("web_assets/index.html").read_text()

    assert '<link rel="manifest" href="/assets/manifest.webmanifest">' in page
    assert '<link rel="icon" href="/assets/icon-192.png"' in page


@pytest.mark.parametrize("path", [
    "/sw.js", "/assets/manifest.webmanifest", "/assets/icon-192.png", "/assets/icon-512.png",
])
def test_install_files_are_public_before_pairing(path):
    from ciel.web import DesktopWebUI

    ui = DesktopWebUI("owner", "hey ciel")
    handler = SimpleNamespace(path=path, command="GET", headers={},
                              client_address=("192.168.1.20", 1234), _json=MagicMock())

    assert ui.channels.authorize(handler)
