"""Runtime cache ownership, lifetime, reset and retention with synthetic pixels only."""

import io
import json
import os
import stat
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from ciel.screen.cache import CaptureCache, ScreenCacheError, clear_runtime_caches
from ciel.screen.capture import Capture, Cursor, Monitor, ScreenImage


def synthetic_capture(identifier="first", *, captured_at=None):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), "red").save(buffer, format="PNG")
    image = ScreenImage("desktop", buffer.getvalue(), 8, 6)
    return Capture(identifier, captured_at if captured_at is not None else time.time(),
                   (Monitor("screen", 0, 0, 8, 6),),
                   Cursor(2, 3, time.time(), True, "screen", 2, 3), (image,))


def test_same_scope_shares_images_but_different_owner_does_not(tmp_path):
    with CaptureCache("owner-a", tmp_path) as one, CaptureCache("owner-a", tmp_path) as two:
        with CaptureCache("owner-b", tmp_path) as other:
            one.add(synthetic_capture())
            assert two.recent()[0].id == "first"
            assert other.recent() == []
            one.close()
            assert two.contains("first")
    with CaptureCache("owner-a", tmp_path) as restarted:
        assert restarted.recent() == []


def test_two_parents_evict_originals_and_crops(tmp_path):
    with CaptureCache("owner", tmp_path) as cache:
        first = synthetic_capture()
        cache.add(first)
        cache.add_crop("first", replace(first.image(), name="detail"))
        cache.add(synthetic_capture("second"))
        cache.add(synthetic_capture("third"))
        assert [capture.id for capture in cache.recent()] == ["second", "third"]
        assert len(list(cache.directory.glob("*.png"))) == 2
        with pytest.raises(ScreenCacheError, match="expired"):
            cache.get_image("first", "detail")


def test_crops_keep_original_timestamp_and_expire_on_read(tmp_path, monkeypatch):
    clock = [time.time()]
    monkeypatch.setattr("ciel.screen.cache.time.time", lambda: clock[0])
    with CaptureCache("owner", tmp_path) as cache:
        capture = synthetic_capture(captured_at=clock[0])
        cache.add(capture)
        clock[0] += 899
        cache.add_crop("first", replace(capture.image(), name="detail"))
        assert cache.recent()[0].captured_at == capture.captured_at
        clock[0] += 1
        assert not cache.contains("first")
        assert list(cache.directory.glob("*.png")) == []
        with pytest.raises(ScreenCacheError, match="expired"):
            cache.add_crop("first", replace(capture.image(), name="later"))


def test_expiry_sweeper_deletes_pixels_while_idle(tmp_path, monkeypatch):
    clock = [time.time()]
    monkeypatch.setattr("ciel.screen.cache.time.time", lambda: clock[0])
    with CaptureCache("owner", tmp_path, sweep_interval=0.05) as cache:
        cache.add(synthetic_capture(captured_at=clock[0]))
        clock[0] += 901
        deadline = time.monotonic() + 2
        while list(cache.directory.glob("*.png")) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert list(cache.directory.glob("*.png")) == []


def test_global_reset_invalidates_live_handles_and_stale_work(tmp_path):
    with CaptureCache("owner", tmp_path) as one, CaptureCache("owner", tmp_path) as two:
        capture = synthetic_capture()
        one.add(capture)
        clear_runtime_caches(tmp_path)
        for cache in (one, two):
            with pytest.raises(ScreenCacheError, match="reset"):
                cache.recent()
            with pytest.raises(ScreenCacheError, match="reset"):
                cache.add(capture)
        with CaptureCache("owner", tmp_path) as fresh:
            assert fresh.recent() == []


def test_restart_generation_does_not_restore_dead_lease(tmp_path, monkeypatch):
    with CaptureCache("owner", tmp_path) as cache:
        cache.add(synthetic_capture())
        state_path = cache.directory / "state.json"
        state = json.loads(state_path.read_text())
        for value in state["leases"].values():
            value["identity"] = "recycled-pid"
        state_path.write_text(json.dumps(state))
        with CaptureCache("owner", tmp_path) as restarted:
            assert restarted.recent() == []
            assert list(cache.directory.glob("*.png")) == []


def test_explicit_service_restart_invalidates_existing_handles(tmp_path):
    with CaptureCache("owner", tmp_path, lifecycle_id="old") as old:
        old.add(synthetic_capture())
        with CaptureCache("owner", tmp_path, lifecycle_id="new") as fresh:
            assert fresh.recent() == []
            with pytest.raises(ScreenCacheError, match="reset"):
                old.contains("first")


def test_default_storage_is_private_runtime_only(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with CaptureCache("owner") as cache:
        cache.add(synthetic_capture())
        assert cache.directory.parent == tmp_path / "ciel-screen"
        for path in (cache.root, cache.directory):
            assert stat.S_IMODE(path.stat().st_mode) == 0o700
        for path in cache.directory.iterdir():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_missing_or_insecure_runtime_refuses_repository_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "missing"))
    with pytest.raises(ScreenCacheError, match="unavailable"):
        CaptureCache("owner")
    tmp_path.chmod(0o755)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with pytest.raises(ScreenCacheError, match="not private"):
        CaptureCache("owner")


def test_image_symlink_is_never_followed(tmp_path):
    with CaptureCache("owner", tmp_path) as cache:
        cache.add(synthetic_capture())
        image = next(cache.directory.glob("*.png"))
        private = tmp_path / "unrelated.txt"
        private.write_text("private unrelated file")
        image.unlink()
        image.symlink_to(private)
        with pytest.raises(ScreenCacheError, match="no longer available"):
            cache.recent()
        assert private.read_text() == "private unrelated file"


def test_metadata_path_traversal_is_never_read(tmp_path):
    with CaptureCache("owner", tmp_path) as cache:
        cache.add(synthetic_capture())
        state_path = cache.directory / "state.json"
        state = json.loads(state_path.read_text())
        state["captures"][0]["images"][0]["file"] = "../private.png"
        state_path.write_text(json.dumps(state))
        with pytest.raises(ScreenCacheError, match="reference is invalid"):
            cache.recent()


def test_clear_and_close_remove_images(tmp_path):
    cache = CaptureCache("owner", tmp_path)
    cache.add(synthetic_capture())
    cache.clear()
    assert not cache.contains("first")
    cache.add(synthetic_capture())
    directory = cache.directory
    cache.close()
    assert list(directory.glob("*.png")) == []
    with pytest.raises(ScreenCacheError, match="closed"):
        cache.recent()


def test_malformed_and_expired_captures_rejected(tmp_path):
    with CaptureCache("owner", tmp_path) as cache:
        with pytest.raises(ScreenCacheError, match="expired"):
            cache.add(synthetic_capture(captured_at=time.time() - 900))
        capture = synthetic_capture()
        with pytest.raises(ScreenCacheError, match="metadata"):
            cache.add(replace(capture, images=(replace(capture.image(), width=12),)))
        assert cache.recent() == []
        assert list(cache.directory.glob("*.png")) == []


def test_cache_never_puts_image_bytes_in_json(tmp_path):
    with CaptureCache("owner", tmp_path) as cache:
        cache.add(synthetic_capture())
        state = (cache.directory / "state.json").read_text()
        assert "base64" not in state and "data" not in state
        assert "owner" not in str(cache.directory)
        assert os.getuid() == cache.directory.stat().st_uid


def child_environment():
    return {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}


def test_separate_frontend_process_reads_shared_cache_without_ending_lifetime(tmp_path):
    with CaptureCache("owner", tmp_path) as cache:
        cache.add(synthetic_capture())
        program = (
            "import sys; from pathlib import Path; "
            "from ciel.screen.cache import CaptureCache; "
            "cache=CaptureCache('owner',Path(sys.argv[1])); "
            "print(cache.recent()[0].id); cache.close()"
        )
        result = subprocess.run(
            [sys.executable, "-c", program, str(tmp_path)], env=child_environment(),
            capture_output=True, text=True, timeout=5, check=True,
        )
        assert result.stdout.strip() == "first"
        assert cache.contains("first")


def test_last_process_crash_cannot_restore_images_after_restart(tmp_path):
    program = """
import io
import os
import sys
import time
from pathlib import Path
from PIL import Image
from ciel.screen.capture import Capture, Cursor, Monitor, ScreenImage
from ciel.screen.cache import CaptureCache
data = io.BytesIO()
Image.new('RGB', (4, 4), 'blue').save(data, format='PNG')
capture = Capture('crashed', time.time(), (Monitor('test', 0, 0, 4, 4),),
                  Cursor(1, 1, time.time(), True),
                  (ScreenImage('desktop', data.getvalue(), 4, 4),))
cache = CaptureCache('owner', Path(sys.argv[1]))
cache.add(capture)
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", program, str(tmp_path)], env=child_environment(),
                   capture_output=True, timeout=5, check=True)
    assert list(tmp_path.glob("*/*.png"))
    with CaptureCache("owner", tmp_path) as restarted:
        assert restarted.recent() == []
        assert list(tmp_path.glob("*/*.png")) == []
