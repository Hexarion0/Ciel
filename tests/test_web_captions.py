"""Web replies are captioned in time with the voice, and / commands match the terminal."""

import json
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ciel.webapp.requests import WebRequest

CAPTION = "Hello there, my friend. It is a fine day."


class TTS:
    def __init__(self, *segments: float) -> None:
        self.segments = segments or (2.0,)

    def synthesize_stream(self, text, **kwargs):
        for seconds in self.segments:
            yield np.zeros(int(16000 * seconds), dtype=np.float32), 16000


def all_chunks(service, job):
    chunks = []
    for _ in range(400):
        try:
            chunk = service.chunk(job, "phone", 1, "tab-one")
        except EOFError:
            return chunks
        if chunk:
            chunks.append(chunk)
        else:
            threading.Event().wait(0.005)
    raise AssertionError("speech job did not finish")


def speak(tts, text=CAPTION, message_id=7):
    from ciel.webapp.speech import BrowserSpeech

    service = BrowserSpeech(tts)
    request = WebRequest("r", "phone", "chat", 1, "text", "hi")
    job = service.speak(text, request, cancel_event=threading.Event(), current=lambda: True,
                        message_id=message_id)
    return service, job


@pytest.mark.parametrize("segments", [(2.0,), (0.6, 0.9, 1.4)])
def test_chunks_carry_contiguous_caption_ranges_that_end_with_the_text(segments):
    service, job = speak(TTS(*segments))

    chunks = all_chunks(service, job)

    assert chunks and all(chunk.startswith(b"RIFF") for chunk in chunks)
    assert chunks[0].caption_from == 0
    for previous, current in zip(chunks, chunks[1:]):
        assert current.caption_from == previous.caption_to
    assert all(chunk.caption_from <= chunk.caption_to for chunk in chunks)
    assert chunks[-1].caption_to == len(CAPTION)
    assert sum(chunk.seconds for chunk in chunks) == pytest.approx(sum(segments))


def test_single_segment_captions_follow_audio_time():
    service, job = speak(TTS(2.0))

    first, second = all_chunks(service, job)

    assert 0.35 * len(CAPTION) < first.caption_to < 0.65 * len(CAPTION)
    assert second.caption_to == len(CAPTION)


def test_caption_offsets_count_like_the_browser_does_for_emoji():
    """Spoken excerpts drop emoji, but a full replay keeps them; JS counts them as two units."""
    from ciel.webapp.speech import BrowserSpeech

    caption = "Done! 🎉 Enjoy the party 🥳 tonight."
    service = BrowserSpeech(TTS(2.0))
    request = WebRequest("r", "phone", "chat", 1, "text", "hi")
    job = service.speak(caption, request, cancel_event=threading.Event(), current=lambda: True,
                        full=True, message_id=1)

    chunks = all_chunks(service, job)

    browser_length = len(caption.encode("utf-16-le")) // 2
    assert chunks[-1].caption_to == browser_length != len(caption)


def test_audio_already_made_still_plays_if_the_next_segment_fails():
    class Failing:
        def synthesize_stream(self, text, **kwargs):
            yield np.zeros(16000, dtype=np.float32), 16000
            raise RuntimeError("voice engine crashed")

    service, job = speak(Failing())

    chunks = []
    with pytest.raises(EOFError):
        for _ in range(400):
            chunk = service.chunk(job, "phone", 1, "tab-one")
            if chunk:
                chunks.append(chunk)
            else:
                threading.Event().wait(0.005)
    assert len(chunks) == 1 and chunks[0].caption_to == len(CAPTION)


def test_jobs_list_their_caption_and_reply_message():
    service, job = speak(TTS(1.0), message_id=42)

    listed = service.list("phone", 1)[0]

    assert listed["id"] == job and listed["message"] == 42
    assert listed["caption"] == CAPTION


def test_spoken_reply_is_flagged_and_returns_its_message_id():
    from ciel.web import DesktopWebUI

    ui = DesktopWebUI("owner", "hey ciel")
    plain = ui.text_reply("Typed only.")
    spoken = ui.text_reply("Said aloud.", spoken=True)

    messages = {message["id"]: message for message in ui.snapshot()["messages"]}
    assert spoken > plain
    assert messages[spoken]["spoken"] is True
    assert "spoken" not in messages[plain]


def test_audio_route_sends_the_caption_range_with_each_chunk():
    from ciel.web import DesktopWebUI

    ui = DesktopWebUI("owner", "hey ciel")
    ui.channels.configure(MagicMock(), TTS(1.0))
    request = WebRequest("r", "pc", "chat", ui.channels.revision(), "text", "hi")
    job = ui.channels.speech.speak(CAPTION, request, cancel_event=threading.Event(),
                                   current=lambda: True, message_id=3)
    sent = {}
    for _ in range(400):
        handler = SimpleNamespace(
            path=f"/api/audio/{job}?client={'c' * 20}", command="GET", headers={},
            client_address=("127.0.0.1", 1), _json=MagicMock(), response_headers={},
            _send=lambda status, body, kind, h=None: sent.update(status=status, body=body),
        )
        handler._send = (lambda handler: lambda status, body, kind: sent.update(
            status=status, headers=dict(handler.response_headers)))(handler)
        assert ui.channels.authorize(handler)
        ui.channels.get(handler)
        if sent.get("status") == 200:
            break
        threading.Event().wait(0.005)
    assert sent["headers"]["X-Caption-From"] == "0"
    assert sent["headers"]["X-Caption-To"] == str(len(CAPTION))


def test_command_catalog_is_served_from_the_terminal_definitions():
    from ciel.terminal import command_catalog, command_suggestions

    catalog = command_catalog()

    names = [row["name"] for row in catalog["commands"]]
    assert names == [name for name, _ in command_suggestions("/")]
    assert catalog["arguments"]["/mute"] == ["on", "off"]
    assert {"/mute", "/fast", "/help"} <= set(catalog["spaced"])


DRAFTS = ["/", "/m", "/mu", "/mute ", "/mute o", "/MO", "/mode a", "/help ", "/help st",
          "/x", "hello", "/fast ", "/observe p", "/reasoning of"]


def test_web_command_picker_matches_the_terminal_suggestions():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to run the browser command picker")
    from ciel.terminal import command_catalog, command_suggestions

    root = Path(__file__).resolve().parents[1]
    script = (
        "const fs=require('fs'),vm=require('vm');const c={};vm.createContext(c);"
        "vm.runInContext(fs.readFileSync('src/ciel/web_assets/commands.js','utf8')"
        "+';globalThis.commandSuggestions=commandSuggestions;',c);"
        "const [catalog,drafts]=JSON.parse(fs.readFileSync(0,'utf8'));"
        "console.log(JSON.stringify(drafts.map(d=>c.commandSuggestions(d,catalog))));"
    )
    result = subprocess.run([node, "-e", script], cwd=root, capture_output=True, text=True,
                            input=json.dumps([command_catalog(), DRAFTS]), timeout=10)
    assert result.returncode == 0, result.stderr
    web = json.loads(result.stdout)
    terminal = [[list(row) for row in command_suggestions(draft)] for draft in DRAFTS]
    assert web == terminal


def test_commands_endpoint_requires_a_paired_device():
    from ciel.web import DesktopWebUI

    ui = DesktopWebUI("owner", "hey ciel")
    handler = SimpleNamespace(path="/api/commands", command="GET", headers={},
                              client_address=("192.168.1.20", 1), _json=MagicMock())

    assert not ui.channels.authorize(handler)


def test_help_panel_uses_the_terminal_groups_and_details():
    from ciel.terminal import COMMAND_GROUPS, command_catalog

    catalog = command_catalog()

    assert [group["title"] for group in catalog["groups"]] == [title for title, _ in COMMAND_GROUPS]
    first = catalog["groups"][0]["commands"][1]
    assert first == {"label": "/mute [on|off]", "name": "/mute",
                     "description": "Toggle or set microphone mute"}
    assert catalog["details"]["/mute"].startswith("/mute [on|off] — ")
