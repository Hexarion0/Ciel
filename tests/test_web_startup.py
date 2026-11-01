"""ciel web loads recognition and the voice model before the web app opens."""

from types import SimpleNamespace

import pytest


class Recorder:
    def __init__(self, stt_ready=True, warmed=True):
        self.events = []
        self.stt = SimpleNamespace(wait_ready=self._wait)
        self.tts = SimpleNamespace(enabled=True, engine="chatterbox_turbo", warmup=self._warm)
        self.stt_ready, self.warmed = stt_ready, warmed

    def _wait(self, timeout=None):
        self.events.append(("stt", timeout))
        return self.stt_ready

    def _warm(self):
        self.events.append("tts")
        return self.warmed


@pytest.mark.parametrize("web_mode", [True, False])
def test_turbo_and_recognition_load_before_anything_opens(web_mode):
    from ciel.runtime import prepare_voice

    run = Recorder()

    prepared = prepare_voice(run.stt, run.tts, stt_timeout=45, web_mode=web_mode)

    assert prepared is True
    assert run.events == [("stt", 45), "tts"]


def test_web_mode_waits_for_recognition_even_with_a_light_voice():
    from ciel.runtime import prepare_voice

    run = Recorder()
    run.tts.engine = "piper"

    assert prepare_voice(run.stt, run.tts, stt_timeout=45, web_mode=True) is None
    assert run.events == [("stt", 45)]


def test_desktop_with_a_light_voice_starts_without_waiting():
    from ciel.runtime import prepare_voice

    run = Recorder()
    run.tts.engine = "piper"

    assert prepare_voice(run.stt, run.tts, stt_timeout=45, web_mode=False) is None
    assert run.events == []


def test_failed_voice_load_reports_fallback_and_still_opens():
    from ciel.runtime import prepare_voice

    run = Recorder(warmed=False)

    assert prepare_voice(run.stt, run.tts, stt_timeout=45, web_mode=True) is False


def test_slow_recognition_does_not_block_startup_forever():
    from ciel.runtime import prepare_voice

    run = Recorder(stt_ready=False)

    assert prepare_voice(run.stt, run.tts, stt_timeout=45, web_mode=True) is None
    assert run.events == [("stt", 45)]


def test_preparation_runs_before_the_web_server_starts():
    import inspect

    from ciel import runtime

    source = inspect.getsource(runtime.run_desktop)
    assert source.index("prepare_voice(") < source.index("url = web_ui.start(")
