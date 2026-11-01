# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CIEL is a Linux desktop voice assistant (Python 3.10+, `src/` layout, package `ciel`). It also
offers browser chat (`ciel web`, phone pairing) and Telegram chat. See `AGENTS.md` for style and
commit conventions; the essentials are repeated here.

## Commands

Use the project venv (`.venv/`). CI (`.github/workflows/checks.yml`) runs on Python 3.10 and 3.14.

```bash
.venv/bin/pip install -e ".[dev]"            # editable install with test/lint/voice deps
.venv/bin/pytest                             # unit tests (integration excluded by default)
.venv/bin/pytest tests/test_router.py        # one file
.venv/bin/pytest tests/test_router.py::test_name   # one test
.venv/bin/pytest -m integration              # needs real audio devices / downloaded models
.venv/bin/ruff check src tests scripts       # lint (CI lints scripts/ too)
.venv/bin/python -m pip wheel . --no-deps -w dist  # build wheel
bash -n setup.sh scripts/*.sh                # CI syntax-checks every shell script
```

Running the app: `python -m ciel start [dev]` (voice; `dev` = verbose logging), `ciel web`,
`ciel pair`, `ciel chat`, `ciel setup`, `ciel doctor`, `ciel config`. Run `ciel --help` for
the full list. Configuration comes from `.env` (template: `.env.example`) via pydantic-settings
in `config.py`. Add new options to `.env.example` with safe placeholders.

## Testing notes

- `pytest` defaults to `-m 'not integration'`. Mark anything that needs hardware, models, or
  training deps with `@pytest.mark.integration`.
- `tests/conftest.py` has autouse fixtures that, for non-integration tests, redirect
  `XDG_RUNTIME_DIR`/`XDG_CONFIG_HOME` to temp dirs, stub `ToolProgress.warm`, and force TTS
  settings to `Settings(_env_file=None, tts_engine="piper")`. Unit tests must not depend on
  the real `.env` or live voice config.
- Frontend JS (`src/ciel/web_assets/*.js`) is tested by Node scripts in `tests/frontend/*.cjs`,
  driven from `tests/test_web_frontend.py`. Node must be on PATH.
- CI also smoke-tests the installed wheel **outside the checkout**. Bundled resources
  (`web_assets/`, the Turbo worker, `voice_profiles/`) must resolve through `ciel.paths`, not
  repo-relative paths. `paths.development_root()` is `None` when installed.

## Architecture

**Entry and CLI.** `__main__.py` parses commands. It avoids importing audio and model
dependencies until they're needed, so `--help` and `--version` stay fast. Voice mode goes
to `runtime.run_desktop()`.

**Voice runtime (`runtime.py`, the largest module).** This module wires everything together:
wake/ambient listening (`audio/listener.py`, `audio/wake.py`, `audio/ambient.py`), recording
and STT (`audio/recorder.py`, `audio/stt.py`, faster-whisper + Silero VAD), the per-turn
pipeline, and streamed TTS playback with interruption (`audio/tts.py`, `audio/engines.py`,
`audio/streaming.py`, `audio/chunk_playback.py`). TTS engines: Piper (default), Pocket TTS,
and a Chatterbox "Turbo" engine that runs in a separate worker process
(`audio/turbo_worker.py`, `audio/chatterbox_worker.py`).

**Per-turn decision flow.**
1. `conversation.py` runs cheap regex intent checks: is CIEL being addressed, farewells, and
   conversation controls.
2. `controller.py` `AIController.decide()` asks a model for a validated `ControllerDecision`:
   reply directly, search the web, look at the screen, run a tool, and so on. `run_web()`
   handles grounded web answers using `internet.py`, `search_*.py`, `parallel_search.py`,
   `research.py`, and `web_answers.py`.
3. `agent.py` `MainAgent` runs the tool-calling loop. It streams, supports cancellation
   through generation counters and events, and validates tool arguments against JSON schemas.
   Tool definitions live in `agent_tools.py` and `memory/tools.py`.
4. `persona.py` / `dialogue.py` / `task_context.py` build the system prompt and dialogue
   context.

**Providers (`providers/`).** `ModelRouter` (`router.py`) classifies prompt complexity and
picks the provider and model. `ProviderManager` (`manager.py`) handles fallback across Groq,
OpenRouter, NVIDIA NIM, and Ollama. Privacy invariant: `ModelRouter.requires_local()` keeps any
conversation that contains `local_only` messages (or legacy `/local` commands) on Ollama. Code
that builds or forwards messages must preserve the `local_only` flag. `vision.py` handles
screen images.

**Text channels (`chat/`).** `chat/session.py` is the text-turn pipeline for CLI and Telegram
(`chat/telegram.py`). It builds its own `MainAgent`/`ModelRouter`, parallel to the voice path
in `runtime.py`. Behavior changes to turn handling often need to land in both places.

**Web app (`webapp/`).** Stdlib `ThreadingHTTPServer` (no framework) with TLS sniffing
(`server.py`, `tls.py`), device pairing and auth (`auth.py`, `pair_cli.py`), browser speech
and audio (`speech.py`, `audio.py`), and conversation listing. The frontend is plain JS/CSS in
`src/ciel/web_assets/`, shipped inside the wheel.

**Memory (`memory/`).** An editable Markdown workspace plus SQLite (`store.py`) with episodic
archives, semantic/term-based recall (`recall.py`, `semantic.py`), lifecycle management, and
`service.py` for conservative fact extraction and explicit forgetting. Manage it through the
CLI with `ciel memory`.

**Screen (`screen/`).** On-request desktop capture and understanding, plus an optional
"initiative" mode. `actions/` holds trusted local actions (see `docs/actions.md` for how to add
one). `platform/` covers Linux audio and system telemetry.

## Repo conventions

- Commit messages use scoped Conventional Commits, e.g. `feat(memory): ...`,
  `fix(audio): ...`. Each release is recorded in `CHANGELOG.md`; the version lives in
  `pyproject.toml`.
- Design docs and plans live in `docs/`. Release checklist: `docs/release-checks.md`. Roadmap:
  `docs/roadmap.md`.
- `data/`, `models/`, `.env`, recordings, and conversation databases are private or local and
  ignored by git.
