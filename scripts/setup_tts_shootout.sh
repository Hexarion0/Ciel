#!/usr/bin/env bash
# Evaluation dependencies only. Ciel and the existing clone environment remain untouched.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export UV_CACHE_DIR="$PWD/data/voice/cache/uv"
export NLTK_DATA="$PWD/data/voice/cache/nltk"
if ! command -v uv >/dev/null; then
    echo "This isolated installer requires the existing uv tool. See docs/tts-quality-shootout.md."
    exit 1
fi
if ! command -v espeak-ng >/dev/null; then
    echo "StyleTTS2 needs the espeak-ng system package for its original phonemizer."
    exit 1
fi
if [[ ! -x data/voice/envs/shootout/bin/python ]]; then
    if [[ -x data/voice/envs/clone/bin/python ]]; then
        uv venv data/voice/envs/shootout --python data/voice/envs/clone/bin/python
    else
        uv venv data/voice/envs/shootout --python 3.12
    fi
fi
uv pip install --python data/voice/envs/shootout/bin/python \
    torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python data/voice/envs/shootout/bin/python \
    -r scripts/voice_requirements/shootout.txt
data/voice/envs/shootout/bin/python -m scripts.tts_shootout.assets "$@"
