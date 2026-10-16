#!/usr/bin/env bash
# Optional evaluation only; never alter Ciel's runtime or existing environments.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export UV_CACHE_DIR="$PWD/data/voice/cache/uv"
# Reuse the existing login in place; never copy credentials into project data.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/token}"
export HF_HOME="$PWD/data/voice/cache/neutts-hf"
export CMAKE_ARGS="-DGGML_CUDA=OFF -DGGML_NATIVE=ON"
export CMAKE_BUILD_PARALLEL_LEVEL=2
if ! command -v uv >/dev/null || ! command -v espeak-ng >/dev/null; then
    echo "This evaluation needs the existing uv and espeak-ng tools."
    exit 1
fi
if [[ ! -x data/voice/envs/neutts/bin/python ]]; then
    uv venv data/voice/envs/neutts --python data/voice/envs/clone/bin/python
fi
uv pip install --python data/voice/envs/neutts/bin/python \
    torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python data/voice/envs/neutts/bin/python \
    -r scripts/voice_requirements/neutts_shootout.txt
data/voice/envs/neutts/bin/python -m scripts.tts_shootout.neutts_assets
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
    data/voice/envs/neutts/bin/python -m scripts.tts_shootout.neutts_reference
