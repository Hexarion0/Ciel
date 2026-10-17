#!/usr/bin/env bash
# ==============================================================================
#  CIEL — Automated Installation & Configuration Setup Script
# ==============================================================================

set -e

# Color helpers
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}"
echo "  ================================================================"
echo "      CIEL — JARVIS-Style Desktop AI Assistant Installer       "
echo "  ================================================================"
echo -e "${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Check Python
echo -e "${BOLD}[1/5] Checking Python environment...${NC}"
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.10 or newer.${NC}"
    exit 1
fi

PY_VER=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "   Found Python ${GREEN}v$PY_VER${NC} ($($PYTHON_CMD --version))"
if ! "$PYTHON_CMD" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    echo -e "${RED}Python 3.10 or newer is required.${NC}"
    exit 1
fi

# 2. Setup Virtual Environment
echo -e "\n${BOLD}[2/5] Setting up virtual environment (.venv)...${NC}"
if [ ! -d ".venv" ]; then
    echo "   Creating virtual environment in $SCRIPT_DIR/.venv..."
    $PYTHON_CMD -m venv .venv
else
    echo "   Existing virtual environment detected."
fi

# Activate virtual environment
source .venv/bin/activate
echo -e "   Active Python: ${GREEN}$(which python)${NC}"

# 3. Install Dependencies
echo -e "\n${BOLD}[3/5] Installing dependencies and packages...${NC}"
python -m pip install --upgrade pip --quiet
# Preinstall CPU torch for Pocket, avoiding Torch's additional CUDA stack.
# Existing torch installations are preserved; Pocket never initializes CUDA.
if ! python -c 'import importlib.util; assert importlib.util.find_spec("torch")'; then
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
python -m pip install -e ".[pocket-tts,gpu]"
echo -e "   ${GREEN}✅ Dependencies successfully installed.${NC}"

# 4. Pre-download Configured Speech Models (Whisper STT & Piper TTS)
echo -e "\n${BOLD}[4/5] Pre-downloading voice and AI models...${NC}"
python - <<'PY'
from pathlib import Path

from faster_whisper import download_model

from ciel.config import get_settings

print('   Checking Piper TTS voice model...')
from piper.download_voices import download_voice
tts_dir = Path('models/tts')
tts_dir.mkdir(parents=True, exist_ok=True)
if not all((tts_dir / name).is_file() for name in (
    'en_US-amy-medium.onnx', 'en_US-amy-medium.onnx.json'
)):
    print('   Downloading default Amy voice model (~60MB)...')
    download_voice('en_US-amy-medium', tts_dir)
print('   Piper voice ready.')

audio = get_settings().audio
for model_name in dict.fromkeys((audio.stt_model, audio.wake_stt_model)):
    model_path = Path(model_name).expanduser()
    if model_path.is_dir():
        print(f'   Using local speech model: {model_path}')
        continue
    if model_path.is_absolute() or model_name.startswith(('./', '../', '~')):
        raise FileNotFoundError(f'Configured speech model directory is missing: {model_path}')
    print(f'   Checking faster-whisper model: {model_name}')
    download_model(model_name)
print('   Command and wake speech models ready (no inference initialized).')
PY
echo -e "   ${GREEN}✅ Neural voice and speech models ready.${NC}"

# 5. Configuration & Setup Wizard
echo -e "\n${BOLD}[5/5] Checking configuration (.env)...${NC}"
if [ ! -f ".env" ]; then
    echo -e "   ${YELLOW}No .env file found. Launching configuration wizard...${NC}"
    python -m ciel setup
else
    echo -e "   ${GREEN}Found existing .env configuration.${NC}"
    read -p "   Do you want to reconfigure settings with the interactive wizard? [y/N]: " reconf
    if [[ "$reconf" =~ ^[Yy]$ ]]; then
        python -m ciel setup
    fi
fi

bash scripts/install_ciel_command.sh

echo -e "\n${CYAN}${BOLD}"
echo "  ================================================================"
echo "      CIEL Installation Finished — Review Setup Results      "
echo "  ================================================================"
echo -e "${NC}"
echo -e "To start listening:"
echo -e "   ${GREEN}ciel start${NC}"
echo -e "Or run directly without activating the environment:"
echo -e "   ${GREEN}.venv/bin/ciel start${NC}\n"
