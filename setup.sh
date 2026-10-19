#!/usr/bin/env bash
# ==============================================================================
#  CIEL — Automated Installation & Configuration Setup Script
# ==============================================================================

set -euo pipefail

# Color helpers
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}"
echo "  ================================================================"
echo "      CIEL — Personal Desktop Voice Assistant       "
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

PY_VER=$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "   Found Python ${GREEN}v$PY_VER${NC} ($("$PYTHON_CMD" --version))"
if ! "$PYTHON_CMD" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    echo -e "${RED}Python 3.10 or newer is required.${NC}"
    exit 1
fi

# 2. Setup Virtual Environment
echo -e "\n${BOLD}[2/5] Setting up virtual environment (.venv)...${NC}"
if [ ! -d ".venv" ]; then
    echo "   Creating virtual environment in $SCRIPT_DIR/.venv..."
    "$PYTHON_CMD" -m venv .venv
else
    echo "   Existing virtual environment detected."
fi

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]] || ! "$VENV_PYTHON" -c \
    'import sys; sys.exit(sys.version_info < (3, 10))'; then
    echo -e "${RED}The existing .venv needs Python 3.10 or newer. Recreate it and rerun setup.${NC}"
    exit 1
fi
echo -e "   Environment Python: ${GREEN}$VENV_PYTHON${NC}"

# 3. Install the core needed to run configuration. The wizard handles optional packages.
echo -e "\n${BOLD}[3/5] Installing CIEL core...${NC}"
"$VENV_PYTHON" -m pip install -e .

# 4. Collect and save choices before installing selected speech packages and models.
echo -e "\n${BOLD}[4/5] Configure CIEL and prepare your selections...${NC}"
echo "   Review your settings before voice packages and models download."
echo "   Quick start keeps saved settings; Customize changes your choices."
if "$VENV_PYTHON" -m ciel setup; then
    SETUP_STATUS=0
else
    SETUP_STATUS=$?
fi

# 5. Install a launcher even if setup needs another attempt.
echo -e "\n${BOLD}[5/5] Installing the ciel command...${NC}"
bash scripts/install_ciel_command.sh

if [[ "$SETUP_STATUS" -ne 0 ]]; then
    echo -e "\n${YELLOW}CIEL core is installed; setup did not finish.${NC}"
    echo "Review the setup messages, then retry: .venv/bin/ciel setup"
    exit "$SETUP_STATUS"
fi

echo -e "\n${GREEN}Configuration and selected components are prepared.${NC}"
echo "Review the wizard's voice-test result before starting."
echo "Start listening: ciel start (or .venv/bin/ciel start)"
echo "Change choices or retry downloads: ciel setup"
