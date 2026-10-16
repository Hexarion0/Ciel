#!/usr/bin/env bash
# Install a user command backed by this checkout's virtual environment.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${HOME}/.local/bin"
if [[ $# -gt 0 ]]; then
    if [[ $# -ne 2 || "$1" != "--bin-dir" ]]; then
        echo "Usage: bash scripts/install_ciel_command.sh [--bin-dir DIRECTORY]" >&2
        exit 2
    fi
    BIN_DIR="$2"
fi

if [[ ! -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
    echo "Install CIEL's .venv first with bash setup.sh." >&2
    exit 1
fi

COMMAND_PATH="${BIN_DIR}/ciel"
MARKER="# CIEL managed launcher"
if [[ -L "${COMMAND_PATH}" || -d "${COMMAND_PATH}" ]]; then
    echo "Refusing to replace ${COMMAND_PATH}; it is not a managed launcher." >&2
    exit 1
fi
if [[ -e "${COMMAND_PATH}" ]]; then
    EXISTING_MARKER="$(sed -n '2p' "${COMMAND_PATH}")"
    if [[ "${EXISTING_MARKER}" != "${MARKER}" ]]; then
        echo "Refusing to replace an existing unrelated command: ${COMMAND_PATH}" >&2
        exit 1
    fi
fi

mkdir -p "${BIN_DIR}"
TEMP_LAUNCHER="$(mktemp "${BIN_DIR}/.ciel.XXXXXX")"
trap 'rm -f "${TEMP_LAUNCHER}"' EXIT
{
    printf '#!/usr/bin/env bash\n%s\n' "${MARKER}"
    printf 'exec bash %q "$@"\n' "${PROJECT_DIR}/scripts/launch_ciel_gpu.sh"
} > "${TEMP_LAUNCHER}"
chmod 755 "${TEMP_LAUNCHER}"
mv "${TEMP_LAUNCHER}" "${COMMAND_PATH}"
echo "Installed ${COMMAND_PATH}; run ciel or ciel start without activating .venv."
case ":${PATH}:" in
    *":${BIN_DIR}:"*) ;;
    *) echo "Add ${BIN_DIR} to your shell's PATH to use the command by name." ;;
esac
