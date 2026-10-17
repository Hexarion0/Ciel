#!/usr/bin/env bash
# Make existing project-local CUDA libraries visible before Python starts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; install CIEL's virtual environment first." >&2
  exit 1
fi

CUDA_LIBRARIES="$("${PYTHON}" - "${ROOT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
search_roots = [Path(sys.prefix) / "lib", root / "training/piper/.venv/lib"]
directories = []
for search_root in search_roots:
    for directory in sorted(search_root.glob("python*/site-packages/nvidia/*/lib")):
        if directory.is_dir() and str(directory) not in directories:
            directories.append(str(directory))
print(":".join(directories))
PY
)"

if [[ -n "${CUDA_LIBRARIES}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_LIBRARIES}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cd "${ROOT}"
if [[ $# -eq 0 ]]; then
  set -- start
fi
exec "${PYTHON}" -m ciel "$@"
