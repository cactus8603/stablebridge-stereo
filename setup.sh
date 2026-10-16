#!/usr/bin/env bash
set -euo pipefail
STEREO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEREO_PYTHON="${PYTHON_BIN:-python3.12}"
STEREO_ENV="${STEREO_VENV:-$STEREO_ROOT/.venv}"
"$STEREO_PYTHON" -c 'import sys; assert sys.version_info[:3]==(3,12,14), "Use Python 3.12.14"'
if [[ -e "$STEREO_ENV" ]]; then echo "Environment exists: $STEREO_ENV" >&2; exit 2; fi
"$STEREO_PYTHON" -m venv "$STEREO_ENV"
"$STEREO_ENV/bin/python" -m pip install pip==26.2.1 setuptools==78.1.0 wheel==0.47.0
"$STEREO_ENV/bin/python" -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r "$STEREO_ROOT/requirements.lock"
echo "Activate: source $STEREO_ENV/bin/activate"
