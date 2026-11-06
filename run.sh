#!/usr/bin/env bash
# Padarth Khoj web app: creates a virtual environment on first run, installs what is needed, starts the server.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
  echo "Python 3.9 or newer is required (set PYTHON=/path/to/python3 to choose one)."; exit 1
fi
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
if ! cmp -s requirements.txt .venv/requirements.installed || ! cmp -s engine/requirements.txt .venv/engine-requirements.installed; then
  echo "Installing packages (first run downloads PyTorch, a few hundred MB)..."
  pip install --upgrade pip >/dev/null
  # On Linux the default PyTorch wheel bundles CUDA (several GB); the CPU build is all this app needs.
  if [ "$(uname -s)" = "Linux" ] && ! python -c "import torch" 2>/dev/null; then
    pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
  pip install -r requirements.txt
  cp requirements.txt .venv/requirements.installed; cp engine/requirements.txt .venv/engine-requirements.installed
fi
exec python server/app.py --open "$@"
