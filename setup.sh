#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# ─────────────────────────────────────────────────────────────
# Bootstrap a native Python environment for the client, installing
# every dependency it needs to run — no manual pip steps, no compile
# step (opencv-python-headless ships prebuilt wheels; no dlib).
#
# Safe to re-run. Prefers `uv` (fast); falls back to python3 -m venv.
# ─────────────────────────────────────────────────────────────

log() { printf '\033[1;34m[setup.sh]\033[0m %s\n' "$*"; }

# 1) Create the virtualenv if it doesn't exist yet.
if [ ! -d .venv ]; then
    if command -v uv >/dev/null 2>&1; then
        log "Creating .venv with uv (Python 3.13)"
        uv venv --python 3.13 .venv || uv venv .venv
    else
        log "uv not found — creating .venv with python3 -m venv"
        python3 -m venv .venv
    fi
else
    log ".venv already exists — reusing it"
fi

# 2) Choose the installer (uv if present, else the venv's pip).
if command -v uv >/dev/null 2>&1; then
    PIP=(uv pip install --python .venv/bin/python)
else
    PIP=(.venv/bin/pip install --upgrade pip)
    "${PIP[@]}" >/dev/null 2>&1 || true
    PIP=(.venv/bin/pip install)
fi

# 3) Install dependencies.
log "Installing dependencies (numpy / onnxruntime / opencv / requests / pyyaml)"
"${PIP[@]}" -r requirements.txt

log "Done. Start the mock server (am-mock-server/run.sh) and register a face, then identify with:"
log "  .venv/bin/python client.py --server <photo.jpg>                          # default sface model"
log "  .venv/bin/python client.py --config config.auraface.yaml --server <photo.jpg>   # auraface (R100)"
