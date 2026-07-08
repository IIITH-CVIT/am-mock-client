#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# ─────────────────────────────────────────────────────────────
# Bootstrap a native Python environment for the client, installing
# every dependency it needs to run — no manual pip steps.
#
# Default ("mock") installs ONLY what the mock-server path needs
# (yunet + mobilefacenet, 512-dim): opencv / onnxruntime / numpy /
# requests / pyyaml. That path uses NO dlib, so there's no ~15-min
# source compile and no C++ build tools required.
#
# Pass --full for the dlib/real-am-master-server path. That pulls in
# dlib (compiles from source — slow, needs cmake/g++/BLAS on the host).
#
# Safe to re-run. Prefers `uv` (fast); falls back to python3 -m venv.
# ─────────────────────────────────────────────────────────────

MODE="mock"
[ "${1:-}" = "--full" ] && MODE="full"

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

# 3) Install dependencies for the chosen path.
if [ "$MODE" = "full" ]; then
    log "FULL install (dlib real-server path) — dlib compiles from source (~15 min, needs cmake/g++/BLAS)"
    "${PIP[@]}" -r requirements.txt
else
    log "MOCK-SERVER install (light, no dlib) — yunet + mobilefacenet path only"
    "${PIP[@]}" \
        onnxruntime==1.27.0 \
        opencv-python-headless==4.13.0.92 \
        numpy==2.4.6 \
        requests==2.34.2 \
        pyyaml==6.0.3
fi

log "Done. Test against the mock server (started via am-mock-server/run.sh) with:"
log "  .venv/bin/python client.py --config config.mock-server.yaml --server <photo.jpg>"
if [ "$MODE" = "mock" ]; then
    log "For the dlib / real-am-master-server path instead, re-run: ./setup.sh --full"
fi
