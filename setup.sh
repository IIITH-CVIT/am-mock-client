#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# ─────────────────────────────────────────────────────────────
# Bootstrap a native Python environment for the client, installing
# every dependency it needs to run — no manual pip steps.
#
# Default: installs EVERYTHING, including dlib — because the default
# config.yaml uses the dlib model. dlib compiles from source (~10-15
# min, needs cmake/g++/BLAS on the host, which most Linux boxes have).
#
# Pass --light (a.k.a. --yunet) to skip dlib and install only what the
# YuNet + MobileFaceNet model needs (opencv / onnxruntime / numpy /
# requests / pyyaml). Fast, no compile — but you must then run with
# --config config.yunet.yaml (not the default dlib config.yaml).
#
# Safe to re-run. Prefers `uv` (fast); falls back to python3 -m venv.
# ─────────────────────────────────────────────────────────────

MODE="full"
case "${1:-}" in
    --light|--yunet) MODE="light" ;;
esac

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
    log "FULL install (default, includes dlib) — dlib compiles from source (~10-15 min, needs cmake/g++/BLAS)"
    "${PIP[@]}" -r requirements.txt
else
    log "LIGHT install (no dlib) — YuNet + MobileFaceNet only. Use --config config.yunet.yaml."
    "${PIP[@]}" \
        onnxruntime==1.27.0 \
        opencv-python-headless==4.13.0.92 \
        numpy==2.4.6 \
        requests==2.34.2 \
        pyyaml==6.0.3
fi

log "Done. Start the mock server (am-mock-server/run.sh) and register a face, then identify with:"
if [ "$MODE" = "full" ]; then
    log "  .venv/bin/python client.py --server <photo.jpg>                        # default dlib model"
    log "  .venv/bin/python client.py --config config.yunet.yaml --server <photo.jpg>   # YuNet+MobileFaceNet"
else
    log "  .venv/bin/python client.py --config config.yunet.yaml --server <photo.jpg>"
    log "For the default dlib model too, re-run: ./setup.sh   (installs dlib)"
fi
