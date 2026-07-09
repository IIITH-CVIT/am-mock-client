#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# ─────────────────────────────────────────────────────────────
# Run the face-recognition client's LIVE CAMERA mode under Podman
# (team standard — not Docker), with the host's cameras and X
# display forwarded in. Builds the image first if it isn't built
# yet. Uses the default config.yaml (dlib model).
#
# Needs an X server (Linux desktop). For single-image identify you
# don't need this or a container at all — see the README's native
# path (./setup.sh + client.py <photo.jpg>).
# ─────────────────────────────────────────────────────────────

log()  { printf '\033[1;34m[run.sh]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[run.sh]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[run.sh] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v podman >/dev/null 2>&1 || die "podman not found. Run ./build.sh first (it installs Podman and builds the image)."

# Build the image on first run (or if it was removed).
if ! podman image exists face-recognition; then
    log "Image 'face-recognition' not found — building it now (first time: ~10-15 min)."
    ./build.sh
fi

# Auto-detect all connected cameras and pass them into the container.
DEVICES=()
for dev in /dev/video*; do
    [ -e "$dev" ] && DEVICES+=(--device "$dev")
done
if [ ${#DEVICES[@]} -eq 0 ]; then
    die "No cameras found at /dev/video*. Plug in a camera and retry."
fi
log "Found cameras: ${DEVICES[*]}"

[ -n "${DISPLAY:-}" ] || die "No DISPLAY set. Live camera mode needs a desktop X session (it opens a window)."

# Allow the container to talk to the local X server, and revoke on exit.
xhost +local: > /dev/null 2>&1 || warn "xhost not available — the camera window may fail to open."
trap 'xhost -local: > /dev/null 2>&1 || true' EXIT

log "Starting live camera (press 'q' in the window to quit)..."
# --group-add keep-groups lets rootless Podman keep your 'video' group so it can
# open the camera devices. --net=host + the X socket mount forward the display.
podman run --rm \
    --net=host \
    --group-add keep-groups \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    "${DEVICES[@]}" \
    face-recognition \
    --config config.yaml \
    --camera
