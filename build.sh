#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# ─────────────────────────────────────────────────────────────
# Build the face-recognition client image under Podman (team
# standard — not Docker), installing Podman first if it's missing.
# Needs sudo + network only when it has to install Podman.
# ─────────────────────────────────────────────────────────────

log()  { printf '\033[1;34m[build.sh]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[build.sh]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[build.sh] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

pkg_install_cmd() {
    if   command -v dnf     >/dev/null 2>&1; then echo "sudo dnf install -y"
    elif command -v yum     >/dev/null 2>&1; then echo "sudo yum install -y"
    elif command -v apt-get >/dev/null 2>&1; then echo "sudo apt-get update && sudo apt-get install -y"
    elif command -v zypper  >/dev/null 2>&1; then echo "sudo zypper install -y"
    elif command -v pacman  >/dev/null 2>&1; then echo "sudo pacman -S --noconfirm"
    elif command -v brew    >/dev/null 2>&1; then echo "brew install"
    fi
}

ensure_podman() {
    if command -v podman >/dev/null 2>&1; then
        log "podman present: $(podman --version)"
        return
    fi
    warn "podman not found — installing it (this needs sudo)."
    local cmd; cmd="$(pkg_install_cmd)"
    [ -n "$cmd" ] || die "No supported package manager found. Install 'podman' manually, then re-run."
    eval "$cmd podman" || die "podman install failed. Install it manually and re-run."
    command -v podman >/dev/null 2>&1 || die "podman install did not take effect. Open a new shell and re-run."
    log "podman installed: $(podman --version)"
}

ensure_podman
log "Building 'face-recognition' image..."
podman build -t face-recognition .
log "Build complete. Run the live camera with: ./run.sh"
