#!/usr/bin/env bash 
set -euo pipefail 
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Stub for the kiosk team's real "main algorithm" (Step 4). This is NOT the final
# implementation, just enough to prove the camera-handoff contract works: it
# must be able to open the same camera device the instant kiosk.py exits.
# Not production code, used only by scripts/demo_kiosk_cycle.sh.

NAME="${1:-Unknown}"
DEVICE="${2:-0}"

echo "[stub-main-algorithm] Running for: $NAME"
echo "[stub-main-algorithm] Attempting to open camera device $DEVICE (proves kiosk.py released it)..."

.venv/bin/python - "$DEVICE" <<'PY'
import sys
import cv2

device = int(sys.argv[1])
cap = cv2.VideoCapture(device)
if not cap.isOpened():
    print(f"[stub-main-algorithm] FAILED to open camera {device} — still held by kiosk.py?", file=sys.stderr)
    sys.exit(1)
print(f"[stub-main-algorithm] Camera {device} opened successfully — handoff OK.")
cap.release()
PY

echo "[stub-main-algorithm] Simulating checkout/interaction (3s)..."
sleep 3
echo "[stub-main-algorithm] Done."
