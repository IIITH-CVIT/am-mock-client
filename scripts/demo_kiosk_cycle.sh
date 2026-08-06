#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Demo orchestrator. Simulates the kiosk team's Step 1 to 5 sequence end-to-end:
#   1. hardware presence detection      (stubbed here as an Enter keypress)
#   2/3. face recognition               (kiosk.py, the real thing)
#   4. their "main algorithm"           (stubbed: scripts/stub_main_algorithm.sh)
#   5. back to presence detection       (loop)
# Not production code, ONLY for demoing/validating the exit-code contract live.
# This is NOT what their real bash script needs to look like; it's a demonstration of what the
# client side promises to do.

ITER="${1:-1}"   # number of cycles to run (default 1; pass a bigger number to loop)

for i in $(seq 1 "$ITER"); do
    echo
    echo "=== Cycle $i/$ITER — Step 1: waiting for presence (stub: press Enter) ==="
    read -r -p "Press Enter to simulate a person stepping up to the kiosk..." _

    echo "=== Step 2/3: starting face recognition (kiosk.py) ==="
    if name=$(.venv/bin/python kiosk.py); then
        echo "=== Step 3 result: recognized '$name' ==="
        echo "=== Step 4: handing off to main algorithm (stub) ==="
        ./scripts/stub_main_algorithm.sh "$name"
    else
        rc=$?
        if [ "$rc" -eq 2 ]; then
            echo "=== Step 3 result: hardware/model ERROR (exit 2) — check camera/models ===" >&2
        else
            echo "=== Step 3 result: no match within timeout (exit 1) ==="
        fi
    fi
    echo "=== Step 5: back to presence detection ==="
done
