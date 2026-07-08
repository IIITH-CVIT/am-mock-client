#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -z "${DISPLAY:-}" ]; then
    echo "SKIP: no DISPLAY set (headless environment). Camera smoke test needs a real X session"
    exit 0
fi

docker build -t face-recognition-test .
xhost +local:docker > /dev/null 2>&1 || true
trap 'xhost -local:docker > /dev/null 2>&1 || true; docker rm -f face-rec-smoketest > /dev/null 2>&1 || true' EXIT

# Run camera mode for 3s in the background, then check the container is still
# alive (i.e. it didn't crash on cv2.imshow) rather than actually verifying
# pixels. This is a smoke test, not a full camera integration test.
docker run -d --rm --name face-rec-smoketest \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    face-recognition-test --camera

sleep 3

if docker ps --filter "name=face-rec-smoketest" --filter "status=running" | grep -q face-rec-smoketest; then
    echo "OK: camera mode container still running after 3s"
else
    echo "FAILED: camera container exited early — check X11 forwarding"
    docker logs face-rec-smoketest || true
    exit 1
fi