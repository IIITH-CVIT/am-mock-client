#!/bin/bash
set -e

# Auto-detect all connected cameras and pass them into the container
DEVICES=""
for dev in /dev/video*; do
    [ -e "$dev" ] && DEVICES="$DEVICES --device $dev"
done

if [ -z "$DEVICES" ]; then
    echo "No cameras found at /dev/video*. Plug in a camera and retry."
    exit 1
fi

echo "Found cameras: $DEVICES"

docker run --rm \
    --network=host \
    $DEVICES \
    face-recognition
