import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("podman") is None, reason="podman not installed")
def test_container_default_command_does_not_crash():
    """Confirms `podman run face-recognition` with no args exits cleanly (prints
    usage) instead of attempting cv2.imshow with no display and crashing.

    Requires the image to be built first (./build.sh). Skips if podman is absent.
    """
    result = subprocess.run(
        ["podman", "run", "--rm", "face-recognition"],
        capture_output=True, timeout=30,
    )
    assert result.returncode == 0
    assert b"usage" in result.stdout.lower()
