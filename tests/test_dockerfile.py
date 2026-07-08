import subprocess

def test_container_default_command_does_not_crash():
    """Confirms `docker run face-recognition` with no args exits cleanly (prints
    usage) instead of attempting cv2.imshow with no display and crashing."""
    result = subprocess.run(
        ["docker", "run", "--rm", "face-recognition"],
        capture_output=True, timeout=10,
    )
    assert result.returncode == 0
    assert b"usage" in result.stdout.lower()