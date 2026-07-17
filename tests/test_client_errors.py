"""Error-path tests for the ClientError refactor.

Every fatal condition that used to call ``sys.exit(1)`` deep in the client now
raises ``ClientError`` instead, so it can be exercised in-process with
``pytest.raises`` without killing the test runner. These tests prove finding #4
(sys.exit blocks testing) is actually closed — not just moved around.
"""

import numpy as np
import pytest

from client import (
    ClientError,
    Config,
    _detect_and_embed,
    _ensure_models,
    _load_image,
)


def _cfg(data):
    """Build a Config without running __init__ (no config file needed)."""
    cfg = Config.__new__(Config)
    cfg._data = data
    return cfg


def test_load_image_missing_raises_clienterror(tmp_path):
    with pytest.raises(ClientError, match="Failed to load image"):
        _load_image(str(tmp_path / "does_not_exist.jpg"))


def test_detect_and_embed_no_face_raises_clienterror():
    class _NoFaceDetector:
        def detect(self, frame):
            return None

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ClientError, match="No face detected"):
        _detect_and_embed(frame, _NoFaceDetector(), embedder=None)


def test_ensure_models_missing_raises_clienterror(tmp_path):
    cfg = _cfg(
        {
            "embedder": {"model": "sface"},
            "models": {
                "face_detector_path": str(tmp_path / "missing_yunet.onnx"),
                "face_recognizer_path": str(tmp_path / "missing_sface.onnx"),
                "face_recognizer_auraface_path": str(tmp_path / "missing_aura.onnx"),
            },
        }
    )
    with pytest.raises(ClientError, match="Missing model"):
        _ensure_models(cfg)


def test_ensure_models_present_does_not_raise(tmp_path):
    """The happy path must stay silent (no false-positive ClientError)."""
    yunet = tmp_path / "y.onnx"
    sface = tmp_path / "s.onnx"
    yunet.write_bytes(b"")
    sface.write_bytes(b"")
    cfg = _cfg(
        {
            "embedder": {"model": "sface"},
            "models": {
                "face_detector_path": str(yunet),
                "face_recognizer_path": str(sface),
                "face_recognizer_auraface_path": str(tmp_path / "unused_aura.onnx"),
            },
        }
    )
    _ensure_models(cfg)  # should not raise


def test_ensure_models_auraface_checks_auraface_path_not_sface(tmp_path):
    """embedder.model: auraface must validate the auraface weights, and must
    NOT require the (unused) sface weights to exist."""
    yunet = tmp_path / "y.onnx"
    aura = tmp_path / "a.onnx"
    yunet.write_bytes(b"")
    aura.write_bytes(b"")
    cfg = _cfg(
        {
            "embedder": {"model": "auraface"},
            "models": {
                "face_detector_path": str(yunet),
                "face_recognizer_path": str(tmp_path / "unused_sface.onnx"),
                "face_recognizer_auraface_path": str(aura),
            },
        }
    )
    _ensure_models(cfg)  # should not raise
