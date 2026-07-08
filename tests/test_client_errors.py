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
            "detection": {"detector": "yunet"},
            "embedder": {"model": "mobilefacenet"},
            "models": {
                "yunet": str(tmp_path / "missing_yunet.onnx"),
                "mobilefacenet": str(tmp_path / "missing_mfn.onnx"),
            },
        }
    )
    with pytest.raises(ClientError, match="Missing model"):
        _ensure_models(cfg)


def test_ensure_models_present_does_not_raise(tmp_path):
    """The happy path must stay silent (no false-positive ClientError)."""
    yunet = tmp_path / "y.onnx"
    mfn = tmp_path / "m.onnx"
    yunet.write_bytes(b"")
    mfn.write_bytes(b"")
    cfg = _cfg(
        {
            "detection": {"detector": "yunet"},
            "embedder": {"model": "mobilefacenet"},
            "models": {"yunet": str(yunet), "mobilefacenet": str(mfn)},
        }
    )
    _ensure_models(cfg)  # should not raise


def test_dlib_backend_missing_dep_raises_clienterror(monkeypatch):
    """When the dlib wheel isn't installed, selecting the dlib backend must fail
    with a clear ClientError (not an AttributeError on a None module)."""
    import client as client_mod

    monkeypatch.setattr(client_mod, "dlib", None)
    cfg = _cfg({"detection": {"detector": "dlib"}, "embedder": {"model": "dlib"}})
    with pytest.raises(ClientError, match="dlib"):
        client_mod.DlibFaceDetector(cfg)


def _fake_face_rec_models(tmp_path, *, with_dats: bool):
    """A stand-in for the face_recognition_models package: an object whose
    __file__ points at a dir with a models/ subdir, optionally holding the two
    .dat weight files."""
    import types

    pkg_dir = tmp_path / "face_recognition_models"
    models = pkg_dir / "models"
    models.mkdir(parents=True)
    if with_dats:
        (models / "dlib_face_recognition_resnet_model_v1.dat").write_bytes(b"")
        (models / "shape_predictor_5_face_landmarks.dat").write_bytes(b"")
    return types.SimpleNamespace(__file__=str(pkg_dir / "__init__.py"))


def test_ensure_models_dlib_dat_missing_raises(tmp_path, monkeypatch):
    """dlib embedder selected, package present but its .dat weights are gone
    (broken/partial install): must fail fast with the clear message here, not an
    obscure dlib RuntimeError later."""
    import client as client_mod

    monkeypatch.setattr(
        client_mod, "_face_rec_models", _fake_face_rec_models(tmp_path, with_dats=False)
    )
    cfg = _cfg({"detection": {"detector": "dlib"}, "embedder": {"model": "dlib"}})
    with pytest.raises(ClientError, match="face_recognition_models"):
        client_mod._ensure_models(cfg)


def test_ensure_models_dlib_dat_present_ok(tmp_path, monkeypatch):
    import client as client_mod

    monkeypatch.setattr(
        client_mod, "_face_rec_models", _fake_face_rec_models(tmp_path, with_dats=True)
    )
    cfg = _cfg({"detection": {"detector": "dlib"}, "embedder": {"model": "dlib"}})
    client_mod._ensure_models(cfg)  # should not raise


def test_ensure_models_dlib_package_absent_raises(monkeypatch):
    """dlib embedder selected but the package isn't importable at all."""
    import client as client_mod

    monkeypatch.setattr(client_mod, "_face_rec_models", None)
    cfg = _cfg({"detection": {"detector": "dlib"}, "embedder": {"model": "dlib"}})
    with pytest.raises(ClientError, match="dlib"):
        client_mod._ensure_models(cfg)
