import os
import sys
from pathlib import Path 
from typing import Any, Dict

import yaml 

from .errors import ClientError

_DEFAULT_CONFIG: Dict[str, Any] = {
    "mode": "server",
    "server": {
        "url": "http://localhost:8000",
        "timeout": 10,
    },
    "diagnostic": {
        "db_path": "./diagnostic_mode/diagnostic.db",
        "faces_dir": "./diagnostic_mode/faces",
        "cosine_threshold": 0.6,
        "topk": 3,
    },
    "models": {
        "face_detector_path": "./models/face_detection_yunet_2026may.onnx",
        "face_recognizer_path": "./models/face_recognition_sface_2021dec.onnx",
        "face_recognizer_auraface_path": "./models/aurar100.onnx",
    },
    "detection": {
        "threshold": 0.5,
        "input_size": 640,
    },
    "embedder": {
        "model": "sface",
    },
    "camera": {
        "device": 0,
        "frame_skip": 10,
        "backend": "opencv",
    },
    "logging": {
        "level": "INFO",
    },
}

# In a PyInstaller bundle __file__ is the exe; use sys.executable dir instead
if getattr(sys, "frozen", False):
    _CONFIG_PATH = os.path.join(os.path.dirname(sys.executable), "config.yaml")
else:
    _CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "config.yaml")

_APP_DIR = os.path.dirname(_CONFIG_PATH)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base (override wins on conflict)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class Config:
    """
    Thin YAML config loader that mirrors AppConfig from the reference (config.py).

    Loads config.yaml, deep-merges with built-in defaults, and exposes
    dot-path access:  cfg.get("server.url")
    """

    def __init__(self, config_path: str = _CONFIG_PATH):
        self._data = dict(_DEFAULT_CONFIG)
        self.config_path = config_path

        if not os.path.isfile(config_path):
            raise ClientError(
                f"Config file not found: {config_path!r}. Check the path and try again."
            )

        with open(config_path) as f:
            loaded = yaml.safe_load(f) or {}
        self._data = _deep_merge(self._data, loaded)
        self._path = config_path

    def get(self, dotpath: str, default=None):
        parts = dotpath.split(".")
        node = self._data
        for p in parts:
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node

    # Convenience accessors (keeps call-sites readable)
    @property
    def mode(self) -> str:
        return str(self.get("mode", "server")).lower()

    @property
    def server_url(self) -> str:
        return str(self.get("server.url", "http://localhost:8000"))

    @property
    def server_timeout(self) -> int:
        return int(self.get("server.timeout", 10))

    @property
    def diag_db_path(self) -> str:
        return os.path.expanduser(str(self.get("diagnostic.db_path", "./diagnostic_mode/diagnostic.db")))

    @property
    def diag_faces_dir(self) -> str:
        return os.path.expanduser(str(self.get("diagnostic.faces_dir", "./diagnostic_mode/faces")))

    @property
    def cosine_threshold(self) -> float:
        return float(self.get("diagnostic.cosine_threshold", 0.6))

    @property
    def topk(self) -> int:
        return int(self.get("diagnostic.topk", 3))

    @staticmethod
    def _resolve_path(path: str) -> str:
        """Expand ~ and resolve relative paths against the app dir (not cwd)."""
        path = os.path.expanduser(str(path))
        if not os.path.isabs(path):
            path = os.path.join(_APP_DIR, path)
        return path

    @property
    def face_detector_path(self) -> str:
        return self._resolve_path(
            self.get("models.face_detector_path", "./models/face_detection_yunet_2026may.onnx")
        )

    @property
    def face_recognizer_path(self) -> str:
        return self._resolve_path(
            self.get("models.face_recognizer_path", "./models/face_recognition_sface_2021dec.onnx")
        )

    @property
    def face_recognizer_auraface_path(self) -> str:
        return self._resolve_path(
            self.get("models.face_recognizer_auraface_path", "./models/aurar100.onnx")
        )

    @property
    def detection_threshold(self) -> float:
        return float(self.get("detection.threshold", 0.5))

    @property
    def face_detector_input_size(self) -> int:
        return int(self.get("detection.input_size", 640))

    @property
    def embedder_model(self) -> str:
        return str(self.get("embedder.model", "sface")).lower()

    @property
    def camera_device(self) -> int:
        return int(self.get("camera.device", 0))

    @property
    def camera_frame_skip(self) -> int:
        return int(self.get("camera.frame_skip", 10))
    
    @property
    def camera_backend(self) -> str:
        return str(self.get("camera.backend", "opencv")).lower()

    @property
    def log_level(self) -> int:
        import logging
        level_str = str(self.get("logging.level", "INFO")).upper()
        return getattr(logging, level_str, logging.INFO)
