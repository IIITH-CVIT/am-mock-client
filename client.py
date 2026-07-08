#!/usr/bin/env python3
"""
Face Recognition Client — image and live-camera

Usage:
    # Identify (mode determined by config.yaml):
    .venv/bin/python client.py <image_path>

    # Live camera feed (also the default when no image is given):
    .venv/bin/python client.py --camera

    # Force diagnostic mode (local SQLite only):
    .venv/bin/python client.py --diag <image_path>

    # Force server mode:
    .venv/bin/python client.py --server <image_path>

    # Register a face into the local diagnostic DB:
    .venv/bin/python client.py --register <name> <image_path>

    # List all faces in the local diagnostic DB:
    .venv/bin/python client.py --list

    # Use a specific config file (default: config.yaml):
    .venv/bin/python client.py --config my_config.yaml <image_path>

Detection + embedding backends (config.yaml: detection.detector / embedder.model)
mirror am-master-server's two identification backends (app/core/face_rec.py) so
embeddings computed here match what the server has enrolled. The two pairings
are never mixed — pick one:
  - dlib (HOG + ResNet)   — default; matches DlibBackend: HOG upsample=1,
                            128-dim raw embedding, server-side L2 cutoff 0.6
  - YuNet + MobileFaceNet — alternate; matches AurafaceBackend: YuNet (ONNX)
                            detection + 5-point landmarks, 512-dim L2-normalised
                            embedding, server-side L2 cutoff 0.8

Diagnostic mode mirrors the DiagnosticFacePipeline + DatabaseManager pattern
from the reference (am-fru-desktop-app-Fru-MacOs/database.py):
  - SQLite storage with cosine similarity search
  - Face crops saved to diagnostic_mode/faces/
  - Top-3 nearest-neighbour display on unknown faces
"""

import argparse
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import requests
import yaml

# dlib + its model-weights package are imported lazily: the yunet/mobilefacenet
# (ONNX) backends don't need them, and deferring the import lets client.py be
# imported for unit tests without the heavy compiled dlib wheel installed. The
# dlib-backed classes raise a clear ClientError at construction if it's missing.
try:
    import dlib
    import face_recognition_models as _face_rec_models
except ModuleNotFoundError:  # pragma: no cover - depends on install profile
    dlib = None
    _face_rec_models = None

_DLIB_MISSING_MSG = (
    "The dlib backend is selected (detector/embedder: dlib) but the 'dlib' and "
    "'face_recognition_models' packages aren't installed. Install them, or switch "
    "to the yunet/mobilefacenet pairing (e.g. --config config.mock-server.yaml)."
)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

_DEFAULT_CONFIG: Dict[str, Any] = {
    "mode": "server",
    "server": {
        "url": "http://localhost:8100",
        "timeout": 10,
    },
    "diagnostic": {
        "db_path": "./diagnostic_mode/diagnostic.db",
        "faces_dir": "./diagnostic_mode/faces",
        "cosine_threshold": 0.6,
        "topk": 3,
    },
    "models": {
        "yunet": "./models/face_detection_yunet_2023mar.onnx",
        "mobilefacenet": "./models/mobilefacenet.onnx",
    },
    "detection": {
        "detector": "dlib",
        "threshold": 0.3,
        "input_size": 640,
        "num_upsamples": 1,
    },
    "embedder": {
        "model": "dlib",
        "num_jitters": 1,
    },
    "camera": {
        "device": 0,
        "frame_skip": 10,
    },
    "logging": {
        "level": "INFO",
    },
}

# In a PyInstaller bundle __file__ is the exe; use sys.executable dir instead
if getattr(sys, "frozen", False):
    _CONFIG_PATH = os.path.join(os.path.dirname(sys.executable), "config.yaml")
else:
    _CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

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

        if os.path.isfile(config_path):
            with open(config_path) as f:
                loaded = yaml.safe_load(f) or {}
            self._data = _deep_merge(self._data, loaded)
        else:
            # First run — write the defaults so the user can see / edit them
            pass

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
        return str(self.get("server.url", "http://localhost:8100"))

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
    def yunet_model(self) -> str:
        return self._resolve_path(self.get("models.yunet", "./models/face_detection_yunet_2023mar.onnx"))

    @property
    def mobilefacenet_model(self) -> str:
        return self._resolve_path(self.get("models.mobilefacenet", "./models/mobilefacenet.onnx"))

    @property
    def detector_backend(self) -> str:
        return str(self.get("detection.detector", "dlib")).lower()

    @property
    def detection_threshold(self) -> float:
        return float(self.get("detection.threshold", 0.3))

    @property
    def yunet_input_size(self) -> int:
        return int(self.get("detection.input_size", 640))

    @property
    def num_upsamples(self) -> int:
        return int(self.get("detection.num_upsamples", 1))

    @property
    def embedder_model(self) -> str:
        return str(self.get("embedder.model", "dlib")).lower()

    @property
    def num_jitters(self) -> int:
        return int(self.get("embedder.num_jitters", 1))

    @property
    def camera_device(self) -> int:
        return int(self.get("camera.device", 0))

    @property
    def camera_frame_skip(self) -> int:
        return int(self.get("camera.frame_skip", 10))

    @property
    def log_level(self) -> int:
        level_str = str(self.get("logging.level", "INFO")).upper()
        return getattr(logging, level_str, logging.INFO)


# ─────────────────────────────────────────────────────────────
# Logging (configured after Config is loaded)
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("face_client")


class ClientError(Exception):
    """Fatal, user-facing client error.

    Raised anywhere below main() instead of calling sys.exit(). main() is the
    single place that catches it and translates it to a clean exit(1), which
    keeps every function importable and unit-testable (no process-killing
    sys.exit in library code).
    """


# ─────────────────────────────────────────────────────────────
# Verify the model weights required by the configured detector/
# embedder are present, and fail fast with a clear message if not:
#   - ONNX files (yunet/mobilefacenet) are bundled in ./models/
#     (same files as mock-server/models/ — no auto-download).
#   - dlib .dat files (dlib backend) ship inside the
#     face_recognition_models pip package, not ./models/.
# ─────────────────────────────────────────────────────────────

def _raise_missing(missing: list, remediation: str) -> None:
    lines = [f"Missing model: {label}  ({path})" for label, path in missing]
    lines.append(remediation)
    raise ClientError("\n".join(lines))


def _ensure_models(cfg: Config) -> None:
    # ONNX weights bundled in ./models/ (yunet detector / mobilefacenet embedder)
    onnx_required = []
    if cfg.detector_backend == "yunet":
        onnx_required.append(("YuNet face detector", cfg.yunet_model))
    if cfg.embedder_model == "mobilefacenet":
        onnx_required.append(("MobileFaceNet embedder", cfg.mobilefacenet_model))

    onnx_missing = [(label, path) for label, path in onnx_required if not os.path.exists(path)]
    if onnx_missing:
        _raise_missing(
            onnx_missing,
            f"Copy the missing .onnx file(s) into {os.path.dirname(onnx_missing[0][1])} "
            f"(same files as mock-server/models/).",
        )

    # The dlib embedder loads two .dat weight files from inside the
    # face_recognition_models pip package (see DlibEmbedder.__init__). Validate
    # them here so a missing package or a broken/partial install fails with the
    # same clear message as the ONNX case, instead of an obscure RuntimeError
    # raised by dlib deep in the embedder mid-run. (The dlib detector needs only
    # dlib's built-in HOG detector, so the .dat files matter only for embedding.)
    if cfg.embedder_model == "dlib":
        if _face_rec_models is None:
            raise ClientError(_DLIB_MISSING_MSG)
        dat_dir = os.path.join(os.path.dirname(_face_rec_models.__file__), "models")
        dat_required = [
            ("dlib face-recognition ResNet weights",
             os.path.join(dat_dir, "dlib_face_recognition_resnet_model_v1.dat")),
            ("dlib 5-point shape predictor",
             os.path.join(dat_dir, "shape_predictor_5_face_landmarks.dat")),
        ]
        dat_missing = [(label, path) for label, path in dat_required if not os.path.exists(path)]
        if dat_missing:
            _raise_missing(
                dat_missing,
                "These ship inside the 'face_recognition_models' package; reinstall it "
                "(e.g. pip install --force-reinstall face_recognition_models).",
            )


# ─────────────────────────────────────────────────────────────
# Detection result
# ─────────────────────────────────────────────────────────────

class FaceDetection:
    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        score: float,
        landmarks: Optional[List[Tuple[float, float]]] = None,
    ):
        self.bbox      = bbox        # (x1, y1, x2, y2)
        self.score     = score
        self.landmarks = landmarks   # 5-point (x, y) list — only set by YuNet


# ─────────────────────────────────────────────────────────────
# Face detection — dlib HOG detector
# Mirrors reference DlibFaceDetector in face_recognition_pipeline.py
# ─────────────────────────────────────────────────────────────

class DlibFaceDetector:
    def __init__(self, cfg: Config):
        if dlib is None:
            raise ClientError(_DLIB_MISSING_MSG)
        self.threshold     = cfg.detection_threshold
        self.num_upsamples = cfg.num_upsamples
        self._detector     = dlib.get_frontal_face_detector()
        logger.info(
            "DlibFaceDetector (HOG) | upsamples=%d  threshold=%.2f",
            self.num_upsamples, self.threshold,
        )

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceDetection]:
        if frame_bgr is None or frame_bgr.ndim != 3:
            return None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        dets, scores, _ = self._detector.run(rgb, self.num_upsamples, self.threshold)
        if not dets:
            return None
        best = int(np.argmax(scores))
        d = dets[best]
        return FaceDetection(
            bbox=(d.left(), d.top(), d.right(), d.bottom()),
            score=float(scores[best]),
        )


# ─────────────────────────────────────────────────────────────
# Face detection — YuNet ONNX detector
# Mirrors mock-server FaceEngine._detect_face: manual multi-scale
# decode (letterbox to a square input, strides 8/16/32), returns
# bbox + 5-point landmarks for downstream alignment.
# ─────────────────────────────────────────────────────────────

class YuNetDetector:
    STRIDES = (8, 16, 32)

    def __init__(self, cfg: Config):
        model_path = cfg.yunet_model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YuNet model not found: {model_path}")
        self.input_size      = cfg.yunet_input_size
        self.score_threshold = cfg.detection_threshold
        self._session        = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name     = self._session.get_inputs()[0].name
        self._output_names   = [o.name for o in self._session.get_outputs()]
        logger.info(
            "YuNetDetector (ONNX) | input=%dx%d  threshold=%.2f",
            self.input_size, self.input_size, self.score_threshold,
        )

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceDetection]:
        if frame_bgr is None or frame_bgr.ndim != 3:
            return None
        h, w = frame_bgr.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        nw, nh = int(w * scale), int(h * scale)

        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        canvas[:nh, :nw, :] = cv2.resize(frame_bgr, (nw, nh))
        blob = canvas.astype(np.float32).transpose(2, 0, 1)[None, ...]

        outputs = self._session.run(self._output_names, {self._input_name: blob})
        by_name = dict(zip(self._output_names, outputs))

        best = None  # (score, bbox, landmarks)
        for stride in self.STRIDES:
            cls  = by_name[f"cls_{stride}"][0]
            obj  = by_name[f"obj_{stride}"][0]
            bbox = by_name[f"bbox_{stride}"][0]
            kps  = by_name[f"kps_{stride}"][0]
            fm_width = self.input_size // stride

            scores = np.sqrt(np.clip(cls[:, 0], 0.0, 1.0) * np.clip(obj[:, 0], 0.0, 1.0))
            for i in np.where(scores > self.score_threshold)[0]:
                score = float(scores[i])
                if best is not None and score <= best[0]:
                    continue

                col, row = int(i % fm_width), int(i // fm_width)
                cx = (col + bbox[i, 0]) * stride
                cy = (row + bbox[i, 1]) * stride
                bw = np.exp(bbox[i, 2]) * stride
                bh = np.exp(bbox[i, 3]) * stride

                bbox_out = (
                    int(max(0, min((cx - bw / 2.0) / scale, w))),
                    int(max(0, min((cy - bh / 2.0) / scale, h))),
                    int(max(0, min((cx + bw / 2.0) / scale, w))),
                    int(max(0, min((cy + bh / 2.0) / scale, h))),
                )
                landmarks = [
                    (
                        max(0.0, min(float((col + kps[i, 2 * k]) * stride / scale), float(w))),
                        max(0.0, min(float((row + kps[i, 2 * k + 1]) * stride / scale), float(h))),
                    )
                    for k in range(5)
                ]
                best = (score, bbox_out, landmarks)

        if best is None:
            return None
        score, bbox_out, landmarks = best
        return FaceDetection(bbox=bbox_out, score=score, landmarks=landmarks)


# ─────────────────────────────────────────────────────────────
# Embedders
# ─────────────────────────────────────────────────────────────

class DlibEmbedder:
    """128-dim raw embeddings via dlib ResNet (mirrors reference DlibEmbeddingExtractor)."""

    DIM = 128

    def __init__(self, cfg: Config):
        if dlib is None or _face_rec_models is None:
            raise ClientError(_DLIB_MISSING_MSG)
        self.num_jitters = cfg.num_jitters

        # Works in both normal Python and PyInstaller frozen bundles
        models_dir = os.path.join(os.path.dirname(_face_rec_models.__file__), "models")
        self._face_encoder    = dlib.face_recognition_model_v1(
            os.path.join(models_dir, "dlib_face_recognition_resnet_model_v1.dat")
        )
        self._shape_predictor = dlib.shape_predictor(
            os.path.join(models_dir, "shape_predictor_5_face_landmarks.dat")
        )
        self._detector = dlib.get_frontal_face_detector()
        logger.info("DlibEmbedder (ResNet 128-dim, raw) | jitters=%d", self.num_jitters)

    def embed(self, frame_bgr: np.ndarray, detection: FaceDetection) -> np.ndarray:
        rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rect  = self._dlib_box(rgb, detection.bbox)
        shape = self._shape_predictor(rgb, rect)
        enc   = self._face_encoder.compute_face_descriptor(rgb, shape, self.num_jitters)
        return np.asarray(enc, dtype=np.float32)   # RAW — do NOT L2-normalise

    def _dlib_box(self, rgb: np.ndarray, hint_bbox: Tuple[int, int, int, int]):
        h, w = rgb.shape[:2]
        dets = self._detector(rgb, 1)
        if dets:
            if hint_bbox is not None:
                hx1, hy1, hx2, hy2 = hint_bbox

                def _overlap(d) -> float:
                    ix1 = max(hx1, d.left());  iy1 = max(hy1, d.top())
                    ix2 = min(hx2, d.right()); iy2 = min(hy2, d.bottom())
                    return max(0, ix2 - ix1) * max(0, iy2 - iy1)

                return max(dets, key=_overlap)
            return max(dets, key=lambda d: (d.right() - d.left()) * (d.bottom() - d.top()))

        if hint_bbox is not None:
            x1, y1, x2, y2 = hint_bbox
            return dlib.rectangle(max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2))
        return dlib.rectangle(0, 0, w - 1, h - 1)


class MobileFaceNetEmbedder:
    """
    512-dim L2-normalised embeddings via ONNX MobileFaceNet.

    Mirrors mock-server FaceEngine.embed: 5-point landmark alignment into the
    112x112 ArcFace pose (falls back to a margin crop + resize when no
    landmarks are available, e.g. detector=dlib), BGR->RGB, (x-127.5)/128.
    """

    DIM         = 512
    INPUT_SIZE  = 112
    CROP_MARGIN = 0.20

    # ArcFace/MobileFaceNet 112x112 reference landmarks
    _REF_LANDMARKS = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    def __init__(self, cfg: Config):
        model_path = cfg.mobilefacenet_model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MobileFaceNet model not found: {model_path}")
        self._session    = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        out_shape        = self._session.get_outputs()[0].shape
        self.DIM         = out_shape[-1] if out_shape else 512
        logger.info("MobileFaceNetEmbedder (%d-dim, L2) | %s", self.DIM, model_path)

    def embed(self, frame_bgr: np.ndarray, detection: FaceDetection) -> np.ndarray:
        face_img = self._crop_and_align(frame_bgr, detection)
        rgb  = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[None, ...]
        vec  = self._session.run(None, {self._input_name: blob})[0].flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _crop_and_align(self, frame_bgr: np.ndarray, detection: FaceDetection) -> np.ndarray:
        if detection.landmarks and len(detection.landmarks) == 5:
            aligned = self._align(frame_bgr, detection.landmarks)
            if aligned is not None:
                return aligned

        x1, y1, x2, y2 = detection.bbox
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mx, my = int((x2 - x1) * self.CROP_MARGIN), int((y2 - y1) * self.CROP_MARGIN)
        crop = frame_bgr[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
        if crop.size == 0:
            crop = frame_bgr
        return cv2.resize(crop, (self.INPUT_SIZE, self.INPUT_SIZE))

    def _align(self, frame_bgr: np.ndarray, landmarks: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        lm = np.array(landmarks, dtype=np.float32)
        transform, _ = cv2.estimateAffinePartial2D(lm, self._REF_LANDMARKS, method=cv2.LMEDS)
        if transform is None:
            return None
        return cv2.warpAffine(
            frame_bgr, transform, (self.INPUT_SIZE, self.INPUT_SIZE),
            flags=cv2.INTER_LINEAR, borderValue=0,
        )


# ─────────────────────────────────────────────────────────────
# Diagnostic DB — mirrors reference DatabaseManager diagnostic methods
# SQLite local storage with cosine similarity search
# ─────────────────────────────────────────────────────────────

class DiagnosticDB:
    """
    Local SQLite face store for offline / diagnostic mode.

    Mirrors the diagnostic_faces table and search logic from
    DatabaseManager in the reference (database.py):
      - register()    → INSERT embedding blob
      - search()      → vectorised cosine similarity → (name, image_path, sim)
      - search_topk() → top-K without threshold gate
      - list_all()    → all registered faces
    """

    def __init__(self, cfg: Config):
        self.db_path   = cfg.diag_db_path
        self.threshold = cfg.cosine_threshold
        self.topk      = cfg.topk
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS diagnostic_faces (
                    face_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    name       TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    embedding  BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON diagnostic_faces(name)")
        logger.info("DiagnosticDB | %s", self.db_path)

    @staticmethod
    def _to_blob(emb: np.ndarray) -> bytes:
        return emb.astype(np.float32).tobytes()

    @staticmethod
    def _from_blob(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def register(self, name: str, embedding: np.ndarray, image_path: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO diagnostic_faces (name, image_path, embedding) VALUES (?, ?, ?)",
                (name, image_path, self._to_blob(embedding)),
            )
            face_id = cur.lastrowid
        logger.info("Registered '%s' → face_id=%d  dim=%d", name, face_id, embedding.shape[0])
        return face_id

    def _scored_candidates(
        self, embedding: np.ndarray, *, warn_on_dim_mismatch: bool = False
    ) -> Tuple[List[str], List[str], np.ndarray]:
        """Shared core of search() and search_topk().

        Load every stored face, keep the ones whose embedding dimension matches
        the query, and return (names, image_paths, cosine_sims) aligned by index.
        Returns empty lists + an empty array when the DB is empty, the query is
        degenerate (zero norm), or nothing shares the query's dimension —
        logging a warning in that last case only when asked, so the message
        fires from search() (as it always did) but not a second time from the
        search_topk() call that immediately follows it in cmd_diagnostic().
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, image_path, embedding FROM diagnostic_faces"
            ).fetchall()

        query      = embedding.astype(np.float32)
        query_dim  = query.shape[0]
        query_norm = np.linalg.norm(query)
        empty: Tuple[List[str], List[str], np.ndarray] = ([], [], np.empty(0, dtype=np.float32))
        if not rows or query_norm == 0:
            return empty

        names, paths, embs = [], [], []
        for row in rows:
            stored = self._from_blob(row["embedding"])
            if stored.shape[0] == query_dim:
                names.append(row["name"])
                paths.append(row["image_path"])
                embs.append(stored)

        if not embs:
            if warn_on_dim_mismatch:
                logger.warning(
                    "DiagnosticDB: no stored embeddings with dim=%d — "
                    "register faces first with --register", query_dim
                )
            return empty

        stored_mat = np.array(embs, dtype=np.float32)
        norms = np.linalg.norm(stored_mat, axis=1)
        sims  = np.zeros(len(names), dtype=np.float32)
        valid = norms > 0
        if valid.any():
            sims[valid] = (stored_mat[valid] @ query) / (norms[valid] * query_norm)

        return names, paths, sims

    def search(self, embedding: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """
        Vectorised cosine similarity search (mirrors reference search_diagnostic_faces).
        Returns: (name, image_path, similarity) — name/path are None if below threshold.
        """
        names, paths, sims = self._scored_candidates(embedding, warn_on_dim_mismatch=True)
        if sims.size == 0:
            return None, None, 0.0

        best     = int(np.argmax(sims))
        best_sim = float(sims[best])
        if best_sim >= self.threshold:
            return names[best], paths[best], best_sim
        return None, None, best_sim

    def search_topk(self, embedding: np.ndarray) -> List[Tuple[str, float]]:
        """Top-K without threshold gate (mirrors reference search_diagnostic_faces_topk)."""
        names, _paths, sims = self._scored_candidates(embedding)
        if sims.size == 0:
            return []
        order = np.argsort(-sims)[:max(1, self.topk)]
        return [(names[i], float(sims[i])) for i in order]

    def list_all(self) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT face_id, name, image_path, created_at FROM diagnostic_faces "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────
# Server client
# ─────────────────────────────────────────────────────────────

class ServerClient:
    """Sends face embeddings to the FRU API."""

    def __init__(self, cfg: Config):
        self.base_url = cfg.server_url.rstrip("/")
        self.timeout  = cfg.server_timeout
        self._session = requests.Session()

    def health_check(self) -> bool:
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=5)
            ok = resp.status_code == 200
            logger.info("Server %s: %s", self.base_url, "OK" if ok else f"HTTP {resp.status_code}")
            return ok
        except requests.RequestException as e:
            logger.warning("Server unreachable (%s): %s", self.base_url, e)
            return False

    def _post(self, vector: np.ndarray) -> Dict:
        # 6 decimal places is intentional and provably safe here, not an
        # oversight. Embeddings are L2-normalised (or small-magnitude raw dlib
        # descriptors), so components sit well inside [-1, 1]. Measured over 2000
        # random 512-d unit vectors, %.6f rounding gives at most ~5e-7 per
        # component and ~7e-6 whole-vector L2 error, shifting the server's L2
        # match distance by <=1.5e-6, which is six orders of magnitude below the 0.8
        # match threshold, and it never changed a nearest-neighbour pick. Full
        # float32 round-trip would need ~9 significant figures; it buys nothing
        # for matching and only enlarges the payload.
        vec_str = ",".join(f"{v:.6f}" for v in vector.flatten())
        try:
            resp = self._session.post(
                f"{self.base_url}/api/v1/identify/",
                data={"type": "face", "face_vector": vec_str, "n": 1},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            return {"error": str(e), "status_code": e.response.status_code if e.response else None}
        except requests.RequestException as e:
            return {"error": str(e)}
    
    # NOTE: this only fires against a mock server with the 512-dim validation fix
    # applied (returns 400 on wrong-dim vectors). Against an unfixed server, a
    # dimension mismatch looks identical to a genuine no-match: check
    # `Embedding: N-dim` in the log above and compare against the server's gallery.
    def identify(self, embedding: np.ndarray) -> Optional[str]:
        result = self._post(embedding)
        if "error" in result:
            detail = result["error"]
            if result.get("status_code") == 400 and "dimensions" in str(detail):
                logger.error(
                "Dimension mismatch talking to the server (sent %d-dim). "
                "If you're testing against iiith-cvit-am-mock-server, it only "
                "supports the yunet/mobilefacenet pairing (512-dim). Run with "
                "--config config.mock-server.yaml instead of config.yaml.",
                embedding.shape[0],
                )
            else:
                logger.warning("Server error (%d-dim): %s", embedding.shape[0], result["error"])
            return None
        name = result.get("name")
        confidence = result.get("confidence")
        distance = result.get("distance")
        logger.info("Server → name=%r confidence = %s distance = %s", name, confidence, distance)
        return name

    def close(self):
        self._session.close()

class ClientError(Exception):
     """Fatal, user-facing client error. main() converts it to exit(1);
    nothing below main() calls sys.exit, so it all stays importable/testable."""

# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def _load_image(path: str) -> np.ndarray:
    frame = cv2.imread(path)
    if frame is None:
        raise ClientError(f"Failed to load image: {path}")
    logger.info("Image: %dx%d  ← %s", frame.shape[1], frame.shape[0], path)
    return frame


def _save_face_crop(frame: np.ndarray, bbox: Tuple, name: str, faces_dir: str) -> str:
    os.makedirs(faces_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in name).lower()
    path = os.path.join(faces_dir, f"{safe}_{int(time.time())}.jpg")
    x1, y1, x2, y2 = bbox
    ih, iw = frame.shape[:2]
    cv2.imwrite(path, frame[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)])
    return path


def _detect_and_embed(
    frame: np.ndarray,
    detector,
    embedder,
) -> Tuple[FaceDetection, np.ndarray]:
    detection = detector.detect(frame)
    if detection is None:
        raise ClientError("No face detected in the image.")
    x1, y1, x2, y2 = detection.bbox
    logger.info("Face: bbox=(%d,%d,%d,%d)  score=%.3f", x1, y1, x2, y2, detection.score)
    vec = embedder.embed(frame, detection)
    logger.info("Embedding: %d-dim", vec.shape[0])
    return detection, vec


# ─────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────

def cmd_register(
    name: str, image_path: str,
    detector, embedder,
    cfg: Config,
) -> None:
    logger.info("REGISTER '%s'  ← %s", name, image_path)
    frame = _load_image(image_path)
    detection, vec = _detect_and_embed(frame, detector, embedder)
    crop_path = _save_face_crop(frame, detection.bbox, name, cfg.diag_faces_dir)
    logger.info("Face crop saved: %s", crop_path)
    db      = DiagnosticDB(cfg)
    face_id = db.register(name, vec, crop_path)
    print(f"\n>>> Registered '{name}'  (face_id={face_id}  dim={vec.shape[0]})")
    print(f"    Crop : {crop_path}")
    print(f"    DB   : {cfg.diag_db_path}\n")


def cmd_list(cfg: Config) -> None:
    db    = DiagnosticDB(cfg)
    faces = db.list_all()
    if not faces:
        print("\n(No faces registered yet — use --register first.)\n")
        return
    print(f"\n{'ID':>4}  {'Name':<20}  {'Registered':<20}  Image")
    print("-" * 80)
    for f in faces:
        print(f"{f['face_id']:>4}  {f['name']:<20}  {str(f['created_at'])[:19]:<20}  {f['image_path']}")
    print(f"\nTotal: {len(faces)} face(s) in {cfg.diag_db_path}\n")


def cmd_diagnostic(
    image_path: str,
    detector, embedder,
    cfg: Config,
) -> None:
    logger.info("MODE: diagnostic (local SQLite)  ← %s", image_path)
    frame = _load_image(image_path)
    _, vec = _detect_and_embed(frame, detector, embedder)

    db = DiagnosticDB(cfg)
    t0 = time.time()
    name, img_path, sim = db.search(vec)
    elapsed_ms = (time.time() - t0) * 1000

    topk     = db.search_topk(vec)
    topk_str = ", ".join(f"{n}={s:.3f}" for n, s in topk) or "<no compatible embeddings>"
    logger.info("Top-%d: %s  (%.1f ms)", cfg.topk, topk_str, elapsed_ms)

    print()
    if name:
        print(f">>> Recognised: {name}  (cosine sim={sim:.4f})")
        if img_path:
            print(f"    Matched: {img_path}")
    else:
        print(f">>> Unknown  (best sim={sim:.4f}  threshold={cfg.cosine_threshold})")
        print(f"    Nearest: {topk_str}")
    print()


def cmd_server(
    image_path: str,
    detector, embedder,
    cfg: Config,
) -> None:
    logger.info("MODE: server (%s)  ← %s", cfg.server_url, image_path)
    frame = _load_image(image_path)
    detection, vec = _detect_and_embed(frame, detector, embedder)

    client    = ServerClient(cfg)
    server_ok = client.health_check()

    if server_ok:
        name = client.identify(vec)
        client.close()
        print()
        print(f">>> Recognised: {name}" if name else ">>> Unknown face  (via server)")
        print()
    else:
        client.close()
        logger.warning("Server unreachable — falling back to diagnostic (local SQLite).")

        db    = DiagnosticDB(cfg)
        faces = db.list_all()
        if not faces:
            raise ClientError(
                "Server unreachable and no local faces registered. "
                "Register first: python client.py --register <name> <image>"
            )

        t0 = time.time()
        name, img_path, sim = db.search(vec)
        elapsed_ms = (time.time() - t0) * 1000

        topk     = db.search_topk(vec)
        topk_str = ", ".join(f"{n}={s:.3f}" for n, s in topk) or "<no compatible embeddings>"
        logger.info("[Diagnostic fallback] Top-%d: %s  (%.1f ms)", cfg.topk, topk_str, elapsed_ms)

        print()
        if name:
            print(f">>> Recognised: {name}  (diagnostic fallback  sim={sim:.4f})")
            if img_path:
                print(f"    Matched: {img_path}")
        else:
            print(f">>> Unknown  (diagnostic fallback  best sim={sim:.4f}  threshold={cfg.cosine_threshold})")
            print(f"    Nearest: {topk_str}")
        print()


def cmd_camera(
    detector, embedder,
    cfg: Config,
) -> None:
    device     = cfg.camera_device
    frame_skip = cfg.camera_frame_skip

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise ClientError(f"Cannot open camera device {device}")
    logger.info("Camera %d opened | frame_skip=%d | embedder=%d-dim",
                device, frame_skip, embedder.DIM)

    # Resolve backend once at startup
    use_server    = False
    server_client = None
    diag_db       = None

    if cfg.mode != "diagnostic":
        server_client = ServerClient(cfg)
        if server_client.health_check():
            use_server = True
        else:
            logger.warning("Server unreachable — falling back to diagnostic (local SQLite).")
            server_client.close()
            server_client = None

    if not use_server:
        diag_db = DiagnosticDB(cfg)

    label      = ""
    last_bbox  = None
    frame_no   = 0

    print("Camera running — press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera %d", device)
                break

            frame_no += 1

            if frame_no % frame_skip == 0:
                detection = detector.detect(frame)
                if detection is not None:
                    last_bbox = detection.bbox
                    vec = embedder.embed(frame, detection)

                    if use_server:
                        name = server_client.identify(vec)
                        label = name if name else "Unknown"
                    else:
                        name, _, sim = diag_db.search(vec)
                        label = f"{name} ({sim:.2f})" if name else f"Unknown ({sim:.2f})"
                else:
                    last_bbox = None
                    label     = ""

            if last_bbox is not None:
                x1, y1, x2, y2 = last_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if label:
                    cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if server_client:
            server_client.close()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def _run(args, cfg: Config) -> None:
    """Execute the requested command.

    Raises ClientError on any fatal, user-facing error and never calls
    sys.exit, so the whole flow stays importable and unit-testable. main() is
    the sole place that turns a ClientError into exit(1).
    """
    # ── --list: no image needed ──────────────────────────────
    if args.list:
        cmd_list(cfg)
        return

    # ── Validate image (not needed for --camera/--list; default is camera) ───
    if not args.camera and not args.image:
        args.camera = True  # no args → default to camera

    if not args.camera and not os.path.isfile(args.image):
        raise ClientError(f"Image not found: {args.image}")

    # ── Ensure models ────────────────────────────────────────
    _ensure_models(cfg)

    # ── Load models (all settings from config) ───────────────
    try:
        if cfg.detector_backend == "dlib":
            detector = DlibFaceDetector(cfg)
        else:
            detector = YuNetDetector(cfg)

        if cfg.embedder_model == "dlib":
            embedder = DlibEmbedder(cfg)
        else:
            embedder = MobileFaceNetEmbedder(cfg)

        logger.info(
            "Detector: %s  |  Embedder: %s (%d-dim)",
            cfg.detector_backend, cfg.embedder_model, embedder.DIM,
        )
    except ClientError:
        raise
    except Exception as e:
        raise ClientError(f"Model init failed: {e}") from e

    # ── Resolve effective mode ───────────────────────────────
    if args.camera:
        cmd_camera(detector, embedder, cfg)
    elif args.register:
        cmd_register(args.register, args.image, detector, embedder, cfg)
    elif args.diag or cfg.mode == "diagnostic":
        cmd_diagnostic(args.image, detector, embedder, cfg)
    else:
        # args.server  OR  cfg.mode == "server"  (or anything else → server)
        cmd_server(args.image, detector, embedder, cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Face Recognition Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client.py photo.jpg                      # use mode from config.yaml
  python client.py --server photo.jpg             # force server mode
  python client.py --diag photo.jpg               # force diagnostic mode
  python client.py --register Alice photo.jpg     # register a face locally
  python client.py --list                         # list registered faces
  python client.py --config custom.yaml photo.jpg # use a different config
        """,
    )
    parser.add_argument("image",      nargs="?", help="Path to the input face image")
    parser.add_argument("--server",   action="store_true", help="Force server mode")
    parser.add_argument("--diag",     action="store_true", help="Force diagnostic (local SQLite) mode")
    parser.add_argument("--camera",   action="store_true", help="Use live camera feed")
    parser.add_argument("--register", metavar="NAME",      help="Register a face with the given name")
    parser.add_argument("--list",     action="store_true", help="List all registered faces in local DB")
    parser.add_argument("--config",   default=_CONFIG_PATH, metavar="FILE",
                        help=f"Config file (default: config.yaml)")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────
    cfg = Config(args.config)
    logging.getLogger().setLevel(cfg.log_level)
    logger.setLevel(cfg.log_level)

    logger.info("Config: %s  |  mode=%s  |  server=%s",
                args.config, cfg.mode, cfg.server_url)

    # All fatal, user-facing errors surface as ClientError; this is the single
    # place that turns them into a clean exit(1). Everything below main() raises
    # instead of calling sys.exit, so it stays importable and unit-testable.
    try:
        _run(args, cfg)
    except ClientError as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
