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

Detection + embedding backends (config.yaml: embedder.model) mirror
am-mock-server's FaceEngine (app/core/face_engine.py) so embeddings computed
here match what the server has enrolled. One detector, two embedders — never
mixed, pick one:
  - Detection: cv2.FaceDetectorYN (YuNet, native OpenCV 5) — bbox + 5-point
               landmarks, used by both embedders below.
  - sface     — default; cv2.FaceRecognizerSF (native OpenCV 5, no
                onnxruntime), 128-dim L2-normalised, aligned via alignCrop().
  - auraface  — alternate; aurar100.onnx (ArcFace-style R100, via
                onnxruntime), 512-dim L2-normalised, aligned via the 112x112
                ArcFace reference landmark warp.

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

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

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
# Verify the model weights required by the configured embedder are
# present, and fail fast with a clear message if not. All weights are
# ONNX files bundled in ./models/ (same files as mock-server/models/
# — no auto-download).
# ─────────────────────────────────────────────────────────────

def _raise_missing(missing: list, remediation: str) -> None:
    lines = [f"Missing model: {label}  ({path})" for label, path in missing]
    lines.append(remediation)
    raise ClientError("\n".join(lines))


def _ensure_models(cfg: Config) -> None:
    required = [("YuNet face detector", cfg.face_detector_path)]
    if cfg.embedder_model == "auraface":
        required.append(("AuraFace R100 embedder", cfg.face_recognizer_auraface_path))
    else:
        required.append(("SFace embedder", cfg.face_recognizer_path))

    missing = [(label, path) for label, path in required if not os.path.exists(path)]
    if missing:
        _raise_missing(
            missing,
            f"Copy the missing .onnx file(s) into {os.path.dirname(missing[0][1])} "
            f"(same files as mock-server/models/).",
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
        raw: Optional[np.ndarray] = None,
    ):
        self.bbox      = bbox        # (x1, y1, x2, y2)
        self.score     = score
        self.landmarks = landmarks   # 5-point (x, y) list
        self.raw       = raw         # native cv2.FaceDetectorYN row — needed by SFace's alignCrop()


# ─────────────────────────────────────────────────────────────
# Face detection — YuNet, native OpenCV 5 (cv2.FaceDetectorYN)
# Mirrors mock-server FaceEngine._detect_face: no manual ONNX decode
# needed any more, OpenCV 5 ships YuNet inference built in. Returns
# bbox + 5-point landmarks (+ the raw detector row, for SFace).
# ─────────────────────────────────────────────────────────────

class FaceDetector:
    def __init__(self, cfg: Config):
        model_path = cfg.face_detector_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YuNet model not found: {model_path}")
        size = cfg.face_detector_input_size
        self._detector = cv2.FaceDetectorYN.create(
            model_path, "", (size, size),
            score_threshold=cfg.detection_threshold,
        )
        logger.info(
            "FaceDetector (YuNet, OpenCV 5 native) | input=%dx%d  threshold=%.2f",
            size, size, cfg.detection_threshold,
        )

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceDetection]:
        if frame_bgr is None or frame_bgr.ndim != 3:
            return None
        h, w = frame_bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None

        face_row = faces[int(np.argmax(faces[:, 14]))]
        x, y, bw, bh = face_row[0:4]
        bbox = (
            max(0, int(x)), max(0, int(y)),
            min(w, int(x + bw)), min(h, int(y + bh)),
        )
        landmarks = [(float(face_row[4 + 2 * k]), float(face_row[5 + 2 * k])) for k in range(5)]
        return FaceDetection(bbox=bbox, score=float(face_row[14]), landmarks=landmarks, raw=face_row)


# ─────────────────────────────────────────────────────────────
# Embedders
# ─────────────────────────────────────────────────────────────

class SFaceEmbedder:
    """128-dim L2-normalised embeddings via cv2.FaceRecognizerSF (native OpenCV 5).

    Mirrors mock-server FaceEngine.embed (sface path): alignCrop() on the raw
    detector row, then feature() for the embedding.
    """

    DIM = 128

    def __init__(self, cfg: Config):
        model_path = cfg.face_recognizer_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SFace model not found: {model_path}")
        self._recognizer = cv2.FaceRecognizerSF.create(model_path, "")
        logger.info("SFaceEmbedder (128-dim, L2, native OpenCV 5) | %s", model_path)

    def embed(self, frame_bgr: np.ndarray, detection: FaceDetection) -> np.ndarray:
        aligned = self._recognizer.alignCrop(frame_bgr, detection.raw)
        vec = self._recognizer.feature(aligned).flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class AuraFaceEmbedder:
    """
    512-dim L2-normalised embeddings via ONNX AuraFace (ArcFace-style R100).

    Mirrors mock-server FaceEngine.embed (auraface path): 5-point landmark
    alignment into the 112x112 ArcFace pose (falls back to a margin crop +
    resize if the warp fails), BGR->RGB, (x-127.5)/128.
    """

    DIM         = 512
    INPUT_SIZE  = 112
    CROP_MARGIN = 0.20

    # ArcFace/AuraFace 112x112 reference landmarks
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
        model_path = cfg.face_recognizer_auraface_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"AuraFace model not found: {model_path}")
        self._session    = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        out_shape        = self._session.get_outputs()[0].shape
        self.DIM         = out_shape[-1] if out_shape else 512
        logger.info("AuraFaceEmbedder (%d-dim, L2, R100) | %s", self.DIM, model_path)

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
        # oversight. Embeddings are L2-normalised, so components sit well
        # inside [-1, 1]. Measured over 2000 random 512-d unit vectors, %.6f
        # rounding gives at most ~5e-7 per component and ~7e-6 whole-vector L2
        # error, shifting the server's L2 match distance by <=1.5e-6, which is
        # six orders of magnitude below the 0.8 match threshold, and it never
        # changed a nearest-neighbour pick. Full float32 round-trip would need
        # ~9 significant figures; it buys nothing for matching and only
        # enlarges the payload.
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

    # NOTE: the server enrols vectors from exactly one embedder at a time
    # (models.embedder_model in its config.yaml), so a 400 "dimensions" error
    # here means this client's embedder.model doesn't match the server's —
    # e.g. this client sent sface (128-dim) but the server is enrolled with
    # auraface (512-dim), or vice versa.
    def identify(self, embedding: np.ndarray) -> Optional[str]:
        result = self._post(embedding)
        if "error" in result:
            detail = result["error"]
            if result.get("status_code") == 400 and "dimensions" in str(detail):
                logger.error(
                "The server rejected the vector dimension (sent %d-dim). Make sure "
                "this client's embedder.model matches the server's models.embedder_model: "
                "sface (128-dim) or auraface (512-dim). The config.yaml / "
                "config.auraface.yaml presets are already set up for each.",
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
        detector = FaceDetector(cfg)

        if cfg.embedder_model == "auraface":
            embedder = AuraFaceEmbedder(cfg)
        else:
            embedder = SFaceEmbedder(cfg)

        logger.info(
            "Detector: YuNet  |  Embedder: %s (%d-dim)",
            cfg.embedder_model, embedder.DIM,
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
