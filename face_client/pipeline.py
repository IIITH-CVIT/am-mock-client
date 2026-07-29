import logging 
import os 
import time     
from pathlib import Path
from typing import List, Dict, Tuple, Optional 

import cv2
import numpy as np

from .config import Config 
from .errors import ClientError 
from .detection import FaceDetection, FaceDetector
from .embedders import create_embedder, get_embedder_spec
from .diagnostic_db import DiagnosticDB
from .server_client import ServerClient

logger = logging.getLogger("face_client")

# Model presence check  

def _raise_missing(missing: list, remediation: str) -> None:
    lines = [f"Missing model: {label} ({path})" for label, path in missing]
    lines.append(remediation)
    raise ClientError("\n".join(lines))

def _ensure_models(cfg: Config) -> None:
    spec = get_embedder_spec(cfg)
    embedder_path = getattr(cfg, spec["path_attr"])
    required = [
        ("YuNet face detector", cfg.face_detector_path),
        (spec["label"], embedder_path),
    ]
    missing = [(label, path) for label, path in required if not os.path.exists(path)]
    if missing:
        _raise_missing(
            missing,
            f"Copy the missing .onnx file(s) into {os.path.dirname(missing[0][1])} "
            f"(same files as mock-server/models/).",
        )

# Shared helpers 

def _load_image(path: str) -> np.ndarray:
    frame = cv2.imread(path)
    if frame is None:
        raise ClientError(f"Failed to load image: {path}")
    logger.info("Image: %dx%d <- %s", frame.shape[1], frame.shape[0], path)
    return frame

def _save_face_crop(frame: np.ndarray, bbox: Tuple, name: str, faces_dir: str) -> str:
    os.makedirs(faces_dir, exist_ok = True)
    safe = "".join(c if c.isalnum() else "_" for c in name).lower()
    path = os.path.join(faces_dir, f"{safe}_{int(time.time())}.jpg")
    x1, y1, x2, y2 = bbox
    ih, iw = frame.shape[:2]
    cv2.imwrite(path, frame[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)])
    return path

def _detect_and_embed(frame: np.ndarray, detector, embedder,) -> Tuple[FaceDetection, np.ndarray]:
    detection = detector.detect(frame)
    if detection is None:
        raise ClientError("No face detected in the image")
    x1, y1, x2, y2 = detection.bbox
    logger.info("Face: bbox = (%d, %d, %d, %d) score = %.3f", x1, y1, x2, y2, detection.score)
    vec = embedder.embed(frame, detection)
    logger.info("Embedding: %d-dim", vec.shape[0])
    return detection, vec 

# High level client

class FaceRecognitionClient:
    """
    High-level face recognition client.

    Usage:
        client = FaceRecognitionClient()
        client.identify("photo.jpg")
    """

    def __init__(self, config_path: Optional[str] = None):
        self.cfg = Config(config_path) if config_path else Config()
        self._detector = None
        self._embedder = None
    
    def _ensure_backend(self):
        """Lazily build the detector + embedder (list_faces() never needs them)."""

        if self._detector is None or self._embedder is None:
            _ensure_models(self.cfg)
            try:
                self._detector = FaceDetector(self.cfg)
                self._embedder = create_embedder(self.cfg)
                logger.info(
                    "Detector: YuNet  |  Embedder: %s (%d-dim)",
                    self.cfg.embedder_model, self._embedder.DIM,
                )
            except ClientError:
                raise 
            except Exception as e:
                raise ClientError(f"Model init failed: {e}") from e

        return self._detector, self._embedder
    
    def identify(self, image_path: str, mode: Optional[str] = None) -> Optional[str]:
        """Identify a face in image_path. mode overrides cfg.mode ('server'/'diagnostic')."""
        
        if not os.path.isfile(image_path):
            raise ClientError(f"Image not found: {image_path}")
        
        detector, embedder = self._ensure_backend()
        effective_mode = (mode or self.cfg.mode).lower()

        frame = _load_image(image_path)
        _, vec = _detect_and_embed(frame, detector, embedder)

        if effective_mode == "diagnostic":
            return self._identify_diagnostic(vec)
        
        return self._identify_server(vec)
    
    def _identify_server(self, vec:np.ndarray) -> Optional[str]:
        logger.info("MODE: server (%s)", self.cfg.server_url)
        client = ServerClient(self.cfg)
        server_ok = client.health_check()

        if server_ok:
            name = client.identify(vec)
            client.close()
            print()
            print(f">>> Recognised: {name}" if name else ">>> Unknown face  (via server)")
            print()
            return name 
        
        client.close()
        logger.warning("Server unreachable — falling back to diagnostic (local SQLite).")
        
        return self._identify_diagnostic(vec, fallback = True)
    
    def _identify_diagnostic(self, vec: np.ndarray, fallback: bool = False) -> Optional[str]:
        cfg = self.cfg
        db = DiagnosticDB(cfg)

        if fallback and not db.list_all():
            raise ClientError(
                "Server unreachable and no local faces registered. "
                "Register first: client.register(name, image_path)"
            )
        
        t0 = time.time()
        name, img_path, sim = db.search(vec)
        elapsed_ms = (time.time() - t0) * 1000

        topk = db.search_topk(vec)
        topk_str = ", ".join(f"{n}={s:.3f}" for n, s in topk) or "<no compatible embeddings>"
        tag = "[Diagnostic fallback] " if fallback else ""
        logger.info("%sTop-%d: %s  (%.1f ms)", tag, cfg.topk, topk_str, elapsed_ms)

        fb = "diagnostic fallback  " if fallback else ""
        print()
        if name:
            print(f">>> Recognised: {name}  ({fb}sim={sim:.4f})" if fallback else f">>> Recognised: {name}  (cosine sim={sim:.4f})")
            if img_path:
                print(f"Matched: {img_path}")
        else:
            print(f">>> Unknown ({fb}best sim={sim:.4f} threshold={cfg.cosine_threshold})")
            print(f"Nearest: {topk_str}")
        print()
        return name 
    
    def register(self, name: str, image_path: str) -> int:
        detector, embedder = self._ensure_backend()
        logger.info("REGISTER '%s' <- %s", name, image_path)
        frame = _load_image(image_path)
        detection, vec = _detect_and_embed(frame, detector, embedder)
        crop_path = _save_face_crop(frame, detection.bbox, name, self.cfg.diag_faces_dir)
        logger.info("Face crop saved: %s", crop_path)
        db = DiagnosticDB(self.cfg)
        face_id = db.register(name, vec, crop_path)
        print(f"\n>>> Registered '{name}' (face_id={face_id} dim={vec.shape[0]})")
        print(f"Crop: {crop_path}")
        print(f"DB: {self.cfg.diag_db_path}\n")

        return face_id

    def list_faces(self) -> List[Dict]:
        db = DiagnosticDB(self.cfg)
        faces = db.list_all()
        if not faces:
            print("\n(No faces registered yet — use register() first.)\n")
            return faces
        print(f"\n{'ID':>4}  {'Name':<20}  {'Registered':<20}  Image")
        print("-" * 80)
        for f in faces:
            print(f"{f['face_id']:>4}  {f['name']:<20}  {str(f['created_at'])[:19]:<20}  {f['image_path']}")
        print(f"\nTotal: {len(faces)} face(s) in {self.cfg.diag_db_path}\n")

        return faces

    def delete_face(self, face_id: int) -> bool:
        db = DiagnosticDB(self.cfg)
        deleted = db.delete(face_id)
        if deleted:
            print(f"\n>>> Deleted face_id={face_id}\n")
        else:
            print(f"\n>>> No face found with face_id={face_id}\n")
        return deleted

    def run_camera(self) -> None:
        detector, embedder = self._ensure_backend()
        cfg = self.cfg
        device = cfg.camera_device
        frame_skip = cfg.camera_frame_skip

        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise ClientError(f"Cannot open camera device {device}")
        logger.info("Camera %d opened | frame_skip=%d | embedder=%d-dim", device, frame_skip, embedder.DIM)

        use_server = False
        server_client = None
        diag_db = None

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

        label = ""
        last_bbox = None
        frame_no = 0

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
                        label = ""

                if last_bbox is not None:
                    x1, y1, x2, y2 = last_bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if label:
                        cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if server_client:
                server_client.close()
