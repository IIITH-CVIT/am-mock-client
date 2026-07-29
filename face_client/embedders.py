import logging 
import os 
from typing import List, Optional, Tuple 

import cv2 
import numpy as np 
import onnxruntime as ort 

from .config import Config 
from .detection import FaceDetection

logger = logging.getLogger("face_client")

class SFaceEmbedder: 
    """128-dim L2-normalised embeddings via cv2.FaceRecognizerSF (native OpenCV 5).

      alignCrop() on the raw detector row, then feature() for the embedding.
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

      5-point landmark alignment into the 112x112 ArcFace pose (falls back to a
      margin crop + resize if the warp fails), BGR->RGB, (x-127.5)/128.
      """

    DIM = 512
    INPUT_SIZE = 112
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
        self._session = ort.InferenceSession(model_path, providers = ["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        out_shape = self._session.get_outputs()[0].shape 
        self.DIM = out_shape[-1] if out_shape else 512
        logger.info("AuraFaceEmbedder (%d-dim, L2, R100) | %s", self.DIM, model_path)
    
    def embed(self, frame_bgr: np.ndarray, detection: FaceDetection) -> np.ndarray:
        face_img = self._crop_and_align(frame_bgr, detection)   
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[None, ...]
        vec = self._session.run(None, {self._input_name: blob})[0].flatten().astype(np.float32)
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
        crop = frame_bgr[max(0, y1 - my):min(h, y2 + my),max(0, x1 - mx):min(w, x2 + mx)]
        if crop.size == 0:
            crop = frame_bgr
        return cv2.resize(crop, (self.INPUT_SIZE, self.INPUT_SIZE))
    
    def _align(self, frame_bgr: np.ndarray, landmarks: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        lm = np.array(landmarks, dtype = np.float32)
        transform, _ = cv2.estimateAffinePartial2D(lm, self._REF_LANDMARKS, method = cv2.LMEDS)
        if transform is None: 
            return None 
        return cv2.warpAffine(
            frame_bgr, transform, (self.INPUT_SIZE, self.INPUT_SIZE),
            flags = cv2.INTER_LINEAR, borderValue = 0,
        )
    
# Registry: which embedder class + model-path property goes with which name
_REGISTRY = {
    "sface": {
        "label": "SFace embedder",
        "path_attr": "face_recognizer_path",
        "cls": SFaceEmbedder
    },
    "auraface": {
        "label": "AuraFace R100 embedder",
        "path_attr": "face_recognizer_auraface_path",
        "cls": AuraFaceEmbedder
    },
}

def get_embedder_spec(cfg:Config) -> dict:
    """Look up the registry entry for cfg.embedder_model (defaults to sface)."""
    return _REGISTRY.get(cfg.embedder_model, _REGISTRY["sface"])

def create_embedder(cfg: Config):
    """Instantiate the configured embedder (sface or auraface)."""
    return get_embedder_spec(cfg)["cls"](cfg)
