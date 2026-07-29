import logging 
import os 
from typing import List, Optional, Tuple 

import cv2 
import numpy as np

from .config import Config 

logger = logging.getLogger("face_client")

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
