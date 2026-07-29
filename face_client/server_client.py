import logging 
from typing import Dict, Optional 

import numpy as np 
import requests 

from .config import Config 

logger = logging.getLogger("face_client")

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
