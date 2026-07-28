import logging 
import os
import sqlite3
from typing import Dict, List, Tuple, Optional 

import numpy as np

from .config import Config 

logger = logging.getLogger("face_client")

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

    def delete(self, face_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT image_path FROM diagnostic_faces WHERE face_id = ?", (face_id,)
            ).fetchone()
            if row is None:
                return False
            conn.execute("DELETE FROM diagnostic_faces WHERE face_id = ?", (face_id,))
        image_path = row["image_path"]
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        logger.info("Deleted face_id=%d (%s)", face_id, image_path)
        return True

