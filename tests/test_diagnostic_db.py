"""Tests for DiagnosticDB.search / search_topk.

These lock the behaviour of the local cosine-similarity store after the
_scored_candidates() dedup refactor (finding #5): a real match above threshold,
top-K ordering, the dimension-mismatch gate, and the empty-DB path. They run
against a throwaway SQLite file in a tmp dir — no server or camera needed.
"""

import numpy as np
import pytest

from client import Config, DiagnosticDB


def _db(tmp_path, threshold=0.5, topk=3):
    cfg = Config.__new__(Config)
    cfg._data = {
        "diagnostic": {
            "db_path": str(tmp_path / "diag.db"),
            "faces_dir": str(tmp_path / "faces"),
            "cosine_threshold": threshold,
            "topk": topk,
        }
    }
    return DiagnosticDB(cfg)


def test_search_returns_best_match_above_threshold(tmp_path):
    db = _db(tmp_path)
    db.register("Alice", np.array([1, 0, 0], dtype=np.float32), "alice.jpg")
    db.register("Bob", np.array([0, 1, 0], dtype=np.float32), "bob.jpg")

    name, path, sim = db.search(np.array([0.9, 0.1, 0.0], dtype=np.float32))
    assert name == "Alice"
    assert path == "alice.jpg"
    assert sim > 0.9


def test_search_below_threshold_returns_none_but_reports_sim(tmp_path):
    db = _db(tmp_path, threshold=0.99)
    db.register("Alice", np.array([1, 0, 0], dtype=np.float32), "alice.jpg")

    name, path, sim = db.search(np.array([0.6, 0.8, 0.0], dtype=np.float32))
    assert name is None and path is None
    assert 0.0 < sim < 0.99  # still reports the best similarity it found


def test_search_topk_orders_by_similarity(tmp_path):
    db = _db(tmp_path, topk=3)
    db.register("Alice", np.array([1, 0, 0], dtype=np.float32), "alice.jpg")
    db.register("Bob", np.array([0, 1, 0], dtype=np.float32), "bob.jpg")

    topk = db.search_topk(np.array([0.9, 0.1, 0.0], dtype=np.float32))
    assert [n for n, _ in topk] == ["Alice", "Bob"]  # only 2 registered
    assert topk[0][1] >= topk[1][1]


def test_dimension_mismatch_gates_both_methods(tmp_path):
    db = _db(tmp_path)
    db.register("Alice", np.ones(128, dtype=np.float32), "alice.jpg")  # 128-dim

    name, path, sim = db.search(np.ones(512, dtype=np.float32))  # query 512-dim
    assert (name, path, sim) == (None, None, 0.0)
    assert db.search_topk(np.ones(512, dtype=np.float32)) == []


def test_empty_db(tmp_path):
    db = _db(tmp_path)
    assert db.search(np.ones(3, dtype=np.float32)) == (None, None, 0.0)
    assert db.search_topk(np.ones(3, dtype=np.float32)) == []


def test_dim_mismatch_warns_once_from_search_not_topk(tmp_path):
    """search() logs the 'no stored embeddings with dim' warning; search_topk()
    must stay silent so cmd_diagnostic (search then topk) doesn't double-warn."""
    db = _db(tmp_path)
    db.register("Alice", np.ones(128, dtype=np.float32), "alice.jpg")
    query = np.ones(512, dtype=np.float32)

    with pytest.MonkeyPatch.context() as mp:
        import client as client_mod

        calls = []
        mp.setattr(client_mod.logger, "warning", lambda *a, **k: calls.append(a))
        db.search(query)
        db.search_topk(query)
        assert len(calls) == 1
