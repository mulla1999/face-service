"""
Identity store for real-time 1:N identification.

Enrolled people -> 512-d embeddings kept in a FAISS inner-product index
(inner product == cosine similarity since vectors are L2-normalized).
Embeddings + metadata persist in SQLite so the service survives restarts.

Scale notes:
  - Up to ~1M identities this exact-search FAISS index answers in ~1-5 ms.
  - Beyond that, swap FlatIP for IVF/HNSW, or use Qdrant / pgvector /
    Milvus as an external vector DB. The interface below stays the same.
"""
from __future__ import annotations

import io
import sqlite3
import threading
import uuid

import faiss
import numpy as np

EMB_DIM = 512


class IdentityStore:
    def __init__(self, db_path: str = "identities.db"):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS identities (
                   id        TEXT PRIMARY KEY,
                   name      TEXT NOT NULL,
                   embedding BLOB NOT NULL,
                   created   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
               )"""
        )
        self._db.commit()
        self._index = faiss.IndexFlatIP(EMB_DIM)
        self._ids: list[str] = []           # index row -> identity id
        self._names: dict[str, str] = {}    # identity id -> name
        self._load()

    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """(Re)build the in-memory index from SQLite. Caller holds the lock, or is __init__."""
        self._index = faiss.IndexFlatIP(EMB_DIM)
        self._ids = []
        self._names = {}
        rows = self._db.execute("SELECT id, name, embedding FROM identities ORDER BY rowid").fetchall()
        if not rows:
            return
        vectors = []
        for iid, name, blob in rows:
            vectors.append(np.load(io.BytesIO(blob), allow_pickle=False))
            self._ids.append(iid)
            self._names[iid] = name
        self._index.add(np.vstack(vectors).astype(np.float32))

    @staticmethod
    def _to_blob(vec: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.save(buf, vec.astype(np.float32), allow_pickle=False)
        return buf.getvalue()

    @staticmethod
    def _row(embedding: np.ndarray) -> np.ndarray:
        vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != EMB_DIM:
            raise ValueError(f"embedding must have {EMB_DIM} dimensions, got {vec.shape[1]}")
        return vec

    # ------------------------------------------------------------------ #

    def enroll(self, name: str, embedding: np.ndarray) -> str:
        """Add one sample. Enroll 2-3 photos per person for robustness:
        each photo becomes its own vector under the same name."""
        vec = self._row(embedding)
        iid = str(uuid.uuid4())
        with self._lock:
            self._db.execute(
                "INSERT INTO identities (id, name, embedding) VALUES (?, ?, ?)",
                (iid, name, self._to_blob(vec[0])),
            )
            self._db.commit()
            self._index.add(vec)
            self._ids.append(iid)
            self._names[iid] = name
        return iid

    def search(self, embedding: np.ndarray, top_k: int = 3) -> list[tuple[str, str, float]]:
        """Return [(name, identity_id, cosine_similarity)] best matches, best first."""
        vec = self._row(embedding)
        with self._lock:  # read results under the lock so a concurrent remove() can't shift rows
            if self._index.ntotal == 0:
                return []
            sims, idxs = self._index.search(vec, min(top_k, self._index.ntotal))
            return [
                (self._names[self._ids[idx]], self._ids[idx], float(sim))
                for sim, idx in zip(sims[0], idxs[0], strict=True)
                if idx != -1
            ]

    def remove(self, identity_id: str) -> bool:
        """Remove one sample (rebuilds the index: fine for <100k entries)."""
        with self._lock:
            cur = self._db.execute("DELETE FROM identities WHERE id = ?", (identity_id,))
            self._db.commit()
            if cur.rowcount == 0:
                return False
            self._load()
        return True

    def list_people(self) -> dict[str, list[str]]:
        """name -> list of enrolled sample ids."""
        with self._lock:
            out: dict[str, list[str]] = {}
            for iid, name in self._names.items():
                out.setdefault(name, []).append(iid)
            return out

    def count(self) -> int:
        with self._lock:
            return len(self._ids)

    def close(self) -> None:
        with self._lock:
            self._db.close()
