"""
memory.py — Temporal visual memory (FAISS)

Keyframe object crops are stored as L2-normalised CLIP embeddings.
Metadata (class, zone, time, frame, global track id, image paths) sits
beside the vectors. Search uses inner product, which equals cosine
similarity for these vectors.

If the faiss package is missing, search falls back to a NumPy dot product
so the rest of the pipeline still runs.
"""

import json
import os
from typing import Optional

import numpy as np

DEFAULT_DIR = "memory"
TEXT_SIM_THRESHOLD = 0.22


def _normalise(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    return vec / (float(np.linalg.norm(vec)) + 1e-8)


class ObservationMemory:
    """Keyframe embedding index for one processed video."""

    def __init__(self, directory: str = DEFAULT_DIR):
        self.directory = directory
        self.meta: list[dict] = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)

    @property
    def _meta_path(self) -> str:
        return os.path.join(self.directory, "meta.json")

    @property
    def _vectors_path(self) -> str:
        return os.path.join(self.directory, "vectors.npy")

    @property
    def _index_path(self) -> str:
        return os.path.join(self.directory, "crops.index")

    def reset(self):
        """Drop the in-memory index and any files from the previous video."""
        self.meta = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        for path in (self._meta_path, self._vectors_path, self._index_path):
            if os.path.exists(path):
                os.remove(path)

    def add(self, embedding: np.ndarray, meta: dict):
        """Append one keyframe crop. ``meta`` is returned unchanged by search."""
        vec = _normalise(embedding)
        if self._vectors.size == 0:
            self._vectors = vec.reshape(1, -1)
        else:
            if vec.shape[0] != self._vectors.shape[1]:
                return
            self._vectors = np.vstack([self._vectors, vec.reshape(1, -1)])
        self.meta.append(dict(meta))

    def save(self):
        """Write vectors, metadata, and a FAISS index when faiss is installed."""
        os.makedirs(self.directory, exist_ok=True)
        np.save(self._vectors_path, self._vectors)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f)
        self._write_faiss()

    def _write_faiss(self):
        if self._vectors.size == 0:
            if os.path.exists(self._index_path):
                os.remove(self._index_path)
            return
        try:
            import faiss
        except ImportError:
            return
        index = faiss.IndexFlatIP(self._vectors.shape[1])
        index.add(np.ascontiguousarray(self._vectors))
        faiss.write_index(index, self._index_path)

    def load(self) -> "ObservationMemory":
        """Load the index saved by the last video run."""
        if os.path.exists(self._meta_path):
            with open(self._meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = []
        if os.path.exists(self._vectors_path):
            self._vectors = np.load(self._vectors_path)
        else:
            self._vectors = np.zeros((0, 0), dtype=np.float32)
        if len(self.meta) != len(self._vectors):
            n = min(len(self.meta), len(self._vectors))
            self.meta = self.meta[:n]
            self._vectors = self._vectors[:n]
        return self

    def search(self, embedding: np.ndarray, k: int = 5) -> list[dict]:
        """Nearest keyframes. Each hit is the stored metadata plus ``score``."""
        if not self.meta or self._vectors.size == 0:
            return []
        query = _normalise(embedding).reshape(1, -1)
        if query.shape[1] != self._vectors.shape[1]:
            return []
        k = min(k, len(self.meta))
        scores, ids = self._search_ids(query, k)
        hits = []
        for score, idx in zip(scores, ids):
            if idx < 0 or idx >= len(self.meta):
                continue
            item = dict(self.meta[idx])
            item["score"] = float(score)
            hits.append(item)
        return hits

    def _search_ids(self, query: np.ndarray, k: int):
        try:
            import faiss
            index = faiss.IndexFlatIP(self._vectors.shape[1])
            index.add(np.ascontiguousarray(self._vectors))
            scores, ids = index.search(np.ascontiguousarray(query), k)
            return scores[0].tolist(), ids[0].astype(int).tolist()
        except ImportError:
            sims = (self._vectors @ query.reshape(-1)).astype(np.float32)
            order = np.argsort(-sims)[:k]
            return sims[order].tolist(), order.astype(int).tolist()


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


def best_text_hit(hits: list[dict], threshold: float = TEXT_SIM_THRESHOLD) -> Optional[dict]:
    """Latest keyframe among hits that clear the text-to-image threshold."""
    kept = [h for h in hits if h.get("score", 0) >= threshold]
    if not kept:
        return None
    return max(kept, key=lambda h: h.get("frame_number") or 0)


if __name__ == "__main__":
    import tempfile

    folder = tempfile.mkdtemp()
    mem = ObservationMemory(folder)
    bottle = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    laptop = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    mem.add(bottle, {
        "object_class": "bottle", "zone": "Desk", "timestamp": "t1",
        "frame_number": 30, "global_track_id": 1,
        "crop_path": "crops/a.jpg", "frame_path": "frames/a.jpg",
        "confidence": 0.9,
    })
    mem.add(laptop, {
        "object_class": "laptop", "zone": "Bed", "timestamp": "t2",
        "frame_number": 90, "global_track_id": 2,
        "crop_path": "crops/b.jpg", "frame_path": "frames/b.jpg",
        "confidence": 0.8,
    })
    mem.save()

    loaded = ObservationMemory(folder).load()
    hit = best_text_hit(loaded.search(bottle, k=2), threshold=0.5)
    assert hit is not None and hit["object_class"] == "bottle", hit
    assert hit["zone"] == "Desk"
    miss = best_text_hit(loaded.search(np.array([0.0, 0.0, 1.0]), k=2), threshold=0.5)
    assert miss is None
    print(f"memory search ok ({'faiss' if _faiss_available() else 'numpy'}): {hit['object_class']} @ {hit['zone']}")
