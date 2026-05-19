"""On-disk store for DINOv2 image embeddings.

Embeddings are kept as a single ``embeddings.npz`` file with two arrays:

* ``ids``     – ``int64[N]``     image IDs (matching ``images.id`` in SQLite)
* ``vectors`` – ``float32[N, D]`` L2-normalised feature vectors

For ~17k images at 384 dims this is ~26 MB on disk and loads in well under
100 ms, which is fast enough that we don't need FAISS for this scale.

Concurrency model: a single ``EmbeddingStore`` instance is intended to be
shared. ``append`` is buffered in memory and an explicit ``save()`` is
required to flush. Reads (``has``, ``load_all``) are cheap and lock-protected.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from app.core.config import EMBEDDING_DIM, EMBEDDINGS_PATH, ensure_data_dirs

logger = logging.getLogger(__name__)


class EmbeddingStore:
    def __init__(
        self,
        path: Path = EMBEDDINGS_PATH,
        dim: int = EMBEDDING_DIM,
    ) -> None:
        ensure_data_dirs()
        self.path = path
        self.dim = dim
        self._lock = threading.RLock()
        self._ids: np.ndarray = np.empty((0,), dtype=np.int64)
        self._vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self._id_to_index: dict[int, int] = {}
        self._dirty = False
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self.path.exists():
            logger.info("EmbeddingStore: no existing file at %s, starting empty", self.path)
            return
        try:
            data = np.load(self.path)
            ids = np.asarray(data["ids"], dtype=np.int64)
            vectors = np.asarray(data["vectors"], dtype=np.float32)
        except (KeyError, ValueError, OSError) as error:
            logger.warning(
                "EmbeddingStore: failed to load %s (%s); starting fresh",
                self.path,
                error,
            )
            return

        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            logger.warning(
                "EmbeddingStore: dimension mismatch on disk (%s vs expected %d), discarding",
                vectors.shape,
                self.dim,
            )
            return

        self._ids = ids
        self._vectors = vectors
        self._id_to_index = {int(image_id): idx for idx, image_id in enumerate(ids)}
        logger.info(
            "EmbeddingStore: loaded %d embeddings (dim=%d) from %s",
            len(ids),
            self.dim,
            self.path,
        )

    def __len__(self) -> int:
        with self._lock:
            return int(self._ids.shape[0])

    def has(self, image_id: int) -> bool:
        with self._lock:
            return int(image_id) in self._id_to_index

    def get(self, image_id: int) -> np.ndarray | None:
        with self._lock:
            idx = self._id_to_index.get(int(image_id))
            if idx is None:
                return None
            return self._vectors[idx].copy()

    def append(self, image_id: int, vector: np.ndarray) -> None:
        """Insert or update one embedding.

        Vector must be 1-D float32 with ``dim`` entries. We assume callers
        have already L2-normalised it; we do not re-normalise here.
        """
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(
                f"Expected vector of dim {self.dim}, got shape {vec.shape}"
            )

        with self._lock:
            idx = self._id_to_index.get(int(image_id))
            if idx is not None:
                self._vectors[idx] = vec
            else:
                self._ids = np.concatenate(
                    [self._ids, np.array([image_id], dtype=np.int64)]
                )
                self._vectors = np.concatenate(
                    [self._vectors, vec[np.newaxis, :]], axis=0
                )
                self._id_to_index[int(image_id)] = self._vectors.shape[0] - 1
            self._dirty = True

    def extend(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        """Bulk insert. Faster than calling ``append`` in a loop."""
        ids_arr = np.asarray(ids, dtype=np.int64).reshape(-1)
        vec_arr = np.asarray(vectors, dtype=np.float32)
        if vec_arr.ndim != 2 or vec_arr.shape[1] != self.dim:
            raise ValueError(
                f"Expected vectors of shape (N, {self.dim}), got {vec_arr.shape}"
            )
        if ids_arr.shape[0] != vec_arr.shape[0]:
            raise ValueError("ids and vectors must have matching length")

        with self._lock:
            for image_id, vec in zip(ids_arr, vec_arr):
                idx = self._id_to_index.get(int(image_id))
                if idx is not None:
                    self._vectors[idx] = vec
                else:
                    new_index = self._vectors.shape[0]
                    self._ids = np.concatenate(
                        [self._ids, np.array([image_id], dtype=np.int64)]
                    )
                    self._vectors = np.concatenate(
                        [self._vectors, vec[np.newaxis, :]], axis=0
                    )
                    self._id_to_index[int(image_id)] = new_index
            self._dirty = True

    def save(self) -> None:
        """Atomically flush the in-memory buffer to disk."""
        with self._lock:
            if not self._dirty and self.path.exists():
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # np.savez auto-appends .npz to string paths, so we keep the
            # extension here and reset to the original name on rename.
            tmp_path = self.path.with_name(self.path.stem + ".tmp" + self.path.suffix)
            np.savez(str(tmp_path), ids=self._ids, vectors=self._vectors)
            tmp_path.replace(self.path)
            self._dirty = False
            logger.info(
                "EmbeddingStore: saved %d embeddings to %s",
                self._ids.shape[0],
                self.path,
            )

    def load_all(self) -> tuple[np.ndarray, np.ndarray]:
        """Return references to ids and vectors as numpy arrays.

        Cheap (returns the live views). Callers must not mutate the returned
        arrays — treat them as read-only.
        """
        with self._lock:
            return self._ids, self._vectors

    def known_ids(self) -> set[int]:
        with self._lock:
            return set(int(i) for i in self._ids)
