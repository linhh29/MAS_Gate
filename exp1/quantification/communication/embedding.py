"""Embedding store with LRU cache via quantification client."""

from __future__ import annotations

import numpy as np

from quantification.clients.factory import QuantificationClient


class EmbeddingStore:
    """Batch embedding with per-text caching."""

    def __init__(self, client: QuantificationClient, cache_size: int = 4096) -> None:
        self.client = client
        self._cache_size = cache_size
        self._cache: dict[str, np.ndarray] = {}

    def _remember(self, text: str, vector: np.ndarray) -> np.ndarray:
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = vector
        return vector

    def embed(self, text: str) -> np.ndarray:
        key = text.strip()
        if not key:
            return np.zeros((0,), dtype=np.float32)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        vectors = self.client.embed_texts([key])
        return self._remember(key, vectors[0])

    def embed_many(self, texts: list[str]) -> np.ndarray:
        cleaned = [t.strip() for t in texts if t and t.strip()]
        if not cleaned:
            return np.zeros((0, 0), dtype=np.float32)

        rows: list[np.ndarray | None] = [None] * len(cleaned)
        misses: list[str] = []
        miss_positions: list[int] = []

        for i, text in enumerate(cleaned):
            cached = self._cache.get(text)
            if cached is not None:
                rows[i] = cached
            else:
                misses.append(text)
                miss_positions.append(i)

        if misses:
            unique_misses = list(dict.fromkeys(misses))
            batch_vectors = self.client.embed_texts(unique_misses)
            for text, vector in zip(unique_misses, batch_vectors, strict=True):
                self._remember(text, vector)
            for i, text in zip(miss_positions, misses, strict=True):
                rows[i] = self._cache[text]

        return np.stack([rows[i] for i in range(len(cleaned))], axis=0)
