"""Semantic retrieval over approved long-term memory.

Memory used to be found by FTS5 keyword match alone, which means a memory
written as "I prefer terse commit messages" never surfaces for "how should you
word my commits". This module gives memory the pipeline the corpus already has —
embed, cosine recall, rerank — while keeping the keyword path as the answer to
every degraded case.

Two properties are deliberate:

* Consent is a separate record from corpus consent. Memories are the user's own
  words about themselves, so opting a code directory into cloud embedding must
  never silently opt these in as well.
* Every failure degrades to FTS rather than raising. Retrieval that returns
  slightly worse memories is a much better outcome than a turn that dies because
  a region was busy.
"""
from __future__ import annotations

import asyncio

from .config import Settings
from .database import Database
from .embeddings import CohereRetrieval


class MemoryIndex:
    """Embeds approved memories and retrieves them by meaning."""

    def __init__(
        self, settings: Settings, database: Database, retrieval: CohereRetrieval
    ) -> None:
        self._settings = settings
        self._db = database
        self._retrieval = retrieval
        self._sync_lock = asyncio.Lock()

    async def consent(self) -> tuple[bool, str | None]:
        return await self._db.get_memory_consent()

    async def set_consent(
        self, consent: bool, reason: str | None
    ) -> tuple[bool, str | None]:
        """Grant or withdraw cloud embedding for memory.

        Granting backfills every existing memory, because a consent that only
        applied to memories created afterwards would leave the user with a
        half-semantic index and no way to tell which half they were getting.
        """
        result = await self._db.set_memory_consent(consent, reason)
        if consent:
            await self.sync()
        return result

    async def available(self) -> bool:
        granted, _ = await self.consent()
        return granted and self._retrieval.available()

    async def stats(self) -> dict[str, object]:
        granted, reason = await self.consent()
        counts = await self._db.memory_vector_stats()
        return {
            "consent": granted,
            "consent_reason": reason,
            "cloud_available": self._retrieval.available(),
            "semantic": granted and self._retrieval.available() and counts["embedded"] > 0,
            **counts,
        }

    async def sync(self) -> int:
        """Embed active memories whose text has no matching vector.

        Safe to call from a background task and safe to call concurrently: the
        lock keeps two callers from paying for the same embeddings, and every
        failure is swallowed because a missing vector only costs retrieval
        quality, never correctness.
        """
        if not await self.available():
            return 0
        async with self._sync_lock:
            pending = await self._db.memories_needing_vectors()
            if not pending:
                return 0
            written = 0
            batch_size = max(1, self._settings.embed_batch)
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                try:
                    vectors = await asyncio.to_thread(
                        self._retrieval.embed,
                        [item["content"] for item in batch],
                        "search_document",
                    )
                except Exception:  # noqa: BLE001 - degrade to the keyword path
                    break
                packed = [
                    (item["id"], _pack(vector), len(vector))
                    for item, vector in zip(batch, vectors, strict=False)
                ]
                written += await self._db.store_memory_vectors(
                    packed, self._settings.oci_embed_model
                )
            return written

    async def search(self, query: str, limit: int = 5) -> list[str]:
        """The best memories for this prompt, by meaning where possible.

        Always returns something usable: an empty semantic result is treated as
        a degraded result and falls through to keyword matching, so enabling the
        feature can never retrieve *less* than it did before.
        """
        query = (query or "").strip()
        if not query:
            return []
        semantic = await self._semantic_search(query, limit)
        if semantic:
            return semantic
        return await self._db.search_memories(query, limit)

    async def _semantic_search(self, query: str, limit: int) -> list[str]:
        if not await self.available():
            return []
        rows = await self._db.memory_search_vectors()
        if not rows:
            return []
        try:
            query_vectors = await asyncio.to_thread(
                self._retrieval.embed, [query], "search_query"
            )
            recall = await asyncio.to_thread(
                self._cosine_recall,
                query_vectors[0],
                rows,
                max(limit * 4, self._settings.corpus_recall_k // 4),
            )
        except Exception:  # noqa: BLE001 - degrade to the keyword path
            return []
        if not recall:
            return []
        contents = await self._db.memories_by_ids([memory_id for memory_id, _ in recall])
        ordered = [
            contents[memory_id] for memory_id, _ in recall if memory_id in contents
        ]
        if not ordered:
            return []
        try:
            ranked = await asyncio.to_thread(
                self._retrieval.rerank, query, ordered, limit
            )
        except Exception:  # noqa: BLE001 - rerank is optional in some regions
            return ordered[:limit]
        return [ordered[index] for index, _ in ranked]

    @staticmethod
    def _cosine_recall(query_vector: list[float], rows, k: int) -> list[tuple[str, float]]:
        import numpy as np

        query = np.asarray(query_vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm
        dim = int(query.shape[0])
        ids: list[str] = []
        vectors: list = []
        for row in rows:
            if int(row["dim"]) != dim:  # a vector from a different embed model
                continue
            ids.append(row["id"])
            vectors.append(np.frombuffer(row["embedding"], dtype=np.float32))
        if not vectors:
            return []
        matrix = np.vstack(vectors)  # stored vectors are already unit-normalized
        similarities = matrix @ query
        top = np.argsort(-similarities)[: max(k, 1)]
        return [(ids[int(i)], float(similarities[int(i)])) for i in top]


def _pack(vector: list[float]) -> bytes:
    import numpy as np

    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm > 0:  # store unit vectors so recall is a plain dot product
        array = array / norm
    return array.astype(np.float32).tobytes()
