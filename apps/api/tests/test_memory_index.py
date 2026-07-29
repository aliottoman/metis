"""Behavioural tests for semantic long-term memory.

The claim being tested is specific: a memory phrased one way must be findable by
a question phrased another way, which keyword search cannot do. Every test uses
the offline FakeRetrieval so the pipeline is exercised without OCI.
"""
from __future__ import annotations

import hashlib
import re

import pytest

from waqil_api.config import Settings
from waqil_api.database import Database
from waqil_api.memory_index import MemoryIndex


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class SynonymRetrieval:
    """Offline embeddings where a few known synonyms share a bucket.

    Real embeddings put "commit message" and "wording of commits" near each
    other; a bag-of-words hash never would. Collapsing a small synonym set is
    the least magical way to reproduce that property deterministically.
    """

    _SYNONYMS = {
        "wording": "message",
        "messages": "message",
        "commits": "commit",
        "phrase": "message",
        "verify": "test",
        "testing": "test",
        "tests": "test",
    }

    def __init__(self, available: bool = True, dim: int = 32) -> None:
        self._available = available
        self._dim = dim
        self.embed_calls = 0
        self.fail_embed = False

    def available(self) -> bool:
        return self._available

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for raw in _tokens(text):
            token = self._SYNONYMS.get(raw, raw)
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._dim
            vector[bucket] += 1.0
        return vector

    def embed(self, texts, input_type: str = "search_document") -> list[list[float]]:
        if self.fail_embed:
            raise RuntimeError("embedding is unavailable in this region")
        self.embed_calls += 1
        return [self._vector(text) for text in texts]

    def rerank(self, query, documents, top_n=None):
        query_tokens = {self._SYNONYMS.get(token, token) for token in _tokens(query)}
        scored = sorted(
            (
                (
                    index,
                    len(
                        query_tokens
                        & {self._SYNONYMS.get(t, t) for t in _tokens(document)}
                    ),
                )
                for index, document in enumerate(documents)
            ),
            key=lambda pair: -pair[1],
        )
        return [(index, float(score)) for index, score in scored[: (top_n or len(documents))]]


async def _index(tmp_path, **settings_kwargs):
    database = Database(tmp_path / "waqil.db")
    await database.open()
    settings = Settings(
        _env_file=None, data_dir=tmp_path / "data", repo_root=tmp_path, **settings_kwargs
    )
    retrieval = SynonymRetrieval()
    return MemoryIndex(settings, database, retrieval), database, retrieval


async def _remember(database: Database, content: str) -> str:
    proposal = await database.create_memory_proposal("project", content, None, 1.0)
    await database.decide_memory_proposal(proposal.id, "approved", None)
    return proposal.id


@pytest.mark.asyncio
async def test_paraphrased_question_outranks_keyword_noise(tmp_path) -> None:
    index, database, _ = await _index(tmp_path)
    await _remember(database, "I prefer terse commit messages in the imperative mood.")
    await _remember(database, "The staging database is reset every Sunday night.")

    question = "how should you phrase the wording of my commits"

    # The prior behaviour, and the reason this feature exists: the question
    # shares no content word with the memory that answers it, so BM25 ranks on
    # incidental overlap and puts an unrelated memory first.
    keyword = await database.search_memories(question)
    assert "terse commit messages" not in keyword[0]

    await index.set_consent(True, "unit test")
    found = await index.search(question)
    assert "terse commit messages" in found[0]


@pytest.mark.asyncio
async def test_nothing_is_embedded_without_consent(tmp_path) -> None:
    index, database, retrieval = await _index(tmp_path)
    await _remember(database, "I prefer terse commit messages in the imperative mood.")

    assert await index.sync() == 0
    assert retrieval.embed_calls == 0
    stats = await index.stats()
    assert stats["consent"] is False and stats["embedded"] == 0
    assert stats["semantic"] is False

    # Retrieval still works; it is simply the keyword path it always was.
    assert await index.search("terse commit messages") != []


@pytest.mark.asyncio
async def test_revoking_consent_purges_every_vector(tmp_path) -> None:
    index, database, _ = await _index(tmp_path)
    await _remember(database, "I prefer terse commit messages in the imperative mood.")
    await index.set_consent(True, "granted")
    assert (await index.stats())["embedded"] == 1

    await index.set_consent(False, "withdrawn")
    stats = await index.stats()
    assert stats["embedded"] == 0
    assert stats["consent"] is False
    # Nothing cloud-derived may outlive the consent that produced it.
    assert await database.memory_search_vectors() == []


@pytest.mark.asyncio
async def test_editing_a_memory_invalidates_its_stored_vector(tmp_path) -> None:
    index, database, _ = await _index(tmp_path)
    await _remember(database, "The deploy target is staging.")
    await index.set_consent(True, "granted")
    assert await index.sync() == 0  # already current

    def rewrite() -> None:
        with database._transaction() as conn:  # noqa: SLF001 - exercising staleness
            conn.execute(
                "UPDATE memory_items SET content = ? WHERE content = ?",
                ("The deploy target is production.", "The deploy target is staging."),
            )

    await database._call(rewrite)  # noqa: SLF001

    # A vector built from the old wording must not keep deciding retrieval.
    assert len(await database.memories_needing_vectors()) == 1
    assert await index.sync() == 1


@pytest.mark.asyncio
async def test_an_embedding_failure_degrades_to_keyword_search(tmp_path) -> None:
    index, database, retrieval = await _index(tmp_path)
    await _remember(database, "I prefer terse commit messages in the imperative mood.")
    await index.set_consent(True, "granted")

    retrieval.fail_embed = True
    # A degraded region must cost retrieval quality, never the turn itself.
    found = await index.search("terse commit messages")
    assert found and "terse commit messages" in found[0]


@pytest.mark.asyncio
async def test_semantic_status_is_honest_about_being_off(tmp_path) -> None:
    index, database, retrieval = await _index(tmp_path)
    await _remember(database, "A durable preference worth keeping.")
    await index.set_consent(True, "granted")
    assert (await index.stats())["semantic"] is True

    # Consent alone is not the same as working: say so rather than implying it.
    retrieval._available = False
    stats = await index.stats()
    assert stats["consent"] is True
    assert stats["cloud_available"] is False
    assert stats["semantic"] is False
