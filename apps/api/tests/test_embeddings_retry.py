"""Transient-failure retry behavior for the OCI Cohere wrapper.

These exercise the real `_with_retry` / `embed` code paths with a stub client
and no network — the exact gap that let a single 'Connection reset by peer'
abort a whole index run.
"""
from __future__ import annotations

import pytest

from waqil_api.config import Settings
from waqil_api.embeddings import CohereRetrieval, _is_transient


def _settings(**kwargs) -> Settings:
    base = dict(
        allow_cloud_embeddings=True,
        oci_compartment_id="ocid1.compartment.oc1..test",
        allow_test_backends=True,
        cloud_retry_base_seconds=0.0,  # no real waiting in tests
        cloud_retry_max_seconds=1.0,
    )
    base.update(kwargs)
    return Settings(**base)


class _Embeddings:
    def __init__(self, vectors):
        self.embeddings = vectors


class _Data:
    def __init__(self, vectors):
        self.data = _Embeddings(vectors)


class _StubClient:
    """Fails transiently a fixed number of times, then returns one vector/input."""

    def __init__(self, fail_times: int, error: Exception):
        self._remaining = fail_times
        self._error = error
        self.calls = 0

    def embed_text(self, details):
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise self._error
        return _Data([[0.1, 0.2, 0.3] for _ in details.inputs])


def test_connection_reset_text_is_classified_transient() -> None:
    reset = ConnectionResetError(54, "Connection reset by peer")
    assert _is_transient(reset)
    assert _is_transient(RuntimeError("Connection aborted."))
    assert _is_transient(RuntimeError("Too Many Requests (429)"))
    # A genuine client error must NOT be retried.
    assert not _is_transient(ValueError("compartment_id is required"))


def test_embed_recovers_after_transient_resets(monkeypatch) -> None:
    monkeypatch.setattr("waqil_api.embeddings.time.sleep", lambda _s: None)
    retrieval = CohereRetrieval(_settings(cloud_max_retries=4))
    stub = _StubClient(
        fail_times=2, error=ConnectionResetError(54, "Connection reset by peer")
    )
    monkeypatch.setattr(retrieval, "_client", lambda: stub)

    vectors = retrieval.embed(["chunk a", "chunk b"], "search_document")

    assert len(vectors) == 2
    assert stub.calls == 3  # two failures, then success


def test_embed_gives_up_after_exhausting_retries(monkeypatch) -> None:
    monkeypatch.setattr("waqil_api.embeddings.time.sleep", lambda _s: None)
    retrieval = CohereRetrieval(_settings(cloud_max_retries=2))
    stub = _StubClient(
        fail_times=99, error=ConnectionResetError(54, "Connection reset by peer")
    )
    monkeypatch.setattr(retrieval, "_client", lambda: stub)

    with pytest.raises(ConnectionResetError):
        retrieval.embed(["chunk"], "search_document")

    assert stub.calls == 3  # 1 initial + 2 retries, then propagate


class _RerankClient:
    """Captures the RerankTextDetails so we can assert the clamped top_n."""

    def __init__(self):
        self.seen_top_n = None

    def rerank_text(self, details):
        self.seen_top_n = details.top_n

        class _Rank:
            def __init__(self, i):
                self.index = i
                self.relevance_score = 1.0 - i * 0.1

        class _Resp:
            data = type("D", (), {"document_ranks": [_Rank(0), _Rank(1)]})()

        return _Resp()


def test_rerank_clamps_top_n_to_document_count(monkeypatch) -> None:
    # Cohere rerank-v4 rejects top_n > len(documents) with a 400. Asking for 8
    # over 2 documents must clamp to 2, not error.
    retrieval = CohereRetrieval(_settings())
    stub = _RerankClient()
    monkeypatch.setattr(retrieval, "_client", lambda endpoint=None: stub)
    ranks = retrieval.rerank("q", ["doc a", "doc b"], top_n=8)
    assert stub.seen_top_n == 2
    assert len(ranks) == 2


def test_embed_does_not_retry_a_non_transient_error(monkeypatch) -> None:
    monkeypatch.setattr("waqil_api.embeddings.time.sleep", lambda _s: None)
    retrieval = CohereRetrieval(_settings(cloud_max_retries=5))
    stub = _StubClient(fail_times=99, error=ValueError("bad request: model not found"))
    monkeypatch.setattr(retrieval, "_client", lambda: stub)

    with pytest.raises(ValueError):
        retrieval.embed(["chunk"], "search_document")

    assert stub.calls == 1  # failed once, not retried
