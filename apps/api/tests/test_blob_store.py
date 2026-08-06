"""Content-addressed blob storage: what is stored must be what was sent."""
from __future__ import annotations

import hashlib

import pytest

from waqil_api.blob_store import BlobStore, BlobTooLargeError


async def test_multi_chunk_stream_stores_every_chunk(tmp_path) -> None:
    """The regression: uploads arrive in 64 KB chunks, and a write outside the
    loop stored only the last one — under a digest computed over all of them,
    so the content-address silently lied about the content."""
    store = BlobStore(tmp_path)
    parts = [b"a" * 1024, b"b" * 1024, b"c" * 7]
    payload = b"".join(parts)

    async def chunks():
        for part in parts:
            yield part

    blob = await store.put_stream(chunks(), max_bytes=1_000_000)
    assert blob.size == len(payload)
    assert blob.path.read_bytes() == payload
    assert blob.sha256 == hashlib.sha256(payload).hexdigest()


async def test_put_bytes_round_trips(tmp_path) -> None:
    store = BlobStore(tmp_path)
    blob = await store.put_bytes(b"metis", max_bytes=1024)
    assert blob.sha256 == hashlib.sha256(b"metis").hexdigest()
    with store.open(blob.sha256) as handle:
        assert handle.read() == b"metis"


async def test_oversized_stream_is_refused_and_leaves_nothing_behind(tmp_path) -> None:
    store = BlobStore(tmp_path)

    async def chunks():
        yield b"x" * 64
        yield b"y" * 64

    with pytest.raises(BlobTooLargeError):
        await store.put_stream(chunks(), max_bytes=100)
    assert list((tmp_path / ".tmp").glob("upload-*")) == []


def test_path_for_rejects_a_non_digest(tmp_path) -> None:
    store = BlobStore(tmp_path)
    with pytest.raises(ValueError):
        store.path_for("../../etc/passwd")
