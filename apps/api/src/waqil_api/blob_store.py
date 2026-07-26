from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterable, Callable


@dataclass(frozen=True, slots=True)
class StoredBlob:
    sha256: str
    size: int
    path: Path


class BlobTooLargeError(ValueError):
    pass


class BlobStore:
    """Content-addressed immutable blob storage."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, digest: str) -> Path:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid SHA-256 digest")
        path = (self.root / digest[:2] / digest[2:]).resolve()
        if self.root not in path.parents:
            raise ValueError("blob path escaped storage root")
        return path

    async def put_stream(
        self,
        chunks: AsyncIterable[bytes],
        *,
        max_bytes: int,
        validate_staged: Callable[[Path], None] | None = None,
    ) -> StoredBlob:
        temp_dir = self.root / ".tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"upload-{os.getpid()}-{uuid.uuid4().hex}"
        digest = hashlib.sha256()
        size = 0
        try:
            with temp_path.open("xb") as handle:
                async for chunk in chunks:
                    size += len(chunk)
                    if size > max_bytes:
                        raise BlobTooLargeError(f"upload exceeds {max_bytes} bytes")
                    digest.update(chunk)
                await asyncio.to_thread(handle.write, chunk)
                await asyncio.to_thread(handle.flush)
                await asyncio.to_thread(os.fsync, handle.fileno())
            if validate_staged is not None:
                await asyncio.to_thread(validate_staged, temp_path)
            hex_digest = digest.hexdigest()
            target = self.path_for(hex_digest)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                temp_path.unlink(missing_ok=True)
            else:
                temp_path.replace(target)
            return StoredBlob(hex_digest, size, target)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

    async def put_bytes(self, content: bytes, *, max_bytes: int) -> StoredBlob:
        async def chunks() -> AsyncIterable[bytes]:
            yield content

        return await self.put_stream(chunks(), max_bytes=max_bytes)

    def open(self, digest: str):
        return self.path_for(digest).open("rb")
