"""Upload handling with the known failure modes already closed.

Every rule maps to a defect observed in a real generated build: client
filenames used in filesystem paths, every image labelled JPEG regardless of
its bytes, no size cap, and temp files leaking on the error path. The
API is deliberately small — save, use, remove in a finally.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# Magic-byte signatures. The client's declared content type and filename are
# untrusted inputs; the first bytes are the ruling.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"%PDF-", "application/pdf"),
)

IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})


def sniff_mime(data: bytes) -> str:
    """The MIME type the bytes actually are, not what anyone claims."""
    for signature, mime in _SIGNATURES:
        if data.startswith(signature):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


class UploadError(ValueError):
    """The upload is refused; the message is safe to show to a user."""


class _AsyncReadable(Protocol):
    async def read(self, size: int = ..., /) -> bytes: ...


@dataclass(frozen=True)
class SavedUpload:
    """An accepted upload on disk, under a generated name."""

    path: Path
    mime: str
    size: int

    def remove(self) -> None:
        """Delete the temp file; safe to call twice, safe in a finally."""
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


async def save_upload(
    upload: _AsyncReadable,
    *,
    max_bytes: int = 10 * 1024 * 1024,
    allowed_mimes: frozenset[str] = IMAGE_MIMES,
) -> SavedUpload:
    """Read an async upload (FastAPI UploadFile shape) into a private temp file.

    The on-disk name is generated, never the client's. The size cap is
    enforced while streaming, so an oversized body is refused without ever
    being held whole in memory. The MIME check trusts the first bytes only.
    Nothing leaks on refusal — the temp file is removed before the raise;
    callers remove() the accepted result in their own finally.
    """
    handle, raw_path = tempfile.mkstemp(prefix="upload-", suffix=".bin")
    path = Path(raw_path)
    size = 0
    head = b""
    try:
        with os.fdopen(handle, "wb") as sink:
            while True:
                chunk = await upload.read(64 * 1024)
                if not chunk:
                    break
                if len(head) < 16:
                    head = (head + chunk)[:16]
                size += len(chunk)
                if size > max_bytes:
                    raise UploadError(
                        f"upload is larger than the {max_bytes // (1024 * 1024)} MB limit"
                    )
                sink.write(chunk)
        if size == 0:
            raise UploadError("upload is empty")
        mime = sniff_mime(head)
        if allowed_mimes and mime not in allowed_mimes:
            accepted = ", ".join(sorted(allowed_mimes))
            raise UploadError(f"unsupported file type {mime}; accepted: {accepted}")
        return SavedUpload(path=path, mime=mime, size=size)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
