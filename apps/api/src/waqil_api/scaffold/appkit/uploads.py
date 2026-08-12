"""Upload handling with the known failure modes already closed.

Every rule maps to a defect observed in a real generated build: client
filenames used in filesystem paths, every image labelled JPEG regardless of
its bytes, no size cap, and temp files leaking on the error path. The
API is deliberately small — save, use, remove in a finally.
"""

from __future__ import annotations

import codecs
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
DOCUMENT_MIMES = IMAGE_MIMES | frozenset({"application/pdf", "text/plain"})

# UTF-8 alone is not enough to call a file a text document: shell scripts and
# native executables can have a perfectly ordinary ASCII prefix. Uploaded
# files are never executed by appkit, but refusing these unmistakable forms is
# a cheap, conservative boundary for routes that opt into DOCUMENT_MIMES.
_EXECUTABLE_PREFIXES: tuple[bytes, ...] = (
    b"#!",
    b"MZ",
    b"\x7fELF",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
)
_TEXT_CONTROLS = frozenset(range(0x20)) - {0x09, 0x0A, 0x0D}


def _known_mime(data: bytes) -> str | None:
    """Return a magic-byte MIME, leaving unknown bytes undecided."""
    for signature, mime in _SIGNATURES:
        if data.startswith(signature):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _has_executable_prefix(data: bytes) -> bool:
    candidate = data.removeprefix(b"\xef\xbb\xbf")  # UTF-8 BOM
    return candidate.startswith(_EXECUTABLE_PREFIXES)


def _has_binary_controls(data: bytes) -> bool:
    return any(byte in _TEXT_CONTROLS for byte in data)


def sniff_mime(data: bytes) -> str:
    """The MIME type the bytes actually are, not what anyone claims."""
    known = _known_mime(data)
    if known is not None:
        return known
    if data and not _has_executable_prefix(data) and not _has_binary_controls(data):
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            return "text/plain"
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
    being held whole in memory. Known formats are identified from magic bytes;
    plain text is validated as UTF-8 across the entire stream. Nothing leaks on
    refusal — the temp file is removed before the raise; callers remove() the
    accepted result in their own finally.
    """
    handle, raw_path = tempfile.mkstemp(prefix="upload-", suffix=".bin")
    path = Path(raw_path)
    size = 0
    head = b""
    text_decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    is_plain_text = True
    try:
        with os.fdopen(handle, "wb") as sink:
            while True:
                chunk = await upload.read(64 * 1024)
                if not chunk:
                    break
                if len(head) < 16:
                    head = (head + chunk)[:16]
                if is_plain_text:
                    if _has_binary_controls(chunk):
                        is_plain_text = False
                    else:
                        try:
                            text_decoder.decode(chunk, final=False)
                        except UnicodeDecodeError:
                            is_plain_text = False
                size += len(chunk)
                if size > max_bytes:
                    raise UploadError(
                        f"upload is larger than the {max_bytes // (1024 * 1024)} MB limit"
                    )
                sink.write(chunk)
        if size == 0:
            raise UploadError("upload is empty")
        if is_plain_text:
            try:
                text_decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                is_plain_text = False
        mime = _known_mime(head)
        if mime is None:
            mime = (
                "text/plain"
                if is_plain_text and not _has_executable_prefix(head)
                else "application/octet-stream"
            )
        if allowed_mimes and mime not in allowed_mimes:
            accepted = ", ".join(sorted(allowed_mimes))
            raise UploadError(f"unsupported file type {mime}; accepted: {accepted}")
        return SavedUpload(path=path, mime=mime, size=size)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
