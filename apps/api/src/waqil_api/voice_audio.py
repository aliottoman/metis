"""Rendered speech, kept so the same words are never paid for twice.

The morning brief is the case this exists for: it changes once a day and gets
replayed as often as the owner wants to hear it, and re-synthesizing it on
every press would spend a cloud call and several seconds to produce a file
that is already on disk, byte for byte.

Keyed on everything that could change what comes out of the speaker — the
words, the voice, the model, and the scope the caller names (the brief's date,
so a new day is a new recording even if the text happens to match yesterday's).
Not content-addressed like the blob store: this is a cache with a key that
must be computable *before* the expensive call, which is the whole point.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Suffix per container, so the cached file says what it is and a reader does
# not have to guess. Anything unrecognized is stored and returned verbatim.
_SUFFIX_BY_TYPE = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "audio/flac": "flac",
}
_TYPE_BY_SUFFIX = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "opus": "audio/opus",
    "flac": "audio/flac",
}


class SpokenAudioCache:
    """One flat directory of rendered speech, addressed by rendition key."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def key(text: str, *, scope: str, voice_id: str, model: str) -> str:
        """The fingerprint of one rendition.

        Every input that changes the audio is in here, and nothing that does
        not. Change the words, the voice, the model or the day and this is a
        different recording; press the button twice on the same morning and it
        is the same one.
        """
        digest = hashlib.sha256()
        for part in (scope, voice_id, model, text):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def _path(self, key: str, suffix: str) -> Path:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("invalid rendition key")
        return self.root / f"{key}.{suffix}"

    def read(self, key: str) -> tuple[bytes, str] | None:
        """The cached rendition, or None when this is the first time."""
        for suffix, media_type in _TYPE_BY_SUFFIX.items():
            path = self._path(key, suffix)
            if path.is_file():
                try:
                    return path.read_bytes(), media_type
                except OSError:
                    # A half-written or unreadable file is a cache miss, not a
                    # failure: the caller re-renders and overwrites it.
                    return None
        return None

    def write(self, key: str, audio: bytes, media_type: str) -> None:
        """Store one rendition. A cache write never fails the call it serves."""
        suffix = _SUFFIX_BY_TYPE.get(media_type.split(";", 1)[0].strip().lower())
        if suffix is None:
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self._path(key, suffix)
            # Written beside the target and moved into place, so a reader can
            # never observe a partial file under a key that claims to be whole.
            staged = path.with_suffix(f".{suffix}.partial")
            staged.write_bytes(audio)
            staged.replace(path)
        except (OSError, ValueError):
            return
